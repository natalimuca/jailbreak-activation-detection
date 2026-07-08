import json
import random

from src.data.splits import apply_manifest, content_hash, save_manifest, stratified_split


def _fake_corpus():
    records = []
    for i in range(20):
        records.append({"text": f"harmful prompt {i}", "label": "harmful", "source": "advbench"})
    for i in range(20):
        records.append({"text": f"harmless prompt {i}", "label": "harmless", "source": "alpaca"})
    return records


def test_split_covers_every_record_exactly_once():
    records = _fake_corpus()
    split = stratified_split(records, train_frac=0.6, val_frac=0.2, seed=0)
    all_texts = [r["text"] for group in split.values() for r in group]
    assert sorted(all_texts) == sorted(r["text"] for r in records)
    assert len(all_texts) == len(set(all_texts))


def test_split_is_stratified_by_source_and_label():
    records = _fake_corpus()
    split = stratified_split(records, train_frac=0.6, val_frac=0.2, seed=0)
    for name in ("train", "val", "test"):
        labels = {r["label"] for r in split[name]}
        assert labels == {"harmful", "harmless"}, f"{name} split lost a stratum"


def test_split_is_deterministic_given_same_seed():
    records = _fake_corpus()
    split_a = stratified_split(records, seed=42)
    split_b = stratified_split(records, seed=42)
    assert [r["text"] for r in split_a["train"]] == [r["text"] for r in split_b["train"]]


def test_manifest_round_trip(tmp_path):
    records = _fake_corpus()
    split = stratified_split(records, seed=0)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(split, path=manifest_path)

    manifest = json.loads(manifest_path.read_text())
    reapplied = apply_manifest(records, manifest)

    assert len(reapplied["unassigned"]) == 0
    for name in ("train", "val", "test"):
        assert sorted(r["text"] for r in reapplied[name]) == sorted(r["text"] for r in split[name])


def test_manifest_round_trip_preserves_exact_order():
    # Regression test: stratified_split() shuffles records within each
    # stratum, so the manifest captures that shuffled order. A real bug had
    # apply_manifest() silently fall back to *input* order instead of the
    # manifest's own order -- same set of records per split, but different
    # order, which breaks any code assuming matching indices across two
    # separately generated caches (see assert_caches_consistent).
    records = _fake_corpus()
    split = stratified_split(records, seed=0)
    manifest = {name: [content_hash(r["text"]) for r in group] for name, group in split.items()}

    reapplied = apply_manifest(records, manifest)
    for name in ("train", "val", "test"):
        assert [r["text"] for r in reapplied[name]] == [r["text"] for r in split[name]]


def test_manifest_reapplication_is_order_independent_of_input_order():
    # The whole point of the fix: even if a second process presents
    # `records` in a completely different order (e.g. unshuffled dedup
    # output vs. the originally-shuffled split), applying the same manifest
    # must produce the exact same per-split order both times.
    records = _fake_corpus()
    split = stratified_split(records, seed=0)
    manifest = {name: [content_hash(r["text"]) for r in group] for name, group in split.items()}

    shuffled_records = records[:]
    random.Random(123).shuffle(shuffled_records)

    reapplied_original_order = apply_manifest(records, manifest)
    reapplied_shuffled_order = apply_manifest(shuffled_records, manifest)

    for name in ("train", "val", "test"):
        assert [r["text"] for r in reapplied_original_order[name]] == [
            r["text"] for r in reapplied_shuffled_order[name]
        ]


def test_manifest_flags_unknown_records_instead_of_dropping_silently():
    records = _fake_corpus()
    split = stratified_split(records, seed=0)
    manifest = {name: [content_hash(r["text"]) for r in group] for name, group in split.items()}

    unseen_record = {"text": "a totally new prompt not in the manifest", "label": "harmful", "source": "advbench"}
    reapplied = apply_manifest(records + [unseen_record], manifest)
    assert len(reapplied["unassigned"]) == 1
    assert reapplied["unassigned"][0]["text"] == unseen_record["text"]
