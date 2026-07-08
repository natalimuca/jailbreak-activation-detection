import json

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


def test_manifest_flags_unknown_records_instead_of_dropping_silently():
    records = _fake_corpus()
    split = stratified_split(records, seed=0)
    manifest = {name: [content_hash(r["text"]) for r in group] for name, group in split.items()}

    unseen_record = {"text": "a totally new prompt not in the manifest", "label": "harmful", "source": "advbench"}
    reapplied = apply_manifest(records + [unseen_record], manifest)
    assert len(reapplied["unassigned"]) == 1
    assert reapplied["unassigned"][0]["text"] == unseen_record["text"]
