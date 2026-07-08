"""Stratified train/val/test split of the deduplicated multi-dataset corpus.

Stratifies by (source, label) so each split preserves the same dataset/label
mix as the full corpus -- otherwise a random split could, say, put almost all
of XSTest's hard safe-but-scary prompts in train and leave test unable to
measure over-refusal at all.

The split is saved as a manifest of content hashes (not raw text -- these are
still real harmful-instruction strings and there's no reason to duplicate them
into a second committed file when the source datasets already publish them).
Reproducing the split later means: re-run the loaders + dedup (deterministic
given the same seed and unchanged upstream data), hash each record, and look
up its assigned split in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "splits" / "corpus_split_v1.json"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def stratified_split(
    records: list[dict], train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 0
) -> dict[str, list[dict]]:
    """records: list of {"text", "label", "source"}. Splits within each
    (source, label) stratum independently so proportions are preserved."""
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        strata[(r["source"], r["label"])].append(r)

    split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for key, group in strata.items():
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        split["train"].extend(group[:n_train])
        split["val"].extend(group[n_train : n_train + n_val])
        split["test"].extend(group[n_train + n_val :])

    return split


def save_manifest(split: dict[str, list[dict]], path: Path = MANIFEST_PATH) -> None:
    manifest = {name: [content_hash(r["text"]) for r in records] for name, records in split.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, list[str]]:
    return json.loads(path.read_text())


def apply_manifest(records: list[dict], manifest: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Assigns records to splits by looking up each record's content hash in
    the manifest.

    Output order follows the manifest's own stored hash order for each
    split, NOT the order `records` happens to be in. This matters:
    `stratified_split()` shuffles records within each stratum before
    slicing, so the first time a manifest is saved it captures that shuffled
    order. If `apply_manifest` just preserved input order instead, a second
    call (e.g. extracting activations for a second model, in a fresh
    process where `records` naturally comes out in unshuffled dedup order)
    would silently produce train/val/test lists containing the *same set*
    of prompts but in a *different order* than the first call -- which
    breaks any code that assumes matching indices across two separately
    generated caches (exactly what `assert_caches_consistent` checks for).
    """
    hash_to_record: dict[str, dict] = {content_hash(r["text"]): r for r in records}

    result: dict[str, list[dict]] = {}
    accounted: set[str] = set()
    for split_name, hashes in manifest.items():
        result[split_name] = []
        for h in hashes:
            if h in hash_to_record:
                result[split_name].append(hash_to_record[h])
                accounted.add(h)
    # Records present now but not in the manifest at all (e.g. upstream
    # dataset changed) -- collected separately rather than silently dropped.
    result["unassigned"] = [r for r in records if content_hash(r["text"]) not in accounted]
    return result
