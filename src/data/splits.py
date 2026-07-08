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
    the manifest. Records whose hash isn't in the manifest (e.g. upstream
    dataset changed) are collected separately rather than silently dropped."""
    hash_to_split: dict[str, str] = {}
    for split_name, hashes in manifest.items():
        for h in hashes:
            hash_to_split[h] = split_name

    result: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "unassigned": []}
    for r in records:
        h = content_hash(r["text"])
        result[hash_to_split.get(h, "unassigned")].append(r)
    return result
