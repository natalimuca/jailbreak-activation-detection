"""Extract and cache residual-stream activations across all four datasets
(AdvBench, HarmBench, JBB-Behaviors, XSTest) for a given model.

Loads the combined corpus, deduplicates it, assigns each record to
train/val/test via the saved split manifest (generating one if it doesn't
exist yet), then extracts activations for every record and caches them
(with the split tag) to disk for downstream work to build on without
re-running the model.

As a sanity check, it computes the harmful/harmless direction on the TRAIN
split only and reports per-layer separation on the held-out VAL split (never
mixing the two, unlike a naive single-split check).

Usage: python scripts/03_extract_all_activations.py <model_name>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import extract_and_cache, load_cache
from src.activations.extract import load_model
from src.data.dedup import deduplicate
from src.data.loaders import load_all_labeled_prompts
from src.data.splits import MANIFEST_PATH, apply_manifest, load_manifest, save_manifest, stratified_split
from src.direction.compute import compute_directions, select_candidate_layers, separation_score

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def get_split_records() -> dict[str, list[dict]]:
    records = load_all_labeled_prompts()
    kept, dropped = deduplicate(records)
    print(f"Loaded {len(records)}, deduplicated to {len(kept)} ({len(dropped)} dropped)")

    if MANIFEST_PATH.exists():
        manifest = load_manifest()
        split = apply_manifest(kept, manifest)
        if split["unassigned"]:
            print(f"WARNING: {len(split['unassigned'])} records not found in the saved manifest "
                  f"(upstream dataset likely changed) -- these are excluded from train/val/test.")
        del split["unassigned"]
    else:
        print("No split manifest found, generating one")
        split = stratified_split(kept)
        save_manifest(split)

    return split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    p.add_argument("--4bit", dest="load_in_4bit", action="store_true", help="Load model in 4-bit (bitsandbytes) -- needed for 7-9B models on a 6GB GPU.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model

    print(f"Loading model: {model_name}" + (" (4-bit)" if args.load_in_4bit else ""))
    model = load_model(model_name, load_in_4bit=args.load_in_4bit)

    split = get_split_records()
    all_records = []
    for split_name, records in split.items():
        for r in records:
            all_records.append({**r, "split": split_name})
        print(f"  {split_name}: {len(records)} ({Counter((r['source'], r['label']) for r in records)})")

    print(f"\nExtracting activations for {len(all_records)} prompts (may take a few minutes)")
    path = extract_and_cache(model, model_name, all_records)
    print(f"Cached activations to {path}")

    # Sanity check: direction estimated on train, separation measured on held-out val.
    cache = load_cache(model_name)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    splits = cache["splits"]
    is_train = torch.tensor([s == "train" for s in splits], dtype=torch.bool)
    is_val = torch.tensor([s == "val" for s in splits], dtype=torch.bool)

    directions = compute_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])

    val_acts = acts[:, is_val, :]
    val_labels = labels[is_val]
    scores = separation_score(val_acts, directions, val_labels)
    candidates = select_candidate_layers(scores, k=5)
    print("\nTop-5 layers by separation score (direction from train, measured on held-out val):")
    for layer in candidates:
        print(f"  layer {layer}: score={scores[layer].item():.3f}")


if __name__ == "__main__":
    main()
