"""Phase 2: extract and cache residual-stream activations across all four
datasets (AdvBench, HarmBench, JBB-Behaviors, XSTest) for a given model.

This is the shared activation cache that Phase 3+ (SAE-feature detector,
baselines, evaluation) will build on, so nothing downstream has to re-run
the model. As a sanity check, it also reports per-layer harmful/harmless
separation on the full combined set and compares the best layer against
what Phase 1 found using only AdvBench+Alpaca.

Usage: python scripts/03_extract_all_activations.py <model_name>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import extract_and_cache
from src.activations.extract import load_model
from src.data.loaders import load_all_labeled_prompts
from src.direction.compute import compute_directions, select_candidate_layers, separation_score

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    print(f"Loading model: {model_name}")
    model = load_model(model_name)

    print("Loading labeled prompts from all four datasets")
    records = load_all_labeled_prompts()
    n_harmful = sum(r["label"] == "harmful" for r in records)
    n_harmless = sum(r["label"] == "harmless" for r in records)
    print(f"Total: {len(records)} ({n_harmful} harmful, {n_harmless} harmless)")
    from collections import Counter
    print("By source/label:", Counter((r["source"], r["label"]) for r in records))

    print("\nExtracting activations (this covers all prompts, may take a few minutes)")
    path = extract_and_cache(model, model_name, records)
    print(f"Cached activations to {path}")

    # Sanity check: per-layer separation on the full combined dataset.
    from src.activations.cache import load_cache
    cache = load_cache(model_name)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)

    harmful_acts = acts[:, labels, :]
    harmless_acts = acts[:, ~labels, :]
    directions = compute_directions(harmful_acts, harmless_acts)
    scores = separation_score(acts, directions, labels)
    candidates = select_candidate_layers(scores, k=5)
    print("\nTop-5 layers by separation score on FULL combined dataset:")
    for layer in candidates:
        print(f"  layer {layer}: score={scores[layer].item():.3f}")


if __name__ == "__main__":
    main()
