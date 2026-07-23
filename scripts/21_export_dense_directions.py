"""Precomputes and persists each model's TRAIN-derived dense refusal
direction as a small standalone artifact (`results/dense_directions.pt`),
for Phase 7's interactive UI backend.

Every existing script (scripts/06, 10, 12, 14) recomputes the direction
on the fly from that model's full activation cache
(`results/activations/<model>.pt`, ~1GB each, gitignored) -- fine for a
one-shot batch analysis, but the UI backend needs to score arbitrary live
prompts on demand and can't justify loading a multi-GB cache into memory
per request just to re-derive a d_model-sized vector that never changes.
This script computes each model's direction once (from the same cached
TRAIN-split activations every other script already uses, via
`resolve_layer_for_model` for the layer choice -- no new selection logic)
and saves just the small (layer, direction) results, mirroring how
scripts/10 persists calibrated thresholds instead of recalibrating at
use time.

Usage: python scripts/21_export_dense_directions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.detectors.dense_direction_detector import resolve_layer_for_model
from src.direction.compute import compute_directions

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_PATH = RESULTS_DIR / "dense_directions.pt"

MODELS = {
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}


def export_direction(cache_label: str, hf_model_name: str) -> dict:
    cache = load_cache(cache_label)
    acts = cache["activations"]
    labels_t = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_train = torch.tensor([s == "train" for s in cache["splits"]], dtype=torch.bool)

    layer = resolve_layer_for_model(hf_model_name)
    direction = compute_directions(
        acts[layer : layer + 1, is_train & labels_t, :],
        acts[layer : layer + 1, is_train & ~labels_t, :],
    )[0]
    return {"layer": layer, "direction": direction}


def main() -> None:
    out = {}
    for cache_label, hf_model_name in MODELS.items():
        print(f"{cache_label}: loading cache, computing direction...")
        out[cache_label] = export_direction(cache_label, hf_model_name)
        print(f"  layer {out[cache_label]['layer']}, direction norm {out[cache_label]['direction'].norm():.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT_PATH)
    print(f"\nSaved {len(out)} directions to {OUT_PATH}")


if __name__ == "__main__":
    main()
