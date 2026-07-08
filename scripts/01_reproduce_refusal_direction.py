"""Phase 1 milestone: reproduce the causal refusal-direction finding
(Arditi et al., 2024) on a small open-weight model.

Pipeline:
  1. Load harmful (AdvBench) / harmless (Alpaca) instruction pairs.
  2. Extract last-token residual-stream activations at every layer.
  3. Compute per-layer difference-of-means direction, pick candidate layers
     via a cheap separation-score proxy.
  4. Causally validate the top candidate: directional ablation should
     suppress refusal on harmful prompts; activation addition should induce
     refusal on harmless prompts.
  5. Save refusal-rate results to results/phase1_reproduction.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import get_last_token_resid_acts, load_model
from src.data.loaders import load_harmful_harmless_split
from src.direction.compute import (
    compute_directions,
    compute_raw_directions,
    select_candidate_layers,
    separation_score,
)
from src.direction.interventions import generate_baseline, generate_with_ablation, generate_with_addition
from src.direction.refusal_classifier import refusal_stats

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    model_slug = model_name.split("/")[-1]
    results_path = RESULTS_DIR / f"phase1_reproduction_{model_slug}.json"

    print(f"Loading model: {model_name}")
    model = load_model(model_name)

    print("Loading harmful/harmless prompt split")
    split = load_harmful_harmless_split(n_train=200, n_val=30)

    print("Extracting activations (train split)")
    harmful_train_acts = get_last_token_resid_acts(model, split["harmful_train"])
    harmless_train_acts = get_last_token_resid_acts(model, split["harmless_train"])

    print("Computing per-layer directions")
    directions = compute_directions(harmful_train_acts, harmless_train_acts)
    raw_directions = compute_raw_directions(harmful_train_acts, harmless_train_acts)

    print("Extracting activations (val split) for layer selection")
    harmful_val_acts = get_last_token_resid_acts(model, split["harmful_val"])
    harmless_val_acts = get_last_token_resid_acts(model, split["harmless_val"])
    all_val_acts = torch.cat([harmful_val_acts, harmless_val_acts], dim=1)
    labels = torch.cat([
        torch.ones(len(split["harmful_val"]), dtype=torch.bool),
        torch.zeros(len(split["harmless_val"]), dtype=torch.bool),
    ])
    scores = separation_score(all_val_acts, directions, labels)
    candidates = select_candidate_layers(scores, k=3)
    print(f"Top candidate layers by separation score: {candidates}")
    print(f"Scores: {[round(scores[i].item(), 3) for i in candidates]}")

    best_layer = candidates[0]
    direction = directions[best_layer]
    raw_direction = raw_directions[best_layer]
    print(f"Raw direction norm at layer {best_layer}: {raw_direction.norm().item():.3f}")

    print(f"\n== Causal validation at layer {best_layer} ==")

    print("Baseline completions on harmful val prompts")
    baseline_harmful = [generate_baseline(model, p) for p in split["harmful_val"]]
    print("Ablated completions on harmful val prompts")
    ablated_harmful = [generate_with_ablation(model, p, direction) for p in split["harmful_val"]]

    print("Baseline completions on harmless val prompts")
    baseline_harmless = [generate_baseline(model, p) for p in split["harmless_val"]]
    print("Direction-added completions on harmless val prompts")
    added_harmless = [
        generate_with_addition(model, p, raw_direction, layer_idx=best_layer) for p in split["harmless_val"]
    ]

    results = {
        "model": model_name,
        "best_layer": best_layer,
        "candidate_layers": candidates,
        "separation_scores": {i: round(scores[i].item(), 4) for i in candidates},
        "refusal_rate": {
            "harmful_baseline": refusal_stats(baseline_harmful),
            "harmful_ablated": refusal_stats(ablated_harmful),
            "harmless_baseline": refusal_stats(baseline_harmless),
            "harmless_direction_added": refusal_stats(added_harmless),
        },
        "samples": {
            "harmful_baseline": list(zip(split["harmful_val"], baseline_harmful)),
            "harmful_ablated": list(zip(split["harmful_val"], ablated_harmful)),
            "harmless_baseline": list(zip(split["harmless_val"], baseline_harmless)),
            "harmless_direction_added": list(zip(split["harmless_val"], added_harmless)),
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))

    print("\n== Results ==")
    print(json.dumps(results["refusal_rate"], indent=2))
    print(f"\nFull results saved to {results_path}")

    rr = results["refusal_rate"]
    reproduced = (
        rr["harmful_baseline"]["rate"] > rr["harmful_ablated"]["rate"]
        and rr["harmless_direction_added"]["rate"] > rr["harmless_baseline"]["rate"]
    )
    print(f"\nFinding reproduced: {reproduced}")


if __name__ == "__main__":
    main()
