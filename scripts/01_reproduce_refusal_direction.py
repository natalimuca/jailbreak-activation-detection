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
from src.direction.compute import compute_directions, select_candidate_layers, separation_score
from src.direction.interventions import generate_baseline, generate_with_ablation, generate_with_addition
from src.direction.refusal_classifier import refusal_rate

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "phase1_reproduction.json"


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = load_model(MODEL_NAME)

    print("Loading harmful/harmless prompt split")
    split = load_harmful_harmless_split(n_train=24, n_val=8)

    print("Extracting activations (train split)")
    harmful_train_acts = get_last_token_resid_acts(model, split["harmful_train"])
    harmless_train_acts = get_last_token_resid_acts(model, split["harmless_train"])

    print("Computing per-layer directions")
    directions = compute_directions(harmful_train_acts, harmless_train_acts)

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

    print(f"\n== Causal validation at layer {best_layer} ==")

    print("Baseline completions on harmful val prompts")
    baseline_harmful = [generate_baseline(model, p) for p in split["harmful_val"]]
    print("Ablated completions on harmful val prompts")
    ablated_harmful = [generate_with_ablation(model, p, direction) for p in split["harmful_val"]]

    print("Baseline completions on harmless val prompts")
    baseline_harmless = [generate_baseline(model, p) for p in split["harmless_val"]]
    print("Direction-added completions on harmless val prompts")
    added_harmless = [
        generate_with_addition(model, p, direction, layer_idx=best_layer) for p in split["harmless_val"]
    ]

    results = {
        "model": MODEL_NAME,
        "best_layer": best_layer,
        "candidate_layers": candidates,
        "separation_scores": {i: round(scores[i].item(), 4) for i in candidates},
        "refusal_rate": {
            "harmful_baseline": refusal_rate(baseline_harmful),
            "harmful_ablated": refusal_rate(ablated_harmful),
            "harmless_baseline": refusal_rate(baseline_harmless),
            "harmless_direction_added": refusal_rate(added_harmless),
        },
        "samples": {
            "harmful_baseline": list(zip(split["harmful_val"], baseline_harmful)),
            "harmful_ablated": list(zip(split["harmful_val"], ablated_harmful)),
            "harmless_baseline": list(zip(split["harmless_val"], baseline_harmless)),
            "harmless_direction_added": list(zip(split["harmless_val"], added_harmless)),
        },
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n== Results ==")
    print(json.dumps(results["refusal_rate"], indent=2))
    print(f"\nFull results saved to {RESULTS_PATH}")

    rr = results["refusal_rate"]
    reproduced = rr["harmful_baseline"] > rr["harmful_ablated"] and rr["harmless_direction_added"] > rr["harmless_baseline"]
    print(f"\nFinding reproduced: {reproduced}")


if __name__ == "__main__":
    main()
