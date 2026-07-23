"""Calibrate the activation-addition coefficient (alpha) per model.

Phase 1 found that alpha=1.0 (adding the raw, unnormalized mean-difference
direction once) induces refusal strongly on Qwen2.5-1.5B-Instruct but only
weakly on SmolLM2-1.7B-Instruct, despite both using the same "natural
magnitude" direction. This sweeps alpha on a calibration split (disjoint
from both the direction-training set and the held-out val split used for
final reporting) to find, per model, the smallest alpha that induces a high
refusal rate without collapsing generation into degenerate repeated tokens.

Usage: python scripts/calibrate_alpha.py <model_name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import get_last_token_resid_acts, load_model
from src.data.loaders import load_harmful_harmless_split
from src.direction.compute import compute_raw_directions, select_candidate_layers, separation_score, compute_directions
from src.direction.interventions import generate_with_addition
from src.direction.refusal_classifier import is_degenerate, is_refusal

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ALPHA_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
TARGET_RATE = 0.8
MAX_DEGENERATE_FRAC = 0.1


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    model_slug = model_name.split("/")[-1]

    print(f"Loading model: {model_name}")
    model = load_model(model_name)

    print("Loading harmful/harmless prompt split (train + calib only)")
    split = load_harmful_harmless_split(n_train=200, n_calib=12, n_val=0)

    print("Extracting activations (train split)")
    harmful_train_acts = get_last_token_resid_acts(model, split["harmful_train"])
    harmless_train_acts = get_last_token_resid_acts(model, split["harmless_train"])

    directions = compute_directions(harmful_train_acts, harmless_train_acts)
    raw_directions = compute_raw_directions(harmful_train_acts, harmless_train_acts)

    print("Extracting activations (calib split) for layer selection")
    harmful_calib_acts = get_last_token_resid_acts(model, split["harmful_calib"])
    harmless_calib_acts = get_last_token_resid_acts(model, split["harmless_calib"])
    all_calib_acts = torch.cat([harmful_calib_acts, harmless_calib_acts], dim=1)
    labels = torch.cat([
        torch.ones(len(split["harmful_calib"]), dtype=torch.bool),
        torch.zeros(len(split["harmless_calib"]), dtype=torch.bool),
    ])
    scores = separation_score(all_calib_acts, directions, labels)
    best_layer = select_candidate_layers(scores, k=1)[0]
    raw_direction = raw_directions[best_layer]
    print(f"Best layer: {best_layer}, raw direction norm: {raw_direction.norm().item():.3f}")

    print(f"\nSweeping alpha over {ALPHA_GRID} on {len(split['harmless_calib'])} calib prompts")
    sweep = []
    for alpha in ALPHA_GRID:
        completions = [
            generate_with_addition(model, p, raw_direction, layer_idx=best_layer, alpha=alpha)
            for p in split["harmless_calib"]
        ]
        n = len(completions)
        n_refusal = sum(is_refusal(c) for c in completions)
        n_degenerate = sum(is_degenerate(c) for c in completions)
        row = {
            "alpha": alpha,
            "refusal_rate": round(n_refusal / n, 4),
            "degenerate_frac": round(n_degenerate / n, 4),
            "sample_completion": completions[0][:100],
        }
        sweep.append(row)
        print(f"  alpha={alpha:5.2f}  refusal_rate={row['refusal_rate']:.2f}  degenerate_frac={row['degenerate_frac']:.2f}")

    # Pick the smallest alpha that clears the target refusal rate without
    # degenerating; fall back to the highest non-degenerate refusal rate.
    viable = [r for r in sweep if r["degenerate_frac"] <= MAX_DEGENERATE_FRAC]
    above_target = [r for r in viable if r["refusal_rate"] >= TARGET_RATE]
    if above_target:
        calibrated = min(above_target, key=lambda r: r["alpha"])
    elif viable:
        calibrated = max(viable, key=lambda r: r["refusal_rate"])
    else:
        calibrated = min(sweep, key=lambda r: r["alpha"])  # everything degenerates; pick smallest

    result = {
        "model": model_name,
        "best_layer": best_layer,
        "raw_direction_norm": round(raw_direction.norm().item(), 4),
        "sweep": sweep,
        "calibrated_alpha": calibrated["alpha"],
        "calibrated_refusal_rate": calibrated["refusal_rate"],
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"alpha_calibration_{model_slug}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\nCalibrated alpha for {model_name}: {calibrated['alpha']} (refusal_rate={calibrated['refusal_rate']})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
