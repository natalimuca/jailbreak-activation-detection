"""Extends Phase 1's sufficiency (activation-addition) test to 7-9B scale
(Qwen3-8B, Llama-3.1-8B-Instruct) -- closes a gap flagged in RESULTS.md's
"Known limitations": necessity (ablation) has since been tested at this
scale (Wave 2, `scripts/17`'s cross-model-transfer test) but sufficiency
hadn't, and remained tested only on the two small Phase 1 models
(Qwen2.5-1.5B, SmolLM2-1.7B).

Reuses each model's already-selected layer (never re-derives layer
selection) and the already-cached full-corpus activations
(`results/activations/<model>.pt`) -- no new activation extraction needed,
only new generation. Mirrors Phase 1's split discipline (scripts/01 + 02)
adapted to this project's train/val/test scheme (no separate "calib"
split exists here, unlike Phase 1's original small-corpus 3-way split):

  1. Alpha-sweep calibration on a modest VAL-split harmless sample
     (n=12, matching scripts/02's calib-split size) -- picks the smallest
     alpha that induces a high refusal rate without degenerating.
  2. Final causal-validation generation test on a TEST-split harmless
     sample (n=50, this project's standing validation-sample convention),
     completely disjoint from the alpha-sweep prompts -- baseline vs.
     addition at the calibrated alpha.

Both models get the identical sampled prompt texts (same corpus, same
splits, same seed) for direct comparability -- not literally the same
activations, since those are model-specific.

Usage: python scripts/23_sufficiency_7b_9b_scale.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.activations.extract import load_model
from src.detectors.dense_direction_detector import resolve_layer_for_model
from src.direction.compute import compute_raw_directions
from src.direction.interventions import generate_baseline, generate_with_addition
from src.direction.refusal_classifier import is_degenerate, is_refusal, refusal_stats

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
ALPHA_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
TARGET_RATE = 0.8
MAX_DEGENERATE_FRAC = 0.1
N_ALPHA_SWEEP = 12
N_VALIDATION = 50
MAX_PROMPT_CHARS = 150
MAX_NEW_TOKENS = 40
SEED = 11


def harmless_prompts(cache: dict, split_name: str, n: int, rng: random.Random) -> list[str]:
    eligible = [
        t for t, lab, sp in zip(cache["texts"], cache["labels"], cache["splits"])
        if lab == "harmless" and sp == split_name and len(t) <= MAX_PROMPT_CHARS
    ]
    return rng.sample(eligible, min(n, len(eligible)))


def calibrate_alpha(model, prompts: list[str], raw_direction: torch.Tensor, layer: int) -> tuple[dict, list[dict]]:
    sweep = []
    for alpha in ALPHA_GRID:
        completions = [
            generate_with_addition(model, p, raw_direction, layer_idx=layer, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS)
            for p in prompts
        ]
        n = len(completions)
        n_refusal = sum(is_refusal(c) for c in completions)
        n_degenerate = sum(is_degenerate(c) for c in completions)
        row = {"alpha": alpha, "refusal_rate": round(n_refusal / n, 4), "degenerate_frac": round(n_degenerate / n, 4)}
        sweep.append(row)
        print(f"    alpha={alpha:5.2f}  refusal_rate={row['refusal_rate']:.2f}  degenerate_frac={row['degenerate_frac']:.2f}")

    viable = [r for r in sweep if r["degenerate_frac"] <= MAX_DEGENERATE_FRAC]
    above_target = [r for r in viable if r["refusal_rate"] >= TARGET_RATE]
    if above_target:
        calibrated = min(above_target, key=lambda r: r["alpha"])
    elif viable:
        calibrated = max(viable, key=lambda r: r["refusal_rate"])
    else:
        calibrated = min(sweep, key=lambda r: r["alpha"])  # everything degenerates; pick smallest
    return calibrated, sweep


def run_model(cache_label: str, hf_model_name: str) -> dict:
    print(f"\n=== {cache_label} ===")
    cache = load_cache(cache_label)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_train = torch.tensor([s == "train" for s in cache["splits"]], dtype=torch.bool)
    raw_directions = compute_raw_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])
    layer = resolve_layer_for_model(hf_model_name)
    raw_direction = raw_directions[layer]
    print(f"Own layer: {layer} (already-selected), raw direction norm: {raw_direction.norm().item():.3f}")

    rng = random.Random(SEED)  # fresh per model -> identical sampled prompt texts across models
    calib_prompts = harmless_prompts(cache, "val", N_ALPHA_SWEEP, rng)
    val_prompts = harmless_prompts(cache, "test", N_VALIDATION, rng)
    print(f"Alpha-sweep on {len(calib_prompts)} VAL-harmless prompts, final validation on {len(val_prompts)} TEST-harmless prompts (disjoint)")

    print(f"Loading model: {hf_model_name} (4-bit)")
    model = load_model(hf_model_name, load_in_4bit=True)

    print("  Alpha sweep")
    calibrated, sweep = calibrate_alpha(model, calib_prompts, raw_direction, layer)
    alpha = calibrated["alpha"]
    print(f"  Calibrated alpha: {alpha} (refusal_rate={calibrated['refusal_rate']} on calib set)")

    print("  Baseline completions (TEST-harmless)")
    baseline_completions = [generate_baseline(model, p, max_new_tokens=MAX_NEW_TOKENS) for p in val_prompts]
    print("  Addition completions (TEST-harmless, calibrated alpha)")
    added_completions = [
        generate_with_addition(model, p, raw_direction, layer_idx=layer, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS)
        for p in val_prompts
    ]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    baseline_stats = refusal_stats(baseline_completions)
    added_stats = refusal_stats(added_completions)
    print(f"  baseline refusal: {baseline_stats['rate']} [{baseline_stats['ci_low']}, {baseline_stats['ci_high']}]")
    print(f"  addition refusal: {added_stats['rate']} [{added_stats['ci_low']}, {added_stats['ci_high']}]")

    return {
        "model": hf_model_name,
        "layer": layer,
        "raw_direction_norm": round(raw_direction.norm().item(), 4),
        "alpha_sweep": sweep,
        "calibrated_alpha": alpha,
        "n_calib": len(calib_prompts),
        "n_validation": len(val_prompts),
        "calib_prompts": calib_prompts,
        "val_prompts": val_prompts,
        "refusal_rate": {"baseline": baseline_stats, "addition": added_stats},
        "degenerate_count": {
            "baseline": sum(is_degenerate(c) for c in baseline_completions),
            "addition": sum(is_degenerate(c) for c in added_completions),
        },
        "samples": {
            "baseline": list(zip(val_prompts, baseline_completions)),
            "addition": list(zip(val_prompts, added_completions)),
        },
    }


def main() -> None:
    results = {}
    for cache_label, hf_model_name in MODELS.items():
        results[cache_label] = run_model(cache_label, hf_model_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "sufficiency_7b_9b_scale.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== Summary ===")
    for cache_label, r in results.items():
        b = r["refusal_rate"]["baseline"]["rate"]
        a = r["refusal_rate"]["addition"]["rate"]
        print(f"  {cache_label}: baseline={b:.3f} -> addition={a:.3f} (alpha={r['calibrated_alpha']})")


if __name__ == "__main__":
    main()
