"""Closes the last un-started item in this project's remaining-gaps list: the
sufficiency (activation-addition) side of cross-model direction transfer.
`scripts/transfer_direction.py` already tested necessity (ablation) transfer between
Qwen3-8B and Llama-3.1-8B-Instruct (the only d_model=4096-matched pair) and found a clean
no-transfer result for Qwen3-8B and an inconclusive one for Llama -- deliberately deferring
sufficiency, since addition needs its own alpha calibration for the *foreign* direction on
the target's residual-stream scale (a fixed alpha has no consistent cross-model meaning, same
reasoning as `src.direction.compute.compute_raw_directions`'s own docstring).

Reuses, rather than regenerates, everything already available:
- Own raw directions/layers: computed identically to `scripts/sufficiency_at_scale.py`
  (`compute_raw_directions` on each model's own TRAIN split, `resolve_layer_for_model` for the
  already-selected layer -- never re-derived).
- Own-direction baseline and addition completions: `results/sufficiency_7b_9b_scale.json`.
  Verified (not assumed) that this script's harmless-prompt sampling (same SEED=11, same
  val/test splits, same n=12/n=50) reproduces that file's `calib_prompts`/`val_prompts`
  exactly -- so baseline behavior doesn't depend on which direction is later added, and
  reusing it saves half the new generation this gap would otherwise need.

Only new generation: alpha-calibration sweep + final validation using the *foreign* raw
direction, added at the *target's* own already-selected layer (same convention as
`scripts/transfer_direction.py`'s cross-model separation-score check: "foreign direction
evaluated at the target's own already-selected layer", extended here from ablation to
addition).

Usage: python scripts/transfer_sufficiency.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import assert_caches_consistent, load_cache
from src.activations.extract import load_model
from src.detectors.dense_direction_detector import resolve_layer_for_model
from src.direction.compute import compute_raw_directions
from src.direction.interventions import generate_with_addition
from src.direction.refusal_classifier import is_degenerate, is_refusal, refusal_stats
from src.eval.detector_metrics import mcnemar_exact

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
EXISTING_SUFFICIENCY_PATH = RESULTS_DIR / "sufficiency_7b_9b_scale.json"
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
SEED = 11  # matches scripts/sufficiency_at_scale.py exactly -- verified to reproduce
# its calib_prompts/val_prompts byte-for-byte before reusing its baseline completions.


def harmless_prompts(cache: dict, split_name: str, n: int, rng: random.Random) -> list[str]:
    eligible = [
        t for t, lab, sp in zip(cache["texts"], cache["labels"], cache["splits"])
        if lab == "harmless" and sp == split_name and len(t) <= MAX_PROMPT_CHARS
    ]
    return rng.sample(eligible, min(n, len(eligible)))


def own_raw_direction_and_layer(cache_label: str, hf_model_name: str) -> tuple[torch.Tensor, int]:
    cache = load_cache(cache_label)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_train = torch.tensor([s == "train" for s in cache["splits"]], dtype=torch.bool)
    raw_directions = compute_raw_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])
    layer = resolve_layer_for_model(hf_model_name)
    return raw_directions[layer], layer


def calibrate_alpha(
    model, prompts: list[str], foreign_raw_direction: torch.Tensor, target_layer: int
) -> tuple[dict, list[dict]]:
    sweep = []
    for alpha in ALPHA_GRID:
        completions = [
            generate_with_addition(
                model, p, foreign_raw_direction, layer_idx=target_layer, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS
            )
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
        calibrated = min(sweep, key=lambda r: r["alpha"])
    return calibrated, sweep


def run_target(
    cache_label: str,
    hf_model_name: str,
    target_layer: int,
    foreign_raw_direction: torch.Tensor,
    calib_prompts: list[str],
    val_prompts: list[str],
) -> dict:
    print(f"\n=== Target: {cache_label} (foreign direction added at its own layer {target_layer}) ===")
    print(f"Loading model: {hf_model_name} (4-bit)")
    model = load_model(hf_model_name, load_in_4bit=True)

    print("  Alpha sweep (foreign direction)")
    calibrated, sweep = calibrate_alpha(model, calib_prompts, foreign_raw_direction, target_layer)
    alpha = calibrated["alpha"]
    print(f"  Calibrated alpha: {alpha} (refusal_rate={calibrated['refusal_rate']} on calib set)")

    print("  Foreign-addition completions (TEST-harmless, calibrated alpha)")
    foreign_completions = [
        generate_with_addition(model, p, foreign_raw_direction, layer_idx=target_layer, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS)
        for p in val_prompts
    ]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    foreign_stats = refusal_stats(foreign_completions)
    print(f"  foreign-addition refusal: {foreign_stats['rate']} [{foreign_stats['ci_low']}, {foreign_stats['ci_high']}]")

    return {
        "layer": target_layer,
        "alpha_sweep": sweep,
        "calibrated_alpha": alpha,
        "refusal_rate": foreign_stats,
        "degenerate_count": sum(is_degenerate(c) for c in foreign_completions),
        "completions": foreign_completions,
    }


def main() -> None:
    assert_caches_consistent(list(MODELS.keys()))
    existing = json.load(open(EXISTING_SUFFICIENCY_PATH, encoding="utf-8"))

    print("=== Own raw directions/layers (reused, not re-derived) ===")
    qwen_direction, qwen_layer = own_raw_direction_and_layer("Qwen3-8B", MODELS["Qwen3-8B"])
    llama_direction, llama_layer = own_raw_direction_and_layer(
        "Llama-3.1-8B-Instruct", MODELS["Llama-3.1-8B-Instruct"]
    )
    print(f"Qwen3-8B layer {qwen_layer}, Llama-3.1-8B layer {llama_layer}")

    print("\n=== Verifying harmless-prompt sampling matches results/sufficiency_7b_9b_scale.json exactly ===")
    prompts_by_model = {}
    for cache_label in MODELS:
        cache = load_cache(cache_label)
        rng = random.Random(SEED)
        calib_prompts = harmless_prompts(cache, "val", N_ALPHA_SWEEP, rng)
        val_prompts = harmless_prompts(cache, "test", N_VALIDATION, rng)
        assert calib_prompts == existing[cache_label]["calib_prompts"], f"{cache_label} calib_prompts mismatch"
        assert val_prompts == existing[cache_label]["val_prompts"], f"{cache_label} val_prompts mismatch"
        prompts_by_model[cache_label] = {"calib": calib_prompts, "val": val_prompts}
        print(f"  {cache_label}: {len(calib_prompts)} calib + {len(val_prompts)} val prompts match existing run")

    print("\n--- Qwen3-8B: foreign direction = Llama's raw direction (its own layer 27) ---")
    qwen_foreign = run_target(
        "Qwen3-8B", MODELS["Qwen3-8B"], qwen_layer, llama_direction,
        prompts_by_model["Qwen3-8B"]["calib"], prompts_by_model["Qwen3-8B"]["val"],
    )
    print("\n--- Llama-3.1-8B-Instruct: foreign direction = Qwen3-8B's raw direction (its own layer 23) ---")
    llama_foreign = run_target(
        "Llama-3.1-8B-Instruct", MODELS["Llama-3.1-8B-Instruct"], llama_layer, qwen_direction,
        prompts_by_model["Llama-3.1-8B-Instruct"]["calib"], prompts_by_model["Llama-3.1-8B-Instruct"]["val"],
    )

    print("\n=== Transfer verdicts (paired McNemar, reusing existing baseline/own-addition completions) ===")
    results = {}
    for cache_label, foreign in (("Qwen3-8B", qwen_foreign), ("Llama-3.1-8B-Instruct", llama_foreign)):
        own = existing[cache_label]
        baseline_completions = [c for _, c in own["samples"]["baseline"]]
        own_completions = [c for _, c in own["samples"]["addition"]]
        foreign_completions = foreign["completions"]

        baseline_refusals = [is_refusal(c) for c in baseline_completions]
        own_refusals = [is_refusal(c) for c in own_completions]
        foreign_refusals = [is_refusal(c) for c in foreign_completions]

        verdict = {
            "baseline_vs_foreign": mcnemar_exact(baseline_refusals, foreign_refusals),
            "baseline_vs_own": mcnemar_exact(baseline_refusals, own_refusals),
            "own_vs_foreign": mcnemar_exact(own_refusals, foreign_refusals),
        }
        print(f"\n{cache_label} (own direction added at layer {foreign['layer']} vs. foreign at same layer):")
        print(f"  baseline: refusal={own['refusal_rate']['baseline']['rate']}")
        print(f"  own addition:     refusal={own['refusal_rate']['addition']['rate']}")
        print(f"  foreign addition: refusal={foreign['refusal_rate']['rate']}")
        print(f"  baseline vs foreign: p={verdict['baseline_vs_foreign']['p_value']}")
        print(f"  baseline vs own:     p={verdict['baseline_vs_own']['p_value']}")
        print(f"  own vs foreign:      p={verdict['own_vs_foreign']['p_value']}")

        results[cache_label] = {
            "layer": foreign["layer"],
            "own_addition_reused_from": str(EXISTING_SUFFICIENCY_PATH.name),
            "baseline": own["refusal_rate"]["baseline"],
            "own_addition": own["refusal_rate"]["addition"],
            "own_calibrated_alpha": own["calibrated_alpha"],
            "foreign_addition": {
                "alpha_sweep": foreign["alpha_sweep"],
                "calibrated_alpha": foreign["calibrated_alpha"],
                "refusal_stats": foreign["refusal_rate"],
                "degenerate_count": foreign["degenerate_count"],
                "completions": list(zip(prompts_by_model[cache_label]["val"], foreign["completions"])),
            },
            "mcnemar": verdict,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "sufficiency_transfer.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
