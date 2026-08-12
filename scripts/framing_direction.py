"""Computes a per-layer "framing direction" -- mean residual-stream activation
under wrapper/framing conditions minus mean under the bare (unwrapped)
condition, from the same 10-core x 5-wrapper factorial already used by
scripts/wrapper_swap_variance.py and scripts/feature_variance_family.py --
and validates that ablating it actually removes the framing signal from
Qwen3-8B's already-known framing-leaning SAE features before trusting it for
anything downstream.

**Motivation.** A prior experiment (reports/DECISIONS.md, "Pre-registration:
a content-weighted SAE detector") tried fixing Qwen3-8B's PAIR vulnerability
by down-weighting its framing-tracking SAE features downstream -- it failed
because those features carry real class-separating signal beyond their
framing-sensitivity, so suppressing them loses genuine harmfulness signal
too. This script builds the ingredient for a different fix instead: an
explicit direction to ablate from the residual stream *upstream*, before
either detector scores it, so the framing component is removed without
discarding whole features. See reports/DECISIONS.md's "Pre-registration:
framing-direction ablation" entry for the full pre-registered design and
validation criterion (fixed before this script's real run).

**Reuses, does not duplicate**: `src.direction.compute.compute_directions`
(the exact function the refusal direction itself uses -- pure
difference-of-means, then L2-normalize) for the direction; a local 3-line
reimplementation of `src.direction.interventions._project_out`'s ablation
math (that function is module-private and this reuse is trivial enough not
to warrant promoting it to shared status); `src.activations.extract
.get_last_token_resid_acts` for multi-layer, single-forward-pass-per-prompt
residual capture; `wrapper_swap_variance.py`'s `CORE_REQUESTS`,
`WRAPPER_TEMPLATES`, `WRAPPER_KEYS`, `build_prompt`, `sum_of_squares`,
`permutation_test_wrapper_effect` for the prompt design and the validation
ANOVA.

Usage: python scripts/framing_direction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import get_last_token_resid_acts, load_model
from src.direction.compute import compute_directions
from scripts.wrapper_swap_variance import (
    CORE_REQUESTS,
    MODELS,
    N_PERMUTATIONS,
    RNG_SEED,
    WRAPPER_KEYS,
    build_prompt,
    permutation_test_wrapper_effect,
    sum_of_squares,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LAYERS = {
    "Qwen3-8B": [23, 24, 25],
    "Llama-3.1-8B-Instruct": [21, 26, 27],
}
BARE_INDEX = WRAPPER_KEYS.index("bare")


def _project_out(act: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Local reimplementation of src.direction.interventions._project_out's
    math (that function is module-private and this reuse is a 3-line pure
    tensor op, not worth a cross-module import)."""
    direction = direction.to(act.dtype)
    proj = (act @ direction).unsqueeze(-1) * direction
    return act - proj


def collect_wrapper_swap_acts(model, layers: list[int]) -> tuple[torch.Tensor, list[int], list[int]]:
    """Returns (acts, bare_idx, wrapped_idx). acts: [len(layers), 50, d_model].
    Flat prompt order matches wrapper_swap_variance.py's grid convention:
    index i*5+j is (CORE_REQUESTS[i], WRAPPER_KEYS[j])."""
    prompts = [build_prompt(core, wrapper_key) for core in CORE_REQUESTS for wrapper_key in WRAPPER_KEYS]
    all_layer_acts = get_last_token_resid_acts(model, prompts)  # [n_layers_all, 50, d_model]
    acts = all_layer_acts[layers]  # [len(layers), 50, d_model]
    bare_idx = [i * len(WRAPPER_KEYS) + BARE_INDEX for i in range(len(CORE_REQUESTS))]
    wrapped_idx = [i * len(WRAPPER_KEYS) + j for i in range(len(CORE_REQUESTS)) for j in range(len(WRAPPER_KEYS)) if j != BARE_INDEX]
    return acts, bare_idx, wrapped_idx


def compute_framing_directions(acts: torch.Tensor, bare_idx: list[int], wrapped_idx: list[int]) -> torch.Tensor:
    """acts: [n_layers, 50, d_model]. Returns [n_layers, d_model], one unit
    direction per layer (wrapped mean minus bare mean, matching
    compute_directions's (group_a, group_b) -> group_a - group_b convention,
    so positive = more "wrapped-like")."""
    wrapped_group = acts[:, wrapped_idx, :]
    bare_group = acts[:, bare_idx, :]
    return compute_directions(wrapped_group, bare_group)


def validate_ablation(
    model, cache_label: str, layers: list[int], directions: torch.Tensor, rng: np.random.Generator
) -> dict:
    """Recomputes the wrapper-effect ANOVA (reusing wrapper_swap_variance.py's
    machinery unmodified) for every framing-leaning feature in this model's
    already-published feature_variance_<model>.json, both before and after
    ablating this script's frozen direction from the same 50 prompts'
    residuals -- the required validation gate before Stage 2 is trusted."""
    with open(RESULTS_DIR / f"feature_variance_{cache_label}.json") as fh:
        family = json.load(fh)
    layer_set = set(layers)
    framing_leaning = [
        f for f in family["features"]
        if f["layer"] in layer_set and f["stats"]["eta_sq_wrapper"] > f["stats"]["eta_sq_core"]
    ]
    print(f"  {len(framing_leaning)} framing-leaning features in layers {layers} to validate against")

    prompts = [build_prompt(core, wrapper_key) for core in CORE_REQUESTS for wrapper_key in WRAPPER_KEYS]
    all_layer_acts = get_last_token_resid_acts(model, prompts)

    per_feature = []
    for f in framing_leaning:
        layer, feature_idx = f["layer"], f["feature"]
        layer_pos = layers.index(layer)
        raw_acts = all_layer_acts[layer]  # [50, d_model]
        ablated_acts = _project_out(raw_acts, directions[layer_pos])

        sae_row = _sae_row_cache(model_hf_name_for(cache_label), layer, feature_idx)
        w_enc, b_enc = sae_row
        original_grid = ((raw_acts @ w_enc) + b_enc).reshape(len(CORE_REQUESTS), len(WRAPPER_KEYS)).numpy()
        ablated_grid = ((ablated_acts @ w_enc) + b_enc).reshape(len(CORE_REQUESTS), len(WRAPPER_KEYS)).numpy()

        original_stats = sum_of_squares(original_grid)
        ablated_stats = sum_of_squares(ablated_grid)
        ablated_p = permutation_test_wrapper_effect(ablated_grid, N_PERMUTATIONS, rng)

        rel_drop = (
            (original_stats["eta_sq_wrapper"] - ablated_stats["eta_sq_wrapper"]) / original_stats["eta_sq_wrapper"]
            if original_stats["eta_sq_wrapper"] > 0 else 0.0
        )
        per_feature.append({
            "layer": layer, "feature": feature_idx,
            "eta_sq_wrapper_original": original_stats["eta_sq_wrapper"],
            "eta_sq_wrapper_ablated": ablated_stats["eta_sq_wrapper"],
            "relative_drop": rel_drop,
            "wrapper_effect_p_ablated": ablated_p,
            "lost_significance": ablated_p >= 0.05,
        })
        print(
            f"    L{layer}/F{feature_idx}: eta_wrapper {original_stats['eta_sq_wrapper']:.3f} -> "
            f"{ablated_stats['eta_sq_wrapper']:.3f} (drop={rel_drop:.1%}), p_ablated={ablated_p:.4f}"
        )

    median_drop = float(np.median([f["relative_drop"] for f in per_feature]))
    n_lost_significance = sum(1 for f in per_feature if f["lost_significance"])
    passed = median_drop >= 0.5 and n_lost_significance >= 10
    return {
        "n_features": len(per_feature), "median_relative_drop": median_drop,
        "n_lost_significance": n_lost_significance, "passed": passed,
        "per_feature": per_feature,
    }


_SAE_CACHE: dict = {}


def model_hf_name_for(cache_label: str) -> str:
    return MODELS[cache_label]


def _sae_row_cache(hf_model_name: str, layer: int, feature_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    from src.sae.registry import SAE_PROVIDERS
    key = (hf_model_name, layer)
    if key not in _SAE_CACHE:
        load_sae_fn = SAE_PROVIDERS[hf_model_name][0]
        _SAE_CACHE[key] = load_sae_fn(layer)
    sae = _SAE_CACHE[key]
    return sae.W_enc[feature_idx].float(), sae.b_enc[feature_idx].float()


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    out_path = RESULTS_DIR / "framing_directions.pt"
    validation_path = RESULTS_DIR / "framing_direction_validation.json"

    directions_out: dict = {}
    validation_out: dict = {}

    for cache_label, hf_model_name in MODELS.items():
        layers = LAYERS[cache_label]
        print(f"\n=== {cache_label} (layers {layers}) ===")
        print(f"Loading model: {hf_model_name} (4-bit)")
        model = load_model(hf_model_name, load_in_4bit=True)

        acts, bare_idx, wrapped_idx = collect_wrapper_swap_acts(model, layers)
        directions = compute_framing_directions(acts, bare_idx, wrapped_idx)
        directions_out[cache_label] = {"layers": layers, "directions": directions}
        print(f"Computed {directions.shape[0]} framing directions (one per layer), norm check: "
              f"{[round(float(directions[i].norm()), 4) for i in range(directions.shape[0])]}")

        print("Validating ablation against known framing-leaning features...")
        validation = validate_ablation(model, cache_label, layers, directions, rng)
        validation_out[cache_label] = validation
        gate = "Qwen3-8B" in cache_label
        status = "PASSED" if validation["passed"] else "FAILED"
        print(
            f"  median relative drop={validation['median_relative_drop']:.1%}, "
            f"lost significance={validation['n_lost_significance']}/{validation['n_features']} "
            f"-> {status}" + (" (gates Stage 2)" if gate else " (diagnostic only)")
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save(directions_out, out_path)
    with open(validation_path, "w") as fh:
        json.dump(validation_out, fh, indent=2)
    print(f"\nSaved directions to {out_path}")
    print(f"Saved validation to {validation_path}")

    if not validation_out["Qwen3-8B"]["passed"]:
        print("\nQwen3-8B validation gate FAILED -- per pre-registration, Stage 2 should NOT run for Qwen3-8B.")
    else:
        print("\nQwen3-8B validation gate PASSED -- Stage 2 (scripts/framing_ablation_eval.py) may proceed.")


if __name__ == "__main__":
    main()
