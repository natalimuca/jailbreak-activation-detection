"""Extends scripts/wrapper_swap_variance.py's rank-1-only wrapper-swap ANOVA
to each model's full top-15 causally-ranked SAE features, not just feature #1.

**Motivation.** The rank-1 result already published (`results/wrapper_swap_variance.json`)
found Qwen3-8B's top feature (layer 25, feature 65291) is dominated by
wrapper/framing identity (eta_sq_wrapper=0.656), while Llama-3.1-8B-Instruct's
top feature (layer 27, feature 13363) is dominated by core-request content
(eta_sq_core=0.407). This script asks the same question of the other 14
features that make up each model's top-15 SAE-feature detector
(`src.detectors.sae_feature_detector`), so a content-weighted variant of that
detector (see `reports/DECISIONS.md`'s "Pre-registration" entry, 2026-08-11)
can be built on real per-feature evidence rather than extrapolating from
feature #1 alone.

**Reuses, does not duplicate**, `wrapper_swap_variance.py`'s grid measurement,
ANOVA decomposition, and per-feature permutation-test machinery (all
model-/feature-agnostic pure functions) -- only the outer loop (which
features to read out, and the new family-wise correction across them) is new.

**Family-wise correction.** 15 features x 2 effects (core, wrapper) per model
is a genuine correlated family (same 50 prompts, adjacent SAE indices, shared
layers) -- plain Bonferroni would be both wrong (ignores the correlation) and
overly conservative. `maxT_correct_family` adapts the maxT/Westfall-Young
pattern already used in `scripts/wrapper_feature_search.py::maxT_family_test`:
one shared permutation draw per Monte Carlo replicate, applied to every
feature's grid simultaneously (preserving their correlation structure in the
null), each feature's null distribution standardized before taking the
family-wide max -- exactly mirroring the existing implementation's
standardize-then-max design, just with a shared-grid-permutation null instead
of a shared-outcome-permutation null (the natural adaptation for a
per-feature-grid family rather than a per-covariate family).

Checkpointed per feature (not just per model): a crash partway through one
model's 15 features only costs re-doing the unfinished features, not
restarting that model, matching this project's established
`token_attribution.py` checkpointing lesson.

Usage: python scripts/feature_variance_family.py
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

from src.activations.extract import load_model
from src.sae.registry import SAE_PROVIDERS
from scripts.wrapper_swap_variance import (
    MODELS,
    N_PERMUTATIONS,
    RNG_SEED,
    measure_activation_grid,
    permutation_test_core_effect,
    permutation_test_wrapper_effect,
    sum_of_squares,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RANKING_PATH = {
    "Qwen3-8B": RESULTS_DIR / "sae_causal_ranking_Qwen3-8B.json",
    "Llama-3.1-8B-Instruct": RESULTS_DIR / "sae_causal_ranking_Llama-3.1-8B-Instruct.json",
}
TOP_K = 15


def load_top_k_features(model_label: str, k: int = TOP_K) -> list[tuple[int, int]]:
    with open(RANKING_PATH[model_label]) as fh:
        ranked = json.load(fh)["ranked_features"]
    return [(r["layer"], r["feature"]) for r in ranked[:k]]


def _feature_key(layer: int, feature: int) -> str:
    return f"{layer}_{feature}"


def maxT_correct_family(
    grids: dict[str, np.ndarray], effect: str, n_perm: int, rng: np.random.Generator
) -> dict[str, float]:
    """Family-wise maxT/Westfall-Young correction across every feature's grid
    in `grids` (keyed by "{layer}_{feature}"). `effect`: "core" or "wrapper".
    One shared permutation draw per replicate is applied to every feature's
    grid so the correlation between features (same 50 prompts) is preserved
    in the null, then each feature's null is standardized before the
    family-wide max is taken -- same structure as
    wrapper_feature_search.py::maxT_family_test, adapted from a
    shared-outcome-permutation family to a shared-grid-permutation one."""
    names = list(grids.keys())
    ss_key = f"ss_{effect}"
    observed = {name: sum_of_squares(grids[name])[ss_key] for name in names}
    n_core, n_wrapper = next(iter(grids.values())).shape

    null_per_feature = {name: np.zeros(n_perm) for name in names}
    for p in range(n_perm):
        if effect == "core":
            row_perm_by_col = [rng.permutation(n_core) for _ in range(n_wrapper)]
            for name in names:
                permuted = np.empty_like(grids[name])
                for j in range(n_wrapper):
                    permuted[:, j] = grids[name][row_perm_by_col[j], j]
                null_per_feature[name][p] = sum_of_squares(permuted)[ss_key]
        else:
            col_perm_by_row = [rng.permutation(n_wrapper) for _ in range(n_core)]
            for name in names:
                permuted = np.empty_like(grids[name])
                for i in range(n_core):
                    permuted[i, :] = grids[name][i, col_perm_by_row[i]]
                null_per_feature[name][p] = sum_of_squares(permuted)[ss_key]

    standardized_null = {}
    standardized_observed = {}
    for name in names:
        mu, sd = null_per_feature[name].mean(), null_per_feature[name].std(ddof=1)
        sd = sd if sd > 0 else 1e-12
        standardized_null[name] = (null_per_feature[name] - mu) / sd
        standardized_observed[name] = (observed[name] - mu) / sd

    family_max_null = np.max(np.stack([standardized_null[name] for name in names]), axis=0)

    adjusted_p = {}
    for name in names:
        adjusted_p[name] = float((np.sum(family_max_null >= standardized_observed[name] - 1e-9) + 1) / (n_perm + 1))
    return adjusted_p


def verify_maxT_on_synthetic() -> None:
    """Null case: 15 independent noise grids, none should be significant
    after correction. Planted case: 14 noise grids + 1 grid with a real,
    strong core-effect signal planted -- only that one should survive
    correction, with no false positives among the other 14 (checked for both
    the core and wrapper effect columns)."""
    rng = np.random.default_rng(123)
    n_core, n_wrapper = 10, 5

    null_grids = {f"noise{i}": rng.normal(size=(n_core, n_wrapper)) for i in range(15)}
    null_core_p = maxT_correct_family(null_grids, "core", 2000, rng)
    null_wrapper_p = maxT_correct_family(null_grids, "wrapper", 2000, rng)
    assert all(p > 0.05 for p in null_core_p.values()), f"null case gave a spuriously significant core effect: {null_core_p}"
    assert all(p > 0.05 for p in null_wrapper_p.values()), f"null case gave a spuriously significant wrapper effect: {null_wrapper_p}"

    planted_grids = {f"noise{i}": rng.normal(size=(n_core, n_wrapper)) for i in range(14)}
    row_effect = rng.normal(scale=3.0, size=n_core)
    planted_grids["planted"] = rng.normal(size=(n_core, n_wrapper)) + row_effect[:, None]
    planted_core_p = maxT_correct_family(planted_grids, "core", 2000, rng)
    assert planted_core_p["planted"] < 0.05, f"planted case failed to detect the real core effect: {planted_core_p['planted']}"
    for name in [n for n in planted_grids if n != "planted"]:
        assert planted_core_p[name] > 0.05, f"planted case gave a false positive on unrelated feature {name}: {planted_core_p[name]}"

    print("maxT_correct_family verified: null case clean, planted case detected with no false positives.")


def verify_rank1_reproduction(model_label: str, rank1_result: dict) -> None:
    """Cross-checks this script's rank-1 feature (recomputed through the new
    multi-feature loop) against the already-published single-feature result
    in results/wrapper_swap_variance.json -- eta-squared values are a
    deterministic function of the (no-grad, no-sampling) forward passes so
    should match near-exactly; permutation p-values carry Monte Carlo noise
    so are checked for the same significance conclusion plus a loose
    numeric tolerance, not bit-for-bit equality."""
    published_path = RESULTS_DIR / "wrapper_swap_variance.json"
    with open(published_path) as fh:
        published = json.load(fh)[model_label]["stats"]

    for key in ["eta_sq_core", "eta_sq_wrapper", "eta_sq_residual"]:
        assert abs(rank1_result["stats"][key] - published[key]) < 1e-6, (
            f"{model_label} rank-1 {key} mismatch: new={rank1_result['stats'][key]}, published={published[key]}"
        )
    for key in ["core_effect_p", "wrapper_effect_p"]:
        new_p, pub_p = rank1_result["stats"][key], published[key]
        assert (new_p < 0.05) == (pub_p < 0.05), (
            f"{model_label} rank-1 {key} significance conclusion changed: new={new_p}, published={pub_p}"
        )
        assert abs(new_p - pub_p) < 0.02, f"{model_label} rank-1 {key} drifted too far: new={new_p}, published={pub_p}"

    print(f"{model_label} rank-1 reproduction verified against published wrapper_swap_variance.json.")


def main() -> None:
    verify_maxT_on_synthetic()

    rng = np.random.default_rng(RNG_SEED)

    for cache_label, hf_model_name in MODELS.items():
        out_path = RESULTS_DIR / f"feature_variance_{cache_label}.json"
        data = {"model": cache_label, "features": []}
        if out_path.exists():
            with open(out_path) as fh:
                data = json.load(fh)

        top_features = load_top_k_features(cache_label)
        done_keys = {_feature_key(f["layer"], f["feature"]) for f in data["features"]}
        remaining = [(rank, layer, feat) for rank, (layer, feat) in enumerate(top_features, start=1)
                     if _feature_key(layer, feat) not in done_keys]

        if not remaining and "maxT_correction" in data:
            print(f"\n=== {cache_label}: already complete (15 features + maxT correction), skipping ===")
            continue

        if remaining:
            print(f"\n=== {cache_label}: {len(remaining)}/{TOP_K} features remaining ===")
            print(f"Loading model: {hf_model_name} (4-bit)")
            model = load_model(hf_model_name, load_in_4bit=True)
            layers_needed = sorted({layer for _, layer, _ in remaining})
            load_sae_fn = SAE_PROVIDERS[hf_model_name][0]
            saes = {layer: load_sae_fn(layer) for layer in layers_needed}

            for rank, layer, feature_idx in remaining:
                print(f"  rank {rank}: layer {layer}, feature {feature_idx}")
                grid = measure_activation_grid(model, layer, feature_idx, saes[layer])
                stats = sum_of_squares(grid)
                stats["core_effect_p"] = permutation_test_core_effect(grid, N_PERMUTATIONS, rng)
                stats["wrapper_effect_p"] = permutation_test_wrapper_effect(grid, N_PERMUTATIONS, rng)
                data["features"].append({
                    "rank": rank, "layer": layer, "feature": feature_idx,
                    "activation_grid": grid.tolist(), "stats": stats,
                })
                with open(out_path, "w") as fh:
                    json.dump(data, fh, indent=2)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if "maxT_correction" not in data:
            print(f"  running family-wise maxT correction across {len(data['features'])} features...")
            grids = {_feature_key(f["layer"], f["feature"]): np.array(f["activation_grid"]) for f in data["features"]}
            core_p_maxT = maxT_correct_family(grids, "core", N_PERMUTATIONS, rng)
            wrapper_p_maxT = maxT_correct_family(grids, "wrapper", N_PERMUTATIONS, rng)
            for f in data["features"]:
                key = _feature_key(f["layer"], f["feature"])
                f["stats"]["core_effect_p_maxT"] = core_p_maxT[key]
                f["stats"]["wrapper_effect_p_maxT"] = wrapper_p_maxT[key]
            data["maxT_correction"] = {"n_permutations": N_PERMUTATIONS, "method": "maxT/Westfall-Young, shared grid permutation per replicate"}
            with open(out_path, "w") as fh:
                json.dump(data, fh, indent=2)

        rank1 = next(f for f in data["features"] if f["rank"] == 1)
        verify_rank1_reproduction(cache_label, rank1)

        print(f"\n{cache_label} summary (raw p / maxT-adjusted p):")
        for f in sorted(data["features"], key=lambda x: x["rank"]):
            s = f["stats"]
            leaning = "content" if s["eta_sq_core"] > s["eta_sq_wrapper"] else "framing"
            print(
                f"  rank {f['rank']:2d} (L{f['layer']}/F{f['feature']}): "
                f"eta_core={s['eta_sq_core']:.3f} (p={s['core_effect_p']:.4f}/{s['core_effect_p_maxT']:.4f}), "
                f"eta_wrapper={s['eta_sq_wrapper']:.3f} (p={s['wrapper_effect_p']:.4f}/{s['wrapper_effect_p_maxT']:.4f}) "
                f"-> {leaning}-leaning"
            )
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
