"""Reframes Phase 3's causally-ranked SAE features as a prompt classifier:
sum of the selected top-K features' encoded activations (across their
respective layers) as a harmfulness score, thresholded the same way as the
dense-direction detector (`src.detectors.dense_direction_detector`).

Reuses the exact (layer, feature) list already selected by causal ranking
(`results/sae_causal_ranking_<model>.json`, default Qwen3-8B, override via
`load_top_features`'s `path` arg for other models) -- this module doesn't
rerun ranking, it's a different downstream use of the same result. `saes`
values may be `TopKSAE` (Qwen-Scope) or `JumpReLUSAE` (LlamaScope/
GemmaScope) -- both only need `.encode()`, so this module works unchanged
across every model with a pretrained SAE suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.eval.detector_metrics import max_accuracy_threshold
from src.sae.jumprelu_sae import JumpReLUSAE
from src.sae.qwen_scope import TopKSAE

AnySAE = TopKSAE | JumpReLUSAE

RANKING_RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "sae_causal_ranking_Qwen3-8B.json"


def load_top_features(k: int = 15, path: Path = RANKING_RESULTS_PATH) -> list[tuple[int, int]]:
    """Top-k (layer, feature) pairs from Phase 3's causal ranking, in ranked
    order. Default k=15 matches reports/RESULTS.md's finding that suppressing the
    top-15 gives the strongest causal-validation effect."""
    with open(path) as fh:
        ranked = json.load(fh)["ranked_features"]
    return [(r["layer"], r["feature"]) for r in ranked[:k]]


def score(
    activations_by_layer: dict[int, torch.Tensor],
    saes: dict[int, AnySAE],
    features: list[tuple[int, int]],
    weights: list[float] | None = None,
) -> torch.Tensor:
    """activations_by_layer: {layer: (n_prompts, d_model)} residual-stream
    activations. saes: {layer: TopKSAE}. features: (layer, feature_idx)
    pairs to sum. weights: optional per-feature multiplier, same order as
    `features` (default: all 1.0, reproducing the original unweighted sum
    exactly -- see `compute_content_weights` for a principled alternative).
    Returns (n_prompts,) score -- weighted sum of each selected feature's
    encoded activation (post-TopK, so zero if that feature didn't survive
    this prompt's top-K sparsity cutoff)."""
    weights = weights if weights is not None else [1.0] * len(features)
    n_prompts = next(iter(activations_by_layer.values())).shape[0]
    total = torch.zeros(n_prompts)
    for (layer, feature_idx), weight in zip(features, weights):
        encoded = saes[layer].encode(activations_by_layer[layer])  # (n_prompts, d_sae)
        total = total + weight * encoded[:, feature_idx].to(total.dtype)
    return total


def calibrate(
    val_activations_by_layer: dict[int, torch.Tensor],
    val_labels: list[bool],
    saes: dict[int, AnySAE],
    features: list[tuple[int, int]],
    weights: list[float] | None = None,
) -> float:
    """Calibrates the decision threshold on a labeled split (VAL, per Phase
    4's split discipline -- see reports/DECISIONS.md) via `max_accuracy_threshold`
    (the project-wide adopted rule since 2026-08-04, see reports/RESULTS.md's
    "Threshold reselection" section -- this docstring previously said
    "Youden's J", which stopped being true once that rule changed)."""
    scores = score(val_activations_by_layer, saes, features, weights).tolist()
    return max_accuracy_threshold(scores, val_labels)


def compute_content_weights(variance_stats: list[dict]) -> list[float]:
    """Primary, pre-registered weighting formula (reports/DECISIONS.md,
    "Pre-registration: a content-weighted SAE detector", 2026-08-11): for
    each feature, w = eta_core / (eta_core + eta_wrapper), a closed-form
    ratio with no tunable threshold. `variance_stats`: list of per-feature
    stats dicts in the same order as the `features` list passed to `score`/
    `calibrate`, each with `eta_sq_core`/`eta_sq_wrapper` keys (the shape
    produced by scripts/feature_variance_family.py)."""
    weights = []
    for s in variance_stats:
        denom = s["eta_sq_core"] + s["eta_sq_wrapper"]
        weights.append(s["eta_sq_core"] / denom if denom > 0 else 0.0)
    return weights


def compute_content_weights_binary(variance_stats: list[dict], alpha: float = 0.05) -> list[float]:
    """Secondary, robustness-check weighting formula (same pre-registration
    entry as `compute_content_weights`): weight 0 for any feature whose
    wrapper effect is maxT-significant and larger than its core effect,
    weight 1 otherwise. Simpler than the primary ratio but introduces a
    significance-threshold hyperparameter, which is why it is not primary."""
    weights = []
    for s in variance_stats:
        drop = s["wrapper_effect_p_maxT"] < alpha and s["eta_sq_wrapper"] > s["eta_sq_core"]
        weights.append(0.0 if drop else 1.0)
    return weights


def is_flagged(s: float, threshold: float) -> bool:
    return s >= threshold
