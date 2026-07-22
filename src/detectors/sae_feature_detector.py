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

from src.eval.detector_metrics import youden_threshold
from src.sae.jumprelu_sae import JumpReLUSAE
from src.sae.qwen_scope import TopKSAE

AnySAE = TopKSAE | JumpReLUSAE

RANKING_RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "sae_causal_ranking_Qwen3-8B.json"


def load_top_features(k: int = 15, path: Path = RANKING_RESULTS_PATH) -> list[tuple[int, int]]:
    """Top-k (layer, feature) pairs from Phase 3's causal ranking, in ranked
    order. Default k=15 matches RESULTS.md's finding that suppressing the
    top-15 gives the strongest causal-validation effect."""
    with open(path) as fh:
        ranked = json.load(fh)["ranked_features"]
    return [(r["layer"], r["feature"]) for r in ranked[:k]]


def score(
    activations_by_layer: dict[int, torch.Tensor],
    saes: dict[int, AnySAE],
    features: list[tuple[int, int]],
) -> torch.Tensor:
    """activations_by_layer: {layer: (n_prompts, d_model)} residual-stream
    activations. saes: {layer: TopKSAE}. features: (layer, feature_idx)
    pairs to sum. Returns (n_prompts,) score -- sum of each selected
    feature's encoded activation (post-TopK, so zero if that feature didn't
    survive this prompt's top-K sparsity cutoff)."""
    n_prompts = next(iter(activations_by_layer.values())).shape[0]
    total = torch.zeros(n_prompts)
    for layer, feature_idx in features:
        encoded = saes[layer].encode(activations_by_layer[layer])  # (n_prompts, d_sae)
        total = total + encoded[:, feature_idx].to(total.dtype)
    return total


def calibrate(
    val_activations_by_layer: dict[int, torch.Tensor],
    val_labels: list[bool],
    saes: dict[int, AnySAE],
    features: list[tuple[int, int]],
) -> float:
    """Calibrates the decision threshold on a labeled split (VAL, per Phase
    4's split discipline -- see DECISIONS.md) via Youden's J."""
    scores = score(val_activations_by_layer, saes, features).tolist()
    return youden_threshold(scores, val_labels)


def is_flagged(s: float, threshold: float) -> bool:
    return s >= threshold
