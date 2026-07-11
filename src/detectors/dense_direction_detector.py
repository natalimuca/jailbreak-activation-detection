"""Reframes Phase 1's dense refusal direction (`src.direction.compute`) as a
prompt classifier: project a prompt's residual-stream activation at the
already-selected layer onto the TRAIN-estimated direction, and flag it as
harmful if the projection clears a calibrated threshold.

This is the same projection `src.direction.compute.separation_score` already
computes internally for layer selection -- what's new here is turning that
projection into an actual per-prompt decision (a threshold, calibrated on a
labeled split) rather than a layer-selection diagnostic.
"""

from __future__ import annotations

import torch

from src.direction.compute import compute_directions, select_candidate_layers, separation_score
from src.eval.detector_metrics import youden_threshold


def project(activations: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """activations: (n_prompts, d_model) at a single layer.
    direction: (d_model,). Returns (n_prompts,) projection scores -- higher
    means more harmful-like, by construction of `compute_directions`
    (harmful mean minus harmless mean)."""
    direction = direction.to(activations.dtype)
    return activations @ direction


def calibrate(val_activations: torch.Tensor, val_labels: list[bool], direction: torch.Tensor) -> float:
    """Calibrates the decision threshold on a labeled split (VAL, per Phase
    4's split discipline -- see DECISIONS.md) via Youden's J."""
    scores = project(val_activations, direction).tolist()
    return youden_threshold(scores, val_labels)


def is_flagged(score: float, threshold: float) -> bool:
    return score >= threshold


def select_layer_and_calibrate(
    acts: torch.Tensor, labels: torch.Tensor, is_train: torch.Tensor, is_val: torch.Tensor
) -> tuple[int, torch.Tensor, float]:
    """One-shot layer selection + threshold calibration for a full-corpus
    activation cache, both done on VAL -- deliberately NOT on TEST (unlike
    `scripts/06_dense_direction_ablation_qwen3.py`'s ablation-layer choice,
    which was selected via TEST-split separation score since VAL was
    already spoken for by Phase 3's causal validation at the time). Reusing
    TEST for both layer selection and final reported metrics is a mild
    leakage pattern this project explicitly avoids elsewhere (see
    METHODOLOGY.md's "Train/calib/val separation" note); doing both steps on
    VAL instead keeps TEST untouched until final reporting. For Qwen3-8B
    specifically, this makes no practical difference (layer 23 is selected
    either way -- verified directly, see DECISIONS.md), but this is the
    correct discipline going forward for additional models.

    acts: [n_layers, n_prompts, d_model]. labels/is_train/is_val: [n_prompts]
    bool tensors. Returns (layer, direction, threshold)."""
    directions = compute_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])
    val_scores = separation_score(acts[:, is_val, :], directions, labels[is_val])
    layer = select_candidate_layers(val_scores, k=1)[0]
    direction = directions[layer]
    threshold = calibrate(acts[layer, is_val, :], labels[is_val].tolist(), direction)
    return layer, direction, threshold
