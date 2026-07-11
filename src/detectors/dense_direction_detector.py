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
