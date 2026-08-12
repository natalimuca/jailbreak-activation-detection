"""Live-scoring wrapper around the pre-registered non-linear SAE-feature
combiner (reports/DECISIONS.md, "Pre-registration: a non-linear SAE-feature
combiner"; fit and evaluated in scripts/nonlinear_combiner_eval.py). Loads
an already-fitted scikit-learn `Pipeline` from disk rather than refitting --
a live prompt's classification must always match what reports/RESULTS.md
already reports, never drift from it (see src.api.inference_manager's own
module docstring for this project's standing rule on that).

Same `score`/`is_flagged` shape as `dense_direction_detector` and
`sae_feature_detector` so this is real, testable `src/` logic, not glue
inlined into the API layer.
"""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.pipeline import Pipeline

from src.detectors.sae_feature_detector import AnySAE, feature_matrix


def load_pipeline(path: Path) -> Pipeline:
    import joblib

    return joblib.load(path)


def score(
    activations_by_layer: dict[int, torch.Tensor],
    saes: dict[int, AnySAE],
    features: list[tuple[int, int]],
    pipeline: Pipeline,
) -> torch.Tensor:
    """Returns (n_prompts,) predicted probabilities -- reuses
    `sae_feature_detector.feature_matrix` (the exact per-feature SAE-encode
    loop every other SAE-based detector uses) then `pipeline.predict_proba`."""
    matrix = feature_matrix(activations_by_layer, saes, features).numpy()
    return torch.tensor(pipeline.predict_proba(matrix)[:, 1])


def is_flagged(s: float, threshold: float) -> bool:
    return s >= threshold
