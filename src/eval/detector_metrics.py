"""Shared classification-metrics harness for Phase 4's detector head-to-head.

Every detector compared in this phase (keyword filter, perplexity filter,
dense-direction projection, SAE-feature score) reduces to the same shape: a
real-valued score per prompt plus a decision threshold. This module gives
all four one shared way to (a) calibrate that threshold on a labeled split
and (b) report accuracy/precision/recall/F1/AUROC -- so detectors don't each
reimplement their own evaluation code, and the numbers are directly
comparable across methods.

Wilson-score CIs on the rate-like metrics (accuracy/precision/recall) match
the style already used for refusal rates in
`src.direction.refusal_classifier.refusal_stats`, for the same reason: well
behaved at small n and at rates near 0% or 100%.
"""

from __future__ import annotations

import math

from sklearn.metrics import roc_auc_score, roc_curve


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def youden_threshold(scores: list[float], labels: list[bool]) -> float:
    """Calibrates a decision threshold on labeled (score, label) pairs: the
    cutoff maximizing Youden's J (TPR - FPR) -- the standard way to turn a
    continuous detector score into a binary decision without an arbitrary
    hand-picked constant, in the same spirit as this project's existing
    alpha-calibration sweep (`scripts/02_calibrate_addition_alpha.py`).
    Intended to be called on VAL (see DECISIONS.md's split-discipline
    entry for Phase 4), never on the split final metrics are reported on."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thresholds[j.argmax()])


def classify(scores: list[float], threshold: float) -> list[bool]:
    return [s >= threshold for s in scores]


def detector_stats(scores: list[float], labels: list[bool], threshold: float, z: float = 1.96) -> dict:
    """Full metrics report for one detector on one evaluation set, at an
    already-calibrated threshold (see `youden_threshold`), plus a
    threshold-independent AUROC."""
    preds = classify(scores, threshold)
    n = len(labels)
    tp = sum(1 for p, l in zip(preds, labels) if p and l)
    fp = sum(1 for p, l in zip(preds, labels) if p and not l)
    fn = sum(1 for p, l in zip(preds, labels) if not p and l)
    tn = sum(1 for p, l in zip(preds, labels) if not p and not l)

    def rate_stat(k: int, denom: int) -> dict:
        if denom == 0:
            return {"n": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        lo, hi = _wilson_ci(k, denom, z)
        return {"n": denom, "rate": round(k / denom, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}

    accuracy = rate_stat(tp + tn, n)
    precision = rate_stat(tp, tp + fp)
    recall = rate_stat(tp, tp + fn)
    # F1 has no simple Wilson-CI form (it's a harmonic mean of two rates,
    # not itself a binomial proportion) -- point estimate only.
    p_point, r_point = precision["rate"], recall["rate"]
    f1 = 2 * p_point * r_point / (p_point + r_point) if (p_point + r_point) > 0 else 0.0

    auc = None
    if len(set(labels)) > 1:
        auc = round(float(roc_auc_score(labels, scores)), 4)

    return {
        "n": n,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": round(f1, 4),
        "auroc": auc,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
