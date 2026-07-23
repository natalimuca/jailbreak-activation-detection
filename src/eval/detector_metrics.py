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

import numpy as np
from scipy.stats import binomtest, chi2, norm
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
    alpha-calibration sweep (`scripts/calibrate_alpha.py`).
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


def mcnemar_exact(preds_a: list[bool], preds_b: list[bool]) -> dict:
    """Exact McNemar's test for two detectors' paired binary decisions on
    the SAME items (e.g. dense-direction vs SAE-feature flag/no-flag calls
    on the same 21 PAIR prompts). This is the statistically correct way to
    ask "do these two detectors differ" on paired data -- comparing two
    independent Wilson CIs for overlap is a weaker, informal proxy for the
    same question and can miss a real paired difference (or, as importantly,
    can't be trusted to rule one out either).

    Only the *discordant* pairs (where the two detectors disagree) carry
    information about which is better; concordant pairs (both flag, or
    both don't) are uninformative and dropped, same as standard McNemar's.
    Under the null hypothesis of no systematic difference, discordant pairs
    should split ~50/50 between "a flags, b doesn't" and "b flags, a
    doesn't" -- tested via an exact two-sided binomial test on that split
    (more appropriate than the chi-squared approximation at the small
    sample sizes this project's adversarial-set conditions have)."""
    only_a = sum(1 for a, b in zip(preds_a, preds_b) if a and not b)
    only_b = sum(1 for a, b in zip(preds_a, preds_b) if b and not a)
    n_discordant = only_a + only_b
    if n_discordant == 0:
        return {"only_a": only_a, "only_b": only_b, "n_discordant": 0, "p_value": 1.0}
    p_value = binomtest(min(only_a, only_b), n_discordant, 0.5).pvalue
    return {"only_a": only_a, "only_b": only_b, "n_discordant": n_discordant, "p_value": round(p_value, 4)}


def _delong_placements(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DeLong et al. (1988)'s structural components: psi(x, y) = 1 if
    x > y, 0.5 if x == y, 0 if x < y, averaged over the opposite class for
    each sample. Mean of `v10` (or equivalently `v01`) is the AUC itself,
    expressed as a Mann-Whitney U-statistic."""
    diff = pos_scores[:, None] - neg_scores[None, :]
    psi = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)
    v10 = psi.mean(axis=1)  # one value per positive sample
    v01 = psi.mean(axis=0)  # one value per negative sample
    return v10, v01


def delong_auc_test(scores_a: list[float], scores_b: list[float], labels: list[bool]) -> dict:
    """Paired DeLong's test comparing two detectors' AUROC on the SAME
    labeled sample (Sun & Xu 2014's structural-components formulation of
    DeLong et al. 1988). The right tool whenever this project compares two
    detectors' AUROC on the same prompt set -- e.g. dense-direction vs
    SAE-feature on TEST-overall -- since an unpaired test (independent
    bootstrap CIs on each AUC) would ignore that both scores come from the
    exact same items and are therefore correlated; ignoring that
    correlation makes an unpaired test systematically too conservative
    (declares real differences non-significant more often than it should).

    Returns each AUC (should match `sklearn.roc_auc_score` exactly), their
    difference, and a two-sided p-value from the resulting z-statistic."""
    labels_arr = np.asarray(labels, dtype=bool)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    pos_a, neg_a = a[labels_arr], a[~labels_arr]
    pos_b, neg_b = b[labels_arr], b[~labels_arr]
    n_pos, n_neg = len(pos_a), len(neg_a)

    v10_a, v01_a = _delong_placements(pos_a, neg_a)
    v10_b, v01_b = _delong_placements(pos_b, neg_b)
    auc_a, auc_b = float(v10_a.mean()), float(v10_b.mean())

    v10 = np.stack([v10_a, v10_b])  # (2, n_pos)
    v01 = np.stack([v01_a, v01_b])  # (2, n_neg)
    s10 = np.cov(v10, ddof=1)
    s01 = np.cov(v01, ddof=1)
    s = s10 / n_pos + s01 / n_neg

    auc_diff = auc_a - auc_b
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var_diff <= 0:
        # Only happens if the two detectors' scores are perfectly
        # correlated (e.g. identical) -- no evidence of a difference either
        # way, reported as non-significant rather than dividing by ~0.
        return {"auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4), "diff": round(auc_diff, 4), "z": 0.0, "p_value": 1.0}
    z = auc_diff / math.sqrt(var_diff)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return {
        "auc_a": round(auc_a, 4),
        "auc_b": round(auc_b, 4),
        "diff": round(auc_diff, 4),
        "z": round(float(z), 4),
        "p_value": round(float(p_value), 4),
    }


def cochrans_q(preds_matrix: list[list[bool]]) -> dict:
    """Cochran's Q test: generalizes McNemar's test from two to k >= 2
    related binary classifiers scored on the SAME n items (e.g. three
    models' dense-direction detectors, each flag/no-flag on the same 21
    PAIR prompts). Tests whether the k classifiers' success rates differ,
    using the repeated-measures structure -- the right tool here instead of
    comparing k independent Wilson CIs pairwise, which both inflates the
    false-positive rate (multiple untested comparisons) and, as with
    McNemar's, ignores that all k classifiers are scored on the same items.

    preds_matrix: k lists of n paired booleans (k classifiers x n items).
    Q ~ chi-squared(k-1) under the null of equal success rates."""
    arr = np.array(preds_matrix, dtype=float)  # (k, n)
    k, n = arr.shape
    item_totals = arr.sum(axis=0)  # how many classifiers flagged each item
    classifier_totals = arr.sum(axis=1)  # each classifier's total flag count

    numerator = k * (k - 1) * np.sum((classifier_totals - classifier_totals.mean()) ** 2)
    denominator = k * np.sum(item_totals) - np.sum(item_totals**2)
    if denominator == 0:
        # Every classifier agreed on every item (all-flag or all-no-flag
        # rows/columns) -- no variation to test, not significant.
        return {"q": 0.0, "df": k - 1, "p_value": 1.0}
    q = numerator / denominator
    p_value = 1 - chi2.cdf(q, df=k - 1)
    return {"q": round(float(q), 4), "df": k - 1, "p_value": round(float(p_value), 4)}
