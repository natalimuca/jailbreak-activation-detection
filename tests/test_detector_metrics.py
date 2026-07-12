import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.eval.detector_metrics import (
    classify,
    cochrans_q,
    delong_auc_test,
    detector_stats,
    mcnemar_exact,
    youden_threshold,
)


def test_youden_threshold_separates_perfectly_separable_scores():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    labels = [False, False, False, True, True, True]
    thr = youden_threshold(scores, labels)
    assert 0.3 < thr <= 0.8


def test_classify_respects_threshold():
    assert classify([0.1, 0.5, 0.9], threshold=0.5) == [False, True, True]


def test_detector_stats_perfect_classifier():
    scores = [0.0, 0.1, 0.9, 1.0]
    labels = [False, False, True, True]
    stats = detector_stats(scores, labels, threshold=0.5)
    assert stats["accuracy"]["rate"] == 1.0
    assert stats["precision"]["rate"] == 1.0
    assert stats["recall"]["rate"] == 1.0
    assert stats["f1"] == 1.0
    assert stats["auroc"] == 1.0
    assert stats["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}


def test_detector_stats_worst_case_classifier():
    scores = [0.0, 0.1, 0.9, 1.0]
    labels = [True, True, False, False]
    stats = detector_stats(scores, labels, threshold=0.5)
    assert stats["accuracy"]["rate"] == 0.0
    assert stats["auroc"] == 0.0


def test_detector_stats_handles_no_positive_predictions():
    scores = [0.0, 0.0, 0.0, 0.0]
    labels = [True, False, True, False]
    stats = detector_stats(scores, labels, threshold=0.5)
    assert stats["precision"] == {"n": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    assert stats["f1"] == 0.0


def test_detector_stats_ci_bounds_are_valid_probabilities():
    scores = [0.1, 0.4, 0.6, 0.7, 0.9] * 4
    labels = [False, False, True, True, True] * 4
    stats = detector_stats(scores, labels, threshold=0.5)
    for metric in ("accuracy", "precision", "recall"):
        m = stats[metric]
        assert 0.0 <= m["ci_low"] <= m["rate"] <= m["ci_high"] <= 1.0


def test_detector_stats_auroc_none_for_single_class():
    scores = [0.1, 0.2, 0.3]
    labels = [True, True, True]
    stats = detector_stats(scores, labels, threshold=0.5)
    assert stats["auroc"] is None


def test_mcnemar_exact_identical_predictions_have_no_discordant_pairs():
    preds = [True, False, True, True, False]
    result = mcnemar_exact(preds, preds)
    assert result == {"only_a": 0, "only_b": 0, "n_discordant": 0, "p_value": 1.0}


def test_mcnemar_exact_symmetric_disagreement_is_not_significant():
    # 3 vs 3 discordant pairs -- perfectly balanced disagreement, should not
    # look significant.
    preds_a = [True, True, True, False, False, False]
    preds_b = [False, False, False, True, True, True]
    result = mcnemar_exact(preds_a, preds_b)
    assert result["n_discordant"] == 6
    assert result["only_a"] == 3
    assert result["only_b"] == 3
    assert result["p_value"] == 1.0


def test_mcnemar_exact_lopsided_disagreement_is_significant():
    # a flags 10 items b doesn't, b flags 0 items a doesn't -- a clearly
    # systematic difference, should read as significant at the usual 0.05.
    preds_a = [True] * 10 + [False] * 10
    preds_b = [False] * 10 + [False] * 10
    result = mcnemar_exact(preds_a, preds_b)
    assert result["only_a"] == 10
    assert result["only_b"] == 0
    assert result["p_value"] < 0.05


def test_delong_auc_test_identical_scores_have_no_difference():
    scores = [0.1, 0.4, 0.3, 0.9, 0.8, 0.95]
    labels = [False, False, False, True, True, True]
    result = delong_auc_test(scores, scores, labels)
    assert result["diff"] == 0.0
    assert result["p_value"] == 1.0


def test_delong_auc_test_aucs_match_sklearn():
    scores_a = [0.2, 0.4, 0.35, 0.9, 0.6, 0.95]
    scores_b = [0.1, 0.3, 0.8, 0.7, 0.85, 0.5]
    labels = [False, False, False, True, True, True]
    result = delong_auc_test(scores_a, scores_b, labels)
    # delong_auc_test rounds to 4 decimals for readability, so compare at
    # that precision rather than exact floating-point equality.
    assert result["auc_a"] == pytest.approx(roc_auc_score(labels, scores_a), abs=1e-4)
    assert result["auc_b"] == pytest.approx(roc_auc_score(labels, scores_b), abs=1e-4)


def test_delong_auc_test_detects_a_real_difference():
    # Classifier A separates the two classes well (with natural overlap/
    # variance, not a degenerate perfect-separation edge case); classifier
    # B is drawn from the same distribution for both classes, i.e. no real
    # separation. Seeded for reproducibility.
    rng = np.random.default_rng(42)
    n = 40
    labels = [False] * n + [True] * n
    scores_a = list(rng.normal(0.25, 0.15, n)) + list(rng.normal(0.75, 0.15, n))
    scores_b = list(rng.normal(0.5, 0.15, n)) + list(rng.normal(0.5, 0.15, n))

    result = delong_auc_test(scores_a, scores_b, labels)
    assert result["auc_a"] > 0.85
    assert 0.3 < result["auc_b"] < 0.7
    assert result["p_value"] < 0.05


def test_cochrans_q_all_agree_is_not_significant():
    preds_matrix = [
        [True, False, True, True, False],
        [True, False, True, True, False],
        [True, False, True, True, False],
    ]
    result = cochrans_q(preds_matrix)
    assert result == {"q": 0.0, "df": 2, "p_value": 1.0}


def test_cochrans_q_detects_a_clear_difference_in_success_rates():
    n = 20
    preds_matrix = [
        [True] * n,  # classifier 1: flags everything
        [False] * n,  # classifier 2: flags nothing
        [True, False] * (n // 2),  # classifier 3: flags half
    ]
    result = cochrans_q(preds_matrix)
    assert result["df"] == 2
    assert result["p_value"] < 0.05


def test_cochrans_q_equal_marginal_rates_gives_zero_q_regardless_of_pattern():
    # Same total flag count (10 of 20) per classifier but different specific
    # items -- Cochran's Q tests marginal success rates, not which items;
    # equal marginals force Q to exactly 0 regardless of the item-level
    # pattern.
    preds_matrix = [
        [True] * 10 + [False] * 10,
        [False] * 10 + [True] * 10,
        [True] * 5 + [False] * 10 + [True] * 5,
    ]
    result = cochrans_q(preds_matrix)
    assert result["q"] == pytest.approx(0.0, abs=1e-9)
