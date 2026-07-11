from src.eval.detector_metrics import classify, detector_stats, mcnemar_exact, youden_threshold


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
