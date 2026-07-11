from src.eval.detector_metrics import classify, detector_stats, youden_threshold


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
