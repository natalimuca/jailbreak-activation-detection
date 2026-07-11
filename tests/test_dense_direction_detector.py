import torch

from src.detectors.dense_direction_detector import calibrate, is_flagged, project


def test_project_returns_dot_product_with_direction():
    activations = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    direction = torch.tensor([1.0, 0.0])
    scores = project(activations, direction)
    assert torch.allclose(scores, torch.tensor([1.0, 0.0, 2.0]))


def test_project_separates_harmful_from_harmless_on_synthetic_data():
    harmful = torch.tensor([[5.0, 0.0], [6.0, 0.0], [5.5, 0.0]])
    harmless = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    direction = torch.tensor([1.0, 0.0])
    harmful_scores = project(harmful, direction)
    harmless_scores = project(harmless, direction)
    assert harmful_scores.min() > harmless_scores.max()


def test_calibrate_finds_separating_threshold():
    activations = torch.tensor([[0.0], [1.0], [8.0], [9.0]])
    labels = [False, False, True, True]
    direction = torch.tensor([1.0])
    threshold = calibrate(activations, labels, direction)
    assert 1.0 < threshold <= 8.0


def test_is_flagged_thresholding():
    assert is_flagged(score=5.0, threshold=3.0)
    assert not is_flagged(score=2.0, threshold=3.0)
