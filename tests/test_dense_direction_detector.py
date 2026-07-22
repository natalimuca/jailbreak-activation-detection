import json

import torch

from src.detectors import dense_direction_detector
from src.detectors.dense_direction_detector import (
    calibrate,
    is_flagged,
    project,
    resolve_layer_for_model,
    select_layer_and_calibrate,
)


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


def test_select_layer_and_calibrate_picks_the_separating_layer_using_val_not_test():
    # 8 prompts: harmful = [0,1,2,3], harmless = [4,5,6,7].
    labels = torch.tensor([True, True, True, True, False, False, False, False])
    is_train = torch.tensor([True, True, False, False, True, True, False, False])
    is_val = torch.tensor([False, False, True, True, False, False, True, True])

    # Layer 0 cleanly separates on train AND val; layer 1 separates on train
    # but is scrambled on val (would look great if layer selection touched
    # TEST/train stats instead of VAL) -- this test would fail if
    # select_layer_and_calibrate used anything other than is_val for scoring.
    layer0 = torch.tensor([10.0, 11.0, 12.0, 13.0, 0.0, 1.0, 2.0, 3.0]).unsqueeze(-1)
    layer1 = torch.tensor([10.0, 11.0, 5.0, 5.0, 0.0, 1.0, 5.0, 5.0]).unsqueeze(-1)
    acts = torch.stack([layer0, layer1])  # [2 layers, 8 prompts, 1 dim]

    layer, direction, threshold = select_layer_and_calibrate(acts, labels, is_train, is_val)
    assert layer == 0
    val_scores = project(acts[0, is_val, :], direction)
    val_labels = labels[is_val]
    # threshold from youden_threshold is one of the observed score values
    # (roc_curve's convention), so the harmful side uses >= (matching
    # `classify`'s own s >= threshold rule) while the harmless side must be
    # strictly below it.
    assert val_scores[val_labels].min() >= threshold > val_scores[~val_labels].max()


def test_resolve_layer_for_model_uses_ablation_file_for_qwen3_8b(tmp_path, monkeypatch):
    ablation_path = tmp_path / "dense_direction_ablation_Qwen3-8B.json"
    ablation_path.write_text(json.dumps({"ablation_layer": 23}))
    monkeypatch.setattr(dense_direction_detector, "QWEN3_ABLATION_LAYER_PATH", ablation_path)

    assert resolve_layer_for_model("Qwen/Qwen3-8B") == 23
    assert resolve_layer_for_model("Qwen3-8B") == 23


def test_resolve_layer_for_model_uses_cross_model_file_for_other_models(tmp_path, monkeypatch):
    cross_model_path = tmp_path / "dense_direction_cross_model.json"
    cross_model_path.write_text(json.dumps({
        "Llama-3.1-8B-Instruct": {"layer": 27, "threshold": 0.5},
        "gemma-2-9b-it": {"layer": 34, "threshold": 0.4},
    }))
    monkeypatch.setattr(dense_direction_detector, "CROSS_MODEL_RESULTS_PATH", cross_model_path)

    assert resolve_layer_for_model("meta-llama/Llama-3.1-8B-Instruct") == 27
    assert resolve_layer_for_model("google/gemma-2-9b-it") == 34
