import json

import torch

from src.detectors.sae_feature_detector import (
    calibrate,
    is_flagged,
    load_top_features,
    score,
)
from src.sae.qwen_scope import TopKSAE


def _make_sae(k: int) -> TopKSAE:
    # d_model=2, d_sae=3. Feature 0 reads dim0, feature 1 reads dim1,
    # feature 2 reads dim0+dim1 -- lets tests target one feature cleanly.
    W_enc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    W_dec = torch.zeros(2, 3)
    b_enc = torch.zeros(3)
    b_dec = torch.zeros(2)
    return TopKSAE(W_enc, W_dec, b_enc, b_dec, k=k)


def test_score_sums_selected_feature_activations_no_truncation():
    sae = _make_sae(k=3)  # keep-all, so no top-k truncation ambiguity
    activations_by_layer = {0: torch.tensor([[5.0, 0.0], [0.0, 3.0]])}
    s = score(activations_by_layer, {0: sae}, features=[(0, 0)])
    assert torch.allclose(s, torch.tensor([5.0, 0.0]))


def test_score_sums_across_multiple_features_and_layers():
    sae0 = _make_sae(k=3)
    sae1 = _make_sae(k=3)
    activations_by_layer = {
        0: torch.tensor([[5.0, 0.0]]),
        1: torch.tensor([[0.0, 4.0]]),
    }
    s = score(activations_by_layer, {0: sae0, 1: sae1}, features=[(0, 0), (1, 1)])
    assert torch.allclose(s, torch.tensor([9.0]))


def test_score_zeroes_features_dropped_by_topk_sparsity():
    sae = _make_sae(k=1)  # only the single largest pre-activation survives
    # dim0=5 makes feature0's pre-act (5) and feature2's pre-act (5) tie for
    # largest, feature1's pre-act (0) is guaranteed dropped either way.
    activations_by_layer = {0: torch.tensor([[0.0, 5.0]])}
    s = score(activations_by_layer, {0: sae}, features=[(0, 0)])
    assert s.item() == 0.0  # feature0's pre-act is 0, never survives top-1


def test_calibrate_finds_separating_threshold():
    sae = _make_sae(k=3)
    val_acts = {0: torch.tensor([[0.0, 0.0], [1.0, 0.0], [8.0, 0.0], [9.0, 0.0]])}
    val_labels = [False, False, True, True]
    threshold = calibrate(val_acts, val_labels, {0: sae}, features=[(0, 0)])
    assert 1.0 < threshold <= 8.0


def test_is_flagged_thresholding():
    assert is_flagged(5.0, threshold=3.0)
    assert not is_flagged(2.0, threshold=3.0)


def test_load_top_features_reads_ranked_json(tmp_path):
    payload = {
        "ranked_features": [
            {"layer": 25, "feature": 65291, "score": 2.2},
            {"layer": 23, "feature": 42331, "score": 1.4},
            {"layer": 24, "feature": 5393, "score": 0.45},
        ]
    }
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps(payload))
    top2 = load_top_features(k=2, path=path)
    assert top2 == [(25, 65291), (23, 42331)]
