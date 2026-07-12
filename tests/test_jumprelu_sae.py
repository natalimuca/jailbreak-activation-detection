import pytest
import torch

from src.sae.jumprelu_sae import JumpReLUSAE


def _toy_sae(d_model=4, d_sae=8, threshold=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(d_sae, d_model, generator=g)
    W_dec = torch.randn(d_model, d_sae, generator=g)
    b_enc = torch.zeros(d_sae)
    b_dec = torch.zeros(d_model)
    return JumpReLUSAE(W_enc, W_dec, b_enc, b_dec, threshold=threshold)


def test_encode_zeroes_pre_activations_at_or_below_threshold():
    sae = _toy_sae(d_model=2, d_sae=3, threshold=0.5)
    sae.W_enc = torch.eye(3, 2)  # feature i reads dim i directly (dim2 unused)
    sae.b_enc = torch.zeros(3)
    residual = torch.tensor([[0.1, 0.6]])  # feature0 pre-act=0.1 (<=thr), feature1=0.6 (>thr)
    acts = sae.encode(residual)
    assert acts[0, 0].item() == 0.0
    assert acts[0, 1].item() == pytest.approx(0.6)


def test_encode_passes_through_unclipped_above_threshold():
    sae = _toy_sae(d_model=2, d_sae=2, threshold=0.5)
    sae.W_enc = torch.eye(2)
    sae.b_enc = torch.zeros(2)
    residual = torch.tensor([[5.0, 0.0]])
    acts = sae.encode(residual)
    # value passes through as-is, not clipped down to the threshold
    assert acts[0, 0].item() == 5.0


def test_encode_supports_per_feature_threshold_tensor():
    sae = _toy_sae(d_model=2, d_sae=2, threshold=torch.tensor([0.1, 10.0]))
    sae.W_enc = torch.eye(2)
    sae.b_enc = torch.zeros(2)
    residual = torch.tensor([[1.0, 1.0]])
    acts = sae.encode(residual)
    assert acts[0, 0].item() == 1.0  # 1.0 > 0.1 threshold -> survives
    assert acts[0, 1].item() == 0.0  # 1.0 <= 10.0 threshold -> zeroed


def test_decode_matches_manual_matmul():
    sae = _toy_sae()
    acts = torch.randn(3, 8)
    decoded = sae.decode(acts)
    expected = acts @ sae.W_dec.T + sae.b_dec
    assert torch.allclose(decoded, expected)


def test_feature_direction_is_decoder_column():
    sae = _toy_sae()
    direction = sae.feature_direction(3)
    assert torch.allclose(direction, sae.W_dec[:, 3])
    assert direction.shape == (4,)


def test_to_moves_all_tensors_including_tensor_threshold():
    sae = _toy_sae(threshold=torch.tensor(0.5))
    sae.to(dtype=torch.float64)
    assert sae.W_enc.dtype == torch.float64
    assert sae.W_dec.dtype == torch.float64
    assert sae.b_enc.dtype == torch.float64
    assert sae.b_dec.dtype == torch.float64
    assert sae.threshold.dtype == torch.float64


def test_to_leaves_scalar_threshold_untouched():
    sae = _toy_sae(threshold=0.5)
    sae.to(dtype=torch.float64)
    assert sae.threshold == 0.5
