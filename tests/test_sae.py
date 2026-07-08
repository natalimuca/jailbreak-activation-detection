import torch

from src.sae.qwen_scope import TopKSAE


def _toy_sae(d_model=4, d_sae=8, k=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(d_sae, d_model, generator=g)
    W_dec = torch.randn(d_model, d_sae, generator=g)
    b_enc = torch.zeros(d_sae)
    b_dec = torch.zeros(d_model)
    return TopKSAE(W_enc, W_dec, b_enc, b_dec, k=k)


def test_encode_keeps_exactly_k_nonzero_per_row():
    sae = _toy_sae(k=2)
    residual = torch.randn(5, 4)
    acts = sae.encode(residual)
    assert acts.shape == (5, 8)
    nonzero_counts = (acts != 0).sum(dim=-1)
    assert (nonzero_counts == 2).all()


def test_encode_keeps_the_actual_top_k_values():
    sae = _toy_sae(k=2)
    residual = torch.randn(1, 4)
    pre_acts = residual @ sae.W_enc.T + sae.b_enc
    acts = sae.encode(residual)
    expected_top2 = pre_acts.topk(2, dim=-1).values.sort(descending=True).values
    actual_nonzero = acts[acts != 0].sort(descending=True).values
    assert torch.allclose(expected_top2.flatten(), actual_nonzero, atol=1e-5)


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


def test_encode_respects_bias():
    sae = _toy_sae(k=1)
    sae.b_enc = torch.tensor([100.0, 0, 0, 0, 0, 0, 0, 0])  # force feature 0 to always win
    residual = torch.randn(2, 4)
    acts = sae.encode(residual)
    assert (acts[:, 0] != 0).all()
    assert (acts[:, 1:] == 0).all()


def test_to_moves_all_tensors():
    sae = _toy_sae()
    sae.to(dtype=torch.float64)
    assert sae.W_enc.dtype == torch.float64
    assert sae.W_dec.dtype == torch.float64
    assert sae.b_enc.dtype == torch.float64
    assert sae.b_dec.dtype == torch.float64
