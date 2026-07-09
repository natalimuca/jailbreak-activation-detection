import torch

from src.sae.feature_selection import top_k0_by_cosine_similarity
from src.sae.qwen_scope import TopKSAE


def _toy_sae(d_model=4, d_sae=8, k=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(d_sae, d_model, generator=g)
    W_dec = torch.randn(d_model, d_sae, generator=g)
    b_enc = torch.zeros(d_sae)
    b_dec = torch.zeros(d_model)
    return TopKSAE(W_enc, W_dec, b_enc, b_dec, k=k)


def test_returns_k0_indices():
    sae = _toy_sae()
    direction = torch.randn(4)
    result = top_k0_by_cosine_similarity(sae, direction, k0=3)
    assert len(result) == 3
    assert len(set(result)) == 3  # no duplicates


def test_picks_the_feature_aligned_with_the_direction():
    d_model, d_sae = 4, 6
    W_dec = torch.randn(d_model, d_sae)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # Force feature 2's decoder column to point exactly along `direction`.
    W_dec[:, 2] = direction * 5.0
    sae = TopKSAE(torch.randn(d_sae, d_model), W_dec, torch.zeros(d_sae), torch.zeros(d_model))

    result = top_k0_by_cosine_similarity(sae, direction, k0=1)
    assert result == [2]


def test_ranking_is_by_signed_not_absolute_similarity():
    d_model, d_sae = 4, 3
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    W_dec = torch.zeros(d_model, d_sae)
    W_dec[:, 0] = direction  # perfectly aligned
    W_dec[:, 1] = -direction  # perfectly anti-aligned
    W_dec[:, 2] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # orthogonal
    sae = TopKSAE(torch.randn(d_sae, d_model), W_dec, torch.zeros(d_sae), torch.zeros(d_model))

    result = top_k0_by_cosine_similarity(sae, direction, k0=1)
    assert result == [0]


def test_k0_larger_than_d_sae_raises():
    sae = _toy_sae(d_sae=4)
    direction = torch.randn(4)
    try:
        top_k0_by_cosine_similarity(sae, direction, k0=10)
        assert False, "expected an error when k0 > d_sae"
    except RuntimeError:
        pass
