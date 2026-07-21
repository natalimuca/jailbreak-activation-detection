"""End-to-end sanity checks for attribution-patching causal ranking.
Uses the small, already-cached SmolLM2 model with toy (randomly initialized)
SAEs -- slow enough to need a real model/GPU-or-CPU forward+backward pass,
so marked separately from the fast default suite (see pytest.ini)."""

import pytest
import torch

from src.direction.refusal_metric import refusal_compliance_token_ids
from src.sae.causal_ranking import feature_ig_attribution, rank_pooled_candidates
from src.sae.qwen_scope import TopKSAE

pytestmark = pytest.mark.model

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


@pytest.fixture(scope="module")
def model():
    from src.activations.extract import load_model

    return load_model(MODEL_NAME)


def _toy_sae(d_model, d_sae=16, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(d_sae, d_model, generator=g) * 0.1
    W_dec = torch.randn(d_model, d_sae, generator=g) * 0.1
    b_enc = torch.zeros(d_sae)
    b_dec = torch.zeros(d_model)
    return TopKSAE(W_enc, W_dec, b_enc, b_dec, k=k)


def test_feature_ig_attribution_returns_a_finite_float(model):
    d_model = model.config.hidden_size
    sae = _toy_sae(d_model)
    refusal_id, compliance_id = refusal_compliance_token_ids(model.tokenizer)
    mid_layer = len(model.model.layers) // 2

    score = feature_ig_attribution(
        model, sae, feature_idx=0, layer_idx=mid_layer,
        prompt="Tell me a fact about the ocean.",
        refusal_id=refusal_id, compliance_id=compliance_id, n_steps=4,
    )
    assert isinstance(score, float)
    assert score == score  # not NaN
    assert abs(score) < 1e6  # not exploded


def test_feature_ig_attribution_micro_batching_matches_full_batch(model):
    """micro_batch_size splits the N interpolation steps into smaller
    forward+backward passes (needed for Gemma-2-9B, which OOMs at full
    batch size on a 6GB GPU -- see DECISIONS.md); this only saves memory if
    it's mathematically equivalent to the full-batch default."""
    d_model = model.config.hidden_size
    sae = _toy_sae(d_model)
    refusal_id, compliance_id = refusal_compliance_token_ids(model.tokenizer)
    mid_layer = len(model.model.layers) // 2
    kwargs = dict(
        model=model, sae=sae, feature_idx=0, layer_idx=mid_layer,
        prompt="Tell me a fact about the ocean.",
        refusal_id=refusal_id, compliance_id=compliance_id, n_steps=4,
    )

    full_batch_score = feature_ig_attribution(**kwargs, micro_batch_size=None)
    chunked_score = feature_ig_attribution(**kwargs, micro_batch_size=2)
    fully_sequential_score = feature_ig_attribution(**kwargs, micro_batch_size=1)

    # rel=1e-2, not bit-exact: different batch sizes take different matmul
    # kernel paths, so floating-point rounding differs slightly even though
    # the underlying computation is the same (matches the "numerically
    # identical" standard the module docstring's own full-vs-single-step
    # probe used).
    assert full_batch_score == pytest.approx(chunked_score, rel=1e-2)
    assert full_batch_score == pytest.approx(fully_sequential_score, rel=1e-2)


def test_rank_pooled_candidates_respects_k_star_and_sorts_descending(model):
    d_model = model.config.hidden_size
    L = len(model.model.layers)
    mid = L // 2
    saes = {mid: _toy_sae(d_model), mid + 1: _toy_sae(d_model, seed=1)}
    candidates = [(mid, i) for i in range(3)] + [(mid + 1, i) for i in range(3)]
    refusal_id, compliance_id = refusal_compliance_token_ids(model.tokenizer)
    prompts = ["Tell me a fact.", "Explain photosynthesis briefly."]

    result = rank_pooled_candidates(
        model, saes, candidates, prompts, refusal_id, compliance_id, k_star=4, n_steps=3,
    )
    assert len(result) == 4
    scores = [s for _, _, s in result]
    assert scores == sorted(scores, reverse=True)
    for layer_idx, feature_idx, _ in result:
        assert (layer_idx, feature_idx) in candidates
