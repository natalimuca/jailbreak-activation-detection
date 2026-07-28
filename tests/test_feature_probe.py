"""Real-model sanity check for feature_value -- mirrors test_causal_ranking.py's
pattern (small cached SmolLM2 + a toy random SAE) since nnsight's model.trace()
context/proxy semantics aren't meaningfully stubbable, unlike plain tensor math."""

import pytest
import torch

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


def test_feature_value_returns_finite_per_row_scores(model):
    from src.sae.feature_probe import feature_value

    d_model = model.config.hidden_size
    sae = _toy_sae(d_model)
    mid_layer = len(model.model.layers) // 2

    input_ids = model.tokenizer(["Tell me a fact about the ocean."], return_tensors="pt")["input_ids"]
    vals = feature_value(model, mid_layer, sae, feature_idx=0, input_ids=input_ids)

    assert vals.shape == (1,)
    assert torch.isfinite(vals).all()


def test_feature_value_matches_manual_encoder_row_math(model):
    """feature_value should equal a plain (resid @ W_enc[feature] + b_enc[feature])
    read at the last token -- verified independently via a hook, not by
    re-deriving the same nnsight trace call."""
    from src.sae.feature_probe import feature_value

    d_model = model.config.hidden_size
    sae = _toy_sae(d_model)
    layer_idx = len(model.model.layers) // 2
    feature_idx = 3

    input_ids = model.tokenizer(["Explain photosynthesis briefly."], return_tensors="pt")["input_ids"]
    input_ids_on_device = input_ids.to(model.device)

    captured = {}

    def _capture(_module, _input, output):
        resid = output[0] if isinstance(output, tuple) else output
        captured["resid"] = resid[:, -1, :].detach()

    handle = model.model.layers[layer_idx].register_forward_hook(_capture)
    try:
        with torch.no_grad():
            model(input_ids_on_device)
    finally:
        handle.remove()

    resid = captured["resid"]
    w_row = sae.W_enc[feature_idx].to(device=resid.device, dtype=resid.dtype)
    bias = sae.b_enc[feature_idx].to(device=resid.device, dtype=resid.dtype)
    expected = resid @ w_row + bias
    actual = feature_value(model, layer_idx, sae, feature_idx, input_ids)

    assert actual == pytest.approx(expected.item(), abs=1e-3)


def test_feature_value_batches_multiple_rows(model):
    from src.sae.feature_probe import feature_value

    d_model = model.config.hidden_size
    sae = _toy_sae(d_model)
    mid_layer = len(model.model.layers) // 2

    input_ids = model.tokenizer(
        ["Tell me a fact.", "Explain photosynthesis briefly."],
        return_tensors="pt", padding=True,
    )["input_ids"]
    vals = feature_value(model, mid_layer, sae, feature_idx=0, input_ids=input_ids)

    assert vals.shape == (2,)
    assert torch.isfinite(vals).all()
