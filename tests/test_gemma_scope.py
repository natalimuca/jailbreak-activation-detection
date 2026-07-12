"""Integration test only -- hits the network to fetch a real GemmaScope
checkpoint, matching the convention in tests/test_llama_scope.py. No
synthetic unit tests here since `download_sae_checkpoint`/`load_sae` are
thin wrappers with no logic of their own beyond what JumpReLUSAE's own
tests already cover (tests/test_jumprelu_sae.py) and the W_enc/W_dec
transpose, which this test verifies directly against the real checkpoint's
raw (untransposed) shapes."""

import pytest
import torch

from src.sae.gemma_scope import download_sae_checkpoint, load_sae
from src.sae.jumprelu_sae import JumpReLUSAE

pytestmark = pytest.mark.network

D_MODEL = 3584
D_SAE = 131072


def test_download_sae_checkpoint_raw_shapes_are_transposed_from_jumprelu_convention():
    # GemmaScope's own params.npz stores W_enc as (d_model, d_sae) and
    # W_dec as (d_sae, d_model) -- the opposite of JumpReLUSAE's expected
    # layout. This test pins that raw layout down so a future GemmaScope
    # release changing it doesn't silently break `load_sae`'s transpose.
    params, hparams = download_sae_checkpoint(layer=34)
    assert params["W_enc"].shape == (D_MODEL, D_SAE)
    assert params["W_dec"].shape == (D_SAE, D_MODEL)
    assert params["threshold"].shape == (D_SAE,)


def test_load_sae_returns_correctly_shaped_jumprelu_sae():
    sae = load_sae(layer=34)
    assert isinstance(sae, JumpReLUSAE)
    assert sae.W_enc.shape == (D_SAE, D_MODEL)
    assert sae.W_dec.shape == (D_MODEL, D_SAE)
    assert sae.b_enc.shape == (D_SAE,)
    assert sae.b_dec.shape == (D_MODEL,)
    assert isinstance(sae.threshold, torch.Tensor)
    assert sae.threshold.shape == (D_SAE,)


def test_load_sae_encode_decode_roundtrip_shapes():
    sae = load_sae(layer=34)
    residual = torch.randn(3, D_MODEL)
    acts = sae.encode(residual)
    assert acts.shape == (3, D_SAE)
    decoded = sae.decode(acts)
    assert decoded.shape == (3, D_MODEL)
