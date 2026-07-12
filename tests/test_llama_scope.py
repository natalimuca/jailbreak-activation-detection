"""Integration test only -- hits the network to fetch a real LlamaScope
checkpoint, matching the convention in tests/test_loaders.py. No synthetic
unit tests here since `download_sae_checkpoint`/`load_sae` are thin
wrappers with no logic of their own beyond what JumpReLUSAE's own tests
already cover (tests/test_jumprelu_sae.py)."""

import pytest
import torch

from src.sae.jumprelu_sae import JumpReLUSAE
from src.sae.llama_scope import load_sae

pytestmark = pytest.mark.network


def test_load_sae_returns_correctly_shaped_jumprelu_sae():
    sae = load_sae(layer=15)
    assert isinstance(sae, JumpReLUSAE)
    assert sae.W_enc.shape == (32768, 4096)
    assert sae.W_dec.shape == (4096, 32768)
    assert sae.b_enc.shape == (32768,)
    assert sae.b_dec.shape == (4096,)
    assert isinstance(sae.threshold, float)


def test_load_sae_encode_decode_roundtrip_shapes():
    sae = load_sae(layer=15)
    residual = torch.randn(3, 4096)
    acts = sae.encode(residual)
    assert acts.shape == (3, 32768)
    decoded = sae.decode(acts)
    assert decoded.shape == (3, 4096)
