import pytest
import torch

from src.sae.eleuther_scope import load_sae
from src.sae.qwen_scope import TopKSAE

pytestmark = pytest.mark.network


def test_load_sae_returns_correctly_shaped_topk_sae():
    sae = load_sae(layer=14)
    assert isinstance(sae, TopKSAE)
    assert sae.W_enc.shape == (65536, 1536)
    assert sae.W_dec.shape == (1536, 65536)
    assert sae.b_enc.shape == (65536,)
    assert sae.b_dec.shape == (1536,)
    assert sae.k == 32


def test_load_sae_encode_decode_roundtrip_shapes():
    sae = load_sae(layer=14)
    residual = torch.randn(3, 1536)
    acts = sae.encode(residual)
    assert acts.shape == (3, 65536)
    assert (acts != 0).sum(dim=-1).tolist() == [32, 32, 32]
    decoded = sae.decode(acts)
    assert decoded.shape == (3, 1536)
