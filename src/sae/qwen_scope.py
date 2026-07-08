"""Loader for Qwen-Scope pretrained SAEs (arXiv:2605.11887).

Repo layout confirmed from `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`'s own
README: one `layer{n}.sae.pt` file per transformer layer (0-35), each a
plain torch dict with W_enc (d_sae, d_model), W_dec (d_model, d_sae),
b_enc (d_sae,), b_dec (d_model,). TopK SAE: at each forward pass, only the
top-K pre-activations survive, everything else is zeroed.

These SAEs are trained on Qwen3-8B-**Base** activations, not the chat model
we actually run (Qwen3-8B). This is a documented, accepted compromise (see
DECISIONS.md and the close-read of arXiv:2505.23556 in LITERATURE.md) --
each `layer{n}.sae.pt` is ~2GB (two 65536x4096 fp32 matrices), so only the
specific layer(s) selected by separation-score analysis are downloaded,
never the full 36-layer set.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
TOP_K = 50
D_MODEL = 4096
D_SAE = 65536


class TopKSAE:
    def __init__(self, W_enc: torch.Tensor, W_dec: torch.Tensor, b_enc: torch.Tensor, b_dec: torch.Tensor, k: int = TOP_K):
        self.W_enc = W_enc
        self.W_dec = W_dec
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.k = k

    def to(self, *args, **kwargs) -> "TopKSAE":
        self.W_enc = self.W_enc.to(*args, **kwargs)
        self.W_dec = self.W_dec.to(*args, **kwargs)
        self.b_enc = self.b_enc.to(*args, **kwargs)
        self.b_dec = self.b_dec.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """residual: (..., d_model) -> sparse feature activations (..., d_sae).
        Matches Qwen-Scope's own reference implementation exactly (see
        REPO_ID's README): no decoder-bias subtraction before the encoder,
        just a plain linear + top-k."""
        pre_acts = residual.to(self.W_enc.dtype) @ self.W_enc.T + self.b_enc
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts

    @torch.no_grad()
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """acts: (..., d_sae) -> reconstructed residual stream (..., d_model)."""
        return acts.to(self.W_dec.dtype) @ self.W_dec.T + self.b_dec

    def feature_direction(self, feature_idx: int) -> torch.Tensor:
        """The residual-stream direction a single feature writes to when
        active -- i.e. the corresponding column of the decoder. This is what
        should be compared (cosine similarity) against a difference-of-means
        refusal direction, not the encoder row."""
        return self.W_dec[:, feature_idx]


def download_sae_checkpoint(layer: int, repo_id: str = REPO_ID) -> dict:
    path = hf_hub_download(repo_id, f"layer{layer}.sae.pt")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_sae(layer: int, repo_id: str = REPO_ID, device: str = "cpu", dtype: torch.dtype = torch.float32) -> TopKSAE:
    ckpt = download_sae_checkpoint(layer, repo_id)
    sae = TopKSAE(ckpt["W_enc"], ckpt["W_dec"], ckpt["b_enc"], ckpt["b_dec"])
    return sae.to(device=device, dtype=dtype)
