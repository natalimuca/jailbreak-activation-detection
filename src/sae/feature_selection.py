"""Restrict an SAE's full feature space down to a small candidate set aligned
with a difference-of-means direction -- step 2 of the methodology in
"Understanding Refusal in Language Models with Sparse Autoencoders"
(arXiv:2505.23556, see LITERATURE.md): top K0=10 features per layer by
cosine similarity between each feature's decoder direction and the refusal
direction, before the (separate, model-dependent) causal ranking step."""

from __future__ import annotations

import torch

from src.sae.qwen_scope import TopKSAE

K0_DEFAULT = 10


def top_k0_by_cosine_similarity(sae: TopKSAE, direction: torch.Tensor, k0: int = K0_DEFAULT) -> list[int]:
    """direction: (d_model,) unit or non-unit vector (e.g. from
    compute_directions/compute_raw_directions). Returns the k0 feature
    indices whose decoder direction (sae.feature_direction(i), i.e. the
    corresponding column of W_dec) has the highest cosine similarity to
    `direction` -- signed, not absolute, since we want features that write
    *along* the harmful direction, not merely correlated with its axis."""
    direction = direction.to(sae.W_dec.dtype)
    decoder_dirs = sae.W_dec  # (d_model, d_sae)
    sims = torch.nn.functional.cosine_similarity(
        decoder_dirs.T, direction.unsqueeze(0).expand(decoder_dirs.shape[1], -1), dim=-1
    )
    topk = sims.topk(k0)
    return topk.indices.tolist()


def pool_top_k0_across_layers(
    saes: dict[int, TopKSAE], directions: dict[int, torch.Tensor], k0: int = K0_DEFAULT
) -> list[tuple[int, int]]:
    """Runs top_k0_by_cosine_similarity independently per layer and pools the
    results into a flat (layer_idx, feature_idx) candidate list -- step 2 of
    the methodology, extended across multiple layers (see DECISIONS.md's
    "Multi-layer K0 pooling" entry for why: K*=20 > K0=10 only makes sense
    if candidates are pooled from more than one layer)."""
    pooled = []
    for layer_idx, sae in saes.items():
        for feature_idx in top_k0_by_cosine_similarity(sae, directions[layer_idx], k0):
            pooled.append((layer_idx, feature_idx))
    return pooled
