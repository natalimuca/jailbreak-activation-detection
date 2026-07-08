"""Difference-of-means refusal direction, per layer, plus a cheap proxy score
for layer selection before running the expensive causal (generation-based)
validation."""

from __future__ import annotations

import torch


def compute_directions(harmful_acts: torch.Tensor, harmless_acts: torch.Tensor) -> torch.Tensor:
    """harmful_acts, harmless_acts: [n_layers, n_prompts, d_model].
    Returns unit directions: [n_layers, d_model]. Used for ablation, where
    only the direction (not magnitude) matters for the projection."""
    diff = harmful_acts.mean(dim=1) - harmless_acts.mean(dim=1)
    return diff / diff.norm(dim=-1, keepdim=True)


def compute_raw_directions(harmful_acts: torch.Tensor, harmless_acts: torch.Tensor) -> torch.Tensor:
    """Same as compute_directions but NOT unit-normalized: the vector's own
    norm already reflects the natural harmful/harmless separation scale at
    that layer, so it's the right thing to add back into the residual stream
    for activation-addition (sufficiency) tests -- a fixed raw magnitude
    (e.g. a hardcoded alpha) has no consistent meaning across layers, since
    residual-stream norms grow substantially with depth."""
    return harmful_acts.mean(dim=1) - harmless_acts.mean(dim=1)


def separation_score(acts: torch.Tensor, directions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-layer separation of harmful vs. harmless projections onto the
    candidate direction, in pooled-std units (a cheap proxy for how causally
    useful a layer's direction is likely to be, before running generation).

    acts: [n_layers, n_prompts, d_model], directions: [n_layers, d_model],
    labels: [n_prompts] bool (True = harmful).
    """
    L = acts.shape[0]
    scores = torch.zeros(L)
    for layer in range(L):
        proj = acts[layer] @ directions[layer]
        harmful_proj = proj[labels]
        harmless_proj = proj[~labels]
        pooled_std = torch.cat([harmful_proj, harmless_proj]).std() + 1e-6
        scores[layer] = (harmful_proj.mean() - harmless_proj.mean()) / pooled_std
    return scores


def select_candidate_layers(scores: torch.Tensor, k: int = 3, skip_frac: float = 0.25) -> list[int]:
    """Top-k layers by separation score, restricted to the middle-to-late
    portion of the network (the original paper finds early layers are noisy
    and late layers are too close to the unembedding to ablate cleanly)."""
    L = len(scores)
    lo = int(L * skip_frac)
    hi = int(L * (1 - skip_frac / 2))
    candidate_idxs = list(range(lo, hi))
    ranked = sorted(candidate_idxs, key=lambda i: scores[i].item(), reverse=True)
    return ranked[:k]
