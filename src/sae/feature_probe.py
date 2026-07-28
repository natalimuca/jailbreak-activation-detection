"""Shared single-feature activation readout, used by any script that needs a
model's raw pre-activation for one SAE feature at the last prompt token
without running a full extraction/generation pass.

Promoted out of `scripts/token_attribution.py`'s private `_feature_value` once
a second script (`scripts/wrapper_swap_variance.py`) needed the identical
helper -- real duplication across two call sites, not a hypothetical one.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def feature_value(model, layer_idx: int, sae, feature_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
    """input_ids: (batch, seq_len). Returns (batch,): the feature's raw
    pre-activation at the last token, per batch row. Matches sae.encode()'s
    own math (linear encoder row + bias) without calling the
    @torch.no_grad()-decorated encode() itself, same reuse reasoning as
    src.sae.causal_ranking._ig_chunk.

    Must stay @torch.no_grad(): omitting it built and retained a full
    autograd graph on every call despite never calling .backward(), causing
    a real cross-iteration memory leak the first time this code was written
    (see scripts/token_attribution.py's module docstring for the full
    debugging account)."""
    with model.trace(input_ids) as tracer:
        resid = model.model.layers[layer_idx].output[:, -1, :]
        feature_row = sae.W_enc[feature_idx].to(device=resid.device, dtype=resid.dtype)
        bias = sae.b_enc[feature_idx].to(device=resid.device, dtype=resid.dtype)
        vals = (resid @ feature_row + bias).save()
    return vals.detach().cpu()
