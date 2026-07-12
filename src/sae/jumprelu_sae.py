"""JumpReLU SAE: JumpReLU(z) = z * (z > threshold), i.e. each pre-activation
is zeroed unless it clears a threshold, otherwise passed through unchanged.
Unlike `qwen_scope.TopKSAE`'s fixed per-forward-pass sparsity (exactly k
features survive every input), JumpReLU thresholds each feature
independently -- the number of surviving features varies per input.

Added for Phase 6 Wave 2: both LlamaScope and GemmaScope's actually-released
checkpoints use JumpReLU (verified directly by inspecting real checkpoint
`hyperparams.json`/metadata this session -- LlamaScope's own paper framing
as "TopK SAEs" describes their training recipe, not the activation function
of the artifacts actually published on Hugging Face). One shared class
here, with per-provider checkpoint loaders (`src/sae/llama_scope.py`,
`src/sae/gemma_scope.py`) supplying the weights and threshold.

Same attribute names and shape conventions as `TopKSAE` (`W_enc`: (d_sae,
d_model), `W_dec`: (d_model, d_sae), `b_enc`: (d_sae,), `b_dec`:
(d_model,)) so it's a drop-in for `src/sae/feature_selection.py`,
`src/sae/causal_ranking.py`, and `src/direction/interventions.py`, all of
which only touch these attributes/methods, not TopKSAE specifically.
"""

from __future__ import annotations

import torch


class JumpReLUSAE:
    def __init__(
        self,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: float | torch.Tensor,
    ):
        """threshold: a scalar (one cutoff for every feature, e.g.
        LlamaScope's checkpoints) or a (d_sae,) tensor (per-feature
        cutoffs, e.g. GemmaScope's)."""
        self.W_enc = W_enc
        self.W_dec = W_dec
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.threshold = threshold

    def to(self, *args, **kwargs) -> "JumpReLUSAE":
        self.W_enc = self.W_enc.to(*args, **kwargs)
        self.W_dec = self.W_dec.to(*args, **kwargs)
        self.b_enc = self.b_enc.to(*args, **kwargs)
        self.b_dec = self.b_dec.to(*args, **kwargs)
        if isinstance(self.threshold, torch.Tensor):
            self.threshold = self.threshold.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """residual: (..., d_model) -> sparse feature activations (..., d_sae).
        Pre-activations at or below the threshold are zeroed; everything
        else passes through unchanged (not clipped to the threshold)."""
        pre_acts = residual.to(self.W_enc.dtype) @ self.W_enc.T + self.b_enc
        mask = pre_acts > self.threshold
        return pre_acts * mask

    @torch.no_grad()
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """acts: (..., d_sae) -> reconstructed residual stream (..., d_model)."""
        return acts.to(self.W_dec.dtype) @ self.W_dec.T + self.b_dec

    def feature_direction(self, feature_idx: int) -> torch.Tensor:
        """The residual-stream direction a single feature writes to when
        active -- the corresponding column of the decoder."""
        return self.W_dec[:, feature_idx]
