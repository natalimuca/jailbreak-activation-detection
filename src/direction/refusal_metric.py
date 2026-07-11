"""Differentiable refusal-vs-compliance logit-difference metric.

Used as the optimization target for attribution-patching-based causal
ranking of SAE features (LITERATURE.md's close-read of arXiv:2505.23556,
step 3) -- NOT for the final suppression validation, which uses the real
`refusal_classifier` on generated text instead (a differentiable metric is
required here only because integrated gradients needs to backprop through
it; a keyword classifier on generated text isn't differentiable).

Token choice: " I" (leading space) vs " Sure" (leading space) -- the same
refusal-vs-compliance first-token contrast used in Arditi et al. 2024
(arXiv:2406.11717), the paper this project's Phase 1 already replicates.
Both are confirmed single tokens for Qwen3-8B's tokenizer.
"""

from __future__ import annotations

import torch

REFUSAL_TOKEN = " I"
COMPLIANCE_TOKEN = " Sure"


def refusal_compliance_token_ids(tokenizer) -> tuple[int, int]:
    refusal_id = tokenizer.encode(REFUSAL_TOKEN, add_special_tokens=False)[0]
    compliance_id = tokenizer.encode(COMPLIANCE_TOKEN, add_special_tokens=False)[0]
    return refusal_id, compliance_id


def refusal_logit_diff(logits: torch.Tensor, refusal_id: int, compliance_id: int) -> torch.Tensor:
    """logits: (..., vocab) -> (...): logit(refusal_token) - logit(compliance_token).
    Higher means the model is more inclined to start its reply with "I ..."
    (refusal) than "Sure ..." (compliance)."""
    return logits[..., refusal_id] - logits[..., compliance_id]
