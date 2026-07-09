"""Causal ranking of candidate SAE features via attribution patching
(integrated gradients, N=10 steps) -- step 3 of the methodology in
LITERATURE.md's close-read of arXiv:2505.23556.

For each candidate feature, estimates the causal effect on the
refusal-vs-compliance logit-diff metric (src/direction/refusal_metric.py)
of ablating that feature (its natural pre-activation value -> 0), via
integrated gradients along the straight-line path from 0 to the feature's
natural value.

Implementation note: the N interpolation steps are batched into a single
forward+backward pass per (feature, prompt) rather than run as N separate
passes -- verified (via a standalone probe, not part of this module) to
give numerically identical gradients to looping one step at a time, at
roughly 1/N the wall-clock cost. This matters on a 6GB GPU where each
forward+backward pass is not cheap.
"""

from __future__ import annotations

import torch
from nnsight import LanguageModel

from src.activations.extract import format_prompt
from src.direction.refusal_metric import refusal_logit_diff
from src.sae.qwen_scope import TopKSAE

N_STEPS_DEFAULT = 10


def feature_ig_attribution(
    model: LanguageModel,
    sae: TopKSAE,
    feature_idx: int,
    layer_idx: int,
    prompt: str,
    refusal_id: int,
    compliance_id: int,
    n_steps: int = N_STEPS_DEFAULT,
) -> float:
    """Integrated-gradients attribution of one SAE feature (at one layer) on
    one prompt: the estimated effect on the refusal-vs-compliance metric of
    ablating this feature (natural pre-activation -> 0)."""
    templated = format_prompt(model, prompt)

    with model.trace([templated] * n_steps) as tracer:
        orig_resid = model.model.layers[layer_idx].output[:, -1, :]
        feature_row = sae.W_enc[feature_idx].to(device=orig_resid.device, dtype=orig_resid.dtype)
        bias = sae.b_enc[feature_idx].to(device=orig_resid.device, dtype=orig_resid.dtype)
        direction = sae.feature_direction(feature_idx).to(device=orig_resid.device, dtype=orig_resid.dtype)

        natural_val = orig_resid[0].detach() @ feature_row + bias
        alphas = torch.arange(1, n_steps + 1, dtype=torch.float32, device=orig_resid.device) / n_steps
        v = (natural_val * alphas).clone().requires_grad_(True)
        perturbed = orig_resid.detach() + (v - natural_val).unsqueeze(-1) * direction
        model.model.layers[layer_idx].output[:, -1, :] = perturbed

        logits = model.lm_head.output[:, -1, :]
        metrics = refusal_logit_diff(logits, refusal_id, compliance_id)
        metrics.sum().backward()
        grad = v.grad.clone().save()
        natural_val_saved = natural_val.clone().save()

    # Left-Riemann approximation of the integral from 0 to natural_val.
    return (natural_val_saved * grad.mean()).item()


def rank_pooled_candidates(
    model: LanguageModel,
    saes: dict[int, TopKSAE],
    candidates: list[tuple[int, int]],
    prompts: list[str],
    refusal_id: int,
    compliance_id: int,
    k_star: int = 20,
    n_steps: int = N_STEPS_DEFAULT,
) -> list[tuple[int, int, float]]:
    """candidates: (layer_idx, feature_idx) pairs pooled from
    src.sae.feature_selection.top_k0_by_cosine_similarity across the
    candidate layers. Averages feature_ig_attribution over `prompts` for
    each candidate, then returns the top `k_star` by score, descending
    (highest positive score = feature whose removal would most reduce the
    refusal-leaning metric, i.e. most causally responsible for refusal)."""
    scored = []
    for layer_idx, feature_idx in candidates:
        sae = saes[layer_idx]
        scores = [
            feature_ig_attribution(model, sae, feature_idx, layer_idx, p, refusal_id, compliance_id, n_steps)
            for p in prompts
        ]
        scored.append((layer_idx, feature_idx, sum(scores) / len(scores)))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:k_star]
