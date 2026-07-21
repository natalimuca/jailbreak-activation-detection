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

`micro_batch_size` (default: batch all N steps at once, matching the above)
lets the N steps be split into smaller chunks, each its own
forward+backward pass, with per-chunk gradients averaged together at the
end -- mathematically identical to full batching (each interpolation
step's gradient is independent; the chunking is purely about how many are
computed in one pass), just slower. Needed for Gemma-2-9B: its 42 layers
leave much less headroom after the quantized weights than Llama-3.1-8B's
32 (confirmed via a real OOM at the default full batch size -- see
DECISIONS.md), so it needs a smaller per-pass batch to fit in 6GB.
"""

from __future__ import annotations

import torch
from nnsight import LanguageModel

from src.activations.extract import format_prompt
from src.direction.refusal_metric import refusal_logit_diff
from src.sae.qwen_scope import TopKSAE

N_STEPS_DEFAULT = 10


def _ig_chunk(
    model: LanguageModel,
    sae: TopKSAE,
    feature_idx: int,
    layer_idx: int,
    templated: str,
    refusal_id: int,
    compliance_id: int,
    chunk_alphas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward+backward pass over a chunk of interpolation steps.
    Returns (natural_val, grad) for this chunk -- natural_val is the same
    regardless of chunk (recomputed each call since it depends on the
    unmodified residual, not on which alphas are in this chunk)."""
    chunk_size = len(chunk_alphas)
    with model.trace([templated] * chunk_size) as tracer:
        orig_resid = model.model.layers[layer_idx].output[:, -1, :]
        feature_row = sae.W_enc[feature_idx].to(device=orig_resid.device, dtype=orig_resid.dtype)
        bias = sae.b_enc[feature_idx].to(device=orig_resid.device, dtype=orig_resid.dtype)
        direction = sae.feature_direction(feature_idx).to(device=orig_resid.device, dtype=orig_resid.dtype)

        natural_val = orig_resid[0].detach() @ feature_row + bias
        alphas = chunk_alphas.to(device=orig_resid.device)
        v = (natural_val * alphas).clone().requires_grad_(True)
        perturbed = orig_resid.detach() + (v - natural_val).unsqueeze(-1) * direction
        model.model.layers[layer_idx].output[:, -1, :] = perturbed

        logits = model.lm_head.output[:, -1, :]
        metrics = refusal_logit_diff(logits, refusal_id, compliance_id)
        metrics.sum().backward()
        grad = v.grad.clone().save()
        natural_val_saved = natural_val.clone().save()

    return natural_val_saved, grad


def feature_ig_attribution(
    model: LanguageModel,
    sae: TopKSAE,
    feature_idx: int,
    layer_idx: int,
    prompt: str,
    refusal_id: int,
    compliance_id: int,
    n_steps: int = N_STEPS_DEFAULT,
    micro_batch_size: int | None = None,
) -> float:
    """Integrated-gradients attribution of one SAE feature (at one layer) on
    one prompt: the estimated effect on the refusal-vs-compliance metric of
    ablating this feature (natural pre-activation -> 0).

    micro_batch_size: if given and < n_steps, splits the n_steps
    interpolation points into chunks of this size, each its own
    forward+backward pass -- see module docstring."""
    templated = format_prompt(model, prompt)
    micro_batch_size = micro_batch_size or n_steps
    alphas = torch.arange(1, n_steps + 1, dtype=torch.float32) / n_steps

    natural_val_saved = None
    grads = []
    for start in range(0, n_steps, micro_batch_size):
        chunk_alphas = alphas[start:start + micro_batch_size]
        natural_val_saved, grad = _ig_chunk(
            model, sae, feature_idx, layer_idx, templated, refusal_id, compliance_id, chunk_alphas
        )
        grads.append(grad)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_grad = torch.cat(grads)
    # Left-Riemann approximation of the integral from 0 to natural_val.
    return (natural_val_saved * full_grad.mean()).item()


def rank_pooled_candidates(
    model: LanguageModel,
    saes: dict[int, TopKSAE],
    candidates: list[tuple[int, int]],
    prompts: list[str],
    refusal_id: int,
    compliance_id: int,
    k_star: int = 20,
    n_steps: int = N_STEPS_DEFAULT,
    micro_batch_size: int | None = None,
) -> list[tuple[int, int, float]]:
    """candidates: (layer_idx, feature_idx) pairs pooled from
    src.sae.feature_selection.top_k0_by_cosine_similarity across the
    candidate layers. Averages feature_ig_attribution over `prompts` for
    each candidate, then returns the top `k_star` by score, descending
    (highest positive score = feature whose removal would most reduce the
    refusal-leaning metric, i.e. most causally responsible for refusal).
    micro_batch_size: passed through to feature_ig_attribution -- see its
    docstring and the module docstring."""
    scored = []
    for i, (layer_idx, feature_idx) in enumerate(candidates):
        sae = saes[layer_idx]
        scores = [
            feature_ig_attribution(
                model, sae, feature_idx, layer_idx, p, refusal_id, compliance_id, n_steps, micro_batch_size
            )
            for p in prompts
        ]
        scored.append((layer_idx, feature_idx, sum(scores) / len(scores)))
        print(f"  [{i+1}/{len(candidates)}] layer {layer_idx}, feature {feature_idx}: "
              f"score={scored[-1][2]:.4f}")
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:k_star]
