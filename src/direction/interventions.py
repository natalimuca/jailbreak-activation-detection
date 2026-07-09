"""Causal interventions on the residual stream via nnsight:

- directional ablation: project the direction out of every layer's residual
  stream output, at every generated token, so the model can never write along
  it -- this is what tests whether the direction is *necessary* for refusal.
- activation addition: add the direction (scaled) at a single layer, at every
  generated token, on prompts the model would normally comply with -- this
  tests whether the direction is *sufficient* to induce refusal.
- SAE feature suppression: the SAE-feature analog of directional ablation,
  for Phase 3's causal validation of causal-ranked candidates.
"""

from __future__ import annotations

import torch
from nnsight import LanguageModel

from src.activations.extract import format_prompt, n_layers
from src.sae.qwen_scope import TopKSAE


def _project_out(act: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction = direction.to(act.dtype).to(act.device)
    proj = (act @ direction).unsqueeze(-1) * direction
    return act - proj


@torch.no_grad()
def generate_with_ablation(
    model: LanguageModel,
    instruction: str,
    direction: torch.Tensor,
    max_new_tokens: int = 40,
) -> str:
    prompt = format_prompt(model, instruction)
    L = n_layers(model)
    with model.generate(
        prompt, min_new_tokens=max_new_tokens, max_new_tokens=max_new_tokens
    ) as tracer:
        for step in tracer.iter[:max_new_tokens]:
            for layer_idx in range(L):
                out = model.model.layers[layer_idx].output
                out[:] = _project_out(out, direction)
        out_ids = model.generator.output.save()
    return model.tokenizer.decode(out_ids[0][-max_new_tokens:], skip_special_tokens=True)


@torch.no_grad()
def generate_with_addition(
    model: LanguageModel,
    instruction: str,
    raw_direction: torch.Tensor,
    layer_idx: int,
    alpha: float = 1.0,
    max_new_tokens: int = 40,
) -> str:
    """raw_direction should be the *unnormalized* mean-difference vector
    (see compute.compute_raw_directions) -- its own norm already reflects
    the natural harmful/harmless separation scale at this layer, so `alpha`
    is a small multiplier on top of that (not a raw activation magnitude)."""
    prompt = format_prompt(model, instruction)
    with model.generate(
        prompt, min_new_tokens=max_new_tokens, max_new_tokens=max_new_tokens
    ) as tracer:
        for step in tracer.iter[:max_new_tokens]:
            out = model.model.layers[layer_idx].output
            out[:] = out + alpha * raw_direction.to(out.dtype).to(out.device)
        out_ids = model.generator.output.save()
    return model.tokenizer.decode(out_ids[0][-max_new_tokens:], skip_special_tokens=True)


@torch.no_grad()
def generate_with_feature_suppression(
    model: LanguageModel,
    instruction: str,
    features_by_layer: dict[int, list[tuple[TopKSAE, int]]],
    max_new_tokens: int = 40,
) -> str:
    """Ablates one or more SAE features (each a (sae, feature_idx) pair,
    grouped by the layer they live at) at every generated token -- the
    SAE-feature analog of generate_with_ablation above, for direct
    causal-validation comparability with Phase 1's methodology.

    Each feature's natural contribution to the residual stream
    (val * feature_direction, where val is its pre-activation from the
    *original*, unmodified residual) is subtracted. When multiple features
    share a layer, all their corrections are computed from that same
    original residual and summed in one combined update -- not applied
    sequentially -- so ablating feature B doesn't shift the baseline that
    feature A's contribution is measured against."""
    prompt = format_prompt(model, instruction)
    # nnsight requires modules to be accessed in the order they actually
    # execute in the forward pass -- iterating features_by_layer in whatever
    # order its keys happen to be in (e.g. ranking-score order) can access a
    # later layer before an earlier one, which raises a MissedProviderError.
    ordered_layers = sorted(features_by_layer.items())
    with model.generate(
        prompt, min_new_tokens=max_new_tokens, max_new_tokens=max_new_tokens
    ) as tracer:
        for step in tracer.iter[:max_new_tokens]:
            for layer_idx, feats in ordered_layers:
                out = model.model.layers[layer_idx].output
                correction = torch.zeros_like(out)
                for sae, feature_idx in feats:
                    feature_row = sae.W_enc[feature_idx].to(out.dtype).to(out.device)
                    bias = sae.b_enc[feature_idx].to(out.dtype).to(out.device)
                    direction = sae.feature_direction(feature_idx).to(out.dtype).to(out.device)
                    val = out @ feature_row + bias
                    correction = correction + val.unsqueeze(-1) * direction
                out[:] = out - correction
        out_ids = model.generator.output.save()
    return model.tokenizer.decode(out_ids[0][-max_new_tokens:], skip_special_tokens=True)


@torch.no_grad()
def generate_baseline(model: LanguageModel, instruction: str, max_new_tokens: int = 40) -> str:
    prompt = format_prompt(model, instruction)
    with model.generate(
        prompt, min_new_tokens=max_new_tokens, max_new_tokens=max_new_tokens
    ) as tracer:
        out_ids = model.generator.output.save()
    return model.tokenizer.decode(out_ids[0][-max_new_tokens:], skip_special_tokens=True)
