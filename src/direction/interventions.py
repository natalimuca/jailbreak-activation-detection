"""Causal interventions on the residual stream via nnsight:

- directional ablation: project the direction out of every layer's residual
  stream output, at every generated token, so the model can never write along
  it -- this is what tests whether the direction is *necessary* for refusal.
- activation addition: add the direction (scaled) at a single layer, at every
  generated token, on prompts the model would normally comply with -- this
  tests whether the direction is *sufficient* to induce refusal.
"""

from __future__ import annotations

import torch
from nnsight import LanguageModel

from src.activations.extract import format_prompt, n_layers


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
    with model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
        for layer_idx in range(L):
            with model.model.layers[layer_idx].all():
                out = model.model.layers[layer_idx].output[0]
                out[:] = _project_out(out, direction)
        out_ids = model.generator.output.save()
    return model.tokenizer.decode(out_ids[0][-max_new_tokens:], skip_special_tokens=True)


@torch.no_grad()
def generate_with_addition(
    model: LanguageModel,
    instruction: str,
    direction: torch.Tensor,
    layer_idx: int,
    alpha: float = 8.0,
    max_new_tokens: int = 40,
) -> str:
    prompt = format_prompt(model, instruction)
    direction = direction / direction.norm()
    with model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
        with model.model.layers[layer_idx].all():
            out = model.model.layers[layer_idx].output[0]
            out[:] = out + alpha * direction.to(out.dtype).to(out.device)
        out_ids = model.generator.output.save()
    return model.tokenizer.decode(out_ids[0][-max_new_tokens:], skip_special_tokens=True)


@torch.no_grad()
def generate_baseline(model: LanguageModel, instruction: str, max_new_tokens: int = 40) -> str:
    prompt = format_prompt(model, instruction)
    with model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
        out_ids = model.generator.output.save()
    return model.tokenizer.decode(out_ids[0][-max_new_tokens:], skip_special_tokens=True)
