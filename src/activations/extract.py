"""Residual-stream activation extraction via nnsight.

Captures the residual stream at the last token position of the (chat-templated)
instruction, at the output of every decoder layer -- this is the standard
extraction point for difference-of-means refusal-direction work.
"""

from __future__ import annotations

import torch
from nnsight import LanguageModel
from tqdm import tqdm


def load_model(model_name: str, device_map: str = "auto", load_in_4bit: bool = False) -> LanguageModel:
    kwargs = dict(device_map=device_map, torch_dtype=torch.float16)
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    return LanguageModel(model_name, **kwargs)


def format_prompt(model: LanguageModel, instruction: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    return model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def n_layers(model: LanguageModel) -> int:
    return len(model.model.layers)


@torch.no_grad()
def get_last_token_resid_acts(
    model: LanguageModel, instructions: list[str]
) -> torch.Tensor:
    """Returns tensor of shape [n_layers, n_instructions, d_model]: the
    residual stream (decoder layer output) at the last prompt token, for
    every layer, for every instruction."""
    L = n_layers(model)
    per_layer_acts = [[] for _ in range(L)]

    for instr in tqdm(instructions, desc="extracting activations"):
        prompt = format_prompt(model, instr)
        with model.trace(prompt) as tracer:
            saved = []
            for layer_idx in range(L):
                out = model.model.layers[layer_idx].output[0][:, -1, :].save()
                saved.append(out)
        for layer_idx in range(L):
            per_layer_acts[layer_idx].append(saved[layer_idx].detach().cpu().float())

    return torch.stack([torch.cat(layer_acts, dim=0) for layer_acts in per_layer_acts])
