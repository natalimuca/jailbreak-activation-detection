"""Residual-stream activation extraction via nnsight.

Captures the residual stream at the last token position of the (chat-templated)
instruction, at the output of every decoder layer -- this is the standard
extraction point for difference-of-means refusal-direction work.
"""

from __future__ import annotations

import os

# Must be set before CUDA initializes. Reduces allocator fragmentation over a
# long run that alternates between very short and very long (~1000-token)
# prompts on a memory-constrained GPU -- see DECISIONS.md.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import BitsAndBytesConfig


def load_model(model_name: str, device_map: str = "auto", load_in_4bit: bool = False) -> LanguageModel:
    kwargs = dict(device_map=device_map, torch_dtype=torch.float16)
    if load_in_4bit:
        # transformers>=5 removed the `load_in_4bit=True` shorthand kwarg;
        # quantization must be passed via BitsAndBytesConfig instead.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # accelerate's device_map="auto" memory estimator is too conservative
        # for a 4-bit 8B model on a 6GB card -- it decides to CPU/disk-offload
        # a few modules even though the model does fit, and bnb refuses that
        # combination outright unless llm_int8_enable_fp32_cpu_offload=True is
        # also set (which routes those modules through bnb's CPU backend --
        # verified to be prohibitively slow, not a real option here). Forcing
        # everything onto the single GPU instead lets it actually fit.
        kwargs["device_map"] = {"": 0}
    return LanguageModel(model_name, **kwargs)


def format_prompt(model: LanguageModel, instruction: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    # enable_thinking=False: Qwen3 models reason inside a <think>...</think>
    # block before answering by default. Without this, the "last prompt
    # token" position (used for direction extraction) sits before the model
    # has even decided whether to think, and the first *generated* token in
    # any causal-validation/ranking step would be the start of that
    # reasoning preamble, not of the actual answer -- silently measuring the
    # wrong thing for Qwen3-8B (see DECISIONS.md). Non-thinking models'
    # chat templates (SmolLM2, Qwen2.5) ignore this unused kwarg.
    return model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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

    for i, instr in enumerate(tqdm(instructions, desc="extracting activations")):
        prompt = format_prompt(model, instr)
        saved = []
        # use_cache=False: we only need one forward pass per prompt (no
        # autoregressive generation), so the KV cache is pure memory overhead
        # -- meaningful on a 6GB GPU with some corpus prompts around 1000 tokens.
        with model.trace(prompt, use_cache=False) as tracer:
            for layer_idx in range(L):
                # NOTE: current transformers decoder layers return the hidden-state
                # tensor directly (not a tuple), so .output is already [batch, seq, d_model].
                out = model.model.layers[layer_idx].output[:, -1, :].save()
                saved.append(out)
        for layer_idx in range(L):
            per_layer_acts[layer_idx].append(saved[layer_idx].detach().cpu().float())
        if torch.cuda.is_available() and i % 50 == 0:
            # Mitigates allocator fragmentation from alternating short/long
            # prompts over a long run near the GPU's memory limit.
            torch.cuda.empty_cache()

    return torch.stack([torch.cat(layer_acts, dim=0) for layer_acts in per_layer_acts])
