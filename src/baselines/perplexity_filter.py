"""Perplexity-based prompt filter, per Alon & Kamfonas 2023 ("Detecting
Language Model Attacks with Perplexity", arXiv:2308.14132). The core
intuition: adversarial-suffix attacks (e.g. GCG) append gibberish token
sequences that are wildly improbable under any language model, so a plain
perplexity threshold catches them cheaply without needing to understand the
prompt's meaning at all.

Included in Phase 4's baseline comparison specifically because this is a
*different* failure mode than the keyword filter's: perplexity should catch
GCG-style suffix attacks (high perplexity) but is expected to miss fluent,
grammatical jailbreak paraphrases like PAIR's roleplay wrappers (low
perplexity, since they read as ordinary text) -- the adversarial paraphrase
set built in `src/eval/adversarial_paraphrase.py` includes both attack types
so this distinction shows up in the results rather than being asserted here.

**Backbone: OLMo-2-0425-1B (AI2, base/pretrained, April 2025), not GPT-2,
GPT-Neo, or an instruction-tuned model** (see DECISIONS.md for the full
history: GPT-2 -> GPT-Neo-1.3B -> Phi-4-mini-instruct -> here). Two
requirements drove this: (1) modern and independent of every model family
already used or reserved as a target in this project (Qwen, SmolLM, and
Llama/Gemma, the latter two reserved for Phase 6 per README.md); (2) a
genuine **base** model, not instruction-tuned. (2) was learned the hard
way -- Phi-4-mini-instruct's XSTest false-positive rate was *worse* than
GPT-Neo-1.3B's despite being newer and larger, because scoring raw,
non-chat-templated text (this function never applies a chat template) is
off-distribution for a model fine-tuned on chat-formatted conversations.
`compute_perplexity` needs a model trained on next-token prediction over
plain text, which is what GPT-2/GPT-Neo were and what OLMo-2-0425-1B is.
Small enough (1B) to run without quantization.
"""

from __future__ import annotations

import math

import torch

DEFAULT_MODEL_NAME = "allenai/OLMo-2-0425-1B"


def load_perplexity_model(device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def compute_perplexity(prompt: str, model, tokenizer) -> float:
    """Perplexity of `prompt` under `model`, via its own next-token
    cross-entropy loss (the standard `labels=input_ids` trick: HF causal LMs
    shift labels internally, so this is the mean per-token NLL of the
    sequence under itself). Reads the model's own device off its
    parameters rather than taking a separate `device` argument, so callers
    can't accidentally score on a different device than the model lives on."""
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < 2:
        # A single-token prompt has no next-token prediction to score --
        # treat as unremarkable (not flagged) rather than dividing by zero.
        return 1.0
    out = model(ids, labels=ids)
    return math.exp(out.loss.item())


def is_flagged(perplexity: float, threshold: float) -> bool:
    return perplexity >= threshold
