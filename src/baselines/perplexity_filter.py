"""GPT-2 perplexity-based prompt filter, per Alon & Kamfonas 2023 ("Detecting
Language Model Attacks with Perplexity", arXiv:2308.14132) -- GPT-2 is their
own reference model too. The core intuition: adversarial-suffix attacks
(e.g. GCG) append gibberish token sequences that are wildly improbable under
any language model, so a plain perplexity threshold catches them cheaply
without needing to understand the prompt's meaning at all.

Included in Phase 4's baseline comparison specifically because this is a
*different* failure mode than the keyword filter's: perplexity should catch
GCG-style suffix attacks (high perplexity) but is expected to miss fluent,
grammatical jailbreak paraphrases like PAIR's roleplay wrappers (low
perplexity, since they read as ordinary text) -- the adversarial paraphrase
set built in `src/eval/adversarial_paraphrase.py` includes both attack types
so this distinction shows up in the results rather than being asserted here.
"""

from __future__ import annotations

import math

import torch

DEFAULT_MODEL_NAME = "gpt2"


def load_gpt2(device: str = "cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def compute_perplexity(prompt: str, model, tokenizer, device: str = "cpu") -> float:
    """Perplexity of `prompt` under `model`, via its own next-token
    cross-entropy loss (the standard `labels=input_ids` trick: HF causal LMs
    shift labels internally, so this is the mean per-token NLL of the
    sequence under itself)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < 2:
        # A single-token prompt has no next-token prediction to score --
        # treat as unremarkable (not flagged) rather than dividing by zero.
        return 1.0
    out = model(ids, labels=ids)
    return math.exp(out.loss.item())


def is_flagged(perplexity: float, threshold: float) -> bool:
    return perplexity >= threshold
