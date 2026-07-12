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

**Backbone: Olmo-3-1025-7B (AI2, base/pretrained, October 2025), not GPT-2,
GPT-Neo, an instruction-tuned model, or OLMo-2-0425-1B** (see DECISIONS.md
for the full history: GPT-2 -> GPT-Neo-1.3B -> Phi-4-mini-instruct ->
OLMo-2-0425-1B -> here). Requirements: (1) as current as reasonably
possible; (2) independent of every model family already used or reserved
as a target in this project (Qwen, SmolLM, and Llama/Gemma, the latter two
reserved for Phase 6 per README.md); (3) a genuine **base** model, not
instruction-tuned -- learned the hard way from Phi-4-mini-instruct, whose
XSTest false-positive rate got *worse* than GPT-Neo-1.3B's because scoring
raw, non-chat-templated text is off-distribution for a model fine-tuned on
chat-formatted conversations. Olmo-3-1025-7B is AI2's next generation after
OLMo-2 (same fully-open lineage: weights, training data, and code all
released), a genuine base checkpoint, and more recent (October 2025 vs.
April 2025). At 7B it doesn't fit unquantized on a 6GB GPU, so it's loaded
4-bit via the same `BitsAndBytesConfig` approach already used for
Qwen3-8B (`src.activations.extract.load_model`).
"""

from __future__ import annotations

import math

import torch

DEFAULT_MODEL_NAME = "allenai/Olmo-3-1025-7B"


def load_perplexity_model(device: str = "cuda", load_in_4bit: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    kwargs = dict(torch_dtype=torch.float16)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME, **kwargs)
    if not load_in_4bit:
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
