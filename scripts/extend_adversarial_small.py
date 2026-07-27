"""Builds the adversarial-paraphrase activation cache (all layers) for
Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct -- the two models
`scripts/extend_qwen_smollm.py` scored on the adversarial set without ever
caching activations to disk (it extracts them transiently at eval time).
Needed here to investigate the PAIR-paraphrase robustness spread across all
5 models, not just the 3 (Qwen3-8B, Llama-3.1-8B, gemma-2-9b-it) that
already have a saved `_adversarial.pt`. Mirrors
`scripts/extend_sae_adversarial.py`'s payload shape exactly (model/
source_model/activations/records) and reuses the same real
`adversarial_paraphrase_manifest.json` prompts, not refetched.

Both models are small enough to load unquantized -- no `--4bit` needed,
and this is forward-pass-only activation extraction (no generation), so
it's fast (seconds, not the multi-hour scale of this project's
causal-validation runs).

Usage: python scripts/extend_adversarial_small.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import get_last_token_resid_acts, load_model

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"
MANIFEST_PATH = RESULTS_DIR / "adversarial_paraphrase_manifest.json"

MODELS = {
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}


def main() -> None:
    with open(MANIFEST_PATH) as fh:
        adversarial = json.load(fh)
    print(f"Reusing {len(adversarial)} adversarial prompts from {MANIFEST_PATH.name} "
          f"(same real JailbreakBench artifacts as every other model's adversarial evaluation)")
    texts = [r["text"] for r in adversarial]

    for cache_label, hf_model_name in MODELS.items():
        print(f"\n=== {cache_label} ===")
        print(f"Loading model: {hf_model_name}")
        model = load_model(hf_model_name, load_in_4bit=False)

        print(f"Extracting activations for {len(texts)} adversarial prompts (forward-pass only, no generation)")
        acts = get_last_token_resid_acts(model, texts)

        payload = {
            "model": f"{cache_label}_adversarial",
            "source_model": hf_model_name,
            "activations": acts,
            "records": adversarial,
        }
        ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = ACTIVATIONS_DIR / f"{cache_label}_adversarial.pt"
        torch.save(payload, cache_path)
        print(f"Saved activation cache to {cache_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
