"""Builds the adversarial-paraphrase activation cache (all layers) for
Llama-3.1-8B-Instruct and gemma-2-9b-it -- the piece Wave 1's dense-direction
extension (`scripts/14`) didn't need, since it only used one layer's
projection and discarded the rest. The SAE-feature detector needs multiple
layers per model (its top-K causally-ranked features span 3 layers each),
so this time the cache is saved to disk, mirroring
`scripts/09_build_adversarial_paraphrase_set.py`'s Qwen3-8B cache exactly
(same payload shape: model/source_model/activations/records).

Reuses the existing `results/adversarial_paraphrase_manifest.json` (real
JailbreakBench PAIR/GCG artifacts, matched to TEST-split JBB goals) --
does not rebuild it, same real prompts as every other model's adversarial
evaluation in this project.

Usage: python scripts/15_extend_sae_adversarial_cache_llama_gemma.py
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
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}


def main() -> None:
    with open(MANIFEST_PATH) as fh:
        adversarial = json.load(fh)
    print(f"Reusing {len(adversarial)} adversarial prompts from {MANIFEST_PATH.name} "
          f"(same real JailbreakBench artifacts as every other model, not refetched)")
    texts = [r["text"] for r in adversarial]

    for cache_label, hf_model_name in MODELS.items():
        print(f"\n=== {cache_label} ===")
        print(f"Loading model: {hf_model_name} (4-bit)")
        model = load_model(hf_model_name, load_in_4bit=True)

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
