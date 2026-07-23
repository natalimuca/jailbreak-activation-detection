"""Builds Phase 4's adversarial paraphrase test set (real JailbreakBench
PAIR/GCG attack artifacts, matched to this project's own TEST-split JBB
harmful goals -- see `src/eval/adversarial_paraphrase.py`) and extracts
Qwen3-8B activations for it.

Forward-pass only, no generation: unlike Phase 1/3's causal generation
experiments, this is a prompt-classification evaluation (does the prompt's
own activation pattern get flagged), so `get_last_token_resid_acts` is all
that's needed -- much cheaper than any of the causal-validation runs.

Usage: python scripts/build_adversarial_set.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import get_last_token_resid_acts, load_model
from src.data.dedup import deduplicate
from src.data.loaders import load_all_labeled_prompts
from src.data.splits import MANIFEST_PATH, apply_manifest, load_manifest
from src.eval.adversarial_paraphrase import build_adversarial_set

DEFAULT_MODEL = "Qwen/Qwen3-8B"
CACHE_MODEL_LABEL = "Qwen3-8B_adversarial"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"


def get_test_split_records() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"No split manifest at {MANIFEST_PATH} -- run scripts/extract_activations.py first."
        )
    records = load_all_labeled_prompts()
    kept, _ = deduplicate(records)
    manifest = load_manifest()
    split = apply_manifest(kept, manifest)
    return split["test"]


def main() -> None:
    test_records = get_test_split_records()
    print(f"TEST split: {len(test_records)} records")

    adversarial = build_adversarial_set(test_records, include_gcg=True)
    if not adversarial:
        raise RuntimeError("No adversarial artifacts matched -- nothing to extract activations for.")
    print(f"Matched {len(adversarial)} real jailbreak artifacts against TEST-split JBB harmful goals")
    print(f"  by method: {dict(Counter(r['method'] for r in adversarial))}")
    print(f"  by target model: {dict(Counter(r['target_model'] for r in adversarial))}")
    print(f"  distinct goals covered: {len(set(r['goal'] for r in adversarial))}")

    print(f"\nLoading model: {DEFAULT_MODEL} (4-bit)")
    model = load_model(DEFAULT_MODEL, load_in_4bit=True)

    texts = [r["text"] for r in adversarial]
    print(f"Extracting activations for {len(texts)} adversarial prompts (forward-pass only, no generation)")
    acts = get_last_token_resid_acts(model, texts)

    payload = {
        "model": CACHE_MODEL_LABEL,
        "source_model": DEFAULT_MODEL,
        "activations": acts,
        "records": adversarial,
    }
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ACTIVATIONS_DIR / f"{CACHE_MODEL_LABEL}.pt"
    torch.save(payload, cache_path)
    print(f"Saved activation cache to {cache_path}")

    manifest_path = RESULTS_DIR / "adversarial_paraphrase_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(adversarial, fh, indent=2)
    print(f"Saved manifest (prompts + provenance) to {manifest_path}")


if __name__ == "__main__":
    main()
