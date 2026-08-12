"""Builds a supplementary PAIR-paraphrase set from TRAIN-split JBB harmful
goals, to add statistical power to the (currently underpowered, n=21)
PAIR-robustness question without touching the officially-reported TEST-based
adversarial set at all.

**Why this is legitimate, not a leak.** TRAIN is only ever used to derive
the refusal direction and the SAE causal ranking (Phase 1-3) -- never a
detection threshold. Threshold calibration happens on VAL. PAIR-paraphrased
versions of TRAIN goals therefore carry none of the calibration-leakage risk
that VAL-goal artifacts would (see reports/DECISIONS.md's "Pre-registration:
framing-direction ablation" entry's own discussion of why VAL was excluded
from that experiment's scope, for the same reasoning). This set is reported
as a separate, clearly-labeled supplementary check, never blended into the
official n=21 TEST-based PAIR metric.

**Why this was possible at all.** JailbreakBench already has successful PAIR
artifacts for 60 of this project's 73 total corpus JBB harmful goals (89
distinct goals exist in JailbreakBench overall) -- not just the ~10-11 that
happen to land in TEST. The project's prior "blocked on JailbreakBench
publishing more artifacts" limitation was imprecise: the real constraint was
always which split a matchable goal happened to fall in, not external data
availability. See reports/RESULTS.md and reports/DECISIONS.md for the
corrected record.

Reuses `src.eval.adversarial_paraphrase.build_adversarial_set` completely
unmodified -- it already takes an arbitrary records list, so passing
TRAIN-split records instead of TEST-split records is a zero-code-change
reuse, exactly like `scripts/build_adversarial_set.py` uses it for TEST.

Usage: python scripts/build_train_pair_set.py
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

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"
TEST_MANIFEST_PATH = RESULTS_DIR / "adversarial_paraphrase_manifest.json"
OUT_MANIFEST_PATH = RESULTS_DIR / "train_pair_manifest.json"

MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}


def get_split_records() -> dict[str, list[dict]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"No split manifest at {MANIFEST_PATH} -- run scripts/extract_activations.py first."
        )
    records = load_all_labeled_prompts()
    kept, _ = deduplicate(records)
    manifest = load_manifest()
    return apply_manifest(kept, manifest)


def main() -> None:
    split = get_split_records()
    train_records = split["train"]
    print(f"TRAIN split: {len(train_records)} records")

    train_pair = build_adversarial_set(train_records, include_gcg=False)
    if not train_pair:
        raise RuntimeError("No TRAIN-goal PAIR artifacts matched -- nothing to extract activations for.")
    print(f"Matched {len(train_pair)} real PAIR artifacts against TRAIN-split JBB harmful goals")
    print(f"  distinct TRAIN goals covered: {len(set(r['goal'] for r in train_pair))}")

    # Safety check: the TRAIN-goal set must be disjoint from the official
    # TEST-based adversarial manifest -- guaranteed by construction (TRAIN
    # and TEST are disjoint splits), but confirmed explicitly rather than
    # assumed before anything downstream trusts it.
    with open(TEST_MANIFEST_PATH) as fh:
        test_manifest = json.load(fh)
    test_goals = {r["goal"].strip().lower() for r in test_manifest}
    train_goals = {r["goal"].strip().lower() for r in train_pair}
    overlap = test_goals & train_goals
    if overlap:
        raise RuntimeError(f"TRAIN-goal set overlaps the official TEST-based manifest on {len(overlap)} goals -- aborting.")
    print(f"Confirmed disjoint from the official TEST-based manifest ({len(test_goals)} goals there).")

    for cache_label, hf_model_name in MODELS.items():
        print(f"\n=== {cache_label} ===")
        print(f"Loading model: {hf_model_name} (4-bit)")
        model = load_model(hf_model_name, load_in_4bit=True)

        texts = [r["text"] for r in train_pair]
        print(f"Extracting activations for {len(texts)} TRAIN-goal PAIR prompts (forward-pass only, no generation)")
        acts = get_last_token_resid_acts(model, texts)

        payload = {
            "model": f"{cache_label}_train_pair",
            "source_model": hf_model_name,
            "activations": acts,
            "records": train_pair,
        }
        ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = ACTIVATIONS_DIR / f"{cache_label}_train_pair.pt"
        torch.save(payload, cache_path)
        print(f"Saved activation cache to {cache_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(OUT_MANIFEST_PATH, "w") as fh:
        json.dump(train_pair, fh, indent=2)
    print(f"\nSaved manifest (prompts + provenance) to {OUT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
