"""Expands the moralize-vs-comply ground-truth set beyond
`scripts/07_sample_completions_for_labeling.py`'s original 45-row
worksheet, which predates the Llama-3.1-8B/gemma-2-9b-it suppression
results and the cross-model-transfer results -- 100% of its `comply`
labels come from Phase 1's smaller models, zero from any Qwen3-8B
intervention experiment, let alone Llama/Gemma. A classifier validated
only against the original worksheet risks learning "which experiment
produced this" rather than "did this text contain harmful content" (see
DECISIONS.md).

Samples non-refuse completions (via `is_refusal`) from the previously-
uncovered files, stratified by (source, condition), same worksheet format
as scripts/07 so the two can be concatenated for validation. Saved to a
SEPARATE file rather than appending to the original, so the original
Qwen3-8B/Phase1 spot-check stays untouched.

Usage: python scripts/18_expand_moralize_comply_worksheet.py [--n-per-source N] [--seed S]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.direction.refusal_classifier import is_refusal

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_PER_SOURCE_DEFAULT = 3


def load_sources() -> list[dict]:
    """Each entry: {source, condition, prompt, completion}. Only
    non-refuse completions -- refuse rows add nothing to a moralize-vs-
    comply ground truth set."""
    entries = []

    for model_label in ["Llama-3.1-8B-Instruct", "gemma-2-9b-it"]:
        path = RESULTS_DIR / f"sae_suppression_validation_{model_label}.json"
        if not path.exists():
            continue
        with open(path) as fh:
            d = json.load(fh)
        for condition, r in d["results"].items():
            for prompt, completion in zip(d["val_prompts"], r["completions"]):
                if not is_refusal(completion):
                    entries.append({
                        "source": f"sae_suppression_{model_label}",
                        "condition": condition,
                        "prompt": prompt,
                        "completion": completion,
                    })

    transfer_path = RESULTS_DIR / "cross_model_direction_transfer.json"
    if transfer_path.exists():
        with open(transfer_path) as fh:
            d = json.load(fh)
        val_prompts = d["val_prompts"]
        for direction_key, payload in d["causal_ablation"].items():
            for condition, r in payload["conditions"].items():
                for prompt, completion in zip(val_prompts, r["completions"]):
                    if not is_refusal(completion):
                        entries.append({
                            "source": f"transfer_{direction_key}",
                            "condition": condition,
                            "prompt": prompt,
                            "completion": completion,
                        })

    return entries


def stratified_sample(entries: list[dict], n_per_source: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_source_condition: dict[tuple, list[dict]] = {}
    for e in entries:
        by_source_condition.setdefault((e["source"], e["condition"]), []).append(e)

    sampled = []
    for key, group in by_source_condition.items():
        k = min(n_per_source, len(group))
        sampled.extend(rng.sample(group, k))
    rng.shuffle(sampled)
    return sampled


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-source", type=int, default=N_PER_SOURCE_DEFAULT)
    p.add_argument("--seed", type=int, default=1)  # different seed from scripts/07
    args = p.parse_args()

    entries = load_sources()
    print(f"Loaded {len(entries)} non-refuse completions across "
          f"{len({(e['source'], e['condition']) for e in entries})} source/condition groups "
          f"(Llama/Gemma suppression + cross-model-transfer, not covered by scripts/07)")

    sampled = stratified_sample(entries, args.n_per_source, args.seed)
    print(f"Sampled {len(sampled)} for labeling")

    worksheet_path = RESULTS_DIR / "classifier_spotcheck_worksheet_expansion.csv"
    with open(worksheet_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "source", "condition", "prompt", "completion", "your_label"])
        for i, e in enumerate(sampled):
            writer.writerow([i, e["source"], e["condition"], e["prompt"], e["completion"], ""])
    print(f"Saved labeling worksheet to {worksheet_path}")
    print("\nFill in 'your_label' for each row with one of: refuse | moralize | partial | comply")
    print("(these are all already non-refuse by is_refusal, so 'refuse' would mean the keyword")
    print("matcher missed a real refusal -- worth flagging if it happens)")


if __name__ == "__main__":
    main()
