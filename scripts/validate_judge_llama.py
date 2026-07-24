"""Retries the moralize-vs-comply classifier with a genuinely larger local
judge: Llama-3.1-8B-Instruct (8B) vs. the two prior failures (Phi-4-mini-
instruct, 3.8B; SmolLM2-1.7B-Instruct, 1.7B -- both defaulted to one
category regardless of content, a real capability ceiling, not a prompt
problem, see `src.direction.moralize_comply_classifier`'s docstring and
DECISIONS.md). Same worksheets, same reporting structure as
`scripts/validate_classifier.py`, so results are directly comparable --
no new human/Claude labeling needed.

**One real difference from the original validation, checked rather than
ignored**: Llama-3.1-8B-Instruct is itself one of this project's causally-
tested target models, unlike Phi-4-mini-instruct (picked specifically to be
independent of the Qwen/Llama/Gemma families). 27 of 98 worksheet rows have
completions Llama-3.1-8B itself generated (`sae_suppression_Llama-3.1-8B-
Instruct`, `transfer_qwen_direction_on_llama`) -- a self-judging case.
Reports a dedicated self-judged-vs-independent breakdown, not just the
aggregate, so any self-serving bias would be visible rather than averaged
away.

Usage: python scripts/validate_judge_llama.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.direction.moralize_comply_classifier import classify, load_judge_model

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
WORKSHEETS = [
    RESULTS_DIR / "classifier_spotcheck_worksheet.csv",
    RESULTS_DIR / "classifier_spotcheck_worksheet_expansion.csv",
]
VALID_LABELS = {"moralize", "comply", "partial", "refuse"}
JUDGE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SELF_JUDGED_SOURCES = {"sae_suppression_Llama-3.1-8B-Instruct", "transfer_qwen_direction_on_llama"}


def load_rows() -> list[dict]:
    rows = []
    for path in WORKSHEETS:
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                label = row["your_label"].strip().lower()
                if label not in VALID_LABELS:
                    continue
                rows.append({
                    "worksheet": path.name,
                    "source": row["source"],
                    "condition": row["condition"],
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "human_label": label,
                    "is_harmful": "harmless" not in row["condition"],
                    "self_judged": row["source"] in SELF_JUDGED_SOURCES,
                })
    return rows


def main() -> None:
    rows = load_rows()
    print(f"Loaded {len(rows)} non-refuse, human-labeled rows across both worksheets")
    print(f"  by human label: {dict(Counter(r['human_label'] for r in rows))}")
    print(f"  self-judged rows (Llama-3.1-8B judging its own completions): {sum(r['self_judged'] for r in rows)}/{len(rows)}")

    print(f"\nLoading judge model (4-bit): {JUDGE_MODEL_NAME}")
    judge_model, judge_tokenizer = load_judge_model(judge_model_name=JUDGE_MODEL_NAME)

    results = []
    for i, r in enumerate(rows):
        verdict = classify(judge_model, judge_tokenizer, r["prompt"], r["completion"])
        results.append({**r, "classifier_verdict": verdict})
        print(f"  [{i+1}/{len(rows)}] human={r['human_label']:>9} classifier={verdict:>12} "
              f"({'harmful' if r['is_harmful'] else 'harmless'}, {r['source']}/{r['condition']})")

    def report(subset: list[dict], label: str) -> None:
        if not subset:
            return
        n_correct = sum(r["human_label"] == r["classifier_verdict"] for r in subset)
        print(f"\n=== {label}: {n_correct}/{len(subset)} ({n_correct/len(subset):.1%}) ===")
        by_human = defaultdict(Counter)
        for r in subset:
            by_human[r["human_label"]]["correct" if r["human_label"] == r["classifier_verdict"] else "incorrect"] += 1
        for human_label in ["moralize", "comply", "partial", "refuse"]:
            counts = by_human.get(human_label)
            if not counts:
                continue
            total = counts["correct"] + counts["incorrect"]
            print(f"  {human_label:>10}: {counts['correct']}/{total} ({counts['correct']/total:.1%})")

    print("\n\n########## VALIDATION RESULTS (judge: Llama-3.1-8B-Instruct) ##########")
    report(results, "Overall (all rows, both worksheets)")
    report([r for r in results if r["is_harmful"]], "Harmful-prompt rows only (the load-bearing case)")
    report([r for r in results if not r["is_harmful"]], "Harmless-prompt rows only (the easier case)")
    report([r for r in results if r["self_judged"]], "Self-judged rows only (Llama judging its own completions)")
    report([r for r in results if not r["self_judged"]], "Independent rows only (judge != completion source)")
    for source in sorted({r["source"] for r in results}):
        report([r for r in results if r["source"] == source], f"Source: {source}")

    verdict_counts = Counter(r["classifier_verdict"] for r in results)
    print(f"\nVerdict distribution: {dict(verdict_counts)}")
    most_common_frac = verdict_counts.most_common(1)[0][1] / len(results)
    print(f"Most-common single verdict accounts for {most_common_frac:.1%} of all outputs "
          f"(both prior judges defaulted to one category ~70%+ of the time -- this checks the same failure mode)")

    unparseable = sum(r["classifier_verdict"] == "unparseable" for r in results)
    print(f"Unparseable judge outputs: {unparseable}/{len(results)}")

    out_path = RESULTS_DIR / "moralize_comply_classifier_validation_llama_judge.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
