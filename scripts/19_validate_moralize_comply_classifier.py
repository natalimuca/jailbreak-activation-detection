"""Validates `src.direction.moralize_comply_classifier` against two
already-labeled worksheets -- no new human/Claude labeling needed:

  - `results/classifier_spotcheck_worksheet.csv` (scripts/07/08's original
    45-row set, Qwen3-8B + Phase 1 sources).
  - `results/classifier_spotcheck_worksheet_expansion.csv` (scripts/18's
    53-row set, Llama-3.1-8B/gemma-2-9b-it suppression + cross-model-
    transfer sources -- the original set had zero coverage of these).

Reports accuracy broken down by source AND by harmful-vs-harmless-prompt
condition, not one aggregate number -- the whole point of this classifier
is not smearing easy cases (harmless-prompt compliance) into the hard,
actually load-bearing case (harmful-prompt post-intervention). This is
the same "don't trust the aggregate" discipline that originally found the
moralize confound (see scripts/06/RESULTS.md).

Usage: python scripts/19_validate_moralize_comply_classifier.py
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
                # "refuse" rows ARE included: they test whether the classifier's
                # refuse safety net correctly catches what is_refusal missed
                # (e.g. the curly-apostrophe bug found while building this
                # worksheet -- see DECISIONS.md), not excluded as out of scope.
                rows.append({
                    "worksheet": path.name,
                    "source": row["source"],
                    "condition": row["condition"],
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "human_label": label,
                    "is_harmful": "harmless" not in row["condition"],
                })
    return rows


def main() -> None:
    rows = load_rows()
    print(f"Loaded {len(rows)} non-refuse, human-labeled rows across both worksheets")
    print(f"  by human label: {dict(Counter(r['human_label'] for r in rows))}")
    print(f"  harmful-prompt rows: {sum(r['is_harmful'] for r in rows)}, "
          f"harmless-prompt rows: {sum(not r['is_harmful'] for r in rows)}")

    print("\nLoading judge model (4-bit)")
    judge_model, judge_tokenizer = load_judge_model()

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

    print("\n\n########## VALIDATION RESULTS ##########")
    report(results, "Overall (all rows, both worksheets)")
    report([r for r in results if r["is_harmful"]], "Harmful-prompt rows only (the load-bearing case)")
    report([r for r in results if not r["is_harmful"]], "Harmless-prompt rows only (the easier case)")
    for source in sorted({r["source"] for r in results}):
        report([r for r in results if r["source"] == source], f"Source: {source}")

    unparseable = sum(r["classifier_verdict"] == "unparseable" for r in results)
    misclassified_as_refuse = sum(r["classifier_verdict"] == "refuse" for r in results)
    print(f"\nUnparseable judge outputs: {unparseable}/{len(results)}")
    print(f"Classifier called 'refuse' on an is_refusal-negative completion: {misclassified_as_refuse}/{len(results)} "
          f"(possible residual is_refusal miss, or judge disagreement)")

    out_path = RESULTS_DIR / "moralize_comply_classifier_validation.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
