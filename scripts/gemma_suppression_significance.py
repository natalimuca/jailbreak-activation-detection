"""Closes a gap flagged in Wave 2's write-up: gemma-2-9b-it's causal-
validation curve (96% baseline -> 82% at top-15/top-20) was reported
descriptively but never run through a formal paired significance test,
unlike Qwen3-8B (whose top-15 vs baseline distinction rests on
non-overlapping Wilson CIs, already a stronger standard) or Llama-3.1-8B
(0% at top-15, an unambiguous floor).

No new GPU compute needed: `scripts/validate_sae_features.py`
already saved every completion for every condition
(`results/sae_suppression_validation_gemma-2-9b-it.json`). This just
reclassifies each of the 50 VAL prompts' completions per condition with
`src.direction.refusal_classifier.is_refusal` and runs McNemar's exact
test (`src.eval.detector_metrics.mcnemar_exact`) between baseline and
each suppression condition -- the correct paired test for "does
suppressing these features change the SAME 50 prompts' refusal calls",
same discipline this project already applies to paired detector
comparisons (see DECISIONS.md's Phase 4/Wave 3 entries).

Usage: python scripts/gemma_suppression_significance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.direction.refusal_classifier import is_refusal
from src.eval.detector_metrics import mcnemar_exact

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
VALIDATION_PATH = RESULTS_DIR / "sae_suppression_validation_gemma-2-9b-it.json"
CONDITIONS = ["top1", "top5", "top10", "top15", "top20"]


def main() -> None:
    with open(VALIDATION_PATH) as fh:
        data = json.load(fh)

    baseline_refusals = [is_refusal(c) for c in data["results"]["baseline"]["completions"]]
    print(f"baseline: refusal={sum(baseline_refusals)}/{len(baseline_refusals)}")

    results = {}
    for condition in CONDITIONS:
        condition_refusals = [is_refusal(c) for c in data["results"][condition]["completions"]]
        mc = mcnemar_exact(baseline_refusals, condition_refusals)
        results[condition] = mc
        print(
            f"{condition:6s}: refusal={sum(condition_refusals)}/{len(condition_refusals)}  "
            f"discordant={mc['n_discordant']} (baseline-only={mc['only_a']}, "
            f"{condition}-only={mc['only_b']})  p={mc['p_value']}"
        )

    out_path = RESULTS_DIR / "sae_suppression_significance_gemma-2-9b-it.json"
    with open(out_path, "w") as fh:
        json.dump({"model": "gemma-2-9b-it", "baseline_vs": results}, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
