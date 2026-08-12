"""Formal paired significance test for each model's SAE-feature causal-
validation (suppression) curve. Originally written for gemma-2-9b-it alone
(its 96%->82% curve was reported descriptively but never formally tested,
unlike Qwen3-8B's non-overlapping-Wilson-CI argument or Llama-3.1-8B's
unambiguous 0% floor -- see reports/DECISIONS.md's "Closing a Wave 2 gap"
entry). Both of those other two arguments are still the same kind of
informal proxy this project's own adversarial-evaluation entry
(reports/DECISIONS.md, 2026-07-11) explicitly flagged as weaker than a
proper paired test for paired data ("comparing two separate Wilson CIs for
overlap ... can miss or wrongly suggest a real paired difference"), so this
now covers all three models the same way, not just gemma.

No new GPU compute needed: `scripts/validate_sae_features.py` already saved
every completion for every condition, per model
(`results/sae_suppression_validation_<model>.json`). This just reclassifies
each of the 50 VAL prompts' completions per condition with
`src.direction.refusal_classifier.is_refusal` and runs McNemar's exact test
(`src.eval.detector_metrics.mcnemar_exact`) between baseline and each
suppression condition -- the correct paired test for "does suppressing
these features change the SAME 50 prompts' refusal calls".

Usage: python scripts/suppression_significance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.direction.refusal_classifier import is_refusal
from src.eval.detector_metrics import mcnemar_exact

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS = ["Qwen3-8B", "Llama-3.1-8B-Instruct", "gemma-2-9b-it"]
CONDITIONS = ["top1", "top5", "top10", "top15", "top20"]


def main() -> None:
    for model in MODELS:
        validation_path = RESULTS_DIR / f"sae_suppression_validation_{model}.json"
        with open(validation_path) as fh:
            data = json.load(fh)

        print(f"\n=== {model} ===")
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

        out_path = RESULTS_DIR / f"sae_suppression_significance_{model}.json"
        with open(out_path, "w") as fh:
            json.dump({"model": model, "baseline_vs": results}, fh, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
