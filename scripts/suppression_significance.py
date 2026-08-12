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

**DeepSeek-R1-Distill-Qwen-1.5B needs a different path** (`reasoning_model:
true` in its own validation JSON): its mandatory `<think>` block means a
real, condition-varying fraction of completions truncate before reaching an
answer (33-60% per condition, already documented in RESULTS.md as making
this model's curve "inconclusive, not negative"), so the raw 15 completions
per condition are NOT already-matched pairs the way the other three models'
are. Naively calling `is_refusal` on all 15 raw completions per condition
(the path used for the other three) would silently include truncated
non-answers as "not a refusal" and, worse, pair up different prompt subsets
across conditions positionally. `resolve_completions_by_index` (already
built for exactly this -- see its own docstring) is used instead, and each
condition's comparison against baseline is restricted to the intersection
of prompt indices that resolved (reached a real answer) in BOTH -- the
correct paired subset, not an assumption that every index survived
everywhere.

Usage: python scripts/suppression_significance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.direction.refusal_classifier import is_refusal, resolve_completions_by_index
from src.eval.detector_metrics import mcnemar_exact

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS = ["Qwen3-8B", "Llama-3.1-8B-Instruct", "gemma-2-9b-it", "DeepSeek-R1-Distill-Qwen-1.5B"]
CONDITIONS = ["top1", "top5", "top10", "top15", "top20"]


def paired_refusals(baseline_by_idx: dict[int, str], condition_by_idx: dict[int, str]) -> tuple[list[bool], list[bool]]:
    shared_idx = sorted(set(baseline_by_idx) & set(condition_by_idx))
    baseline_refusals = [is_refusal(baseline_by_idx[i]) for i in shared_idx]
    condition_refusals = [is_refusal(condition_by_idx[i]) for i in shared_idx]
    return baseline_refusals, condition_refusals


def main() -> None:
    for model in MODELS:
        validation_path = RESULTS_DIR / f"sae_suppression_validation_{model}.json"
        with open(validation_path) as fh:
            data = json.load(fh)
        reasoning_model = data.get("reasoning_model", False)

        print(f"\n=== {model} ===")
        results = {}

        if not reasoning_model:
            baseline_refusals = [is_refusal(c) for c in data["results"]["baseline"]["completions"]]
            print(f"baseline: refusal={sum(baseline_refusals)}/{len(baseline_refusals)}")
            for condition in CONDITIONS:
                condition_refusals = [is_refusal(c) for c in data["results"][condition]["completions"]]
                mc = mcnemar_exact(baseline_refusals, condition_refusals)
                results[condition] = mc
                print(
                    f"{condition:6s}: refusal={sum(condition_refusals)}/{len(condition_refusals)}  "
                    f"discordant={mc['n_discordant']} (baseline-only={mc['only_a']}, "
                    f"{condition}-only={mc['only_b']})  p={mc['p_value']}"
                )
        else:
            baseline_by_idx = resolve_completions_by_index(data["results"]["baseline"]["completions"])
            print(f"baseline: {len(baseline_by_idx)}/{len(data['results']['baseline']['completions'])} resolved")
            for condition in CONDITIONS:
                condition_by_idx = resolve_completions_by_index(data["results"][condition]["completions"])
                baseline_refusals, condition_refusals = paired_refusals(baseline_by_idx, condition_by_idx)
                mc = mcnemar_exact(baseline_refusals, condition_refusals)
                mc["n_paired"] = len(baseline_refusals)
                results[condition] = mc
                print(
                    f"{condition:6s}: {len(condition_by_idx)} resolved, {len(baseline_refusals)} paired with baseline, "
                    f"refusal={sum(condition_refusals)}/{len(condition_refusals)}  "
                    f"discordant={mc['n_discordant']} (baseline-only={mc['only_a']}, "
                    f"{condition}-only={mc['only_b']})  p={mc['p_value']}"
                )

        out_path = RESULTS_DIR / f"sae_suppression_significance_{model}.json"
        with open(out_path, "w") as fh:
            json.dump({"model": model, "reasoning_model": reasoning_model, "baseline_vs": results}, fh, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
