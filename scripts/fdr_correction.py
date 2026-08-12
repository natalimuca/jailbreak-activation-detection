"""Closes a gap `reports/RESULTS.md`'s "Known limitations (cross-model
dense-direction comparison)" section flags honestly: BH-FDR correction is
applied in exactly one place in this project (the wrapper-swap maxT scheme)
and nowhere else, even though several other families of paired tests exist.
A few of those sit close enough to 0.05 that a family-wise correction could
matter -- the LLM-judge PAIR comparisons and the threshold-rule-vs-judge
comparison, both named in that limitations bullet.

Every p-value below is already published; this computes no new statistics
from raw data, only a BH-FDR correction over already-reported numbers.
Sourced from the actual result JSONs wherever one exists (not rounded
prose): `results/threshold_recalibration.json`, `results/paraphrase_decay_sae.json`,
`results/llama_causal_gap_decomposition.json`. DeLong AUROC and
PAIR-McNemar-vs-judge were never persisted to a JSON (computed ad hoc when
first written up); `reports/DECISIONS.md`'s "A wrong paired test, caught and
corrected" entry (2026-08-04) confirms both were unaffected by the McNemar
bug found there, so they're used as-is at their published precision.

**Family design, fixed here rather than left implicit** (see
reports/DECISIONS.md for the full reasoning): five families, none pooled
across the boundary the limitations bullet's own imprecise phrasing
suggested. Notably, the bullet's "threshold-rule-vs-judge comparison
(p=0.0201, p=0.0225)" bundles two different kinds of test -- 0.0201 is
Qwen3-8B's now-superseded youden_j-vs-judge accuracy comparison (the same
underlying question as Family A's final Qwen entry, just at an abandoned
threshold), 0.0225 is max_accuracy-vs-youden_j (no judge involved at all).
0.0201 is excluded from every family below rather than double-counted.

Usage: python scripts/fdr_correction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

FAMILIES = {
    "A_detector_vs_judge": {
        "description": "Dense-direction vs. LLM-judge, at each model's adopted (max_accuracy) threshold",
        "tests": {
            "Qwen3-8B DeLong AUROC": 0.0041,
            "Llama-3.1-8B DeLong AUROC": 0.0015,
            "gemma-2-9b-it DeLong AUROC": 0.0053,
            "Qwen3-8B accuracy McNemar": 0.3833,
            "Llama-3.1-8B accuracy McNemar": 0.8388,
            "gemma-2-9b-it accuracy McNemar": 0.8238,
            "Qwen3-8B PAIR McNemar": 0.031,
            "Llama-3.1-8B PAIR McNemar": 0.69,
            "gemma-2-9b-it PAIR McNemar": 0.039,
        },
    },
    "B_reselection_vs_youden": {
        "description": "max_accuracy threshold vs. youden_j threshold, TEST accuracy (same detector, no judge)",
        "tests": {
            "Qwen3-8B max_accuracy vs youden_j": 0.0225,
            "Llama-3.1-8B max_accuracy vs youden_j": 1.0,
            "gemma-2-9b-it max_accuracy vs youden_j": 1.0,
        },
    },
    "C_sae_paraphrase_decay": {
        "description": "SAE-feature paraphrase-decay Wilcoxon, top-1 feature delta and full top-15 score delta",
        "tests": {
            "Llama-3.1-8B top-1 feature": 0.1678,
            "Llama-3.1-8B full top-15 score": 0.0038,
            "Qwen3-8B top-1 feature": 0.0,
            "Qwen3-8B full top-15 score": 0.0,
            "gemma-2-9b-it top-1 feature": 0.0033,
            "gemma-2-9b-it full top-15 score": 0.0001,
        },
    },
    "D1_component_decomp_llama": {
        "description": "Llama component-decomposition post-hoc pairwise McNemar (following Llama's own omnibus Cochran's Q)",
        "tests": {
            "baseline_vs_own_ablation": 0.5,
            "baseline_vs_parallel_ablation": 0.0,
            "baseline_vs_orthogonal_ablation": 1.0,
            "own_ablation_vs_parallel_ablation": 0.0,
            "own_ablation_vs_orthogonal_ablation": 0.5,
            "parallel_ablation_vs_orthogonal_ablation": 0.0,
        },
    },
    "D2_component_decomp_qwen": {
        "description": "Qwen3-8B component-decomposition post-hoc pairwise McNemar (following Qwen3-8B's own omnibus Cochran's Q)",
        "tests": {
            "baseline_vs_own_ablation": 0.0,
            "baseline_vs_parallel_ablation": 0.0,
            "baseline_vs_orthogonal_ablation": 0.0,
            "own_ablation_vs_parallel_ablation": 0.125,
            "own_ablation_vs_orthogonal_ablation": 0.125,
            "parallel_ablation_vs_orthogonal_ablation": 1.0,
        },
    },
}


def correct_family(tests: dict[str, float]) -> dict[str, dict]:
    names = list(tests.keys())
    raw_p = [tests[n] for n in names]
    adjusted = false_discovery_control(raw_p, method="bh")
    return {
        name: {
            "raw_p": raw_p[i],
            "bh_adjusted_p": float(adjusted[i]),
            "significant_before": raw_p[i] < 0.05,
            "significant_after": float(adjusted[i]) < 0.05,
        }
        for i, name in enumerate(names)
    }


def main() -> None:
    out = {}
    for family_key, family in FAMILIES.items():
        print(f"\n=== {family_key}: {family['description']} ===")
        result = correct_family(family["tests"])
        out[family_key] = {"description": family["description"], "tests": result}
        for name, r in result.items():
            flipped = r["significant_before"] != r["significant_after"]
            flag = " <-- CONCLUSION CHANGES" if flipped else ""
            print(
                f"  {name:45s} raw_p={r['raw_p']:<8g} bh_q={r['bh_adjusted_p']:.4g}"
                f"  sig_before={r['significant_before']}  sig_after={r['significant_after']}{flag}"
            )

    out_path = RESULTS_DIR / "multiple_comparisons_correction.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
