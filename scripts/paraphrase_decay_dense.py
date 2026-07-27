"""First half of a mechanistic dig into why PAIR-paraphrase robustness varies
by model: among the 3 SAE-having models, dense-direction PAIR robustness
ranks Llama-3.1-8B (66.7%) > gemma-2-9b-it (47.6%) > Qwen3-8B (42.9%) --
the *same order* as how causally concentrated each model's SAE-suppression
effect is (Llama: one dominant feature; gemma: modest/gradual; Qwen3-8B:
distributed, top-1 alone does nothing). This rules out a naive "redundancy
protects" story (the most-distributed model is the *least* robust, not the
most) and leaves "the concentrated feature is itself unusually paraphrase-
invariant" as the standing candidate -- see `scripts/paraphrase_decay_sae.py`
for the feature-level half of this test.

This script does the dense-direction half, across all 5 models: rather than
comparing *aggregate* margins across two unrelated prompt sets (what
`scripts/pair_margin_analysis.py` did), it uses the fact that every
PAIR-adversarial record is matched, by construction
(`src.eval.adversarial_paraphrase.build_adversarial_set`'s own goal-text
matching), to one exact original (unparaphrased) TEST-split harmful prompt
already sitting in the main activation cache. This gives 21 **matched
pairs** per model -- same underlying request, only the surface phrasing
differs -- so the paraphrase's effect on the dense-direction projection can
be measured directly per pair, not just compared across distributions.

Needs no new GPU generation -- pure lookup + linear algebra over
already-cached activations (`results/activations/{model}.pt` and
`_adversarial.pt`) and the already-persisted unit directions
(`results/dense_directions.pt`).

Limitation stated up front, not just at the end: 21 matched pairs come from
only 10 unique underlying goals (some goals have multiple PAIR paraphrase
variants) -- the Wilcoxon test below is on 21 pairs, not 21 independent
prompts.

Usage: python scripts/paraphrase_decay_dense.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DENSE_DIRECTIONS_PATH = RESULTS_DIR / "dense_directions.pt"
CROSS_MODEL_PATH = RESULTS_DIR / "dense_direction_cross_model.json"
QWEN3_THRESHOLDS_PATH = RESULTS_DIR / "detector_thresholds_Qwen3-8B.json"

# Known PAIR detection rates (RESULTS.md), for side-by-side comparison only.
KNOWN_PAIR_DETECTION_RATE = {
    "SmolLM2-1.7B-Instruct": 0.905,
    "Llama-3.1-8B-Instruct": 0.667,
    "gemma-2-9b-it": 0.476,
    "Qwen3-8B": 0.429,
    "Qwen2.5-1.5B-Instruct": 0.381,
    "DeepSeek-R1-Distill-Qwen-1.5B": 0.095,
}
MODELS = [
    "Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct", "Qwen3-8B",
    "Llama-3.1-8B-Instruct", "gemma-2-9b-it", "DeepSeek-R1-Distill-Qwen-1.5B",
]


def layer_and_threshold(cache_label: str) -> tuple[int, float]:
    if cache_label == "Qwen3-8B":
        with open(QWEN3_THRESHOLDS_PATH) as fh:
            d = json.load(fh)
        return d["dense_direction_layer"], d["thresholds"]["dense_direction"]
    with open(CROSS_MODEL_PATH) as fh:
        d = json.load(fh)
    return d[cache_label]["layer"], d[cache_label]["threshold"]


def find_matched_original_indices(cache: dict, pair_records: list[dict]) -> list[int]:
    """Matches each PAIR record's `goal` text against the model's own main
    corpus (source=='jbb', label=='harmful', split=='test'), same
    case/whitespace-normalized exact-match logic as
    `src.eval.adversarial_paraphrase.build_adversarial_set`. Asserts every
    record matches -- a silent miss here would silently corrupt every
    downstream number, not something to let slide."""
    eligible = {
        t.strip().lower(): i
        for i, (t, src, lab, sp) in enumerate(zip(cache["texts"], cache["sources"], cache["labels"], cache["splits"]))
        if src == "jbb" and lab == "harmful" and sp == "test"
    }
    indices = []
    for r in pair_records:
        goal = r["goal"].strip().lower()
        assert goal in eligible, f"PAIR record's goal not found in TEST-split jbb prompts: {goal!r}"
        indices.append(eligible[goal])
    return indices


def analyze_model(cache_label: str, direction: torch.Tensor, layer: int, threshold: float) -> dict:
    cache = load_cache(cache_label)
    acts = cache["activations"][layer].float()
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_val = torch.tensor([s == "val" for s in cache["splits"]], dtype=torch.bool)
    val_proj = acts[is_val] @ direction
    pooled_std = torch.cat([val_proj[labels[is_val]], val_proj[~labels[is_val]]]).std().item()

    adv_cache = torch.load(RESULTS_DIR / "activations" / f"{cache_label}_adversarial.pt", map_location="cpu", weights_only=False)
    records = adv_cache["records"]
    pair_idx = [i for i, r in enumerate(records) if r["method"] == "PAIR"]
    pair_records = [records[i] for i in pair_idx]
    original_idx = find_matched_original_indices(cache, pair_records)

    original_proj = acts[original_idx] @ direction
    paraphrased_proj = (adv_cache["activations"][layer].float()[pair_idx]) @ direction

    original_margin = (original_proj - threshold) / pooled_std
    paraphrased_margin = (paraphrased_proj - threshold) / pooled_std
    delta = original_margin - paraphrased_margin  # positive = paraphrase pushed toward harmless

    stat, p_value = wilcoxon(original_margin.tolist(), paraphrased_margin.tolist())

    return {
        "layer": layer,
        "n_pairs": len(pair_idx),
        "n_unique_goals": len(set(r["goal"].strip().lower() for r in pair_records)),
        "original_margin_mean": round(original_margin.mean().item(), 4),
        "paraphrased_margin_mean": round(paraphrased_margin.mean().item(), 4),
        "delta_mean": round(delta.mean().item(), 4),
        "wilcoxon": {"statistic": round(float(stat), 4), "p_value": round(float(p_value), 4), "n": len(pair_idx)},
        "known_pair_detection_rate": KNOWN_PAIR_DETECTION_RATE[cache_label],
    }


def exact_spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """scipy.stats.spearmanr's default p-value uses a t-distribution
    approximation that is not valid at n=5 (this project's full model
    count, not a sample to grow) -- computes the exact permutation p-value
    instead (all n! rankings of y against the fixed rank order of x),
    the correct tool at this sample size, same discipline as this
    project's other paired-test choices (McNemar exact, not asymptotic)."""
    rx, ry = rankdata(x), rankdata(y)
    observed = np.corrcoef(rx, ry)[0, 1]
    count = sum(
        1 for perm in itertools.permutations(ry) if abs(np.corrcoef(rx, perm)[0, 1]) >= abs(observed) - 1e-9
    )
    return observed, count / math.factorial(len(x))


def main() -> None:
    directions = torch.load(DENSE_DIRECTIONS_PATH, map_location="cpu", weights_only=False)
    results = {}
    print(f"{'model':<24} {'layer':>5} {'orig margin':>12} {'paraphr margin':>15} {'delta':>8} {'wilcoxon p':>11} {'known PAIR detect':>18}")
    for cache_label in MODELS:
        layer, threshold = layer_and_threshold(cache_label)
        direction = directions[cache_label]["direction"].float()
        r = analyze_model(cache_label, direction, layer, threshold)
        results[cache_label] = r
        print(f"{cache_label:<24} {r['layer']:>5} {r['original_margin_mean']:>12.3f} "
              f"{r['paraphrased_margin_mean']:>15.3f} {r['delta_mean']:>8.3f} "
              f"{r['wilcoxon']['p_value']:>11.4f} {r['known_pair_detection_rate']:>18.3f}")

    deltas = [results[m]["delta_mean"] for m in MODELS]
    known_rates = [results[m]["known_pair_detection_rate"] for m in MODELS]
    rho, p_value = exact_spearman(deltas, known_rates)
    print(f"\nSpearman rank correlation (matched-pair delta vs. known detection rate, n={len(MODELS)}, "
          f"exact permutation p): rho={rho:.4f}, p={p_value:.4f}")

    out = {"per_model": results, "spearman_delta_vs_detection_rate": {"rho": round(float(rho), 4), "p_value_exact": round(float(p_value), 4), "n": len(MODELS)}}
    out_path = RESULTS_DIR / "paraphrase_decay_dense.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
