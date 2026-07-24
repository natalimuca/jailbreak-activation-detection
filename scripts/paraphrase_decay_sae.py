"""Second half of the mechanistic dig started in `scripts/paraphrase_decay_dense.py`:
that script found a striking result at the dense-direction level -- across
all 5 models, the matched-pair projection delta (original harmful prompt vs.
its real PAIR paraphrase) ranks in *exactly* the same order as known
PAIR-detection robustness (Spearman rho=-1.0, exact permutation p=0.0167,
n=5). This script goes one level deeper, at the SAE-feature level, for the
3 models with a trained SAE (Qwen3-8B, Llama-3.1-8B-Instruct, gemma-2-9b-it)
-- the ones whose causal-suppression effect ranges from concentrated in one
dominant feature (Llama) to fully distributed (Qwen3-8B) to modest/gradual
with no standout (gemma), see RESULTS.md's SAE cross-model section.

Two related but distinct questions, kept separate rather than conflated:

1. **Does the model's own top-ranked causal feature's activation survive
   paraphrasing better in the model whose refusal mechanism is more
   concentrated in it?** (the "this specific feature is unusually
   paraphrase-invariant" hypothesis -- the dense-direction result above
   already rules out the competing "redundancy protects" story, since the
   *most distributed* model, Qwen3-8B, is the *least* robust, not the
   most.)
2. **Does the full SAE-feature-detector score (sum of top-15, the actual
   published-detector quantity) decay differently from the single top
   feature within the same model?** -- if concentration is genuinely about
   one feature carrying the signal, the full-ensemble score and the
   top-1-feature score should decay similarly for Llama (since one feature
   already carries ~everything) but not necessarily for Qwen3-8B (where
   top-1 alone carries nothing).

Same matched-pair method as the dense-direction script: every PAIR record
is matched, by construction, to one exact original TEST-split harmful
prompt already in the main activation cache -- no new GPU generation,
reuses `src.detectors.sae_feature_detector.score` unchanged (the exact
function behind the published SAE-feature PAIR-detection rates) and
`src.sae.registry.SAE_PROVIDERS` for per-model SAE loading.

Limitation stated up front: n=3 models here, not 5 -- Qwen2.5-1.5B and
SmolLM2 have no trained SAE, so the two most extreme dense-direction
PAIR-robustness models can never be checked at this feature-level
resolution.

Usage: python scripts/paraphrase_decay_sae.py
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
from src.detectors.sae_feature_detector import load_top_features, score
from src.sae.registry import SAE_PROVIDERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}
# (layer, feature) of each model's own rank-1 causally-ranked feature, from
# results/sae_causal_ranking_{model}.json -- repeated here as a lookup, not recomputed.
TOP_FEATURE = {
    "Qwen3-8B": (25, 65291),
    "Llama-3.1-8B-Instruct": (27, 13363),
    "gemma-2-9b-it": (35, 52410),
}
# Known SAE-feature-detector PAIR detection rates (RESULTS.md), for comparison only.
KNOWN_SAE_PAIR_DETECTION_RATE = {"Qwen3-8B": 0.333, "Llama-3.1-8B-Instruct": 0.809, "gemma-2-9b-it": 0.238}


def find_matched_original_indices(cache: dict, pair_records: list[dict]) -> list[int]:
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


def exact_spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    rx, ry = rankdata(x), rankdata(y)
    observed = np.corrcoef(rx, ry)[0, 1]
    count = sum(
        1 for perm in itertools.permutations(ry) if abs(np.corrcoef(rx, perm)[0, 1]) >= abs(observed) - 1e-9
    )
    return observed, count / math.factorial(len(x))


def analyze_model(cache_label: str, hf_model_name: str) -> dict:
    layer, feature_idx = TOP_FEATURE[cache_label]
    load_sae_fn = SAE_PROVIDERS[hf_model_name][0]
    sae = load_sae_fn(layer)

    cache = load_cache(cache_label)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_val = torch.tensor([s == "val" for s in cache["splits"]], dtype=torch.bool)

    adv_cache = torch.load(RESULTS_DIR / "activations" / f"{cache_label}_adversarial.pt", map_location="cpu", weights_only=False)
    records = adv_cache["records"]
    pair_idx = [i for i, r in enumerate(records) if r["method"] == "PAIR"]
    pair_records = [records[i] for i in pair_idx]
    original_idx = find_matched_original_indices(cache, pair_records)

    # --- Question 1: single top-ranked feature's own activation ---
    val_harmful_acts = acts[layer, is_val, :].float()[labels[is_val]]
    val_harmful_feature_acts = sae.encode(val_harmful_acts)[:, feature_idx]
    feature_std = val_harmful_feature_acts.std().item()

    original_layer_acts = acts[layer, original_idx, :].float()
    paraphrased_layer_acts = adv_cache["activations"][layer].float()[pair_idx]
    original_feature_val = sae.encode(original_layer_acts)[:, feature_idx]
    paraphrased_feature_val = sae.encode(paraphrased_layer_acts)[:, feature_idx]

    original_feature_norm = original_feature_val / feature_std
    paraphrased_feature_norm = paraphrased_feature_val / feature_std
    feature_delta = original_feature_norm - paraphrased_feature_norm
    feature_stat, feature_p = wilcoxon(original_feature_norm.tolist(), paraphrased_feature_norm.tolist())

    # --- Question 2: full top-15 SAE-feature-detector score ---
    with open(RESULTS_DIR / f"detector_thresholds_{cache_label}.json") as fh:
        thresholds = json.load(fh)
    score_layers = thresholds["sae_feature_layers"]
    features = load_top_features(k=thresholds["sae_feature_k"], path=RESULTS_DIR / f"sae_causal_ranking_{cache_label}.json")
    saes = {l: load_sae_fn(l) for l in score_layers}

    val_by_layer = {l: acts[l, is_val, :].float()[labels[is_val]] for l in score_layers}
    val_scores = score(val_by_layer, saes, features)
    score_std = val_scores.std().item()

    original_by_layer = {l: acts[l, original_idx, :].float() for l in score_layers}
    paraphrased_by_layer = {l: adv_cache["activations"][l].float()[pair_idx] for l in score_layers}
    original_scores = score(original_by_layer, saes, features) / score_std
    paraphrased_scores = score(paraphrased_by_layer, saes, features) / score_std
    score_delta = original_scores - paraphrased_scores
    score_stat, score_p = wilcoxon(original_scores.tolist(), paraphrased_scores.tolist())

    return {
        "top_feature": {"layer": layer, "feature": feature_idx},
        "top_feature_original_mean": round(original_feature_norm.mean().item(), 4),
        "top_feature_paraphrased_mean": round(paraphrased_feature_norm.mean().item(), 4),
        "top_feature_delta_mean": round(feature_delta.mean().item(), 4),
        "top_feature_wilcoxon": {"statistic": round(float(feature_stat), 4), "p_value": round(float(feature_p), 4)},
        "full_score_original_mean": round(original_scores.mean().item(), 4),
        "full_score_paraphrased_mean": round(paraphrased_scores.mean().item(), 4),
        "full_score_delta_mean": round(score_delta.mean().item(), 4),
        "full_score_wilcoxon": {"statistic": round(float(score_stat), 4), "p_value": round(float(score_p), 4)},
        "n_pairs": len(pair_idx),
        "known_sae_pair_detection_rate": KNOWN_SAE_PAIR_DETECTION_RATE[cache_label],
    }


def main() -> None:
    results = {}
    print(f"{'model':<24} {'top-feat delta':>15} {'top-feat p':>11} {'full-score delta':>17} {'full-score p':>13} {'known SAE PAIR detect':>22}")
    for cache_label, hf_model_name in MODELS.items():
        r = analyze_model(cache_label, hf_model_name)
        results[cache_label] = r
        print(f"{cache_label:<24} {r['top_feature_delta_mean']:>15.3f} {r['top_feature_wilcoxon']['p_value']:>11.4f} "
              f"{r['full_score_delta_mean']:>17.3f} {r['full_score_wilcoxon']['p_value']:>13.4f} "
              f"{r['known_sae_pair_detection_rate']:>22.3f}")

    models_ordered = list(MODELS.keys())
    top_feature_deltas = [results[m]["top_feature_delta_mean"] for m in models_ordered]
    known_rates = [results[m]["known_sae_pair_detection_rate"] for m in models_ordered]
    rho, p_value = exact_spearman(top_feature_deltas, known_rates)
    print(f"\nSpearman (top-feature delta vs. known SAE-feature PAIR detection rate, n={len(models_ordered)}, "
          f"exact permutation p): rho={rho:.4f}, p={p_value:.4f}")

    out = {
        "per_model": results,
        "spearman_top_feature_delta_vs_detection_rate": {"rho": round(float(rho), 4), "p_value_exact": round(float(p_value), 4), "n": len(models_ordered)},
    }
    out_path = RESULTS_DIR / "paraphrase_decay_sae.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
