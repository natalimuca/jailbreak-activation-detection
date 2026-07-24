"""Investigates why Llama-3.1-8B-Instruct's dense-direction approach is weak on both
necessity (own-ablation null at n=75, McNemar p=1.0, see scripts/replicate_llama_ablation.py)
and sufficiency (activation addition capped at 10%->34%, degenerates from alpha=2.0, see
scripts/sufficiency_at_scale.py), while its SAE-feature approach ablates refusal
dramatically (98%->10% from a single feature, see scripts/rank_sae_features.py +
scripts/validate_sae_features.py Wave 2 results) -- unlike this project's other cross-model
work, this needs no new generation: everything here is a lookup/computation over
already-cached activations (results/activations/*.pt) and already-downloaded SAE checkpoints
(via src.sae.registry), run once on CPU.

Tests two candidate explanations against the data already on disk and reports both, including
the one that did NOT survive:

1. "Llama's raw diff-in-means direction is just small, so ablating/adding it barely
   perturbs the residual stream." Checked by normalizing each model's raw_direction_norm
   (already saved per-model in results/phase1_reproduction_*.json and
   results/sufficiency_7b_9b_scale.json where available; computed fresh here for
   gemma-2-9b-it, which has neither file, since this ratio needs no causal-generation
   test to compute) by that same layer's own ambient activation norm, computed directly
   from the cached activation tensors. Result: Llama's ratio is the LARGEST of the models
   tested (not the smallest) -- this hypothesis does not survive and is reported as ruled
   out, not smoothed over.

2. "The dense direction and the top causal SAE feature point in different directions,
   so the classifier axis and the causal lever are simply different objects for Llama."
   Checked by loading the real SAE decoder weight for each model's own top causal feature
   (LlamaScope layer 27/feature 13363, Qwen-Scope layer 25/feature 65291, GemmaScope layer
   35/feature 52410) and computing cosine similarity against the dense direction at that
   same layer (recomputed from the TRAIN-split cached activations, matching
   src.direction.compute.compute_directions exactly), against a random-200-feature null
   baseline for scale. Result: Llama and Qwen3-8B show near-identical, modest alignment
   (~0.20, vs ~0.013-0.016 for random features) -- this does NOT differentiate the two
   models and is reported as inconclusive, not as confirming evidence.

Extended to gemma-2-9b-it (2026-07-24) as a third data point for the SAE causal-effect-
concentration spread (Llama concentrated in 1 feature, Qwen3-8B distributed, gemma modest/
gradual with no sharp ranking-score standout at all -- see RESULTS.md). Note: gemma has no
own-direction dense-ablation causal-generation test in this project at all (only Qwen3-8B,
Llama-3.1-8B were tested that way, plus the two small Phase-1 models) -- this script's two
checks don't depend on that test existing, but the necessity/sufficiency correlation framing
from the Llama-vs-Qwen3-8B comparison can't be extended to gemma the same way, reported
honestly rather than glossed over.

What the data DOES support, pulled directly from already-saved results (no new computation
needed): a genuine concentration-vs-distribution asymmetry already documented in RESULTS.md's
SAE cross-model section -- Llama's causal effect collapses into a single dominant feature
(top-1 alone: 86%->10%) where Qwen3-8B's is spread across the ranked set (top-1 alone: no
effect, 82%->84%) and gemma's declines modestly and gradually (96%->82% by top-15/20, no
single feature dominates). This asymmetry co-occurs with which model's *dense*-direction
necessity/sufficiency is clean vs. null for the two models where that causal test exists,
but this script does not claim to have found the mechanism connecting the two -- see
RESULTS.md for the full, hedged write-up.

Usage: python scripts/analyze_llama_causal_gap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.direction.compute import compute_directions, compute_raw_directions
from src.sae.registry import SAE_PROVIDERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_RANDOM_BASELINE = 200
RANDOM_SEED = 0

# (cache_label, layer, raw_direction_norm) -- raw_direction_norm already reported in
# results/phase1_reproduction_*.json and results/sufficiency_7b_9b_scale.json where available;
# None means "no prior published value -- compute fresh from cache below" (true for
# gemma-2-9b-it, which was never run through either of those scripts).
NECESSITY_MODELS = [
    ("Qwen2.5-1.5B-Instruct", 23, 75.3),
    ("SmolLM2-1.7B-Instruct", 20, 279.6),
    ("Qwen3-8B", 23, 99.3335),
    ("Llama-3.1-8B-Instruct", 27, 19.1452),
    ("gemma-2-9b-it", 34, None),
]

# (cache_label, hf_model_name, sae_layer, top_causal_feature) -- sae_layer/feature from
# results/sae_causal_ranking_{model}.json's rank-1 entry.
SAE_ALIGNMENT_MODELS = [
    ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", 27, 13363),
    ("Qwen3-8B", "Qwen/Qwen3-8B", 25, 65291),
    ("gemma-2-9b-it", "google/gemma-2-9b-it", 35, 52410),
]


def ambient_norm_at_layer(cache_label: str, layer: int) -> dict:
    cache = load_cache(cache_label)
    layer_acts = cache["activations"][layer].float()
    norms = layer_acts.norm(dim=-1)
    return {"mean": round(norms.mean().item(), 4), "std": round(norms.std().item(), 4)}


def dense_direction_at_layer(cache_label: str, layer: int) -> torch.Tensor:
    cache = load_cache(cache_label)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_train = torch.tensor([s == "train" for s in cache["splits"]], dtype=torch.bool)
    directions = compute_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])
    return directions[layer].float()


def raw_direction_norm_at_layer(cache_label: str, layer: int) -> float:
    cache = load_cache(cache_label)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_train = torch.tensor([s == "train" for s in cache["splits"]], dtype=torch.bool)
    raw_directions = compute_raw_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])
    return round(raw_directions[layer].float().norm().item(), 4)


def feature_vector(W_dec: torch.Tensor, d_model: int, feature: int) -> torch.Tensor:
    # W_dec is (d_model, d_sae) for both LlamaScope and Qwen-Scope; d_sae >> d_model in both,
    # but check against the actual residual-stream width rather than assuming shape order.
    return W_dec[:, feature] if W_dec.shape[0] == d_model else W_dec[feature]


def decoder_vector(hf_model_name: str, layer: int, feature: int, d_model: int) -> torch.Tensor:
    load_sae_fn = SAE_PROVIDERS[hf_model_name][0]
    sae = load_sae_fn(layer)
    W_dec = sae.W_dec.float()
    return feature_vector(W_dec, d_model, feature)


def cosine_alignment_check(cache_label: str, hf_model_name: str, layer: int, feature: int) -> dict:
    dense_dir = dense_direction_at_layer(cache_label, layer)
    d_model = dense_dir.shape[0]
    feat_vec = decoder_vector(hf_model_name, layer, feature, d_model)
    top_feature_cos = torch.nn.functional.cosine_similarity(dense_dir, feat_vec, dim=0).item()

    load_sae_fn = SAE_PROVIDERS[hf_model_name][0]
    sae = load_sae_fn(layer)
    W_dec = sae.W_dec.float()
    d_sae = W_dec.shape[1] if W_dec.shape[0] == d_model else W_dec.shape[0]

    rng = torch.Generator().manual_seed(RANDOM_SEED)
    rand_feats = torch.randint(0, d_sae, (N_RANDOM_BASELINE,), generator=rng)
    rand_cos = torch.tensor([
        torch.nn.functional.cosine_similarity(dense_dir, feature_vector(W_dec, d_model, int(f)), dim=0).item()
        for f in rand_feats
    ])
    return {
        "layer": layer,
        "top_causal_feature": feature,
        "top_causal_feature_cosine": round(top_feature_cos, 4),
        "random_baseline_mean_abs_cosine": round(rand_cos.abs().mean().item(), 4),
        "random_baseline_max_abs_cosine": round(rand_cos.abs().max().item(), 4),
        "n_random": N_RANDOM_BASELINE,
    }


def main() -> None:
    print("=== Hypothesis 1: raw direction norm vs. ambient activation norm ===")
    magnitude_results = []
    for cache_label, layer, raw_norm in NECESSITY_MODELS:
        freshly_computed = raw_norm is None
        if freshly_computed:
            raw_norm = raw_direction_norm_at_layer(cache_label, layer)
        ambient = ambient_norm_at_layer(cache_label, layer)
        ratio = round(raw_norm / ambient["mean"], 4)
        note = " (freshly computed, no prior published value)" if freshly_computed else ""
        print(f"  {cache_label} layer {layer}: ambient_norm_mean={ambient['mean']} "
              f"raw_direction_norm={raw_norm} ratio={ratio}{note}")
        magnitude_results.append({
            "model": cache_label, "layer": layer, "raw_direction_norm": raw_norm,
            "raw_direction_norm_freshly_computed": freshly_computed,
            "ambient_activation_norm": ambient, "ratio": ratio,
        })

    print("\n=== Hypothesis 2: dense direction vs. top causal SAE feature, cosine alignment ===")
    alignment_results = {}
    for cache_label, hf_model_name, layer, feature in SAE_ALIGNMENT_MODELS:
        result = cosine_alignment_check(cache_label, hf_model_name, layer, feature)
        print(f"  {cache_label} layer {layer} feature {feature}: "
              f"cos={result['top_causal_feature_cosine']} "
              f"(random baseline mean|cos|={result['random_baseline_mean_abs_cosine']}, "
              f"max|cos|={result['random_baseline_max_abs_cosine']})")
        alignment_results[cache_label] = result

    out = {
        "magnitude_vs_ambient_norm": magnitude_results,
        "dense_vs_sae_cosine_alignment": alignment_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "llama_causal_gap_analysis.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
