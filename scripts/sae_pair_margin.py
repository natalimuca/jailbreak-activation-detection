"""Investigates the one case in this project where SAE-feature numerically
beats dense-direction on PAIR-paraphrase detection: Llama-3.1-8B-Instruct
(SAE 80.9% vs. dense 66.7%, McNemar p=0.25 -- not significant at n=21, see
RESULTS.md's cross-model SAE-feature extension section). Applies the same
continuous-margin method used for the dense-direction PAIR-robustness
analysis (`scripts/pair_margin_analysis.py`) to the SAE-feature score
instead, to see whether a finer-grained measure shows a real underlying
signal or whether the raw-rate flip is consistent with noise at this
sample size.

Needs no new GPU generation -- reuses Llama's already-cached main and
adversarial activations (`results/activations/Llama-3.1-8B-Instruct*.pt`),
already-selected top-15 (layer, feature) list
(`results/sae_causal_ranking_Llama-3.1-8B-Instruct.json`), already-downloaded
LlamaScope SAE checkpoints, and the already-calibrated threshold
(`results/detector_thresholds_Llama-3.1-8B-Instruct.json`). Reuses
`src.detectors.sae_feature_detector.score` unchanged -- the exact same
scoring function the published 80.9% detection rate came from, not a
reimplementation.

Usage: python scripts/sae_pair_margin.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.detectors.sae_feature_detector import load_top_features, score
from src.sae.llama_scope import load_sae

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_CACHE_LABEL = "Llama-3.1-8B-Instruct"
RANKING_PATH = RESULTS_DIR / "sae_causal_ranking_Llama-3.1-8B-Instruct.json"
THRESHOLDS_PATH = RESULTS_DIR / "detector_thresholds_Llama-3.1-8B-Instruct.json"
CROSS_MODEL_PATH = RESULTS_DIR / "dense_direction_cross_model.json"
DENSE_DIRECTIONS_PATH = RESULTS_DIR / "dense_directions.pt"

# Already known from RESULTS.md's dense-direction PAIR-margin analysis (scripts/pair_margin_analysis.py),
# repeated here only for side-by-side comparison, not recomputed.
DENSE_HARMFUL_MARGIN = 0.936
DENSE_PAIR_MARGIN = 0.332
DENSE_PAIR_DETECTION_RATE = 0.667
KNOWN_SAE_PAIR_DETECTION_RATE = 0.809


def dense_pair_margins(pair_idx: list[int]) -> torch.Tensor:
    """Per-prompt dense-direction margins on the same 21 PAIR prompts, for a
    proper paired test against the SAE-feature margins below -- not just a
    comparison of two aggregate means."""
    with open(CROSS_MODEL_PATH) as fh:
        cm = json.load(fh)[MODEL_CACHE_LABEL]
    layer, threshold = cm["layer"], cm["threshold"]
    direction = torch.load(DENSE_DIRECTIONS_PATH, map_location="cpu", weights_only=False)[MODEL_CACHE_LABEL]["direction"].float()

    cache = load_cache(MODEL_CACHE_LABEL)
    acts = cache["activations"][layer].float()
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_val = torch.tensor([s == "val" for s in cache["splits"]], dtype=torch.bool)
    val_proj = acts[is_val] @ direction
    val_labels_bool = labels[is_val]
    pooled_std = torch.cat([val_proj[val_labels_bool], val_proj[~val_labels_bool]]).std().item()

    adv_cache = torch.load(RESULTS_DIR / "activations" / f"{MODEL_CACHE_LABEL}_adversarial.pt", map_location="cpu", weights_only=False)
    adv_acts = adv_cache["activations"][layer].float()
    pair_proj = adv_acts[pair_idx] @ direction
    return (pair_proj - threshold) / pooled_std


def activations_by_layer(cache_activations: torch.Tensor, layers: list[int], idx: list[int] | None = None) -> dict[int, torch.Tensor]:
    result = {}
    for layer in layers:
        acts = cache_activations[layer].float()
        result[layer] = acts[idx] if idx is not None else acts
    return result


def main() -> None:
    with open(THRESHOLDS_PATH) as fh:
        thresholds = json.load(fh)
    threshold = thresholds["thresholds"]["sae_feature"]
    layers = thresholds["sae_feature_layers"]
    features = load_top_features(k=thresholds["sae_feature_k"], path=RANKING_PATH)
    print(f"Layers: {layers}, k={thresholds['sae_feature_k']}, threshold={threshold}")

    saes = {layer: load_sae(layer) for layer in layers}

    cache = load_cache(MODEL_CACHE_LABEL)
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_val = torch.tensor([s == "val" for s in cache["splits"]], dtype=torch.bool)
    val_idx = is_val.nonzero(as_tuple=True)[0].tolist()
    val_by_layer = activations_by_layer(cache["activations"], layers, val_idx)
    val_scores = score(val_by_layer, saes, features)
    val_labels_bool = labels[is_val]

    pooled_std = torch.cat([val_scores[val_labels_bool], val_scores[~val_labels_bool]]).std().item()
    harmful_margin_mean = ((val_scores[val_labels_bool] - threshold) / pooled_std).mean().item()

    adv_cache = torch.load(RESULTS_DIR / "activations" / f"{MODEL_CACHE_LABEL}_adversarial.pt", map_location="cpu", weights_only=False)
    records = adv_cache["records"]
    pair_idx = [i for i, r in enumerate(records) if r["method"] == "PAIR"]
    pair_by_layer = activations_by_layer(adv_cache["activations"], layers, pair_idx)
    pair_scores = score(pair_by_layer, saes, features)
    pair_margins = (pair_scores - threshold) / pooled_std
    pair_margin_mean = pair_margins.mean().item()
    pair_detected_rate = (pair_scores >= threshold).float().mean().item()

    print(f"\n{'':20} {'harmful margin':>15} {'PAIR margin':>12} {'PAIR/harmful':>13} {'PAIR detect (repro/known)':>28}")
    print(f"{'SAE-feature':<20} {harmful_margin_mean:>15.3f} {pair_margin_mean:>12.3f} "
          f"{pair_margin_mean / harmful_margin_mean:>13.3f} {pair_detected_rate:>13.3f} / {KNOWN_SAE_PAIR_DETECTION_RATE:<13.3f}")
    print(f"{'dense-direction':<20} {DENSE_HARMFUL_MARGIN:>15.3f} {DENSE_PAIR_MARGIN:>12.3f} "
          f"{DENSE_PAIR_MARGIN / DENSE_HARMFUL_MARGIN:>13.3f} {'(known)':>13} {DENSE_PAIR_DETECTION_RATE:<13.3f}")

    print("\n=== Paired test: SAE-feature vs. dense-direction margins on the SAME 21 PAIR prompts ===")
    dense_margins = dense_pair_margins(pair_idx)
    stat, p_value = wilcoxon(pair_margins.tolist(), dense_margins.tolist())
    print(f"Wilcoxon signed-rank (n=21 pairs): statistic={stat:.4f}, p={p_value:.4f}")
    print(f"SAE margin mean={pair_margins.mean().item():.4f}, dense margin mean={dense_margins.mean().item():.4f}")

    out = {
        "model": MODEL_CACHE_LABEL,
        "sae_feature": {
            "layers": layers,
            "threshold": threshold,
            "pooled_std": round(pooled_std, 4),
            "harmful_val_margin_mean": round(harmful_margin_mean, 4),
            "pair_margin_mean": round(pair_margin_mean, 4),
            "pair_margin_as_frac_of_harmful": round(pair_margin_mean / harmful_margin_mean, 4),
            "pair_detected_rate_reproduced": round(pair_detected_rate, 4),
            "known_pair_detection_rate": KNOWN_SAE_PAIR_DETECTION_RATE,
        },
        "dense_direction_for_comparison": {
            "harmful_val_margin_mean": DENSE_HARMFUL_MARGIN,
            "pair_margin_mean": DENSE_PAIR_MARGIN,
            "pair_margin_as_frac_of_harmful": round(DENSE_PAIR_MARGIN / DENSE_HARMFUL_MARGIN, 4),
            "known_pair_detection_rate": DENSE_PAIR_DETECTION_RATE,
        },
        "paired_wilcoxon_sae_vs_dense_margins": {"statistic": round(float(stat), 4), "p_value": round(float(p_value), 4), "n": 21},
    }
    out_path = RESULTS_DIR / "sae_pair_margin_llama.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
