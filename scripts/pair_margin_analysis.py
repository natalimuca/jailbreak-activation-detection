"""Investigates the real, formally-significant PAIR-paraphrase robustness spread
across all 5 models (Cochran's Q = 19.52, df=4, p=0.0006 -- SmolLM2 90.5% >
Llama-3.1-8B 66.7% > gemma-2-9b-it 47.6% > Qwen3-8B 42.9% > Qwen2.5-1.5B 38.1%,
see RESULTS.md's cross-model dense-direction section). The one candidate
hypothesis floated there (SmolLM2's weaker/less-linear baseline refusal
correlating with its own robustness) was explicitly flagged as not explaining
Llama's position (a strong, "linear"-looking refusal model that's also
comparatively robust) -- this script looks for a different, continuous signal
instead of the binary detected/not-detected rate already known.

Rather than the binary "flagged or not" rate (already published), this measures
each model's dense-direction projection *margin* on the 21 real PAIR prompts,
normalized into pooled-std units (matching src.direction.compute.separation_score's
own normalization) so it's comparable across models with very different raw
activation scales -- then expresses that margin as a fraction of the model's own
*genuine harmful-prompt* margin, giving a "how much of the harmful signal
survives paraphrasing, relative to this model's own typical harmful signal"
number per model.

Needs no new GPU generation -- reuses each model's already-persisted unit
direction (results/dense_directions.pt) and already-calibrated threshold
(results/dense_direction_cross_model.json for 4 models, plus
results/detector_thresholds_Qwen3-8B.json for the one model recorded
separately). Two of the five models (Qwen2.5-1.5B, SmolLM2) needed a one-time
adversarial-activation extraction first (scripts/extend_adversarial_small.py,
forward-pass only, ~10s each) since scripts/extend_qwen_smollm.py never cached
them; the other three already had `results/activations/{model}_adversarial.pt`.

Usage: python scripts/pair_margin_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DENSE_DIRECTIONS_PATH = RESULTS_DIR / "dense_directions.pt"
CROSS_MODEL_PATH = RESULTS_DIR / "dense_direction_cross_model.json"
QWEN3_THRESHOLDS_PATH = RESULTS_DIR / "detector_thresholds_Qwen3-8B.json"

# Known PAIR detection rates (RESULTS.md's cross-model dense-direction section),
# repeated here only for side-by-side comparison against the new margin numbers,
# not recomputed.
KNOWN_PAIR_DETECTION_RATE = {
    "SmolLM2-1.7B-Instruct": 0.905,
    "Llama-3.1-8B-Instruct": 0.667,
    "gemma-2-9b-it": 0.476,
    "Qwen3-8B": 0.429,
    "Qwen2.5-1.5B-Instruct": 0.381,
}

MODELS = [
    "Qwen2.5-1.5B-Instruct",
    "SmolLM2-1.7B-Instruct",
    "Qwen3-8B",
    "Llama-3.1-8B-Instruct",
    "gemma-2-9b-it",
]


def layer_and_threshold(cache_label: str) -> tuple[int, float]:
    if cache_label == "Qwen3-8B":
        with open(QWEN3_THRESHOLDS_PATH) as fh:
            d = json.load(fh)
        return d["dense_direction_layer"], d["thresholds"]["dense_direction"]
    with open(CROSS_MODEL_PATH) as fh:
        d = json.load(fh)
    return d[cache_label]["layer"], d[cache_label]["threshold"]


def analyze_model(cache_label: str, direction: torch.Tensor, layer: int, threshold: float) -> dict:
    cache = load_cache(cache_label)
    acts = cache["activations"][layer].float()
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_val = torch.tensor([s == "val" for s in cache["splits"]], dtype=torch.bool)

    val_acts = acts[is_val]
    val_labels = labels[is_val]
    val_proj = val_acts @ direction
    pooled_std = torch.cat([val_proj[val_labels], val_proj[~val_labels]]).std().item()
    harmful_val_proj = val_proj[val_labels]
    harmful_margin_mean = ((harmful_val_proj - threshold) / pooled_std).mean().item()

    adv_cache = torch.load(RESULTS_DIR / "activations" / f"{cache_label}_adversarial.pt", map_location="cpu", weights_only=False)
    adv_acts = adv_cache["activations"][layer].float()
    records = adv_cache["records"]
    pair_idx = [i for i, r in enumerate(records) if r["method"] == "PAIR"]
    pair_proj = adv_acts[pair_idx] @ direction
    pair_margins = (pair_proj - threshold) / pooled_std
    pair_margin_mean = pair_margins.mean().item()
    pair_detected_rate = (pair_proj > threshold).float().mean().item()

    return {
        "layer": layer,
        "threshold": threshold,
        "pooled_std": round(pooled_std, 4),
        "harmful_val_margin_mean": round(harmful_margin_mean, 4),
        "pair_margin_mean": round(pair_margin_mean, 4),
        "pair_margin_as_frac_of_harmful": round(pair_margin_mean / harmful_margin_mean, 4),
        "pair_detected_rate_reproduced": round(pair_detected_rate, 4),
        "known_pair_detection_rate": KNOWN_PAIR_DETECTION_RATE[cache_label],
    }


def main() -> None:
    directions = torch.load(DENSE_DIRECTIONS_PATH, map_location="cpu", weights_only=False)
    results = {}
    print(f"{'model':<24} {'layer':>5} {'harmful margin':>15} {'PAIR margin':>12} {'PAIR/harmful':>13} {'PAIR detect (repro/known)':>28}")
    for cache_label in MODELS:
        layer, threshold = layer_and_threshold(cache_label)
        direction = directions[cache_label]["direction"].float()
        r = analyze_model(cache_label, direction, layer, threshold)
        results[cache_label] = r
        print(f"{cache_label:<24} {r['layer']:>5} {r['harmful_val_margin_mean']:>15.3f} "
              f"{r['pair_margin_mean']:>12.3f} {r['pair_margin_as_frac_of_harmful']:>13.3f} "
              f"{r['pair_detected_rate_reproduced']:>13.3f} / {r['known_pair_detection_rate']:<13.3f}")

    margins = [results[m]["pair_margin_mean"] for m in MODELS]
    known_rates = [results[m]["known_pair_detection_rate"] for m in MODELS]
    rho, p_value = spearmanr(margins, known_rates)
    print(f"\nSpearman rank correlation (PAIR margin vs. known detection rate, n={len(MODELS)}): "
          f"rho={rho:.4f}, p={p_value:.4f}")

    out = {"per_model": results, "spearman": {"rho": round(float(rho), 4), "p_value": round(float(p_value), 4), "n": len(MODELS)}}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pair_margin_analysis.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
