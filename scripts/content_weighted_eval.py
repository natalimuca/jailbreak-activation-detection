"""Evaluates whether a content-weighted variant of the top-15 SAE-feature
detector improves PAIR-paraphrase robustness without hurting clean TEST
accuracy -- the diagnosis-to-intervention experiment pre-registered in
reports/DECISIONS.md ("Pre-registration: a content-weighted SAE detector",
2026-08-11). Needs no new GPU generation: rescores already-cached TEST and
adversarial activations (`results/activations/<model>*.pt`) using the
already-established `src.detectors.sae_feature_detector.score`/`calibrate`
(unchanged) plus the two weighting formulas fixed in that pre-registration
entry, computed from `scripts/feature_variance_family.py`'s per-feature
wrapper-swap ANOVA output.

Three detector variants per model, calibrated identically (VAL,
`max_accuracy_threshold`, the project-wide adopted rule) and compared
pairwise against the vanilla unweighted detector:
  - vanilla: the existing top-15 SAE-feature detector, unchanged.
  - primary: `compute_content_weights` (continuous eta_core-ratio weighting).
  - binary: `compute_content_weights_binary` (drop significantly
    framing-dominant features, robustness check).

Qwen3-8B is the primary hypothesis (14/15 of its top features are
framing-leaning, per Stage 1); Llama-3.1-8B-Instruct is the negative control
(11/15 content-leaning) -- if reweighting moves Llama's numbers as much as
Qwen3-8B's, that would undermine the claim that any Qwen3-8B effect is real
and specific rather than an artifact of reweighting sums in general.

Usage: python scripts/content_weighted_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.detectors.sae_feature_detector import (
    calibrate,
    compute_content_weights,
    compute_content_weights_binary,
    score,
)
from src.eval.detector_metrics import classify, delong_auc_test, detector_stats, mcnemar_accuracy, mcnemar_exact
from src.sae.registry import SAE_PROVIDERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
VARIANTS = ["vanilla", "primary", "binary"]


def activations_by_layer(cache_activations: torch.Tensor, layers: list[int], idx: list[int] | None = None) -> dict[int, torch.Tensor]:
    result = {}
    for layer in layers:
        acts = cache_activations[layer].float()
        result[layer] = acts[idx] if idx is not None else acts
    return result


def load_family(cache_label: str) -> tuple[list[tuple[int, int]], list[dict]]:
    with open(RESULTS_DIR / f"feature_variance_{cache_label}.json") as fh:
        data = json.load(fh)
    ordered = sorted(data["features"], key=lambda f: f["rank"])
    features = [(f["layer"], f["feature"]) for f in ordered]
    variance_stats = [f["stats"] for f in ordered]
    return features, variance_stats


def main() -> None:
    out: dict = {}

    for cache_label, hf_model_name in MODELS.items():
        print(f"\n=== {cache_label} ===")
        features, variance_stats = load_family(cache_label)
        layers = sorted({layer for layer, _ in features})
        weights = {
            "vanilla": None,
            "primary": compute_content_weights(variance_stats),
            "binary": compute_content_weights_binary(variance_stats),
        }
        n_dropped_binary = sum(1 for w in weights["binary"] if w == 0.0)
        print(f"  binary variant drops {n_dropped_binary}/{len(features)} features")

        load_sae_fn = SAE_PROVIDERS[hf_model_name][0]
        saes = {layer: load_sae_fn(layer) for layer in layers}

        cache = load_cache(hf_model_name)
        labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
        is_val = torch.tensor([s == "val" for s in cache["splits"]], dtype=torch.bool)
        is_test = torch.tensor([s == "test" for s in cache["splits"]], dtype=torch.bool)
        val_idx = is_val.nonzero(as_tuple=True)[0].tolist()
        test_idx = is_test.nonzero(as_tuple=True)[0].tolist()
        val_by_layer = activations_by_layer(cache["activations"], layers, val_idx)
        test_by_layer = activations_by_layer(cache["activations"], layers, test_idx)
        val_labels = labels[is_val].tolist()
        test_labels = labels[is_test].tolist()

        adv_cache = torch.load(RESULTS_DIR / "activations" / f"{cache_label}_adversarial.pt", map_location="cpu", weights_only=False)
        pair_idx = [i for i, r in enumerate(adv_cache["records"]) if r["method"] == "PAIR"]
        pair_by_layer = activations_by_layer(adv_cache["activations"], layers, pair_idx)

        model_out = {"features": features, "n_dropped_binary": n_dropped_binary, "variants": {}}
        thresholds, test_preds, pair_preds = {}, {}, {}

        for variant in VARIANTS:
            w = weights[variant]
            thresholds[variant] = calibrate(val_by_layer, val_labels, saes, features, w)
            test_scores = score(test_by_layer, saes, features, w).tolist()
            pair_scores = score(pair_by_layer, saes, features, w).tolist()
            test_preds[variant] = classify(test_scores, thresholds[variant])
            pair_preds[variant] = classify(pair_scores, thresholds[variant])
            model_out["variants"][variant] = {
                "threshold": thresholds[variant],
                "test": detector_stats(test_scores, test_labels, thresholds[variant]),
                "pair_detection_rate": sum(pair_preds[variant]) / len(pair_preds[variant]),
                "pair_n": len(pair_preds[variant]),
            }

        for variant in ["primary", "binary"]:
            model_out["variants"][variant]["vs_vanilla"] = {
                "test_accuracy_mcnemar": mcnemar_accuracy(test_preds[variant], test_preds["vanilla"], test_labels),
                "test_auroc_delong": delong_auc_test(
                    score(test_by_layer, saes, features, weights[variant]).tolist(),
                    score(test_by_layer, saes, features, weights["vanilla"]).tolist(),
                    test_labels,
                ),
                "pair_mcnemar": mcnemar_exact(pair_preds[variant], pair_preds["vanilla"]),
            }

        out[cache_label] = model_out

        print(f"  {'variant':10s} {'threshold':>10s} {'TEST acc':>9s} {'TEST AUROC':>11s} {'PAIR rate':>10s}")
        for variant in VARIANTS:
            v = model_out["variants"][variant]
            print(
                f"  {variant:10s} {v['threshold']:10.3f} {v['test']['accuracy']['rate'] * 100:8.1f}% "
                f"{v['test']['auroc']:11.4f} {v['pair_detection_rate'] * 100:9.1f}%"
            )
        for variant in ["primary", "binary"]:
            vs = model_out["variants"][variant]["vs_vanilla"]
            print(
                f"  {variant} vs vanilla: TEST-acc McNemar p={vs['test_accuracy_mcnemar']['p_value']:.4f}, "
                f"TEST-AUROC DeLong p={vs['test_auroc_delong']['p_value']:.4f}, "
                f"PAIR McNemar p={vs['pair_mcnemar']['p_value']:.4f}"
            )

        del saes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "content_weighted_eval.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
