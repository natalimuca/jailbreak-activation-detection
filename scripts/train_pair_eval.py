"""Adds statistical power to the non-linear combiner's inconclusive PAIR
result (reports/DECISIONS.md, "Closing the pre-registration: the non-linear
combiner clears both gates, result genuinely inconclusive") using the
supplementary TRAIN-goal PAIR set (scripts/build_train_pair_set.py) --
NOT a new experiment. The pipeline and VAL-derived threshold are refit
exactly as already pre-registered (same deterministic code path,
random_state=0, same VAL data) -- nothing here is re-tuned or re-derived
based on this new data. The official TEST-based n=21 PAIR metric is
untouched; this is a separate, clearly-labeled supplementary check reported
alongside it.

Usage: python scripts/train_pair_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.detectors.sae_feature_detector import calibrate, feature_matrix, load_top_features, score
from src.eval.detector_metrics import classify, mcnemar_exact, max_accuracy_threshold
from src.sae.registry import SAE_PROVIDERS
from scripts.nonlinear_combiner_eval import MODELS, activations_by_layer, build_pipeline

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Known TEST-based PAIR rates, for the consistency check (reports/RESULTS.md,
# "A third fix attempt" and the SAE-feature detector's cross-model section).
KNOWN_TEST_PAIR_RATE = {
    "Qwen3-8B": {"vanilla": 0.524},
    "Llama-3.1-8B-Instruct": {"vanilla": 0.810},
}


def main() -> None:
    out: dict = {}

    for cache_label, hf_model_name in MODELS.items():
        print(f"\n=== {cache_label} ===")
        features = load_top_features(k=15, path=RESULTS_DIR / f"sae_causal_ranking_{cache_label}.json")
        layers = sorted({layer for layer, _ in features})
        load_sae_fn = SAE_PROVIDERS[hf_model_name][0]
        saes = {layer: load_sae_fn(layer) for layer in layers}

        cache = load_cache(hf_model_name)
        labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
        is_val = torch.tensor([s == "val" for s in cache["splits"]], dtype=torch.bool)
        val_idx = is_val.nonzero(as_tuple=True)[0].tolist()
        val_by_layer = activations_by_layer(cache["activations"], layers, val_idx)
        val_labels = labels[is_val].tolist()

        # Reproduce the already pre-registered fit exactly -- same code,
        # same data, same random_state=0 -- not a new fit.
        X_val = feature_matrix(val_by_layer, saes, features).numpy()
        pipeline = build_pipeline()
        pipeline.fit(X_val, val_labels)
        val_scores_nl = pipeline.predict_proba(X_val)[:, 1].tolist()
        threshold_nl = max_accuracy_threshold(val_scores_nl, val_labels)
        threshold_vanilla = calibrate(val_by_layer, val_labels, saes, features)

        train_pair_cache = torch.load(RESULTS_DIR / "activations" / f"{cache_label}_train_pair.pt", map_location="cpu", weights_only=False)
        n_prompts = train_pair_cache["activations"].shape[1]
        tp_by_layer = activations_by_layer(train_pair_cache["activations"], layers, list(range(n_prompts)))

        scores_vanilla = score(tp_by_layer, saes, features).tolist()
        X_tp = feature_matrix(tp_by_layer, saes, features).numpy()
        scores_nl = pipeline.predict_proba(X_tp)[:, 1].tolist()

        preds_vanilla = classify(scores_vanilla, threshold_vanilla)
        preds_nl = classify(scores_nl, threshold_nl)
        rate_vanilla = sum(preds_vanilla) / len(preds_vanilla)
        rate_nl = sum(preds_nl) / len(preds_nl)

        known_vanilla = KNOWN_TEST_PAIR_RATE[cache_label]["vanilla"]
        print(f"  n={n_prompts} TRAIN-goal PAIR prompts (distinct goals: {len(set(r['goal'] for r in train_pair_cache['records']))})")
        print(f"  Consistency check -- vanilla PAIR rate: {rate_vanilla:.1%} on this set vs. {known_vanilla:.1%} known (TEST-based, n=21)")
        print(f"  non-linear PAIR rate on this set: {rate_nl:.1%}")

        comparison = mcnemar_exact(preds_nl, preds_vanilla)
        print(f"  non-linear vs vanilla, McNemar (n={n_prompts}): {comparison}")

        out[cache_label] = {
            "n_prompts": n_prompts,
            "distinct_goals": len(set(r["goal"] for r in train_pair_cache["records"])),
            "consistency_check": {"vanilla_rate_here": rate_vanilla, "vanilla_rate_known_test": known_vanilla},
            "vanilla_pair_rate": rate_vanilla,
            "nonlinear_pair_rate": rate_nl,
            "nonlinear_vs_vanilla_mcnemar": comparison,
        }

        del saes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "train_pair_eval.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
