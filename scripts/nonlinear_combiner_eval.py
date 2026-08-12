"""Evaluates a non-linear (pairwise-interaction) combiner over Qwen3-8B's
top-15 SAE features -- the third diagnosis-to-intervention attempt this
session, after two linear fixes (downstream reweighting, upstream direction
ablation) both failed for separately-verified reasons (see
reports/DECISIONS.md's two prior "Pre-registration" entries and their
closing "Closing the pre-registration" entries). Design fixed in advance:
see reports/DECISIONS.md's "Pre-registration: a non-linear SAE-feature
combiner" -- the model class, hyperparameters, and overfitting-gate
threshold below are copied verbatim from that entry, not tuned here.

Needs no new GPU generation: rescores already-cached VAL/TEST/adversarial
activations (`results/activations/<model>*.pt`) via
`src.detectors.sae_feature_detector.feature_matrix` (the same per-feature
SAE-encode loop `score()` itself uses) and a scikit-learn pipeline fit on
VAL only.

Usage: python scripts/nonlinear_combiner_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.detectors.sae_feature_detector import calibrate, feature_matrix, load_top_features, score
from src.eval.detector_metrics import classify, delong_auc_test, detector_stats, mcnemar_accuracy, mcnemar_exact, max_accuracy_threshold
from src.sae.registry import SAE_PROVIDERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

# Fixed pre-registration values (reports/DECISIONS.md, "Pre-registration:
# a non-linear SAE-feature combiner") -- not tuned here.
LOGREG_C = 0.1
CV_FOLDS = 5
CV_SEED = 0
OVERFIT_GAP_THRESHOLD = 0.05


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scale1", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("scale2", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=LOGREG_C, solver="lbfgs", max_iter=2000, random_state=0)),
    ])


def activations_by_layer(cache_activations: torch.Tensor, layers: list[int], idx: list[int] | None = None) -> dict[int, torch.Tensor]:
    result = {}
    for layer in layers:
        acts = cache_activations[layer].float()
        result[layer] = acts[idx] if idx is not None else acts
    return result


def overfit_gate(X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Stratified 5-fold CV accuracy vs. in-sample accuracy on VAL. Pass
    requires in_sample - mean_cv <= OVERFIT_GAP_THRESHOLD."""
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    cv_scores = cross_val_score(pipeline, X_val, y_val, cv=cv, scoring="accuracy")
    mean_cv = float(cv_scores.mean())

    pipeline.fit(X_val, y_val)
    in_sample = float(pipeline.score(X_val, y_val))

    gap = in_sample - mean_cv
    return {
        "in_sample_accuracy": in_sample, "mean_cv_accuracy": mean_cv,
        "cv_fold_accuracies": cv_scores.tolist(), "gap": gap,
        "passed": gap <= OVERFIT_GAP_THRESHOLD,
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
        is_test = torch.tensor([s == "test" for s in cache["splits"]], dtype=torch.bool)
        val_idx = is_val.nonzero(as_tuple=True)[0].tolist()
        test_idx = is_test.nonzero(as_tuple=True)[0].tolist()
        val_by_layer = activations_by_layer(cache["activations"], layers, val_idx)
        test_by_layer = activations_by_layer(cache["activations"], layers, test_idx)
        val_labels_bool = labels[is_val]
        val_labels = val_labels_bool.tolist()
        test_labels = labels[is_test].tolist()

        adv_cache = torch.load(RESULTS_DIR / "activations" / f"{cache_label}_adversarial.pt", map_location="cpu", weights_only=False)
        pair_idx = [i for i, r in enumerate(adv_cache["records"]) if r["method"] == "PAIR"]
        pair_by_layer = activations_by_layer(adv_cache["activations"], layers, pair_idx)

        X_val = feature_matrix(val_by_layer, saes, features).numpy()
        y_val = np.array(val_labels, dtype=bool)

        print("Running overfitting gate (5-fold CV vs. in-sample VAL accuracy)...")
        gate = overfit_gate(X_val, y_val)
        print(
            f"  in-sample={gate['in_sample_accuracy']:.4f}, mean-CV={gate['mean_cv_accuracy']:.4f}, "
            f"gap={gate['gap']:.4f} -> {'PASSED' if gate['passed'] else 'FAILED'}"
        )

        model_out = {"features": features, "overfit_gate": gate}

        if not gate["passed"]:
            print("  Overfit gate failed -- per pre-registration, TEST/PAIR are not evaluated for this model.")
            out[cache_label] = model_out
            del saes
            continue

        pipeline = build_pipeline()
        pipeline.fit(X_val, y_val)

        X_test = feature_matrix(test_by_layer, saes, features).numpy()
        X_pair = feature_matrix(pair_by_layer, saes, features).numpy()
        val_scores_nl = pipeline.predict_proba(X_val)[:, 1].tolist()
        test_scores_nl = pipeline.predict_proba(X_test)[:, 1].tolist()
        pair_scores_nl = pipeline.predict_proba(X_pair)[:, 1].tolist()
        threshold_nl = max_accuracy_threshold(val_scores_nl, val_labels)

        vanilla_threshold = calibrate(val_by_layer, val_labels, saes, features)
        test_scores_vanilla = score(test_by_layer, saes, features).tolist()
        pair_scores_vanilla = score(pair_by_layer, saes, features).tolist()

        test_preds_nl = classify(test_scores_nl, threshold_nl)
        test_preds_vanilla = classify(test_scores_vanilla, vanilla_threshold)
        pair_preds_nl = classify(pair_scores_nl, threshold_nl)
        pair_preds_vanilla = classify(pair_scores_vanilla, vanilla_threshold)

        model_out["vanilla"] = {
            "threshold": vanilla_threshold,
            "test": detector_stats(test_scores_vanilla, test_labels, vanilla_threshold),
            "pair_detection_rate": sum(pair_preds_vanilla) / len(pair_preds_vanilla),
        }
        model_out["nonlinear"] = {
            "threshold": threshold_nl,
            "test": detector_stats(test_scores_nl, test_labels, threshold_nl),
            "pair_detection_rate": sum(pair_preds_nl) / len(pair_preds_nl),
            "vs_vanilla": {
                "test_accuracy_mcnemar": mcnemar_accuracy(test_preds_nl, test_preds_vanilla, test_labels),
                "test_auroc_delong": delong_auc_test(test_scores_nl, test_scores_vanilla, test_labels),
                "pair_mcnemar": mcnemar_exact(pair_preds_nl, pair_preds_vanilla),
            },
        }
        out[cache_label] = model_out

        print(f"  {'variant':10s} {'threshold':>10s} {'TEST acc':>9s} {'TEST AUROC':>11s} {'PAIR rate':>10s}")
        for variant in ["vanilla", "nonlinear"]:
            v = model_out[variant]
            print(
                f"  {variant:10s} {v['threshold']:10.3f} {v['test']['accuracy']['rate'] * 100:8.1f}% "
                f"{v['test']['auroc']:11.4f} {v['pair_detection_rate'] * 100:9.1f}%"
            )
        vs = model_out["nonlinear"]["vs_vanilla"]
        print(
            f"  nonlinear vs vanilla: TEST-acc McNemar p={vs['test_accuracy_mcnemar']['p_value']:.4f}, "
            f"TEST-AUROC DeLong p={vs['test_auroc_delong']['p_value']:.4f}, "
            f"PAIR McNemar p={vs['pair_mcnemar']['p_value']:.4f}"
        )

        del saes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "nonlinear_combiner_eval.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
