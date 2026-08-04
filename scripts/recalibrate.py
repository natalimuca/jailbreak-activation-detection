"""Usage: python scripts/recalibrate.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.baselines.llm_judge import DEFAULT_MODEL as JUDGE_MODEL
from src.baselines.llm_judge import _cache_key
from src.detectors.dense_direction_detector import project as dense_project
from src.direction.compute import compute_directions
from src.eval.detector_metrics import classify, detector_stats, mcnemar_exact, youden_threshold

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}


def _candidate_thresholds(scores: list[float]) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=float))
    if unique.size < 2:
        return unique
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.concatenate([[unique[0] - 1e-9], midpoints, [unique[-1] + 1e-9]])


def _best_threshold(scores: list[float], labels: list[bool], metric: str) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    best_t, best_v = float(s.min()), -1.0
    for t in _candidate_thresholds(scores):
        pred = s >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        tn = int((~pred & ~y).sum())
        if metric == "accuracy":
            v = (tp + tn) / len(y)
        elif metric == "f1":
            v = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
        else:
            raise ValueError(metric)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


def main() -> None:
    judge_cache = json.loads((RESULTS_DIR / "llm_judge_cache.json").read_text())
    out: dict = {}

    for label, hf_name in MODELS.items():
        payload = json.loads((RESULTS_DIR / f"detector_thresholds_{label}.json").read_text())
        layer = payload["dense_direction_layer"]
        judge_threshold = payload["thresholds"]["llm_judge"]

        cache = load_cache(hf_name)
        acts = cache["activations"]
        labels = [l == "harmful" for l in cache["labels"]]
        splits = cache["splits"]
        texts = cache["texts"]

        is_train = torch.tensor([s == "train" for s in splits], dtype=torch.bool)
        is_val = torch.tensor([s == "val" for s in splits], dtype=torch.bool)
        is_test = torch.tensor([s == "test" for s in splits], dtype=torch.bool)
        lab_t = torch.tensor(labels, dtype=torch.bool)

        direction = compute_directions(
            acts[layer : layer + 1, is_train & lab_t, :],
            acts[layer : layer + 1, is_train & ~lab_t, :],
        )[0]

        val_scores = dense_project(acts[layer, is_val, :], direction).tolist()
        val_labels = [l for l, m in zip(labels, is_val.tolist()) if m]
        test_scores = dense_project(acts[layer, is_test, :], direction).tolist()
        test_labels = [l for l, m in zip(labels, is_test.tolist()) if m]
        test_texts = [t for t, m in zip(texts, is_test.tolist()) if m]
        judge_test = [judge_cache[_cache_key(t, JUDGE_MODEL)] for t in test_texts]

        rules = {
            "youden_j": youden_threshold(val_scores, val_labels),
            "max_accuracy": _best_threshold(val_scores, val_labels, "accuracy"),
            "max_f1": _best_threshold(val_scores, val_labels, "f1"),
        }

        model_out = {"layer": layer, "rules": {}}
        baseline_preds = classify(test_scores, rules["youden_j"])
        for rule, threshold in rules.items():
            stats_val = detector_stats(val_scores, val_labels, threshold)
            stats_test = detector_stats(test_scores, test_labels, threshold)
            preds = classify(test_scores, threshold)
            model_out["rules"][rule] = {
                "threshold": threshold,
                "val_accuracy": stats_val["accuracy"]["rate"],
                "test_accuracy": stats_test["accuracy"]["rate"],
                "test_f1": stats_test["f1"],
                "test_auroc": stats_test["auroc"],
                "vs_youden_mcnemar": mcnemar_exact(preds, baseline_preds),
                "vs_judge_mcnemar": mcnemar_exact(preds, classify(judge_test, judge_threshold)),
            }
        judge_stats = detector_stats(judge_test, test_labels, judge_threshold)
        model_out["judge_test_accuracy"] = judge_stats["accuracy"]["rate"]
        out[label] = model_out

        print(f"\n=== {label} (layer {layer}) ===")
        print(f"  judge TEST accuracy: {judge_stats['accuracy']['rate'] * 100:.1f}%")
        for rule, r in model_out["rules"].items():
            mc_y = r["vs_youden_mcnemar"]["p_value"]
            mc_j = r["vs_judge_mcnemar"]["p_value"]
            print(
                f"  {rule:13s} thr={r['threshold']:9.3f}  VAL acc={r['val_accuracy'] * 100:5.1f}%  "
                f"TEST acc={r['test_accuracy'] * 100:5.1f}%  f1={r['test_f1']:.3f}  "
                f"vs-youden p={mc_y:.4f}  vs-judge p={mc_j:.4f}"
            )

    out_path = RESULTS_DIR / "threshold_recalibration.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
