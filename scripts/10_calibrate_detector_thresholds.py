"""Calibrates decision thresholds for all four Phase 4 detectors (keyword
filter, perplexity filter, dense-direction projection, SAE-feature score) on
the VAL split -- never touched by direction estimation, SAE causal ranking,
or the final TEST-split reporting (see DECISIONS.md's Phase 4 split-
discipline entry). Reuses the layer already selected in
`scripts/06_dense_direction_ablation_qwen3.py` and the top-15 SAE features
already ranked in Phase 3, rather than re-deriving either.

Usage: python scripts/10_calibrate_detector_thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.baselines.keyword_filter import score as keyword_score
from src.baselines.perplexity_filter import compute_perplexity, load_perplexity_model
from src.detectors.dense_direction_detector import calibrate as calibrate_dense
from src.detectors.sae_feature_detector import calibrate as calibrate_sae, load_top_features
from src.direction.compute import compute_directions
from src.eval.detector_metrics import youden_threshold
from src.sae.qwen_scope import load_sae

MODEL_LABEL = "Qwen3-8B"
DENSE_ABLATION_RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "dense_direction_ablation_Qwen3-8B.json"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SAE_K = 15


def main() -> None:
    cache = load_cache(MODEL_LABEL)
    acts = cache["activations"]
    labels_bool = [l == "harmful" for l in cache["labels"]]
    splits = cache["splits"]
    texts = cache["texts"]

    is_train = torch.tensor([s == "train" for s in splits], dtype=torch.bool)
    is_val = torch.tensor([s == "val" for s in splits], dtype=torch.bool)
    labels_t = torch.tensor(labels_bool, dtype=torch.bool)

    val_mask = is_val.tolist()
    val_texts = [t for t, m in zip(texts, val_mask) if m]
    val_labels = [l for l, m in zip(labels_bool, val_mask) if m]
    print(f"VAL split: {len(val_texts)} prompts ({sum(val_labels)} harmful, {len(val_labels) - sum(val_labels)} harmless)")

    thresholds = {}

    print("\n--- keyword filter ---")
    kw_scores = [keyword_score(t) for t in val_texts]
    thresholds["keyword"] = youden_threshold(kw_scores, val_labels)
    print(f"threshold: {thresholds['keyword']}")

    print("\n--- perplexity filter ---")
    print("Loading Olmo-3-1025-7B (4-bit)")
    ppl_model, ppl_tok = load_perplexity_model()
    ppl_scores = [compute_perplexity(t, ppl_model, ppl_tok) for t in val_texts]
    thresholds["perplexity"] = youden_threshold(ppl_scores, val_labels)
    print(f"threshold: {thresholds['perplexity']:.2f}")

    print("\n--- dense-direction detector ---")
    with open(DENSE_ABLATION_RESULTS_PATH) as fh:
        dense_layer = json.load(fh)["ablation_layer"]
    direction = compute_directions(
        acts[dense_layer : dense_layer + 1, is_train & labels_t, :],
        acts[dense_layer : dense_layer + 1, is_train & ~labels_t, :],
    )[0]
    val_acts_dense = acts[dense_layer, is_val, :]
    thresholds["dense_direction"] = calibrate_dense(val_acts_dense, val_labels, direction)
    print(f"layer {dense_layer} (reused from scripts/06), threshold: {thresholds['dense_direction']:.3f}")

    print("\n--- SAE-feature detector ---")
    features = load_top_features(k=SAE_K)
    unique_layers = sorted({layer for layer, _ in features})
    print(f"top-{SAE_K} features span layers {unique_layers}, loading SAEs")
    saes = {layer: load_sae(layer) for layer in unique_layers}
    val_acts_by_layer = {layer: acts[layer, is_val, :] for layer in unique_layers}
    thresholds["sae_feature"] = calibrate_sae(val_acts_by_layer, val_labels, saes, features)
    print(f"threshold: {thresholds['sae_feature']:.3f}")

    payload = {
        "model": MODEL_LABEL,
        "n_val": len(val_texts),
        "dense_direction_layer": dense_layer,
        "sae_feature_k": SAE_K,
        "sae_feature_layers": unique_layers,
        "thresholds": thresholds,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "detector_thresholds_Qwen3-8B.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved calibrated thresholds to {out_path}")


if __name__ == "__main__":
    main()
