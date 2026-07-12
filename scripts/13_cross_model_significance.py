"""Cochran's Q test for the 5-model dense-direction PAIR-paraphrase
comparison (scripts/12 and scripts/14's cross-model extensions) -- the same
repeated-measures structure as `mcnemar_exact`, generalized from 2 to *k*
classifiers scored on the SAME 21 PAIR prompts. Replaces the informal
"non-overlapping Wilson CIs" argument for cross-model PAIR-robustness
differences with a real, formal significance test -- the same discipline
already applied once this session (McNemar's for dense-direction vs
SAE-feature) to a claim that had only been argued from eyeballing
confidence intervals.

Reuses cached activations where possible: Qwen3-8B's PAIR prompts already
have activations cached (`scripts/09`); every other model needs a fresh
(cheap, forward-pass-only) extraction since scripts/12/14 didn't persist
per-prompt scores, only aggregate stats. Llama-3.1-8B-Instruct and
Gemma-2-9B-it need `load_in_4bit=True` (8-9B params on a 6GB GPU);
Qwen2.5-1.5B/SmolLM2-1.7B don't.

Usage: python scripts/13_cross_model_significance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.activations.extract import get_last_token_resid_acts, load_model
from src.detectors.dense_direction_detector import project
from src.direction.compute import compute_directions
from src.eval.detector_metrics import classify, cochrans_q

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ADVERSARIAL_MANIFEST_PATH = RESULTS_DIR / "adversarial_paraphrase_manifest.json"
CROSS_MODEL_RESULTS_PATH = RESULTS_DIR / "dense_direction_cross_model.json"
QWEN3_THRESHOLDS_PATH = RESULTS_DIR / "detector_thresholds_Qwen3-8B.json"
QWEN3_ADVERSARIAL_CACHE_PATH = RESULTS_DIR / "activations" / "Qwen3-8B_adversarial.pt"

SMALL_MODELS = {
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

LARGE_MODELS = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}


def _train_direction(cache_label: str, layer: int) -> torch.Tensor:
    cache = load_cache(cache_label)
    acts = cache["activations"]
    labels_t = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_train = torch.tensor([s == "train" for s in cache["splits"]], dtype=torch.bool)
    return compute_directions(
        acts[layer : layer + 1, is_train & labels_t, :],
        acts[layer : layer + 1, is_train & ~labels_t, :],
    )[0]


def qwen3_pair_predictions(pair_indices: list[int]) -> list[bool]:
    with open(QWEN3_THRESHOLDS_PATH) as fh:
        calib = json.load(fh)
    layer = calib["dense_direction_layer"]
    threshold = calib["thresholds"]["dense_direction"]
    direction = _train_direction("Qwen3-8B", layer)

    adv_cache = torch.load(QWEN3_ADVERSARIAL_CACHE_PATH, weights_only=False)
    pair_acts = adv_cache["activations"][layer][pair_indices]  # (n_pair, d_model)
    scores = project(pair_acts, direction).tolist()
    return classify(scores, threshold)


def model_pair_predictions(
    cache_label: str, hf_model_name: str, pair_texts: list[str], load_in_4bit: bool = False
) -> list[bool]:
    with open(CROSS_MODEL_RESULTS_PATH) as fh:
        cross_model = json.load(fh)
    layer = cross_model[cache_label]["layer"]
    threshold = cross_model[cache_label]["threshold"]
    direction = _train_direction(cache_label, layer)

    model = load_model(hf_model_name, load_in_4bit=load_in_4bit)
    pair_acts = get_last_token_resid_acts(model, pair_texts)[layer]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores = project(pair_acts, direction).tolist()
    return classify(scores, threshold)


def main() -> None:
    with open(ADVERSARIAL_MANIFEST_PATH) as fh:
        adv_records = json.load(fh)
    pair_indices = [i for i, r in enumerate(adv_records) if r["method"] == "PAIR"]
    pair_texts = [adv_records[i]["text"] for i in pair_indices]
    print(f"{len(pair_texts)} PAIR prompts")

    model_names = ["Qwen3-8B"]
    preds_matrix = [qwen3_pair_predictions(pair_indices)]

    for cache_label, hf_model_name in SMALL_MODELS.items():
        print(f"Scoring {cache_label}")
        preds_matrix.append(model_pair_predictions(cache_label, hf_model_name, pair_texts))
        model_names.append(cache_label)

    for cache_label, hf_model_name in LARGE_MODELS.items():
        print(f"Scoring {cache_label} (4-bit)")
        preds_matrix.append(model_pair_predictions(cache_label, hf_model_name, pair_texts, load_in_4bit=True))
        model_names.append(cache_label)

    result = cochrans_q(preds_matrix)

    print("\n=== Cochran's Q: 5-model dense-direction PAIR comparison ===")
    for name, preds in zip(model_names, preds_matrix):
        rate = sum(preds) / len(preds)
        print(f"  {name:24s} detection rate: {rate:.3f} ({sum(preds)}/{len(preds)})")
    print(f"  Q={result['q']}, df={result['df']}, p={result['p_value']}")

    out_path = RESULTS_DIR / "cross_model_pair_cochrans_q.json"
    with open(out_path, "w") as fh:
        json.dump(
            {"model_names": model_names, "predictions": preds_matrix, "cochrans_q": result},
            fh,
            indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
