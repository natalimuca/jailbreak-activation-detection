"""Extends the dense-direction detector to Llama-3.1-8B-Instruct and
Gemma-2-9B-it (Phase 6) -- the two additional model families named in the
original project roadmap, chosen specifically because pretrained SAE
suites exist for both (LlamaScope, GemmaScope), unlike Qwen2.5/SmolLM2.
This script only extends the dense-direction detector (mirrors
`scripts/12_extend_dense_direction_qwen25_smollm2.py`'s pattern exactly);
extending the SAE-feature detector is a separate, substantially larger
follow-up (effectively repeating Phase 3's methodology per model) and is
not done here.

Both models are ~8-9B params, so both are loaded 4-bit (matching Qwen3-8B's
approach) -- unlike Qwen2.5-1.5B/SmolLM2-1.7B, which fit unquantized.

Merges into the existing `results/dense_direction_cross_model.json` file
(created by scripts/12) rather than overwriting it, so all four
cross-model results (Qwen2.5, SmolLM2, Llama-3.1, Gemma-2) live in one
place.

Usage: python scripts/14_extend_dense_direction_llama_gemma.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.activations.extract import get_last_token_resid_acts, load_model
from src.detectors.dense_direction_detector import project, select_layer_and_calibrate
from src.eval.detector_metrics import detector_stats

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ADVERSARIAL_MANIFEST_PATH = RESULTS_DIR / "adversarial_paraphrase_manifest.json"
CROSS_MODEL_RESULTS_PATH = RESULTS_DIR / "dense_direction_cross_model.json"

MODELS = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}


def evaluate_model(cache_label: str, hf_model_name: str, adversarial_records: list[dict]) -> dict:
    cache = load_cache(cache_label)
    acts = cache["activations"]
    labels_bool = [l == "harmful" for l in cache["labels"]]
    sources = cache["sources"]
    splits = cache["splits"]
    labels_t = torch.tensor(labels_bool, dtype=torch.bool)

    is_train = torch.tensor([s == "train" for s in splits], dtype=torch.bool)
    is_val = torch.tensor([s == "val" for s in splits], dtype=torch.bool)
    is_test = torch.tensor([s == "test" for s in splits], dtype=torch.bool)

    layer, direction, threshold = select_layer_and_calibrate(acts, labels_t, is_train, is_val)
    print(f"  layer {layer} (selected on VAL), threshold {threshold:.3f}")

    test_labels = labels_t[is_test].tolist()
    test_scores = project(acts[layer, is_test, :], direction).tolist()
    test_stats = detector_stats(test_scores, test_labels, threshold)

    is_xstest_safe = torch.tensor(
        [s == "test" and src == "xstest" and not lab for s, src, lab in zip(splits, sources, labels_bool)],
        dtype=torch.bool,
    )
    n_xstest_safe = int(is_xstest_safe.sum().item())
    xstest_stats = None
    if n_xstest_safe > 0:
        xstest_scores = project(acts[layer, is_xstest_safe, :], direction).tolist()
        xstest_stats = detector_stats(xstest_scores, [False] * n_xstest_safe, threshold)

    print(f"  loading model (4-bit) for adversarial-set activation extraction: {hf_model_name}")
    model = load_model(hf_model_name, load_in_4bit=True)
    adv_texts = [r["text"] for r in adversarial_records]
    adv_acts = get_last_token_resid_acts(model, adv_texts)
    adv_scores = project(adv_acts[layer], direction).tolist()
    adv_stats_pooled = detector_stats(adv_scores, [True] * len(adv_texts), threshold)

    adv_stats_by_method = {}
    for method in sorted({r["method"] for r in adversarial_records}):
        idx = [i for i, r in enumerate(adversarial_records) if r["method"] == method]
        adv_stats_by_method[method] = detector_stats([adv_scores[i] for i in idx], [True] * len(idx), threshold)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "layer": layer,
        "threshold": threshold,
        "test_overall": test_stats,
        "xstest_safe_false_positive_check": xstest_stats,
        "adversarial_paraphrase_pooled": adv_stats_pooled,
        "adversarial_paraphrase_by_method": adv_stats_by_method,
    }


def main() -> None:
    with open(ADVERSARIAL_MANIFEST_PATH) as fh:
        adversarial_records = json.load(fh)
    print(
        f"Reusing {len(adversarial_records)} adversarial prompts from {ADVERSARIAL_MANIFEST_PATH.name} "
        f"(same real JailbreakBench artifacts as Qwen3-8B, not refetched)"
    )

    results = {}
    if CROSS_MODEL_RESULTS_PATH.exists():
        with open(CROSS_MODEL_RESULTS_PATH) as fh:
            results = json.load(fh)
        print(f"Loaded existing cross-model results for: {list(results.keys())}")

    for cache_label, hf_model_name in MODELS.items():
        print(f"\n=== {cache_label} ===")
        results[cache_label] = evaluate_model(cache_label, hf_model_name, adversarial_records)

    print("\n=== Dense-direction detector: cross-model comparison (all models) ===")
    for cache_label, r in results.items():
        acc = r["test_overall"]["accuracy"]
        auc = r["test_overall"]["auroc"]
        print(
            f"  {cache_label:24s} layer={r['layer']:2d}  TEST accuracy={acc['rate']:.3f} "
            f"[{acc['ci_low']:.3f},{acc['ci_high']:.3f}]  auroc={auc}"
        )
        if r["xstest_safe_false_positive_check"]:
            x = r["xstest_safe_false_positive_check"]["accuracy"]
            print(f"    XSTest-safe correctly-not-flagged: {x['rate']:.3f} [{x['ci_low']:.3f},{x['ci_high']:.3f}]")
        pooled = r["adversarial_paraphrase_pooled"]["accuracy"]
        print(f"    adversarial (pooled) detection rate: {pooled['rate']:.3f} [{pooled['ci_low']:.3f},{pooled['ci_high']:.3f}]")
        for method, stats in r["adversarial_paraphrase_by_method"].items():
            m = stats["accuracy"]
            print(f"      {method}: {m['rate']:.3f} [{m['ci_low']:.3f},{m['ci_high']:.3f}] (n={m['n']})")

    with open(CROSS_MODEL_RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved merged results to {CROSS_MODEL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
