"""Phase 4 head-to-head: all four detectors (keyword filter, perplexity
filter, dense-direction, SAE-feature), evaluated on:

  (a) TEST split overall (harmful vs harmless) -- the core accuracy/F1/AUROC
      comparison, on a split none of the four detectors were calibrated on.
  (b) TEST's XSTest-safe subset -- false-positive / over-refusal check (this
      project's "safety tax" angle): these are harmless-but-scary-looking
      prompts, so a good detector should NOT flag them.
  (c) the adversarial paraphrase set (scripts/09) -- real published jailbreak
      prompts, all harmful by construction, so the relevant number is
      detection/flag rate (recall), not full precision/AUROC (no negatives
      in this set).

Thresholds are loaded from scripts/10's VAL-only calibration, never re-tuned
here.

Usage: python scripts/11_head_to_head_detectors.py
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
from src.detectors.dense_direction_detector import project as dense_project
from src.detectors.sae_feature_detector import load_top_features
from src.detectors.sae_feature_detector import score as sae_score
from src.direction.compute import compute_directions
from src.eval.detector_metrics import classify, delong_auc_test, detector_stats, mcnemar_exact
from src.sae.qwen_scope import load_sae

MODEL_LABEL = "Qwen3-8B"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
THRESHOLDS_PATH = RESULTS_DIR / "detector_thresholds_Qwen3-8B.json"
ADVERSARIAL_CACHE_PATH = RESULTS_DIR / "activations" / "Qwen3-8B_adversarial.pt"


def main() -> None:
    with open(THRESHOLDS_PATH) as fh:
        calib = json.load(fh)
    thresholds = calib["thresholds"]
    dense_layer = calib["dense_direction_layer"]
    sae_layers = calib["sae_feature_layers"]
    sae_k = calib["sae_feature_k"]

    cache = load_cache(MODEL_LABEL)
    acts = cache["activations"]
    labels_bool = [l == "harmful" for l in cache["labels"]]
    sources = cache["sources"]
    splits = cache["splits"]
    texts = cache["texts"]
    labels_t = torch.tensor(labels_bool, dtype=torch.bool)

    is_train = torch.tensor([s == "train" for s in splits], dtype=torch.bool)
    is_test = torch.tensor([s == "test" for s in splits], dtype=torch.bool)

    # Direction re-derived from TRAIN at the already-selected layer -- same
    # deterministic recomputation as scripts/10, not re-tuned here.
    direction = compute_directions(
        acts[dense_layer : dense_layer + 1, is_train & labels_t, :],
        acts[dense_layer : dense_layer + 1, is_train & ~labels_t, :],
    )[0]

    features = load_top_features(k=sae_k)
    saes = {layer: load_sae(layer) for layer in sae_layers}

    print("Loading Olmo-3-1025-7B (4-bit) for perplexity scoring")
    ppl_model, ppl_tok = load_perplexity_model()

    def score_all(idx_mask: torch.Tensor) -> dict[str, list[float]]:
        idx = idx_mask.nonzero(as_tuple=True)[0].tolist()
        subset_texts = [texts[i] for i in idx]
        return {
            "keyword": [float(keyword_score(t)) for t in subset_texts],
            "perplexity": [compute_perplexity(t, ppl_model, ppl_tok) for t in subset_texts],
            "dense_direction": dense_project(acts[dense_layer, idx_mask, :], direction).tolist(),
            "sae_feature": sae_score(
                {layer: acts[layer, idx_mask, :] for layer in sae_layers}, saes, features
            ).tolist(),
        }

    results: dict = {}

    print("Scoring TEST split overall")
    test_labels = [l for l, m in zip(labels_bool, is_test.tolist()) if m]
    test_scores = score_all(is_test)
    results["test_overall"] = {
        name: detector_stats(scores, test_labels, thresholds[name]) for name, scores in test_scores.items()
    }

    # dense-direction (AUROC 0.983) vs SAE-feature (0.975) on TEST-overall is
    # close enough to be worth a formal paired test rather than eyeballing
    # the gap -- DeLong's test is the right tool for two AUROCs measured on
    # the same items (see DECISIONS.md's significance-testing entry).
    results["test_overall_delong_dense_vs_sae"] = delong_auc_test(
        test_scores["dense_direction"], test_scores["sae_feature"], test_labels
    )

    print("Scoring TEST's XSTest-safe subset (false-positive check)")
    is_xstest_safe = torch.tensor(
        [s == "test" and src == "xstest" and not lab for s, src, lab in zip(splits, sources, labels_bool)],
        dtype=torch.bool,
    )
    n_xstest_safe = int(is_xstest_safe.sum().item())
    if n_xstest_safe > 0:
        xstest_labels = [False] * n_xstest_safe
        xstest_scores = score_all(is_xstest_safe)
        results["xstest_safe_false_positive_check"] = {
            name: detector_stats(scores, xstest_labels, thresholds[name]) for name, scores in xstest_scores.items()
        }
    else:
        print("  WARNING: no XSTest-safe prompts found in TEST -- skipping this condition")

    print("Scoring adversarial paraphrase set")
    adv_cache = torch.load(ADVERSARIAL_CACHE_PATH, weights_only=False)
    adv_acts = adv_cache["activations"]
    adv_records = adv_cache["records"]
    adv_texts = [r["text"] for r in adv_records]
    adv_labels = [True] * len(adv_texts)
    adv_scores = {
        "keyword": [float(keyword_score(t)) for t in adv_texts],
        "perplexity": [compute_perplexity(t, ppl_model, ppl_tok) for t in adv_texts],
        "dense_direction": dense_project(adv_acts[dense_layer], direction).tolist(),
        "sae_feature": sae_score({layer: adv_acts[layer] for layer in sae_layers}, saes, features).tolist(),
    }
    results["adversarial_paraphrase"] = {
        name: detector_stats(scores, adv_labels, thresholds[name]) for name, scores in adv_scores.items()
    }

    # PAIR (fluent paraphrase) and GCG (gibberish suffix) are different
    # failure modes for a prompt classifier -- pooling them into one number
    # hides which method is actually driving each detector's flag rate (e.g.
    # a perplexity filter catching GCG suffixes says nothing about whether
    # it can catch fluent paraphrase, the case this phase is named for).
    # Same "don't trust the aggregate, break it down" discipline as the
    # Phase 3 head-to-head's moralize-vs-comply finding (see RESULTS.md).
    results["adversarial_paraphrase_by_method"] = {}
    for method in sorted({r["method"] for r in adv_records}):
        method_idx = [i for i, r in enumerate(adv_records) if r["method"] == method]
        method_labels = [True] * len(method_idx)
        results["adversarial_paraphrase_by_method"][method] = {
            name: detector_stats([scores[i] for i in method_idx], method_labels, thresholds[name])
            for name, scores in adv_scores.items()
        }
    results["adversarial_paraphrase_n_by_method"] = {
        method: sum(1 for r in adv_records if r["method"] == method)
        for method in sorted({r["method"] for r in adv_records})
    }

    # Dense-direction vs SAE-feature is the specific comparison this
    # project's write-up makes a "no significant difference" claim about on
    # PAIR -- worth a real paired test (McNemar's exact, on the SAME
    # prompts scored by both detectors), not just eyeballing whether two
    # independent Wilson CIs overlap (see DECISIONS.md).
    dense_preds_pooled = classify(adv_scores["dense_direction"], thresholds["dense_direction"])
    sae_preds_pooled = classify(adv_scores["sae_feature"], thresholds["sae_feature"])
    results["dense_vs_sae_mcnemar"] = {"pooled": mcnemar_exact(dense_preds_pooled, sae_preds_pooled)}
    for method in sorted({r["method"] for r in adv_records}):
        method_idx = [i for i, r in enumerate(adv_records) if r["method"] == method]
        results["dense_vs_sae_mcnemar"][method] = mcnemar_exact(
            [dense_preds_pooled[i] for i in method_idx], [sae_preds_pooled[i] for i in method_idx]
        )

    def print_condition(label: str, detectors: dict) -> None:
        print(f"\n-- {label} --")
        for name, stats in detectors.items():
            acc = stats["accuracy"]
            auc = stats["auroc"]
            auc_str = f"{auc:.3f}" if auc is not None else "n/a"
            print(
                f"  {name:16s} n={stats['n']:4d}  rate={acc['rate']:.3f} "
                f"[{acc['ci_low']:.3f},{acc['ci_high']:.3f}]  f1={stats['f1']:.3f}  auroc={auc_str}"
            )

    print("\n=== Head-to-head detector comparison ===")
    print_condition("test_overall", results["test_overall"])
    dl = results["test_overall_delong_dense_vs_sae"]
    print(
        f"\n  DeLong test (dense-direction vs SAE-feature AUROC, TEST-overall): "
        f"auc_a={dl['auc_a']} auc_b={dl['auc_b']} diff={dl['diff']} p={dl['p_value']}"
    )
    if "xstest_safe_false_positive_check" in results:
        print_condition("xstest_safe_false_positive_check", results["xstest_safe_false_positive_check"])
    print_condition("adversarial_paraphrase (pooled)", results["adversarial_paraphrase"])
    for method, detectors in results["adversarial_paraphrase_by_method"].items():
        n = results["adversarial_paraphrase_n_by_method"][method]
        print_condition(f"adversarial_paraphrase / {method} (n={n})", detectors)

    print("\n-- dense-direction vs SAE-feature: paired McNemar's exact test --")
    for condition, mc in results["dense_vs_sae_mcnemar"].items():
        print(
            f"  {condition:8s} discordant={mc['n_discordant']:3d} "
            f"(dense-only={mc['only_a']}, sae-only={mc['only_b']})  p={mc['p_value']}"
        )

    out_path = RESULTS_DIR / "detector_head_to_head_Qwen3-8B.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
