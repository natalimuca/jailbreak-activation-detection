"""Tests the salience hypothesis from LITERATURE.md's literature-context note
(2026-07-29): does a core request's real-world discourse salience predict how
strongly it interacts with wrapper framing in Llama-3.1-8B-Instruct's dominant
causal SAE feature? Operationalized as raw perplexity under this project's own
existing GCG-detection reference LM (src/baselines/perplexity_filter.py,
Olmo-3-1025-7B, built for an unrelated purpose -- catching gibberish suffixes
-- so reusing it here is not circular). Lower perplexity approximates higher
salience/commonness in typical text.

No new SAE-feature measurement needed: the 48 core requests and their already-
computed interaction_range (per model) come straight from
results/wrapper_feature_search.json. This only adds one new feature (cheap,
CPU/one-model-load) and correlates it against data already collected.

Honesty note, stated here and in the write-up: this feature was proposed AFTER
seeing that word_count/avg_word_length/keyword_filter_score/source_dataset all
failed (results/wrapper_feature_search.json) and after a literature search
motivated by explaining that same failure -- so it is NOT a blind,
pre-registered-before-any-result test the way Tier A was. It is also not
reverse-engineered from the specific outlier words ("Trump", "fake news
article") the way Tier B's proper_noun_count was -- it comes from an
independent literature finding (SAE/feature-frequency work) about training-data
salience shaping representation geometry generally. Reported as its own single,
honestly-labeled post-hoc test with its own uncorrected permutation p-value,
not folded into the original Tier-A family (whose correction was computed
before this feature existed) and not treated as equivalent to a blind result.

Usage: python scripts/wrapper_salience_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.perplexity_filter import compute_perplexity, load_perplexity_model

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
IN_PATH = RESULTS_DIR / "wrapper_feature_search.json"
OUT_PATH = RESULTS_DIR / "wrapper_salience_test.json"

RNG_SEED = 0
N_PERMUTATIONS = 20_000


def permutation_test_spearman(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    observed = spearmanr(x, y).statistic
    count_ge = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        stat = abs(spearmanr(x, y_perm).statistic)
        if stat >= abs(observed) - 1e-12:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)
    return {"spearman_rho": float(observed), "p_value": float(p_value)}


def main() -> None:
    with open(IN_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    print("Loading perplexity reference model (Olmo-3-1025-7B, 4-bit)...")
    model, tokenizer = load_perplexity_model()

    core_texts = [c["text"] for c in data["Llama-3.1-8B-Instruct"]["cores"]]
    perplexities = {}
    for text in core_texts:
        perplexities[text] = compute_perplexity(text, model, tokenizer)

    rng = np.random.default_rng(RNG_SEED)
    results = {"perplexities": perplexities}

    for model_label in data:
        cores = data[model_label]["cores"]
        interaction_range = np.array(data[model_label]["interaction_range"])
        perplexity_arr = np.array([perplexities[c["text"]] for c in cores])

        test_result = permutation_test_spearman(perplexity_arr, interaction_range, N_PERMUTATIONS, rng)
        results[model_label] = test_result
        print(f"\n=== {model_label} ===")
        print(f"perplexity vs. interaction_range: rho={test_result['spearman_rho']:.3f}, "
              f"p={test_result['p_value']:.4g}")

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
