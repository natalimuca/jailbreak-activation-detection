"""Data-driven follow-up to scripts/wrapper_task_type.py's clean rejection of
the "task type" hypothesis for Llama-3.1-8B-Instruct's core x category
interaction. Rather than hand-picking a third narrative from 2 new outliers
(the same failure mode already rejected once), this samples 48 new real core
requests (never authored, none overlapping the 22 already used) and tests
several OBJECTIVE, pre-committed textual features against each core's
interaction range -- split into Tier A (blind, corrected as a family via a
maxT permutation test + a BH-FDR cross-check) and Tier B (explicitly
non-blind: a task-type replication check, and a feature transparently
motivated by the "Trump"/"fake news article" outliers, reported with a raw
p-value and a stated caveat rather than silently discarded).

Sampling: 12 cores per source (advbench/harmbench/jbb/xstest) -- equal, not
proportional, allocation, since source_dataset is itself a tested predictor.
Excludes near-duplicates of the 22 already-used cores via the same
normalize/content-word-overlap gate src.data.dedup.deduplicate() uses.

Also re-runs balanced_anova()/freedman_lane_interaction_test() on the new
48-core grid directly (core x category, no feature reshaping) as a third
independent replication check of the underlying phenomenon.

Usage: python scripts/wrapper_feature_search.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import false_discovery_control, kruskal, spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import format_prompt, load_model
from src.baselines.keyword_filter import score as keyword_score
from src.data.dedup import _content_word_overlap, _normalize
from src.data.loaders import load_all_labeled_prompts
from src.sae.feature_probe import feature_value
from src.sae.registry import SAE_PROVIDERS
from scripts.wrapper_swap_variance import CORE_REQUESTS, MODELS, TOP_FEATURE
from scripts.wrapper_swap_replication import CATEGORY_KEYS, WRAPPER_PHRASINGS, balanced_anova, freedman_lane_interaction_test
from scripts.wrapper_interaction_cells import cell_interactions
from scripts.wrapper_task_type import ALL_CORES as TASK_TYPE_CORES

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_PATH = RESULTS_DIR / "wrapper_feature_search.json"
CHECKPOINT_PATH = RESULTS_DIR / "_wrapper_feature_search_checkpoint.json"

RNG_SEED = 0
N_PERMUTATIONS = 20_000
N_PER_SOURCE = 12
SOURCES = ["advbench", "harmbench", "jbb", "xstest"]
EXCLUDED_CORES = list(CORE_REQUESTS) + list(TASK_TYPE_CORES)


def is_near_duplicate(candidate: str, existing: list[str]) -> bool:
    norm_c = _normalize(candidate)
    for text in existing:
        norm_e = _normalize(text)
        if norm_c == norm_e:
            return True
        import difflib
        char_ratio = difflib.SequenceMatcher(None, norm_c, norm_e).ratio()
        if char_ratio >= 0.9 and _content_word_overlap(norm_c, norm_e) >= 0.5:
            return True
    return False


def sample_new_cores(seed: int = RNG_SEED) -> list[dict]:
    records = load_all_labeled_prompts(seed=seed)
    harmful_by_source = {s: [] for s in SOURCES}
    for r in records:
        if r["label"] == "harmful" and r["source"] in harmful_by_source:
            harmful_by_source[r["source"]].append(r["text"])

    kept: list[dict] = []
    kept_texts: list[str] = []
    for source in SOURCES:
        count = 0
        for text in harmful_by_source[source]:
            if count >= N_PER_SOURCE:
                break
            if is_near_duplicate(text, EXCLUDED_CORES) or is_near_duplicate(text, kept_texts):
                continue
            kept.append({"text": text, "source": source})
            kept_texts.append(text)
            count += 1
        assert count == N_PER_SOURCE, f"only found {count}/{N_PER_SOURCE} usable cores for {source}"

    assert len(kept) == len(SOURCES) * N_PER_SOURCE
    for r in kept:
        assert not is_near_duplicate(r["text"], EXCLUDED_CORES)
    return kept


_VERB_ARTIFACT = re.compile(r"^(Write|Draft|Compose|Generate)\b")
_VERB_PROCEDURAL = re.compile(r"^(Explain|Provide|Describe|Outline)\b")


def starting_verb_category(text: str) -> str:
    if _VERB_ARTIFACT.match(text):
        return "direct_artifact"
    if _VERB_PROCEDURAL.match(text):
        return "procedural"
    return "other"


def proper_noun_count(text: str) -> int:
    words = text.split()
    return sum(1 for i, w in enumerate(words) if i > 0 and w[:1].isupper() and w.isalpha())


def compute_tier_a_features(text: str) -> dict:
    words = text.split()
    letters_only = re.sub(r"[^A-Za-z]", "", text)
    return {
        "word_count": len(words),
        "avg_word_length": len(letters_only) / len(words) if words else 0.0,
        "keyword_filter_score": keyword_score(text),
    }


def compute_tier_b_features(text: str) -> dict:
    return {
        "starting_verb_category": starting_verb_category(text),
        "proper_noun_count": proper_noun_count(text),
    }


def measure_grid(model, layer_idx: int, sae, feature_idx: int, cores: list[str]) -> np.ndarray:
    grid = np.zeros((len(cores), len(CATEGORY_KEYS), 3))
    for i, core in enumerate(cores):
        for j, category in enumerate(CATEGORY_KEYS):
            for k, template in enumerate(WRAPPER_PHRASINGS[category]):
                prompt = template.format(request=core)
                templated = format_prompt(model, prompt)
                input_ids = model.tokenizer(templated, return_tensors="pt")["input_ids"]
                grid[i, j, k] = feature_value(model, layer_idx, sae, feature_idx, input_ids)[0].item()
    return grid


def maxT_family_test(outcome: np.ndarray, feature_matrix: dict[str, np.ndarray], n_perm: int, rng: np.random.Generator) -> dict:
    names = list(feature_matrix.keys())

    def raw_stats(y: np.ndarray) -> dict:
        out = {}
        for name in names:
            x = feature_matrix[name]
            if x.dtype.kind in "US":
                groups = [y[x == level] for level in np.unique(x)]
                stat = kruskal(*groups).statistic if all(len(g) > 1 for g in groups) else 0.0
            else:
                stat = abs(spearmanr(x, y).statistic)
            out[name] = stat
        return out

    observed = raw_stats(outcome)
    null_stats = {name: np.zeros(n_perm) for name in names}
    for p in range(n_perm):
        y_perm = rng.permutation(outcome)
        s = raw_stats(y_perm)
        for name in names:
            null_stats[name][p] = s[name]

    standardized_observed = {}
    standardized_null = {}
    for name in names:
        mu, sd = null_stats[name].mean(), null_stats[name].std(ddof=1)
        sd = sd if sd > 0 else 1e-12
        standardized_observed[name] = (observed[name] - mu) / sd
        standardized_null[name] = (null_stats[name] - mu) / sd

    family_max_null = np.max(np.stack([standardized_null[name] for name in names]), axis=0)

    adjusted_p = {}
    raw_p = {}
    for name in names:
        adjusted_p[name] = float((np.sum(family_max_null >= standardized_observed[name] - 1e-9) + 1) / (n_perm + 1))
        raw_p[name] = float((np.sum(null_stats[name] >= observed[name] - 1e-9) + 1) / (n_perm + 1))

    return {
        "observed_stat": {k: float(v) for k, v in observed.items()},
        "raw_p": raw_p,
        "maxT_adjusted_p": adjusted_p,
    }


def verify_maxT_on_synthetic() -> None:
    rng = np.random.default_rng(123)
    n = 48
    y_null = rng.normal(size=n)
    features_null = {f"feat{i}": rng.normal(size=n) for i in range(4)}
    result_null = maxT_family_test(y_null, features_null, 2000, rng)
    assert all(p > 0.05 for p in result_null["maxT_adjusted_p"].values()), \
        f"null case gave a spuriously significant feature: {result_null['maxT_adjusted_p']}"

    y_planted = rng.normal(size=n)
    features_planted = {f"feat{i}": rng.normal(size=n) for i in range(3)}
    features_planted["planted"] = y_planted * 2 + rng.normal(scale=0.3, size=n)
    result_planted = maxT_family_test(y_planted, features_planted, 2000, rng)
    assert result_planted["maxT_adjusted_p"]["planted"] < 0.05, \
        f"planted case failed to detect the real correlation: {result_planted['maxT_adjusted_p']}"
    for name in ["feat0", "feat1", "feat2"]:
        assert result_planted["maxT_adjusted_p"][name] > 0.05, \
            f"planted case gave a false positive on an unrelated feature {name}"

    print("maxT_family_test verified: null case clean, planted case detected with no false positives.")


def main() -> None:
    verify_maxT_on_synthetic()

    cores = sample_new_cores()
    print(f"Sampled {len(cores)} new cores, {N_PER_SOURCE} per source, zero overlap with the 22 excluded.")

    tier_a = {c["text"]: compute_tier_a_features(c["text"]) for c in cores}
    tier_b = {c["text"]: compute_tier_b_features(c["text"]) for c in cores}

    results = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as fh:
            results = json.load(fh)

    rng = np.random.default_rng(RNG_SEED)
    core_texts = [c["text"] for c in cores]

    for cache_label, hf_model_name in MODELS.items():
        if cache_label in results:
            print(f"\n=== {cache_label}: already complete in checkpoint, skipping ===")
            continue
        layer_idx, feature_idx = TOP_FEATURE[cache_label]
        print(f"\n=== {cache_label} (layer {layer_idx}, feature {feature_idx}) ===")
        model = load_model(hf_model_name, load_in_4bit=True)
        load_sae_fn = SAE_PROVIDERS[hf_model_name][0]
        sae = load_sae_fn(layer_idx)

        grid = measure_grid(model, layer_idx, sae, feature_idx, core_texts)
        interactions = cell_interactions(grid)
        interaction_range = interactions.max(axis=1) - interactions.min(axis=1)

        replication_stats = balanced_anova(grid)
        replication_stats["p_interaction_permutation"] = freedman_lane_interaction_test(grid, N_PERMUTATIONS, rng)

        feature_matrix_a = {
            "word_count": np.array([tier_a[t]["word_count"] for t in core_texts], dtype=float),
            "avg_word_length": np.array([tier_a[t]["avg_word_length"] for t in core_texts], dtype=float),
            "keyword_filter_score": np.array([tier_a[t]["keyword_filter_score"] for t in core_texts], dtype=float),
            "source_dataset": np.array([c["source"] for c in cores]),
        }
        tier_a_result = maxT_family_test(interaction_range, feature_matrix_a, N_PERMUTATIONS, rng)
        bh_adjusted = false_discovery_control(list(tier_a_result["raw_p"].values()))
        tier_a_result["bh_fdr_adjusted_p"] = dict(zip(tier_a_result["raw_p"].keys(), bh_adjusted.tolist()))

        feature_matrix_b = {
            "starting_verb_category": np.array([tier_b[t]["starting_verb_category"] for t in core_texts]),
            "proper_noun_count": np.array([tier_b[t]["proper_noun_count"] for t in core_texts], dtype=float),
        }
        tier_b_raw = maxT_family_test(interaction_range, feature_matrix_b, N_PERMUTATIONS, rng)

        results[cache_label] = {
            "layer": layer_idx,
            "feature": feature_idx,
            "cores": cores,
            "tier_a_features": tier_a,
            "tier_b_features": tier_b,
            "grid": grid.tolist(),
            "replication_check_stats": replication_stats,
            "interaction_range": interaction_range.tolist(),
            "tier_a_result": tier_a_result,
            "tier_b_non_blind": {"raw_p": tier_b_raw["raw_p"], "observed_stat": tier_b_raw["observed_stat"]},
        }
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

        print(f"replication check: eta_sq_interaction={replication_stats['eta_sq_interaction']:.3f}, "
              f"F-test p={replication_stats['p_interaction_Ftest']:.4g}, "
              f"permutation p={replication_stats['p_interaction_permutation']:.4g}")
        print("Tier A (blind, corrected):")
        for name in feature_matrix_a:
            print(f"  {name}: raw_p={tier_a_result['raw_p'][name]:.4g}, "
                  f"maxT_p={tier_a_result['maxT_adjusted_p'][name]:.4g}, "
                  f"BH_p={tier_a_result['bh_fdr_adjusted_p'][name]:.4g}")
        print("Tier B (non-blind, raw only):")
        for name in feature_matrix_b:
            print(f"  {name}: raw_p={tier_b_raw['raw_p'][name]:.4g}")

        del model
        torch.cuda.empty_cache()

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
