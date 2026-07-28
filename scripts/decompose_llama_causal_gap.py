"""Goes one level deeper than scripts/analyze_llama_causal_gap.py's cosine-alignment
check. That script found Llama-3.1-8B's dense direction and its own top causal SAE
feature (layer 27/13363) have cosine similarity ~0.201 -- essentially identical to
Qwen3-8B's own dense-direction-vs-top-feature cosine (~0.196) -- so raw alignment
doesn't differentiate why Llama's dense direction is a weak causal lever (ablation:
92%->88%, not significant even at n=75) while Qwen3-8B's is a strong one (84%->8%).

This script decomposes each model's own unit-normalized dense direction into two
orthogonal pieces relative to its own top causal SAE feature's decoder direction --
the component parallel to the feature's own axis, and everything orthogonal to it --
then causally ablates each piece SEPARATELY (same generate_with_ablation() intervention
used everywhere else in this project) to see whether the small feature-aligned sliver
or the large orthogonal remainder is doing (or not doing) the causal work.

**Framing, stated explicitly rather than left implicit**: both components are
renormalized to unit length before ablation (generate_with_ablation's projection-removal
math requires a unit-norm input and does no internal renormalization). This means the
question being asked is "is this axis alone -- independent of how little of the
original vector's mass sits on it -- a sufficient/necessary causal lever," not "how
much does this component contribute at its true small weight." The parallel and
orthogonal ablation results below are NOT directly commensurate with the already-
published 92%->88% (Llama) / 84%->8% (Qwen3-8B) full-direction numbers -- they test
axis identity, not a magnitude-faithful split of the real vector's effect.

**A real data bug found and fixed while building this**: results/cross_model_direction_
transfer.json stores STALE pre-bugfix refusal stats for every Llama condition (baseline
stored as 0.80, own_ablation as 0.86) -- the curly-apostrophe is_refusal fix (see
DECISIONS.md, 2026-07-23) was applied to RESULTS.md's published numbers but never
written back into this JSON file. Qwen3-8B's conditions in the same file are unaffected
(ASCII apostrophes throughout). This script reloads the file's raw completions and
rescores them fresh via is_refusal()/refusal_stats() rather than trusting the file's
stored refusal_stats field, for both models, before using either as a baseline for a
new McNemar/Cochran's Q comparison -- confirmed the fresh rescore reproduces RESULTS.md's
published 92%/88% (Llama) and 84%/8% (Qwen3-8B) exactly before trusting the new
conditions' comparisons against them.

**Significance testing**: an omnibus Cochran's Q across all 4 conditions (baseline,
full-direction, parallel-component, orthogonal-component) per model first, then the
6 pairwise McNemar tests reported explicitly as post-hoc -- running 6 uncorrected
pairwise tests on the same paired 50-item sample without an omnibus test first would
risk the same false-positive inflation this project's own cochrans_q was built to
address (previously only ever applied across k different models on the same items,
not k conditions within one model -- a new but direct use of the same function here).

Reuses (does not reimplement) dense_direction_at_layer/decoder_vector/feature_vector
from scripts/analyze_llama_causal_gap.py, and the identical 50-prompt held-out VAL set
already saved in results/cross_model_direction_transfer.json (seed=2, sampled once from
Qwen3-8B's cache, reused verbatim for both models).

Usage: python scripts/decompose_llama_causal_gap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import load_model
from src.direction.interventions import generate_with_ablation
from src.direction.refusal_classifier import is_degenerate, is_refusal, refusal_stats
from src.eval.detector_metrics import cochrans_q, mcnemar_exact
from scripts.analyze_llama_causal_gap import dense_direction_at_layer, decoder_vector

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TRANSFER_JSON = RESULTS_DIR / "cross_model_direction_transfer.json"
OUT_PATH = RESULTS_DIR / "llama_causal_gap_decomposition.json"

MAX_NEW_TOKENS = 40

# (cache_label, hf_model_name, layer, feature) -- own top causally-ranked feature per model.
MODELS = [
    ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", 27, 13363),
    ("Qwen3-8B", "Qwen/Qwen3-8B", 25, 65291),
]
# maps cache_label -> the causal_ablation.json key holding that model's own baseline/own_ablation
TRANSFER_KEY = {
    "Llama-3.1-8B-Instruct": "qwen_direction_on_llama",
    "Qwen3-8B": "llama_direction_on_qwen",
}


def decompose_direction(cache_label: str, hf_model_name: str, layer: int, feature: int) -> dict:
    """Returns dense_dir (unit), d_parallel_unit, d_orthogonal_unit, plus the sanity-check
    numbers (cosine, norms, reconstruction error) -- computed once on CPU from already-
    cached activations and already-downloaded SAE weights, no GPU/model load needed."""
    dense_dir = dense_direction_at_layer(cache_label, layer)  # already unit-normalized
    d_model = dense_dir.shape[0]
    feat_vec = decoder_vector(hf_model_name, layer, feature, d_model)
    feat_unit = feat_vec / feat_vec.norm()

    cos = torch.dot(dense_dir, feat_unit).item()
    d_parallel = cos * feat_unit
    d_orthogonal = dense_dir - d_parallel

    parallel_norm = d_parallel.norm().item()
    orthogonal_norm = d_orthogonal.norm().item()
    reconstruction_error = (d_parallel + d_orthogonal - dense_dir).norm().item()
    pythagorean_sum = parallel_norm ** 2 + orthogonal_norm ** 2

    print(f"  {cache_label}: cos={cos:.4f} ||parallel||={parallel_norm:.4f} "
          f"||orthogonal||={orthogonal_norm:.4f} pythagorean_sum={pythagorean_sum:.6f} "
          f"reconstruction_error={reconstruction_error:.2e}")
    assert abs(pythagorean_sum - 1.0) < 1e-4, "decomposition failed Pythagorean sanity check"
    assert reconstruction_error < 1e-4, "decomposition does not reconstruct the dense direction"

    return {
        "dense_dir": dense_dir,
        "d_parallel_unit": d_parallel / parallel_norm,
        "d_orthogonal_unit": d_orthogonal / orthogonal_norm,
        "cosine": round(cos, 4),
        "parallel_norm": round(parallel_norm, 4),
        "orthogonal_norm": round(orthogonal_norm, 4),
        "reconstruction_error": reconstruction_error,
    }


def rescore_stale_condition(transfer_data: dict, cache_label: str, condition: str) -> dict:
    """Reloads a condition's raw completions from cross_model_direction_transfer.json and
    rescores fresh via is_refusal()/refusal_stats() -- do NOT trust the file's stored
    refusal_stats field, which is stale (pre-apostrophe-bugfix) for every Llama condition."""
    key = TRANSFER_KEY[cache_label]
    completions = transfer_data["causal_ablation"][key]["conditions"][condition]["completions"]
    stats = refusal_stats(completions)
    degenerate_count = sum(is_degenerate(c) for c in completions)
    return {"refusal_stats": stats, "degenerate_count": degenerate_count, "completions": completions}


CHECKPOINT_PATH = RESULTS_DIR / "_decompose_llama_causal_gap_checkpoint.json"


def run_new_conditions(hf_model_name: str, cache_label: str, prompts: list[str], components: dict) -> dict:
    """Checkpoints per-prompt to CHECKPOINT_PATH and resumes any partially-completed
    condition on rerun -- a background run of this exact script was killed mid-generation
    by an unrelated environment disruption with zero progress saved the first time this was
    attempted, losing a full model load. Any long GPU script here should checkpoint
    incrementally, not just at the very end (established lesson, see DECISIONS.md)."""
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as fh:
            checkpoint = json.load(fh)

    condition_specs = [
        ("parallel_ablation", components["d_parallel_unit"]),
        ("orthogonal_ablation", components["d_orthogonal_unit"]),
    ]
    if all(len(checkpoint.get(cache_label, {}).get(name, [])) >= len(prompts) for name, _ in condition_specs):
        print(f"  {cache_label}: both new conditions already complete in checkpoint, skipping model load")
        model = None
    else:
        print(f"Loading model: {hf_model_name} (4-bit)")
        model = load_model(hf_model_name, load_in_4bit=True)

    conditions = {}
    for name, direction in condition_specs:
        saved = checkpoint.get(cache_label, {}).get(name, [])
        completions = list(saved)
        if len(completions) < len(prompts):
            print(f"  condition: {name} (resuming from {len(completions)}/{len(prompts)})")
            for prompt in prompts[len(completions):]:
                completions.append(generate_with_ablation(model, prompt, direction, max_new_tokens=MAX_NEW_TOKENS))
                checkpoint.setdefault(cache_label, {})[name] = completions
                with open(CHECKPOINT_PATH, "w") as fh:
                    json.dump(checkpoint, fh)
        else:
            print(f"  condition: {name} (already complete in checkpoint)")
        stats = refusal_stats(completions)
        degenerate_count = sum(is_degenerate(c) for c in completions)
        print(f"    refusal_rate={stats['rate']} [{stats['ci_low']}, {stats['ci_high']}], "
              f"degenerate={degenerate_count}/{len(completions)}")
        conditions[name] = {"refusal_stats": stats, "degenerate_count": degenerate_count, "completions": completions}

    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return conditions


def significance(all_conditions: dict) -> dict:
    """Omnibus Cochran's Q across all 4 conditions first, then the 6 pairwise McNemar
    tests reported explicitly as post-hoc (not independent pre-registered comparisons)."""
    names = ["baseline", "own_ablation", "parallel_ablation", "orthogonal_ablation"]
    bools = {n: [is_refusal(c) for c in all_conditions[n]["completions"]] for n in names}

    omnibus = cochrans_q([bools[n] for n in names])

    pairwise = {}
    pairs = [
        ("baseline", "parallel_ablation"), ("baseline", "orthogonal_ablation"),
        ("own_ablation", "parallel_ablation"), ("own_ablation", "orthogonal_ablation"),
        ("parallel_ablation", "orthogonal_ablation"), ("baseline", "own_ablation"),
    ]
    for a, b in pairs:
        pairwise[f"{a}_vs_{b}"] = mcnemar_exact(bools[a], bools[b])

    return {"omnibus_cochrans_q": omnibus, "posthoc_pairwise_mcnemar": pairwise}


def main() -> None:
    with open(TRANSFER_JSON) as fh:
        transfer_data = json.load(fh)
    val_prompts = transfer_data["val_prompts"]
    print(f"Reusing {len(val_prompts)} held-out VAL prompts from {TRANSFER_JSON.name} (seed={transfer_data['seed']})")

    print("\n=== Decomposing each model's own dense direction vs. its own top causal SAE feature ===")
    results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as fh:
            results = json.load(fh)

    for cache_label, hf_model_name, layer, feature in MODELS:
        if cache_label in results:
            print(f"\n{cache_label}: already complete in {OUT_PATH.name}, skipping")
            continue
        components = decompose_direction(cache_label, hf_model_name, layer, feature)

        print(f"  Rescoring baseline/own_ablation from {TRANSFER_JSON.name} fresh (not trusting stored stats)")
        baseline = rescore_stale_condition(transfer_data, cache_label, "baseline")
        own_ablation = rescore_stale_condition(transfer_data, cache_label, "own_ablation")
        print(f"    baseline refusal={baseline['refusal_stats']['rate']} "
              f"own_ablation refusal={own_ablation['refusal_stats']['rate']}")

        print(f"\n  Running 2 new conditions on {cache_label}...")
        new_conditions = run_new_conditions(hf_model_name, cache_label, val_prompts, components)

        all_conditions = {"baseline": baseline, "own_ablation": own_ablation, **new_conditions}
        sig = significance(all_conditions)
        print(f"  omnibus Cochran's Q: {sig['omnibus_cochrans_q']}")
        for pair, r in sig["posthoc_pairwise_mcnemar"].items():
            print(f"    {pair}: p={r['p_value']}")

        results[cache_label] = {
            "layer": layer, "feature": feature,
            "decomposition": {k: v for k, v in components.items() if k not in ("dense_dir", "d_parallel_unit", "d_orthogonal_unit")},
            "conditions": all_conditions,
            "significance": sig,
        }
        with open(OUT_PATH, "w") as fh:
            json.dump(results, fh, indent=2)

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
