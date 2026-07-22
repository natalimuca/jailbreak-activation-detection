"""Closes this project's biggest remaining gap: does a refusal direction
found on one model do anything applied to a *different* model? Every prior
phase fit and tested a direction only within the model it came from.

Scoped to Qwen3-8B <-> Llama-3.1-8B-Instruct (both d_model=4096, so a raw
direction vector is dimensionally injectable into either model's residual
stream) -- gemma-2-9b-it (d_model=3584) is excluded rather than attempting
a learned cross-dimension mapping, which would confound "does it transfer"
with "is the mapping any good" (see DECISIONS.md).

A raw copied vector is only "the same direction" if the two models'
residual-stream bases happen to be meaningfully aligned -- not guaranteed
a priori for two independently-trained models. That's the hypothesis under
test, not a flaw in the design: a negative result is informative, not a
failed experiment.

Two tests:
  1. Cross-model separation score (cheap, cache-only, no GPU generation):
     does a foreign direction, evaluated at the target's own
     already-selected layer, still separate harmful/harmless activations?
  2. Cross-model causal ablation (real GPU compute, the definitive test):
     does ablating a foreign direction from generation change refusal
     behavior the way ablating the model's own direction does? Three
     paired McNemar tests per model turn this into a clear verdict
     (no/partial/full transfer), not a pile of numbers -- see the
     own-ablation-vs-foreign-ablation comparison below.

Necessity (ablation) only, not sufficiency (addition) -- addition needs a
calibrated alpha for the foreign direction on the target's residual-stream
scale, real additional scope, deferred. SAE-feature transfer is out of
scope too -- an SAE's feature basis is specific to that trained
autoencoder, not a well-posed transfer question the way a single vector is.

Usage: python scripts/17_cross_model_direction_transfer.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import assert_caches_consistent, load_cache
from src.activations.extract import load_model
from src.detectors.dense_direction_detector import resolve_layer_for_model
from src.direction.compute import compute_directions, separation_score
from src.direction.interventions import generate_baseline, generate_with_ablation
from src.direction.refusal_classifier import is_degenerate, is_refusal, refusal_stats
from src.eval.detector_metrics import mcnemar_exact

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
N_VAL_PROMPTS = 50
MAX_PROMPT_CHARS = 150
MAX_NEW_TOKENS = 40
SEED = 2


def own_directions_and_layer(cache_label: str) -> tuple[torch.Tensor, int, dict]:
    """Returns (per-layer directions [n_layers, d_model], own selected
    layer, cache dict) for a model, computed from its own TRAIN split."""
    cache = load_cache(cache_label)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_train = torch.tensor([s == "train" for s in cache["splits"]], dtype=torch.bool)
    directions = compute_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])
    layer = resolve_layer_for_model(cache_label)
    return directions, layer, cache


def cross_model_separation(
    source_direction: torch.Tensor,
    target_cache: dict,
    target_own_directions: torch.Tensor,
    target_own_layer: int,
) -> dict:
    """Evaluates a foreign direction against the target's own VAL
    activations at every layer (broadcast, since separation_score's math
    is linear in the direction's scale -- a single vector repeated across
    layers is a sound reuse, not a hack). Primary number is the score at
    the target's own already-selected layer, NOT whichever layer scores
    highest in the sweep -- picking post-hoc would be uncontrolled layer
    selection, inconsistent with this project's split discipline."""
    labels = torch.tensor([l == "harmful" for l in target_cache["labels"]], dtype=torch.bool)
    is_val = torch.tensor([s == "val" for s in target_cache["splits"]], dtype=torch.bool)
    val_acts = target_cache["activations"][:, is_val, :]
    val_labels = labels[is_val]
    n_layers = val_acts.shape[0]

    own_sweep = separation_score(val_acts, target_own_directions, val_labels)
    foreign_broadcast = source_direction.unsqueeze(0).expand(n_layers, -1)
    foreign_sweep = separation_score(val_acts, foreign_broadcast, val_labels)

    return {
        "own_score_at_own_layer": own_sweep[target_own_layer].item(),
        "foreign_score_at_own_layer": foreign_sweep[target_own_layer].item(),
        "own_sweep_all_layers_diagnostic_only": own_sweep.tolist(),
        "foreign_sweep_all_layers_diagnostic_only": foreign_sweep.tolist(),
    }


def run_generation_conditions(
    hf_model_name: str, prompts: list[str], own_direction: torch.Tensor, foreign_direction: torch.Tensor
) -> dict:
    print(f"Loading model: {hf_model_name} (4-bit)")
    model = load_model(hf_model_name, load_in_4bit=True)

    conditions = {}
    for name, direction in (("baseline", None), ("own_ablation", own_direction), ("foreign_ablation", foreign_direction)):
        print(f"  condition: {name}")
        completions = []
        for i, prompt in enumerate(prompts):
            if direction is None:
                out = generate_baseline(model, prompt, max_new_tokens=MAX_NEW_TOKENS)
            else:
                out = generate_with_ablation(model, prompt, direction, max_new_tokens=MAX_NEW_TOKENS)
            completions.append(out)
        stats = refusal_stats(completions)
        degenerate_count = sum(is_degenerate(c) for c in completions)
        print(f"    refusal_rate={stats['rate']} [{stats['ci_low']}, {stats['ci_high']}], "
              f"degenerate={degenerate_count}/{len(completions)}")
        conditions[name] = {"refusal_stats": stats, "degenerate_count": degenerate_count, "completions": completions}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return conditions


def transfer_verdict(conditions: dict) -> dict:
    """Three paired McNemar tests on the same prompts, matching
    scripts/16's pattern -- turns the raw numbers into a clear verdict
    rather than a pile of p-values."""
    baseline_refusals = [is_refusal(c) for c in conditions["baseline"]["completions"]]
    own_refusals = [is_refusal(c) for c in conditions["own_ablation"]["completions"]]
    foreign_refusals = [is_refusal(c) for c in conditions["foreign_ablation"]["completions"]]

    return {
        "baseline_vs_foreign": mcnemar_exact(baseline_refusals, foreign_refusals),
        "baseline_vs_own": mcnemar_exact(baseline_refusals, own_refusals),
        "own_vs_foreign": mcnemar_exact(own_refusals, foreign_refusals),
    }


def main() -> None:
    assert_caches_consistent(list(MODELS.keys()))

    print("=== Test 1: cross-model separation score (cache-only, no generation) ===")
    qwen_directions, qwen_layer, qwen_cache = own_directions_and_layer("Qwen3-8B")
    llama_directions, llama_layer, llama_cache = own_directions_and_layer("Llama-3.1-8B-Instruct")
    print(f"Qwen3-8B own layer: {qwen_layer}, Llama-3.1-8B own layer: {llama_layer}")

    qwen_own_direction = qwen_directions[qwen_layer]
    llama_own_direction = llama_directions[llama_layer]

    separation_results = {
        "llama_direction_on_qwen": cross_model_separation(
            llama_own_direction, qwen_cache, qwen_directions, qwen_layer
        ),
        "qwen_direction_on_llama": cross_model_separation(
            qwen_own_direction, llama_cache, llama_directions, llama_layer
        ),
    }
    for name, r in separation_results.items():
        print(f"  {name}: own={r['own_score_at_own_layer']:.4f}  foreign={r['foreign_score_at_own_layer']:.4f}")

    print("\n=== Test 2: cross-model causal ablation ===")
    eligible = [
        t for t, lab, sp in zip(qwen_cache["texts"], qwen_cache["labels"], qwen_cache["splits"])
        if lab == "harmful" and sp == "val" and len(t) <= MAX_PROMPT_CHARS
    ]
    rng = random.Random(SEED)
    val_prompts = rng.sample(eligible, N_VAL_PROMPTS)
    print(f"Sampled {len(val_prompts)} VAL prompts (seed={SEED}), reused identically for both models "
          f"(caches confirmed consistent above)")

    print("\n--- Qwen3-8B: baseline / own (layer 23) / foreign (Llama's layer-27 direction) ---")
    qwen_conditions = run_generation_conditions(
        MODELS["Qwen3-8B"], val_prompts, qwen_own_direction, llama_own_direction
    )
    print("\n--- Llama-3.1-8B-Instruct: baseline / own (layer 27) / foreign (Qwen's layer-23 direction) ---")
    llama_conditions = run_generation_conditions(
        MODELS["Llama-3.1-8B-Instruct"], val_prompts, llama_own_direction, qwen_own_direction
    )

    qwen_verdict = transfer_verdict(qwen_conditions)
    llama_verdict = transfer_verdict(llama_conditions)

    print("\n=== Transfer verdicts (paired McNemar, n=50) ===")
    for label, conditions, verdict in (
        ("Llama's direction applied to Qwen3-8B", qwen_conditions, qwen_verdict),
        ("Qwen's direction applied to Llama-3.1-8B", llama_conditions, llama_verdict),
    ):
        print(f"\n{label}:")
        print(f"  baseline:  refusal={conditions['baseline']['refusal_stats']['rate']}")
        print(f"  own:       refusal={conditions['own_ablation']['refusal_stats']['rate']}")
        print(f"  foreign:   refusal={conditions['foreign_ablation']['refusal_stats']['rate']}")
        print(f"  baseline vs foreign: p={verdict['baseline_vs_foreign']['p_value']}")
        print(f"  baseline vs own:     p={verdict['baseline_vs_own']['p_value']}")
        print(f"  own vs foreign:      p={verdict['own_vs_foreign']['p_value']}")

    payload = {
        "models": MODELS,
        "qwen_layer": qwen_layer,
        "llama_layer": llama_layer,
        "n_val_prompts": N_VAL_PROMPTS,
        "seed": SEED,
        "val_prompts": val_prompts,
        "separation_score": separation_results,
        "causal_ablation": {
            "llama_direction_on_qwen": {"conditions": qwen_conditions, "mcnemar": qwen_verdict},
            "qwen_direction_on_llama": {"conditions": llama_conditions, "mcnemar": llama_verdict},
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cross_model_direction_transfer.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
