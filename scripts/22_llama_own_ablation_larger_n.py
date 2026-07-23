"""Re-runs Llama-3.1-8B-Instruct's own dense-direction causal ablation
(baseline vs. own-direction ablation, NOT the cross-model foreign-direction
comparison) at a larger sample -- closes a specific gap flagged in
RESULTS.md's cross-model-direction-transfer section: `scripts/17`'s n=50
found this effect real but small (92% -> 88% refusal) and NOT significant
via paired McNemar's test, "worth an independent re-run at a larger sample
before leaning on it further."

n=50 wasn't underpowered by accident -- it's this project's standing
validation-sample convention (SAE causal validation, Gemma suppression
significance test). The problem here is specifically that Llama's own
effect size is much smaller than those (an ~8-point refusal-rate shift vs.
SAE-feature's 84%->18%), so McNemar's test -- which only draws information
from *discordant* pairs -- only had a handful to work with at n=50. This
script draws a fresh, independent, 1.5x-larger sample (n=75, new seed --
not an extension of the original 50 prompts) to see whether the same small
effect reaches significance with more statistical power. (An initial n=150
attempt was killed after running far slower than expected -- ~2 tok/s on
plain baseline generation alone, no hooks yet -- and not worth the wall-
clock cost; n=75 is still a real improvement in power over n=50 at a
fraction of the runtime.)

Usage: python scripts/22_llama_own_ablation_larger_n.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.activations.extract import load_model
from src.detectors.dense_direction_detector import resolve_layer_for_model
from src.direction.compute import compute_directions
from src.direction.interventions import generate_baseline, generate_with_ablation
from src.direction.refusal_classifier import is_degenerate, is_refusal, refusal_stats
from src.eval.detector_metrics import mcnemar_exact

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_LABEL = "Llama-3.1-8B-Instruct"
N_VAL_PROMPTS = 75
MAX_PROMPT_CHARS = 150
MAX_NEW_TOKENS = 40
SEED = 7  # different from scripts/17's SEED=2 -- an independent draw, not an extension


def main() -> None:
    print(f"Loading cache for {CACHE_LABEL} to compute own direction/layer from TRAIN split")
    cache = load_cache(CACHE_LABEL)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    is_train = torch.tensor([s == "train" for s in cache["splits"]], dtype=torch.bool)
    directions = compute_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])
    layer = resolve_layer_for_model(HF_MODEL_NAME)
    direction = directions[layer]
    print(f"Own layer: {layer} (already-selected, not re-derived)")

    eligible = [
        t for t, lab, sp in zip(cache["texts"], cache["labels"], cache["splits"])
        if lab == "harmful" and sp == "val" and len(t) <= MAX_PROMPT_CHARS
    ]
    rng = random.Random(SEED)
    n = min(N_VAL_PROMPTS, len(eligible))
    val_prompts = rng.sample(eligible, n)
    print(f"Sampled {len(val_prompts)} VAL prompts (seed={SEED}, independent of scripts/17's n=50 draw)")

    print(f"\nLoading model: {HF_MODEL_NAME} (4-bit)")
    model = load_model(HF_MODEL_NAME, load_in_4bit=True)

    print("Baseline completions")
    baseline_completions = [generate_baseline(model, p, max_new_tokens=MAX_NEW_TOKENS) for p in val_prompts]
    print("Own-direction ablation completions")
    ablated_completions = [generate_with_ablation(model, p, direction, max_new_tokens=MAX_NEW_TOKENS) for p in val_prompts]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    baseline_stats = refusal_stats(baseline_completions)
    ablated_stats = refusal_stats(ablated_completions)
    baseline_refusals = [is_refusal(c) for c in baseline_completions]
    ablated_refusals = [is_refusal(c) for c in ablated_completions]
    verdict = mcnemar_exact(baseline_refusals, ablated_refusals)

    print(f"\nbaseline refusal:      {baseline_stats['rate']} [{baseline_stats['ci_low']}, {baseline_stats['ci_high']}]")
    print(f"own-ablation refusal:  {ablated_stats['rate']} [{ablated_stats['ci_low']}, {ablated_stats['ci_high']}]")
    print(f"McNemar's exact test: {verdict}")
    print(f"degenerate (baseline): {sum(is_degenerate(c) for c in baseline_completions)}/{len(val_prompts)}")
    print(f"degenerate (ablated):  {sum(is_degenerate(c) for c in ablated_completions)}/{len(val_prompts)}")

    payload = {
        "model": HF_MODEL_NAME,
        "layer": layer,
        "n_val_prompts": len(val_prompts),
        "seed": SEED,
        "val_prompts": val_prompts,
        "baseline": {
            "refusal_stats": baseline_stats,
            "degenerate_count": sum(is_degenerate(c) for c in baseline_completions),
            "completions": baseline_completions,
        },
        "own_ablation": {
            "refusal_stats": ablated_stats,
            "degenerate_count": sum(is_degenerate(c) for c in ablated_completions),
            "completions": ablated_completions,
        },
        "mcnemar": verdict,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "llama_own_ablation_larger_n.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
