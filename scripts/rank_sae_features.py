"""SAE-detector feature selection: attribution-patching causal ranking.

Pipeline (see DECISIONS.md for the layer/prompt-count/length-cap rationale):
  1. Load the cached activations for the given model, compute per-layer
     difference-in-means directions on TRAIN.
  2. Load the model's top-3-by-separation-score layers' SAE checkpoints
     (per src.sae.registry.SAE_PROVIDERS -- 23/25/24 for Qwen3-8B, 27/26/21
     for Llama-3.1-8B-Instruct, 34/35/33 for gemma-2-9b-it).
  3. Restrict each layer's SAE to its top K0=10 features by cosine
     similarity to that layer's direction, pool into ~30 candidates.
  4. Rank the pooled candidates by attribution-patching causal effect
     (integrated gradients, N=10 steps) on the refusal-vs-compliance logit
     diff, averaged over a length-capped sample of harmful TRAIN prompts.
  5. Save the top K*=20 to results/sae_causal_ranking_<model>.json.

Usage: python scripts/rank_sae_features.py [model]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.activations.extract import load_model
from src.direction.compute import compute_directions
from src.direction.refusal_metric import refusal_compliance_token_ids
from src.sae.causal_ranking import rank_pooled_candidates
from src.sae.feature_selection import pool_top_k0_across_layers
from src.sae.registry import SAE_PROVIDERS

DEFAULT_MODEL = "Qwen/Qwen3-8B"
K0 = 10
K_STAR = 20
N_STEPS = 10
N_EVAL_PROMPTS = 16
MAX_PROMPT_CHARS = 150  # see DECISIONS.md: backward-pass memory risk on long outliers
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
    load_sae, LAYERS, MICRO_BATCH_SIZE = SAE_PROVIDERS[model_name]

    cache = load_cache(model_name)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    splits = cache["splits"]
    is_train = torch.tensor([s == "train" for s in splits], dtype=torch.bool)

    directions = compute_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])
    dir_by_layer = {l: directions[l] for l in LAYERS}

    print(f"Loading SAE checkpoints for layers {LAYERS}...")
    saes = {l: load_sae(l) for l in LAYERS}

    candidates = pool_top_k0_across_layers(saes, dir_by_layer, k0=K0)
    print(f"Pooled {len(candidates)} candidates from {len(LAYERS)} layers (K0={K0} each)")

    eligible_prompts = [
        t for t, lab, sp in zip(cache["texts"], cache["labels"], splits)
        if lab == "harmful" and sp == "train" and len(t) <= MAX_PROMPT_CHARS
    ]
    rng = random.Random(args.seed)
    eval_prompts = rng.sample(eligible_prompts, N_EVAL_PROMPTS)
    print(f"Sampled {len(eval_prompts)} eval prompts (seed={args.seed}, max_chars={MAX_PROMPT_CHARS}) "
          f"from {len(eligible_prompts)} eligible")

    print(f"Loading model: {model_name} (4-bit)")
    model = load_model(model_name, load_in_4bit=True)
    # SAEs stay on CPU (fp32) here -- feature_ig_attribution only ever
    # indexes a single row/column per candidate (sae.W_enc[feature_idx],
    # sae.feature_direction(feature_idx)) and moves *that* tiny slice to
    # GPU itself, so keeping the full W_enc/W_dec matrices resident on GPU
    # for the whole ranking pass was pure waste -- confirmed as the actual
    # cause of a real OOM on gemma-2-9b-it (its width_131k SAEs are ~1.9GB
    # each; three of them left almost no headroom on a 6GB card). See
    # DECISIONS.md.
    refusal_id, compliance_id = refusal_compliance_token_ids(model.tokenizer)

    print(f"Ranking {len(candidates)} candidates over {len(eval_prompts)} prompts "
          f"(n_steps={N_STEPS}, micro_batch_size={MICRO_BATCH_SIZE})...")
    ranked = rank_pooled_candidates(
        model, saes, candidates, eval_prompts, refusal_id, compliance_id,
        k_star=K_STAR, n_steps=N_STEPS, micro_batch_size=MICRO_BATCH_SIZE,
    )

    print(f"\nTop-{K_STAR} SAE features by causal (IG) effect:")
    for layer_idx, feature_idx, score in ranked:
        print(f"  layer {layer_idx}, feature {feature_idx}: score={score:.4f}")

    payload = {
        "model": model_name,
        "layers": LAYERS,
        "k0": K0,
        "k_star": K_STAR,
        "n_steps": N_STEPS,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "n_eval_prompts": N_EVAL_PROMPTS,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "seed": args.seed,
        "eval_prompts": eval_prompts,
        "pooled_candidates": candidates,
        "ranked_features": [
            {"layer": l, "feature": f, "score": s} for l, f, s in ranked
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"sae_causal_ranking_{model_name.split('/')[-1]}.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
