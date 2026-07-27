"""SAE-detector causal validation via feature suppression.

Mirrors Phase 1's directional-ablation validation methodology (necessity:
does suppressing the feature(s) reduce refusal on harmful prompts?), using
the real refusal_classifier on generated text (not the differentiable
logit-diff proxy used for ranking in scripts/rank_sae_features.py) -- this is the rigorous
validation step, the ranking pass was only ever a cheap screening step.

Conditions compared, on a held-out VAL sample never used for direction
extraction or feature ranking:
  - baseline (no intervention)
  - suppress top-1 feature alone
  - suppress top-5 features (combined)
  - suppress top-10 features (combined)
  - suppress top-15 features (combined)
  - suppress top-20 features (combined, the full ranked set)

Per DECISIONS.md's capability-check requirement: also checks completions
for degeneracy (src.direction.refusal_classifier.is_degenerate) alongside
the refusal rate, since a refusal-rate drop caused by incoherent garbage
output isn't a real causal finding.

Usage: python scripts/validate_sae_features.py [model]
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
from src.direction.interventions import generate_baseline, generate_with_feature_suppression
from src.direction.refusal_classifier import is_degenerate, refusal_stats, resolve_completions
from src.sae.registry import SAE_PROVIDERS, hookpoint_for

DEFAULT_MODEL = "Qwen/Qwen3-8B"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_VAL_PROMPTS = 50
MAX_PROMPT_CHARS = 150  # consistent with scripts/rank_sae_features.py's ranking-pass cap; see DECISIONS.md
MAX_NEW_TOKENS = 40
NEEDS_4BIT = {"Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=1)  # different seed from scripts/rank_sae_features.py's eval sample
    p.add_argument("--reasoning-model", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--n-val-prompts", type=int, default=N_VAL_PROMPTS)
    return p.parse_args()


def features_by_layer_for(ranked: list[dict], saes: dict, top_n: int) -> dict:
    subset = ranked[:top_n]
    out: dict = {}
    for f in subset:
        out.setdefault(f["layer"], []).append((saes[f["layer"]], f["feature"]))
    return out


def main() -> None:
    args = parse_args()
    model_name = args.model
    load_sae, _, _ = SAE_PROVIDERS[model_name]
    ranking_path = RESULTS_DIR / f"sae_causal_ranking_{model_name.split('/')[-1]}.json"

    with open(ranking_path) as fh:
        ranking = json.load(fh)
    ranked = ranking["ranked_features"]  # already sorted descending by score
    layers = sorted({f["layer"] for f in ranked})

    cache = load_cache(model_name)
    eligible = [
        t for t, lab, sp in zip(cache["texts"], cache["labels"], cache["splits"])
        if lab == "harmful" and sp == "val" and len(t) <= MAX_PROMPT_CHARS
    ]
    rng = random.Random(args.seed)
    val_prompts = rng.sample(eligible, args.n_val_prompts)
    print(f"Sampled {len(val_prompts)} held-out VAL prompts (seed={args.seed}) "
          f"from {len(eligible)} eligible")

    print(f"Loading SAE checkpoints for layers {layers}...")
    saes = {l: load_sae(l) for l in layers}

    load_in_4bit = model_name in NEEDS_4BIT
    print(f"Loading model: {model_name}" + (" (4-bit)" if load_in_4bit else ""))
    model = load_model(model_name, load_in_4bit=load_in_4bit)
    hookpoint = hookpoint_for(model_name)
    for l in layers:
        saes[l].to(device="cuda:0", dtype=torch.float16)

    conditions = {
        "baseline": None,
        "top1": features_by_layer_for(ranked, saes, 1),
        "top5": features_by_layer_for(ranked, saes, 5),
        "top10": features_by_layer_for(ranked, saes, 10),
        "top15": features_by_layer_for(ranked, saes, 15),
        "top20": features_by_layer_for(ranked, saes, 20),
    }

    results = {}
    for name, features_by_layer in conditions.items():
        print(f"\nRunning condition: {name}")
        completions = []
        for i, prompt in enumerate(val_prompts):
            if features_by_layer is None:
                out = generate_baseline(model, prompt, max_new_tokens=args.max_new_tokens)
            else:
                out = generate_with_feature_suppression(
                    model, prompt, features_by_layer, max_new_tokens=args.max_new_tokens, hookpoint=hookpoint
                )
            completions.append(out)
            print(f"  [{i+1}/{len(val_prompts)}] {out[:60]!r}")

        if args.reasoning_model:
            resolved, n_truncated = resolve_completions(completions)
        else:
            resolved, n_truncated = completions, 0
        stats = refusal_stats(resolved)
        degenerate_count = sum(is_degenerate(c) for c in resolved)
        results[name] = {
            "refusal_stats": stats,
            "degenerate_count": degenerate_count,
            "n_truncated": n_truncated,
            "completions": completions,
        }
        print(f"  -> refusal_rate={stats['rate']} (95% CI [{stats['ci_low']}, {stats['ci_high']}]), "
              f"degenerate={degenerate_count}/{len(resolved)}, truncated={n_truncated}")

    print("\n=== Summary ===")
    for name, r in results.items():
        s = r["refusal_stats"]
        print(f"  {name}: refusal={s['rate']} [{s['ci_low']}, {s['ci_high']}], "
              f"degenerate={r['degenerate_count']}/{s['n']}")

    payload = {
        "model": model_name,
        "n_val_prompts": args.n_val_prompts,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_model": args.reasoning_model,
        "seed": args.seed,
        "val_prompts": val_prompts,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"sae_suppression_validation_{model_name.split('/')[-1]}.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
