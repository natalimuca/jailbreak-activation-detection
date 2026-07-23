"""Head-to-head: dense single-direction ablation (Phase 1's method) vs the
SAE-feature suppression results, on the SAME model (Qwen3-8B) and the SAME
50 held-out VAL prompts already used in
results/sae_suppression_validation_Qwen3-8B.json -- so the two approaches
are compared under identical conditions, not just independently validated.

Split discipline (avoids the leakage Phase 1 was built to avoid): direction
estimated on TRAIN (as always), but the ablation layer is selected by
separation score on TEST -- not VAL, since VAL is the exact prompt set used
for causal validation here (and was already used to pick Phase 3's SAE
layers). Phase 3 never touched TEST at all, so this is a clean, unused
split for layer selection.

Baseline refusal rate is NOT re-measured -- reused directly from the SAE
suppression validation JSON (same model, same prompts, same do_sample=False
settings), since re-running it would just reproduce that number at extra
compute cost.

Usage: python scripts/ablate_qwen3_direction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import load_cache
from src.activations.extract import load_model
from src.direction.compute import compute_directions, select_candidate_layers, separation_score
from src.direction.interventions import generate_with_ablation
from src.direction.refusal_classifier import is_degenerate, refusal_stats

DEFAULT_MODEL = "Qwen/Qwen3-8B"
SUPPRESSION_RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "sae_suppression_validation_Qwen3-8B.json"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MAX_NEW_TOKENS = 40


def main() -> None:
    model_name = DEFAULT_MODEL

    cache = load_cache(model_name)
    acts = cache["activations"]
    labels = torch.tensor([l == "harmful" for l in cache["labels"]], dtype=torch.bool)
    splits = cache["splits"]
    is_train = torch.tensor([s == "train" for s in splits], dtype=torch.bool)
    is_test = torch.tensor([s == "test" for s in splits], dtype=torch.bool)

    directions = compute_directions(acts[:, is_train & labels, :], acts[:, is_train & ~labels, :])

    test_acts = acts[:, is_test, :]
    test_labels = labels[is_test]
    scores = separation_score(test_acts, directions, test_labels)
    best_layer = select_candidate_layers(scores, k=1)[0]
    print(f"Ablation layer selected via TEST-split separation score: layer {best_layer} "
          f"(score={scores[best_layer].item():.3f})")
    direction = directions[best_layer]

    with open(SUPPRESSION_RESULTS_PATH) as fh:
        suppression_results = json.load(fh)
    val_prompts = suppression_results["val_prompts"]
    baseline_stats = suppression_results["results"]["baseline"]["refusal_stats"]
    print(f"Reusing {len(val_prompts)} VAL prompts and baseline stats from {SUPPRESSION_RESULTS_PATH.name}")
    print(f"Baseline (reused): refusal_rate={baseline_stats['rate']}")

    print(f"Loading model: {model_name} (4-bit)")
    model = load_model(model_name, load_in_4bit=True)

    completions = []
    for i, prompt in enumerate(val_prompts):
        out = generate_with_ablation(model, prompt, direction, max_new_tokens=MAX_NEW_TOKENS)
        completions.append(out)
        print(f"  [{i+1}/{len(val_prompts)}] {out[:60]!r}")

    stats = refusal_stats(completions)
    degenerate_count = sum(is_degenerate(c) for c in completions)
    print(f"\nDense-direction ablation (layer {best_layer}): refusal_rate={stats['rate']} "
          f"(95% CI [{stats['ci_low']}, {stats['ci_high']}]), degenerate={degenerate_count}/{len(completions)}")

    print("\n=== Head-to-head summary ===")
    print(f"  baseline (reused):         refusal={baseline_stats['rate']} "
          f"[{baseline_stats['ci_low']}, {baseline_stats['ci_high']}]")
    print(f"  dense-direction ablation:  refusal={stats['rate']} [{stats['ci_low']}, {stats['ci_high']}]")
    for name in ("top1", "top5", "top10", "top15", "top20"):
        s = suppression_results["results"][name]["refusal_stats"]
        print(f"  SAE suppress {name:>5}:       refusal={s['rate']} [{s['ci_low']}, {s['ci_high']}]")

    payload = {
        "model": model_name,
        "ablation_layer": best_layer,
        "ablation_layer_score": scores[best_layer].item(),
        "n_val_prompts": len(val_prompts),
        "val_prompts": val_prompts,
        "baseline_reused_from": str(SUPPRESSION_RESULTS_PATH),
        "baseline_refusal_stats": baseline_stats,
        "ablation_refusal_stats": stats,
        "ablation_degenerate_count": degenerate_count,
        "completions": completions,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"dense_direction_ablation_{model_name.split('/')[-1]}.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
