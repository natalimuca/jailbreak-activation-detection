"""Extends scripts/transfer_direction.py's cross-model necessity/ablation transfer
test to a SECOND architecture-matched pair: Qwen2.5-1.5B-Instruct <-> DeepSeek-R1-
Distill-Qwen-1.5B, both d_model=1536 (confirmed via results/dense_directions.pt),
activation caches confirmed fully consistent (same 1922-prompt corpus, order,
splits -- assert_caches_consistent passes). No new model downloads needed.

Necessity (ablation) only, not sufficiency (addition) -- DeepSeek's own-direction
addition is already a thoroughly-swept genuine null elsewhere in this project
(alpha 0.25-32, 2 layers, both whole-generation and answer-only application modes,
zero induced refusal everywhere), so a foreign direction is a priori unlikely to
succeed where the model's own already fails, and sufficiency-transfer needs its own
expensive alpha-calibration sweep at DeepSeek's 2048-token budget before any
validation generation even starts -- low expected information value for high cost.

**Asymmetric budget/precision, not a config swap of transfer_direction.py**: Qwen2.5-
1.5B-Instruct is ablated/generated at the standard 40-token budget used everywhere
else in this project (it never reasons); DeepSeek-R1-Distill-Qwen-1.5B always reasons
inside a mandatory <think> block with no disable switch, so it needs the 2048-token
budget and whole-generation intervention already established for its own Phase 1
reproduction (scripts/reproduce_direction.py --reasoning-model), plus resolve_
completions()-style truncation handling before scoring refusal. Both models loaded
fp16 (no 4-bit -- neither needs it at 1.5B scale, matches Phase 1's own precedent;
using 4-bit here would introduce an uncontrolled variable versus the already-
published DeepSeek Phase 1 baseline this experiment should be comparable to).

**A real correctness bug avoided by design, not discovered at runtime**: resolve_
completions() (src/direction/refusal_classifier.py) drops a DIFFERENT subset of
prompts per condition (whichever failed to close <think> in THAT specific
condition). Naively resolving each condition independently then pairing completions
by list position for McNemar would silently misalign prompts across conditions once
their truncation patterns differ. Fixed here via resolve_completions_by_index()
(returns {original_index: resolved_text}, not a compacted list) -- every McNemar
comparison between two conditions restricts to the INTERSECTION of indices that
resolved in BOTH conditions being compared, not just one shared list computed once.
Qwen2.5-1.5B's completions never contain "</think>" at all (it doesn't reason), so
this resolution step is applied ONLY to DeepSeek's conditions; Qwen2.5-1.5B's raw
completions are scored directly with is_refusal(), matching how transfer_direction.py
already handles every non-reasoning model.

N=50 harmful VAL prompts sampled once from Qwen2.5-1.5B's cache (seed=2, <=150 chars,
matching transfer_direction.py's convention exactly) -- Qwen2.5-1.5B's 3 conditions
use all 50 (cheap, 40 tokens); DeepSeek's 3 conditions use only the first 30 of that
same 50-prompt list (a real cost concern: DeepSeek's own SAE-suppression-validation
work was deliberately reduced from N=50 to N=15 per condition specifically because
N=50 x 6 conditions was projected at ~13 hours at this budget; N=30 x 3 conditions
here keeps roughly the same total-cost envelope as that already-accepted N=15 x 6
precedent while preserving more resolved-n per comparison than N=15 would, given
DeepSeek's Phase 1 numbers show 30-47% truncation).

Checkpoints per-prompt within every condition (this project has been burned twice in
one day already by a long GPU script that only wrote output at the very end).

Usage: python scripts/transfer_direction_deepseek.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.cache import assert_caches_consistent, load_cache
from src.activations.extract import load_model
from src.direction.interventions import generate_baseline, generate_with_ablation
from src.direction.refusal_classifier import (
    is_degenerate, is_refusal, refusal_stats, resolve_completions_by_index,
)
from src.eval.detector_metrics import mcnemar_exact
from scripts.transfer_direction import own_directions_and_layer

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CHECKPOINT_PATH = RESULTS_DIR / "_transfer_direction_deepseek_checkpoint.json"
OUT_PATH = RESULTS_DIR / "transfer_direction_deepseek.json"

MODELS = {
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}
N_VAL_PROMPTS = 50
N_DEEPSEEK = 30  # subset of the same 50, not a separate sample -- see module docstring
MAX_PROMPT_CHARS = 150
SEED = 2

QWEN_MAX_NEW_TOKENS = 40
DEEPSEEK_MAX_NEW_TOKENS = 2048


def run_conditions(
    hf_model_name: str, cache_label: str, prompts: list[str],
    own_direction: torch.Tensor, foreign_direction: torch.Tensor,
    max_new_tokens: int, is_reasoning_model: bool,
) -> dict:
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as fh:
            checkpoint = json.load(fh)

    condition_specs = [("baseline", None), ("own_ablation", own_direction), ("foreign_ablation", foreign_direction)]
    if all(len(checkpoint.get(cache_label, {}).get(name, [])) >= len(prompts) for name, _ in condition_specs):
        print(f"  {cache_label}: all 3 conditions already complete in checkpoint, skipping model load")
        model = None
    else:
        print(f"Loading model: {hf_model_name} (fp16, no quantization)")
        model = load_model(hf_model_name, load_in_4bit=False)

    conditions = {}
    for name, direction in condition_specs:
        saved = checkpoint.get(cache_label, {}).get(name, [])
        completions = list(saved)
        if len(completions) < len(prompts):
            print(f"  condition: {name} (resuming from {len(completions)}/{len(prompts)})")
            for prompt in prompts[len(completions):]:
                if direction is None:
                    out = generate_baseline(model, prompt, max_new_tokens=max_new_tokens)
                else:
                    out = generate_with_ablation(model, prompt, direction, max_new_tokens=max_new_tokens)
                completions.append(out)
                checkpoint.setdefault(cache_label, {})[name] = completions
                with open(CHECKPOINT_PATH, "w") as fh:
                    json.dump(checkpoint, fh)
        else:
            print(f"  condition: {name} (already complete in checkpoint)")

        if is_reasoning_model:
            resolved_by_idx = resolve_completions_by_index(completions)
            stats = refusal_stats(list(resolved_by_idx.values()))
            n_truncated = len(completions) - len(resolved_by_idx)
            print(f"    refusal_rate={stats['rate']} [{stats['ci_low']}, {stats['ci_high']}] "
                  f"n_resolved={stats['n']} n_truncated={n_truncated}")
            conditions[name] = {
                "refusal_stats": stats, "n_truncated": n_truncated,
                "completions": completions, "resolved_by_index": resolved_by_idx,
            }
        else:
            stats = refusal_stats(completions)
            degenerate_count = sum(is_degenerate(c) for c in completions)
            print(f"    refusal_rate={stats['rate']} [{stats['ci_low']}, {stats['ci_high']}] "
                  f"degenerate={degenerate_count}/{len(completions)}")
            conditions[name] = {"refusal_stats": stats, "degenerate_count": degenerate_count, "completions": completions}

    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return conditions


def paired_bools(cond_a: dict, cond_b: dict, is_reasoning_model: bool) -> tuple[list[bool], list[bool], int]:
    """Returns (bools_a, bools_b, paired_n) aligned on whichever prompt indices
    are valid for BOTH conditions -- for a reasoning model this is the
    intersection of resolved-by-index keys (a real subset, per-comparison, not
    a single shared list computed once); for a non-reasoning model every index
    is always valid, so this is just the full zip."""
    if is_reasoning_model:
        shared = sorted(set(cond_a["resolved_by_index"].keys()) & set(cond_b["resolved_by_index"].keys()))
        bools_a = [is_refusal(cond_a["resolved_by_index"][i]) for i in shared]
        bools_b = [is_refusal(cond_b["resolved_by_index"][i]) for i in shared]
        return bools_a, bools_b, len(shared)
    bools_a = [is_refusal(c) for c in cond_a["completions"]]
    bools_b = [is_refusal(c) for c in cond_b["completions"]]
    return bools_a, bools_b, len(bools_a)


def transfer_verdict(conditions: dict, is_reasoning_model: bool) -> dict:
    verdict = {}
    for pair_name, (a, b) in (
        ("baseline_vs_own", ("baseline", "own_ablation")),
        ("baseline_vs_foreign", ("baseline", "foreign_ablation")),
        ("own_vs_foreign", ("own_ablation", "foreign_ablation")),
    ):
        bools_a, bools_b, paired_n = paired_bools(conditions[a], conditions[b], is_reasoning_model)
        result = mcnemar_exact(bools_a, bools_b)
        result["paired_n"] = paired_n
        verdict[pair_name] = result
    return verdict


def main() -> None:
    assert_caches_consistent(list(MODELS.keys()))
    print("Cache consistency confirmed for Qwen2.5-1.5B-Instruct <-> DeepSeek-R1-Distill-Qwen-1.5B")

    qwen_directions, qwen_layer, _ = own_directions_and_layer("Qwen2.5-1.5B-Instruct")
    deepseek_directions, deepseek_layer, deepseek_cache = own_directions_and_layer("DeepSeek-R1-Distill-Qwen-1.5B")
    print(f"Qwen2.5-1.5B-Instruct own layer: {qwen_layer}, DeepSeek own layer: {deepseek_layer}")

    qwen_own_direction = qwen_directions[qwen_layer]
    deepseek_own_direction = deepseek_directions[deepseek_layer]

    eligible = [
        t for t, lab, sp in zip(deepseek_cache["texts"], deepseek_cache["labels"], deepseek_cache["splits"])
        if lab == "harmful" and sp == "val" and len(t) <= MAX_PROMPT_CHARS
    ]
    rng = random.Random(SEED)
    val_prompts = rng.sample(eligible, N_VAL_PROMPTS)
    print(f"Sampled {len(val_prompts)} VAL prompts (seed={SEED})")

    print("\n--- Qwen2.5-1.5B-Instruct: baseline / own / foreign (DeepSeek's direction), N=50, 40 tokens ---")
    qwen_conditions = run_conditions(
        MODELS["Qwen2.5-1.5B-Instruct"], "Qwen2.5-1.5B-Instruct", val_prompts,
        qwen_own_direction, deepseek_own_direction, QWEN_MAX_NEW_TOKENS, is_reasoning_model=False,
    )
    print(f"\n--- DeepSeek-R1-Distill-Qwen-1.5B: baseline / own / foreign (Qwen's direction), "
          f"N={N_DEEPSEEK}, {DEEPSEEK_MAX_NEW_TOKENS} tokens ---")
    deepseek_conditions = run_conditions(
        MODELS["DeepSeek-R1-Distill-Qwen-1.5B"], "DeepSeek-R1-Distill-Qwen-1.5B", val_prompts[:N_DEEPSEEK],
        deepseek_own_direction, qwen_own_direction, DEEPSEEK_MAX_NEW_TOKENS, is_reasoning_model=True,
    )

    qwen_verdict = transfer_verdict(qwen_conditions, is_reasoning_model=False)
    deepseek_verdict = transfer_verdict(deepseek_conditions, is_reasoning_model=True)

    print("\n=== Transfer verdicts ===")
    for label, conditions, verdict in (
        ("DeepSeek's direction applied to Qwen2.5-1.5B", qwen_conditions, qwen_verdict),
        ("Qwen2.5-1.5B's direction applied to DeepSeek", deepseek_conditions, deepseek_verdict),
    ):
        print(f"\n{label}:")
        for cname in ("baseline", "own_ablation", "foreign_ablation"):
            print(f"  {cname}: refusal={conditions[cname]['refusal_stats']['rate']}")
        for pair, r in verdict.items():
            print(f"  {pair}: p={r['p_value']} (paired_n={r['paired_n']})")

    # Drop the bulky resolved_by_index maps before saving the final payload
    # (redundant with completions + is_refusal, only needed transiently above).
    for conditions in (qwen_conditions, deepseek_conditions):
        for c in conditions.values():
            c.pop("resolved_by_index", None)

    payload = {
        "models": MODELS,
        "qwen_layer": qwen_layer, "deepseek_layer": deepseek_layer,
        "n_val_prompts": N_VAL_PROMPTS, "n_deepseek": N_DEEPSEEK, "seed": SEED,
        "val_prompts": val_prompts,
        "qwen_max_new_tokens": QWEN_MAX_NEW_TOKENS, "deepseek_max_new_tokens": DEEPSEEK_MAX_NEW_TOKENS,
        "causal_ablation": {
            "deepseek_direction_on_qwen": {"conditions": qwen_conditions, "mcnemar": qwen_verdict},
            "qwen_direction_on_deepseek": {"conditions": deepseek_conditions, "mcnemar": deepseek_verdict},
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
