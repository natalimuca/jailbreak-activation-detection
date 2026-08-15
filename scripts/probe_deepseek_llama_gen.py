"""Quick, cheap diagnostic before committing to the real alpha-calibration
sweep: the first real attempt at `calibrate_alpha.py --reasoning-model
--max-new-tokens 2048` on DeepSeek-R1-Distill-Llama-8B ran for 3h47m without
completing even its first `generate_with_addition` call (the corpus
extraction phases that preceded it only took ~8.5 minutes total, so the
entire delay was inside intervention-hooked generation). This checks two
things at small, bounded cost before re-running anything expensive:

1. Real per-token throughput for PLAIN generation (no intervention hook) --
   isolates whether the slowdown is generic to this model/hardware or
   specific to `generate_with_addition`'s per-token hook.
2. Real per-token throughput WITH the addition intervention active, same
   comparison.
3. Reasoning-trace length distribution (`</think>` position), same idea as
   `scripts/think_length_probe.py` but with this model's own verified token
   ID (128014, confirmed by direct tokenizer inspection -- NOT DeepSeek-1.5B's
   151649, different tokenizer/architecture).

Small N, small budget, purely diagnostic -- not the final calibration run.

Usage: python scripts/probe_deepseek_llama_gen.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import format_prompt, get_last_token_resid_acts, load_model
from src.data.loaders import load_harmful
from src.direction.compute import compute_raw_directions, select_candidate_layers, separation_score, compute_directions
from src.direction.interventions import generate_with_addition

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
THINK_CLOSE_ID = 128014
PROBE_BUDGET = 512
N_PROBE = 4


@torch.no_grad()
def plain_generate_timed(model, instruction: str, max_new_tokens: int) -> tuple[list[int], float]:
    prompt = format_prompt(model, instruction)
    t0 = time.time()
    with model.generate(prompt, min_new_tokens=1, max_new_tokens=max_new_tokens, do_sample=False) as tracer:
        out_ids = model.generator.output.save()
    elapsed = time.time() - t0
    generated = out_ids[0][-max_new_tokens:].tolist()
    return generated, elapsed


def main() -> None:
    print(f"Loading model: {MODEL_NAME} (4-bit)")
    model = load_model(MODEL_NAME, load_in_4bit=True)

    prompts = load_harmful(n=N_PROBE, seed=0)

    print(f"\n=== Plain generation (no intervention), budget={PROBE_BUDGET} ===")
    for i, p in enumerate(prompts):
        generated, elapsed = plain_generate_timed(model, p, PROBE_BUDGET)
        think_pos = next((j for j, t in enumerate(generated) if t == THINK_CLOSE_ID), None)
        n_gen = len(generated)
        rate = n_gen / elapsed if elapsed > 0 else float("nan")
        status = f"</think> at {think_pos}" if think_pos is not None else "no </think> found in budget"
        print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}s for {n_gen} tokens ({rate:.2f} tok/s) -- {status} -- {p[:50]!r}")

    print(f"\n=== Addition-intervention generation, budget={PROBE_BUDGET} ===")
    print("Computing a direction to intervene with (small N, diagnostic only -- magnitude doesn't matter for a pure timing probe, the hook's compute cost is the same regardless)...")
    harmful_acts = get_last_token_resid_acts(model, prompts)
    other_prompts = load_harmful(n=N_PROBE, seed=1)
    other_acts = get_last_token_resid_acts(model, other_prompts)
    directions = compute_raw_directions(harmful_acts, other_acts)
    layer = 15  # mid-network, arbitrary for a timing probe

    for i, p in enumerate(prompts):
        t0 = time.time()
        completion = generate_with_addition(model, p, directions[layer], layer_idx=layer, alpha=1.0, max_new_tokens=PROBE_BUDGET)
        elapsed = time.time() - t0
        # generate_with_addition returns decoded text, not token ids -- approximate token count via whitespace
        approx_tokens = len(completion.split())
        rate = approx_tokens / elapsed if elapsed > 0 else float("nan")
        print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}s (~{approx_tokens} words, ~{rate:.2f} words/s) -- {completion[:80]!r}")


if __name__ == "__main__":
    main()
