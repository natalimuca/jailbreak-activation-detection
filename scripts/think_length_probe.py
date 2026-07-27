from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import format_prompt, load_model
from src.data.loaders import load_harmful

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
THINK_CLOSE_ID = 151649
PROBE_BUDGET = 3072
N_PROMPTS = 30
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "think_length_probe.json"


@torch.no_grad()
def generate_and_find_think_close(model, instruction: str, max_new_tokens: int) -> int | None:
    prompt = format_prompt(model, instruction)
    with model.generate(
        prompt, min_new_tokens=1, max_new_tokens=max_new_tokens, do_sample=False
    ) as tracer:
        out_ids = model.generator.output.save()
    generated = out_ids[0][-max_new_tokens:].tolist()
    for i, tok_id in enumerate(generated):
        if tok_id == THINK_CLOSE_ID:
            return i
    return None


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = load_model(MODEL_NAME)

    prompts = load_harmful(n=N_PROMPTS, seed=0)
    print(f"Probing think-trace length on {len(prompts)} harmful prompts "
          f"(budget={PROBE_BUDGET} tokens)")

    positions: list[int | None] = []
    for i, p in enumerate(prompts):
        pos = generate_and_find_think_close(model, p, PROBE_BUDGET)
        positions.append(pos)
        status = f"</think> at token {pos}" if pos is not None else "TRUNCATED (no </think> found)"
        print(f"  [{i+1}/{len(prompts)}] {status} -- {p[:60]!r}")

    found = [p for p in positions if p is not None]
    truncated = sum(1 for p in positions if p is None)

    result = {
        "model": MODEL_NAME,
        "n_prompts": len(prompts),
        "probe_budget": PROBE_BUDGET,
        "think_close_positions": positions,
        "n_truncated": truncated,
    }
    if found:
        sorted_found = sorted(found)
        n = len(sorted_found)
        result["stats"] = {
            "min": sorted_found[0],
            "median": sorted_found[n // 2],
            "p95": sorted_found[min(n - 1, int(0.95 * n))],
            "max": sorted_found[-1],
        }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print("\n== Summary ==")
    print(json.dumps({k: v for k, v in result.items() if k != "think_close_positions"}, indent=2))
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
