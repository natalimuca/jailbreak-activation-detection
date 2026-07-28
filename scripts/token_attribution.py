"""Tier 2 of the mechanistic dig started in `scripts/paraphrase_decay_sae.py`:
that script found Llama-3.1-8B's own top-ranked causal SAE feature (layer
27/feature 13363) shows no statistically detectable activation shift when a
harmful prompt is paraphrased into its real PAIR-attack form (Wilcoxon
p=0.168) -- the only one of the three SAE-having models where that's true.
That's a real matched-pair result, but still a "what," not a "why": it
doesn't say *what about the paraphrased text* the feature does or doesn't
react to. This script asks that directly, comparing Llama-3.1-8B (feature
invariant) against Qwen3-8B (feature not invariant, its own top feature
layer25/65291 decays significantly under paraphrase) on the 21 real
PAIR-paraphrased prompts.

**Method: single-token leave-one-out ablation, not gradient-based
attribution.** Two successive attempts at embedding-level Integrated
Gradients both failed a numerical correctness check before any real
prompt was trusted:
1. Multiplicative interpolation from a zero baseline (`v = alpha *
   natural_embedding`, mirroring `src.sae.causal_ranking`'s convention for
   a single scalar feature) -- failed IG's completeness property by ~100x.
   Root cause, confirmed by direct measurement: Llama-3.1/Qwen3 apply
   RMSNorm immediately after the embedding layer, and RMSNorm is
   scale-invariant (`RMSNorm(alpha*x) == RMSNorm(x)` for any alpha>0), so
   scaling the embedding barely changes anything except exactly at
   alpha=0 -- IG's discretized integral never samples that exact point and
   misses almost the entire true effect.
2. Fixed to additive interpolation toward a real baseline (each model's
   own EOS-token embedding, tiled across positions) -- completeness still
   failed, now with an even larger discrepancy, most likely because a
   repeated-EOS-token input is itself a wildly out-of-distribution point
   the straight-line path from natural to baseline can't approximate well
   in 10 IG steps (or a deeper nnsight/4-bit-quantization gradient
   interaction at the embedding layer -- not fully isolated).

Given two failed gradient-based attempts, switched to a simpler, more
robust method with no backward pass at all: for each token position,
replace just that token with a neutral baseline token (each model's own
EOS token) and re-run a clean forward pass, comparing the target feature's
raw pre-activation with vs. without that token
(`natural_value - ablated_value`, positive = this token was contributing
positively to the feature firing). Standard occlusion/leave-one-out
saliency -- no gradients, no interpolation path to get wrong, no
quantization-backward fragility. Batches all `seq_len` single-token
ablations for a prompt into one forward pass (no backward needed at all).

Note: unlike Integrated Gradients, single-token ablation has no
"completeness" invariant to check against (leave-one-out effects don't sum
to the full-prompt effect in general, since tokens can be redundant with
each other) -- verification here is a basic sanity check (finite values,
real variance across positions, plausible scale relative to the natural
value), not a formal decomposition property.

Deliberately qualitative, not another aggregate p-value: reports the
top-5 highest-importance tokens per prompt for direct reading, not run
through an automated "core request vs. roleplay wrapper" classifier --
this project's own moralize-vs-comply saga is a standing cautionary tale
about trusting an unvalidated automated classifier over direct reading.

Usage: python scripts/token_attribution.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# Windows' default console codec (cp1252) can't encode several BPE tokenizer
# glyphs (e.g. the "Ġ"-style space marker) that show up when printing raw
# token strings -- reconfigure stdout to UTF-8 rather than let a print()
# crash the whole run partway through, discovered when this crashed after a
# genuinely successful ~5-minute computation on the very first prompt.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import format_prompt, load_model
from src.sae.feature_probe import feature_value
from src.sae.registry import SAE_PROVIDERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TOP_K_TOKENS = 5
# Small: the 4-bit 8B model alone leaves very little of the 6GB card free,
# and bnb's on-the-fly dequantization plus nnsight's trace overhead pushed
# a batch of 32 into OOM directly (measured, not assumed) -- same
# "needs a smaller per-pass batch to fit in 6GB" situation this project
# already hit for gemma-2-9b-it's causal ranking (see reports/DECISIONS.md).
# Measured 4 vs. 8: nearly identical wall-clock (~300s/prompt either way --
# throughput is not batch-bound here), so 8 halves the number of trace()
# calls for free.
ABLATION_BATCH_SIZE = 8

MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
# (layer, feature) of each model's own rank-1 causally-ranked feature.
TOP_FEATURE = {"Qwen3-8B": (25, 65291), "Llama-3.1-8B-Instruct": (27, 13363)}


def token_ablation_importance(
    model, layer_idx: int, sae, feature_idx: int, prompt: str, baseline_token_id: int,
    batch_size: int = ABLATION_BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Returns (importance (seq_len,), input_ids (seq_len,), natural_val).
    importance_i = natural_value - value_with_token_i_replaced -- positive
    means token i was contributing positively to the feature's activation."""
    templated = format_prompt(model, prompt)
    input_ids = model.tokenizer(templated, return_tensors="pt")["input_ids"]  # (1, seq_len)
    seq_len = input_ids.shape[1]

    natural_val = feature_value(model, layer_idx, sae, feature_idx, input_ids)[0]

    ablated_ids = input_ids.repeat(seq_len, 1).clone()
    positions = torch.arange(seq_len)
    ablated_ids[positions, positions] = baseline_token_id

    ablated_vals = []
    for start in range(0, seq_len, batch_size):
        chunk = ablated_ids[start:start + batch_size]
        ablated_vals.append(feature_value(model, layer_idx, sae, feature_idx, chunk))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ablated_vals = torch.cat(ablated_vals, dim=0)  # (seq_len,)

    importance = natural_val - ablated_vals
    return importance, input_ids[0], natural_val.item()


def sanity_check(importance: torch.Tensor, natural_val: float) -> bool:
    """No completeness property to check (see module docstring) -- just
    basic non-degeneracy: finite values and real variance across
    positions, not all collapsed to zero or identical."""
    finite = torch.isfinite(importance).all().item()
    has_variance = importance.std().item() > 1e-6
    print(f"  Sanity check: finite={finite}, std={importance.std().item():.4f}, "
          f"range=[{importance.min().item():.4f}, {importance.max().item():.4f}], natural_val={natural_val:.4f}")
    return finite and has_variance


def content_span_mask(model, templated: str, prompt: str, seq_len: int) -> torch.Tensor:
    """True for token positions that fall inside the raw instruction text
    within the full chat-templated prompt -- everything else is template
    scaffolding (role markers, <think> prefill, newlines). Found necessary
    after a first real run: the un-masked top tokens were dominated by
    boilerplate near the readout position ('<think>', 'assistant', 'ĊĊ'),
    not the actual paraphrased content -- exactly the tokens this
    investigation is not about."""
    start_char = templated.find(prompt)
    assert start_char != -1, "instruction text not found verbatim in the chat-templated prompt"
    end_char = start_char + len(prompt)
    # Must match token_ablation_importance's own tokenizer call exactly
    # (default add_special_tokens=True) -- passing add_special_tokens=False
    # here caused a real off-by-one for Llama-3.1 specifically (its
    # tokenizer prepends a BOS special token by default; Qwen3-8B's
    # doesn't, which is why this only broke on the second model, not the
    # first) -- measured directly, not assumed.
    enc = model.tokenizer(templated, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    assert len(offsets) == seq_len, f"offset mapping length {len(offsets)} != seq_len {seq_len}"
    return torch.tensor([not (o[1] <= start_char or o[0] >= end_char) for o in offsets], dtype=torch.bool)


def top_tokens(
    model, input_ids: torch.Tensor, importance: torch.Tensor, templated: str, prompt: str, k: int = TOP_K_TOKENS
) -> list[tuple[str, float]]:
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    assert len(tokens) == len(importance), f"token/score length mismatch: {len(tokens)} vs {len(importance)}"
    mask = content_span_mask(model, templated, prompt, len(tokens))
    content_pairs = [(t, s) for t, s, m in zip(tokens, importance.tolist(), mask.tolist()) if m]
    scored = sorted(content_pairs, key=lambda t: t[1], reverse=True)
    return scored[:k]


def main() -> None:
    with open(RESULTS_DIR / "adversarial_paraphrase_manifest.json") as fh:
        manifest = json.load(fh)
    pair_records = [r for r in manifest if r["method"] == "PAIR"]
    print(f"{len(pair_records)} real PAIR-adversarial prompts")

    # Resume from checkpoint: skip any model that already has a complete
    # (all pair_records) entry from a prior run -- avoids redoing Qwen3-8B's
    # already-finished 21 prompts after a crash partway through Llama's.
    checkpoint_path = RESULTS_DIR / "token_attribution.json"
    results = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as fh:
            results = json.load(fh)

    for cache_label, hf_model_name in MODELS.items():
        if len(results.get(cache_label, [])) >= len(pair_records):
            print(f"\n=== {cache_label}: already complete in checkpoint ({len(results[cache_label])}/{len(pair_records)}), skipping ===")
            continue
        layer_idx, feature_idx = TOP_FEATURE[cache_label]
        print(f"\n=== {cache_label} (layer {layer_idx}, feature {feature_idx}) ===")
        print(f"Loading model: {hf_model_name} (4-bit)")
        model = load_model(hf_model_name, load_in_4bit=True)
        load_sae_fn = SAE_PROVIDERS[hf_model_name][0]
        sae = load_sae_fn(layer_idx)
        baseline_token_id = model.tokenizer.eos_token_id

        print("Sanity check on first prompt:")
        importance0, input_ids0, natural_val0 = token_ablation_importance(
            model, layer_idx, sae, feature_idx, pair_records[0]["text"], baseline_token_id
        )
        ok = sanity_check(importance0, natural_val0)
        if not ok:
            print(f"  SANITY CHECK FAILED for {cache_label} -- aborting, results would not be trustworthy.")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        model_results = []
        for i, r in enumerate(pair_records):
            importance, input_ids, _ = token_ablation_importance(
                model, layer_idx, sae, feature_idx, r["text"], baseline_token_id
            )
            templated = format_prompt(model, r["text"])
            # Save the raw per-token importance (not just top-K) so a bug in
            # top_tokens/content_span_mask -- there already was one -- can't
            # lose the expensive GPU computation, only the cheap CPU-side
            # reporting step built on top of it.
            record = {
                "goal": r["goal"], "text": r["text"],
                "tokens": model.tokenizer.convert_ids_to_tokens(input_ids.tolist()),
                "importance": importance.tolist(),
            }
            try:
                record["top_tokens"] = top_tokens(model, input_ids, importance, templated, r["text"])
                print(f"  [{i+1}/{len(pair_records)}] top tokens: {[t for t, s in record['top_tokens']]}")
            except Exception as e:
                record["top_tokens_error"] = str(e)
                print(f"  [{i+1}/{len(pair_records)}] top_tokens FAILED ({e}) -- raw importance still saved")
            model_results.append(record)

            # Checkpoint after every prompt -- this run has already lost a
            # completed model's results to a downstream bug once; cheap
            # file I/O, not worth risking again.
            results[cache_label] = model_results
            with open(RESULTS_DIR / "token_attribution.json", "w") as fh:
                json.dump(results, fh, indent=2)

        results[cache_label] = model_results
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "token_attribution.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
