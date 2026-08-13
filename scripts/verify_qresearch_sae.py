"""Required verification gate from `reports/DECISIONS.md`'s pre-registration
entry ("DeepSeek-R1-Distill-Llama-8B vs. Llama-3.1-8B-Instruct"), before any
causal-ranking work trusts `src/sae/qresearch_scope.py`'s SAE.

That checkpoint has no ground truth to verify against (its cited training
repo is dead), unlike every other SAE provider in this project. What can be
checked directly: does forcing hard top-k activation at inference time
reconstruct this model's *real* layer-19 activations well? Good
reconstruction means the checkpoint is a usable feature basis regardless of
its exact training-time sparsity mechanism; poor reconstruction at every
plausible k means something is mismatched (layer, hookpoint, or the
checkpoint genuinely isn't a clean top-k SAE) and it should not be used.

Sweeps k around the model card's reported "final L0 of 93", plus an
unsparsified upper bound (k=d_sae, i.e. plain linear encode/decode with no
zeroing) as a sanity ceiling -- if reconstruction is bad even with zero
sparsity, the problem is architectural (wrong layer/hookpoint/dtype), not a
bad k choice.

Usage: python scripts/verify_qresearch_sae.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import get_last_token_resid_acts, load_model
from src.sae.qresearch_scope import LAYER, REPORTED_L0, load_sae
from scripts.wrapper_swap_variance import CORE_REQUESTS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
HF_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
K_SWEEP = [32, 64, REPORTED_L0, 128, 192, 256, 512]

# First pass at k=93/layer=19 alone was catastrophic (FVE=-2.27, and even
# FVE=-64192 with zero sparsity) -- MSE barely moved across a 16x k range,
# the signature of a scale/normalization or layer-indexing mismatch, not a
# bad k choice. This second pass checks both: neighboring layers (off-by-one
# indexing conventions differ between training setups) and unit-RMS-norm
# rescaling (a common SAE preprocessing step this checkpoint's dead training
# repo can't confirm or rule out).
LAYER_SWEEP = [17, 18, 19, 20, 21]


def fraction_variance_explained(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    resid_var = (x - x_hat).pow(2).sum().item()
    total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum().item()
    return 1.0 - resid_var / total_var if total_var > 0 else float("nan")


def reconstruct(sae, x: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        with torch.no_grad():
            return sae.decode(sae.encode(x))
    d_model = x.shape[-1]
    norm = x.norm(dim=-1, keepdim=True)
    x_scaled = x / norm * (d_model ** 0.5)
    with torch.no_grad():
        x_hat_scaled = sae.decode(sae.encode(x_scaled))
    return x_hat_scaled / (d_model ** 0.5) * norm


def main() -> None:
    print(f"Loading model: {HF_MODEL} (4-bit)")
    model = load_model(HF_MODEL, load_in_4bit=True)

    print(f"Extracting all-layer activations for {len(CORE_REQUESTS)} real prompts...")
    all_layer_acts = get_last_token_resid_acts(model, CORE_REQUESTS)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {}
    print(f"\n=== Layer x normalization sweep at k={REPORTED_L0} ===")
    for layer in LAYER_SWEEP:
        x = all_layer_acts[layer].float()
        try:
            sae = load_sae(layer=LAYER, k=REPORTED_L0)  # SAE itself is only trained for layer 19
        except Exception as e:
            print(f"  layer {layer}: SAE load failed: {e}")
            continue
        for normalize in [False, True]:
            x_hat = reconstruct(sae, x, normalize)
            fve = fraction_variance_explained(x, x_hat)
            mse = (x - x_hat).pow(2).mean().item()
            key = f"layer{layer}_{'normalized' if normalize else 'raw'}"
            results[key] = {"fve": fve, "mse": mse}
            print(f"  layer={layer:2d} {'normalized' if normalize else 'raw       '}  FVE={fve:.4f}  MSE={mse:.4f}")

    print(f"\n=== k sweep at layer {LAYER}, both raw and normalized ===")
    x19 = all_layer_acts[LAYER].float()
    d_sae = 65536
    for k in K_SWEEP + [d_sae]:
        sae = load_sae(layer=LAYER, k=k)
        for normalize in [False, True]:
            x_hat = reconstruct(sae, x19, normalize)
            fve = fraction_variance_explained(x19, x_hat)
            mse = (x19 - x_hat).pow(2).mean().item()
            key = f"k{k}_{'normalized' if normalize else 'raw'}"
            results[key] = {"fve": fve, "mse": mse}
            label = "no sparsity" if k == d_sae else f"k={k}"
            print(f"  {label:14s} {'normalized' if normalize else 'raw       '}  FVE={fve:.4f}  MSE={mse:.4f}")

    out_path = RESULTS_DIR / "qresearch_sae_verification.json"
    with open(out_path, "w") as fh:
        json.dump({"model": HF_MODEL, "n_prompts": len(CORE_REQUESTS), "results": results}, fh, indent=2)
    print(f"\nSaved to {out_path}")

    best_key = max(results, key=lambda k: results[k]["fve"])
    best_fve = results[best_key]["fve"]
    print(f"\nBest configuration overall: {best_key}, FVE={best_fve:.4f}")
    if best_fve >= 0.5:
        print("VERDICT: a usable configuration exists -- gate PASSES for that configuration, proceed with causal-ranking work using it.")
    else:
        print("VERDICT: no configuration tested reaches usable reconstruction (FVE>=0.5) -- gate FAILS. Report as unusable, proceed with dense-direction-only work.")


if __name__ == "__main__":
    main()
