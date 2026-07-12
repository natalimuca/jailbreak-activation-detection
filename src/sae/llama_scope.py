"""Loader for LlamaScope pretrained SAEs (Llama Scope, arXiv:2410.20526).

Repo layout confirmed directly this session by inspecting the real
checkpoint (not assumed): `fnlp/Llama3_1-8B-Base-LXR-8x` is one HF repo
containing every layer's SAE for the "8x expansion, residual stream"
config, laid out as
`Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors` plus a
sibling `hyperparams.json`.

Despite the paper's "improved TopK SAEs" framing (which describes the
*training* recipe), the released checkpoints' own `hyperparams.json` report
`"act_fn": "jumprelu"` with a single scalar `"jump_relu_threshold"` --
verified by downloading and inspecting an actual checkpoint, not assumed
from the paper abstract. So these load as `JumpReLUSAE` with a scalar
threshold, not `TopKSAE`.

Tensor keys inside `final.safetensors`: `encoder.weight` (d_sae, d_model),
`encoder.bias` (d_sae,), `decoder.weight` (d_model, d_sae), `decoder.bias`
(d_model,) -- same shape convention as Qwen-Scope's W_enc/W_dec, just a
different file format (safetensors, not a plain torch `.pt`) and key
names.
"""

from __future__ import annotations

import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.sae.jumprelu_sae import JumpReLUSAE

REPO_ID = "fnlp/Llama3_1-8B-Base-LXR-8x"
EXPANSION = 8


def _checkpoint_dir(layer: int, expansion: int = EXPANSION) -> str:
    return f"Llama3_1-8B-Base-L{layer}R-{expansion}x"


def download_sae_checkpoint(
    layer: int, repo_id: str = REPO_ID, expansion: int = EXPANSION
) -> tuple[dict, dict]:
    ckpt_dir = _checkpoint_dir(layer, expansion)
    weights_path = hf_hub_download(repo_id, f"{ckpt_dir}/checkpoints/final.safetensors")
    hparams_path = hf_hub_download(repo_id, f"{ckpt_dir}/hyperparams.json")
    weights = load_file(weights_path)
    with open(hparams_path) as fh:
        hparams = json.load(fh)
    return weights, hparams


def load_sae(
    layer: int,
    repo_id: str = REPO_ID,
    expansion: int = EXPANSION,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> JumpReLUSAE:
    weights, hparams = download_sae_checkpoint(layer, repo_id, expansion)
    sae = JumpReLUSAE(
        W_enc=weights["encoder.weight"],
        W_dec=weights["decoder.weight"],
        b_enc=weights["encoder.bias"],
        b_dec=weights["decoder.bias"],
        threshold=hparams["jump_relu_threshold"],
    )
    return sae.to(device=device, dtype=dtype)
