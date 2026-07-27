from __future__ import annotations

import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.sae.qwen_scope import TopKSAE

REPO_ID = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"


def _layer_dir(layer: int) -> str:
    return f"layers.{layer}.mlp"


def download_sae_checkpoint(layer: int, repo_id: str = REPO_ID) -> tuple[dict, dict]:
    d = _layer_dir(layer)
    weights_path = hf_hub_download(repo_id, f"{d}/sae.safetensors")
    cfg_path = hf_hub_download(repo_id, f"{d}/cfg.json")
    weights = load_file(weights_path)
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    return weights, cfg


def load_sae(layer: int, repo_id: str = REPO_ID, device: str = "cpu", dtype: torch.dtype = torch.float32) -> TopKSAE:
    weights, cfg = download_sae_checkpoint(layer, repo_id)
    sae = TopKSAE(
        W_enc=weights["encoder.weight"],
        W_dec=weights["W_dec"].T.contiguous(),
        b_enc=weights["encoder.bias"],
        b_dec=weights["b_dec"],
        k=cfg["k"],
    )
    return sae.to(device=device, dtype=dtype)
