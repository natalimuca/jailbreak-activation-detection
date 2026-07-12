"""Loader for GemmaScope pretrained SAEs (arXiv:2408.05147).

Repo layout confirmed directly this session by inspecting real files (not
assumed): `google/gemma-scope-9b-pt-res` lays out checkpoints as
`layer_{n}/width_{w}/average_l0_{l0}/params.npz` (a numpy archive, not
safetensors or a plain torch `.pt`), with a near-empty sibling
`hparams.json` (just `sparsity_lambda` -- unlike LlamaScope's, it does not
carry the JumpReLU threshold).

**Checkpoint choice**: `width_131k` (~36.6x expansion for Gemma-2-9B's
d_model=3584), matching the ~"expansion 32" GemmaScope config the
arXiv:2505.23556 paper this project's SAE methodology is adapted from used
(see LITERATURE.md) -- the only close alternative available for the
layers this project needs is `width_16k` (~4.5x), a clearly worse match.
`average_l0_51`, matching this project's own Qwen-Scope precedent of
L0=50 (`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`) as closely as an available
option allows.

**Tensor layout differs from Qwen-Scope/LlamaScope**: GemmaScope's own
`params.npz` stores `W_enc` as (d_model, d_sae) and `W_dec` as (d_sae,
d_model) -- the *opposite* of this project's convention (`JumpReLUSAE`/
`TopKSAE` expect `W_enc`: (d_sae, d_model), `W_dec`: (d_model, d_sae)).
Transposed on load below; confirmed via direct inspection, not assumed.
`threshold` is a genuine per-feature (d_sae,) array here (unlike
LlamaScope's scalar), consistent with GemmaScope's own JumpReLU SAEs
paper describing per-feature adaptive thresholds.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from src.sae.jumprelu_sae import JumpReLUSAE

REPO_ID = "google/gemma-scope-9b-pt-res"
WIDTH = "131k"
AVERAGE_L0 = 51


def _checkpoint_dir(layer: int, width: str = WIDTH, average_l0: int = AVERAGE_L0) -> str:
    return f"layer_{layer}/width_{width}/average_l0_{average_l0}"


def download_sae_checkpoint(
    layer: int, repo_id: str = REPO_ID, width: str = WIDTH, average_l0: int = AVERAGE_L0
) -> tuple[dict, dict]:
    ckpt_dir = _checkpoint_dir(layer, width, average_l0)
    params_path = hf_hub_download(repo_id, f"{ckpt_dir}/params.npz")
    hparams_path = hf_hub_download(repo_id, f"{ckpt_dir}/hparams.json")
    params = dict(np.load(params_path))
    with open(hparams_path) as fh:
        hparams = json.load(fh)
    return params, hparams


def load_sae(
    layer: int,
    repo_id: str = REPO_ID,
    width: str = WIDTH,
    average_l0: int = AVERAGE_L0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> JumpReLUSAE:
    params, _hparams = download_sae_checkpoint(layer, repo_id, width, average_l0)
    sae = JumpReLUSAE(
        W_enc=torch.from_numpy(params["W_enc"]).T.contiguous(),
        W_dec=torch.from_numpy(params["W_dec"]).T.contiguous(),
        b_enc=torch.from_numpy(params["b_enc"]),
        b_dec=torch.from_numpy(params["b_dec"]),
        threshold=torch.from_numpy(params["threshold"]),
    )
    return sae.to(device=device, dtype=dtype)
