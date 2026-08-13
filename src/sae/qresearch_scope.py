"""Loader for the qresearch third-party SAE trained on
DeepSeek-R1-Distill-Llama-8B (`qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19`).

Layer 19 only -- the sole layer this checkpoint covers, no "top-3 by
separation score" choice available the way this project's other providers
have. Checkpoint verified directly (not assumed from the model card): a
plain `.pt` state dict, `encoder.weight (65536,4096)`, `encoder.bias
(65536,)`, `decoder.weight (4096,65536)`, `decoder.bias (4096,)` -- same
key names and shape convention as LlamaScope's checkpoints
(`src/sae/llama_scope.py`), no transpose needed.

**Activation function is not verified from source, unlike every other SAE
provider in this project.** The model card reports a "final L0 of 93"
during training but the checkpoint stores no separate threshold tensor (a
real JumpReLU checkpoint needs one -- see `src/sae/jumprelu_sae.py`'s own
docstring on LlamaScope/GemmaScope's actual-JumpReLU-despite-paper-framing
precedent), which is evidence against JumpReLU and consistent with hard
top-k -- but the model card's cited GitHub training-code repo is dead (404,
confirmed via the GitHub API), so there is no ground truth to check this
against. `k=93` (the reported L0) is used as `TopKSAE`'s default here, but
this is an assumption pending the empirical reconstruction-quality check
`reports/DECISIONS.md`'s pre-registration entry requires before any
causal-ranking work trusts it -- see that entry, not this module, for the
verification result.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download

from src.sae.qwen_scope import TopKSAE

REPO_ID = "qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19"
FILENAME = "DeepSeek-R1-Distill-Llama-8B-SAE-l19.pt"
LAYER = 19
REPORTED_L0 = 93


def download_sae_checkpoint(repo_id: str = REPO_ID, filename: str = FILENAME) -> dict:
    path = hf_hub_download(repo_id, filename)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_sae(
    layer: int = LAYER,
    k: int = REPORTED_L0,
    repo_id: str = REPO_ID,
    filename: str = FILENAME,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TopKSAE:
    if layer != LAYER:
        raise ValueError(f"qresearch checkpoint only covers layer {LAYER}, got layer {layer}")
    ckpt = download_sae_checkpoint(repo_id, filename)
    sae = TopKSAE(
        W_enc=ckpt["encoder.weight"],
        W_dec=ckpt["decoder.weight"],
        b_enc=ckpt["encoder.bias"],
        b_dec=ckpt["decoder.bias"],
        k=k,
    )
    return sae.to(device=device, dtype=dtype)
