"""Per-model dispatch for the SAE-feature detector pipeline (scripts/04,
scripts/05): which provider's `load_sae` to use, which layers to pool
candidates from, and (for scripts/04's attribution-patching ranking only)
what micro-batch size to use for the integrated-gradients backward pass.
Shared by both scripts so a layer-selection update can't drift between the
ranking and validation steps.

Layers are the top-3 by separation score (VAL-scored, TRAIN-derived
direction -- see DECISIONS.md), verified via HfApi().list_repo_files to
actually have checkpoints in the corresponding provider's repo before being
hardcoded here.

micro_batch_size is None (batch all N_STEPS integrated-gradients steps into
one forward+backward pass, scripts/04's original behavior) for every model
except gemma-2-9b-it: its 42 layers leave much less headroom after the
quantized weights than Llama-3.1-8B's 32, and a real OOM at the default
full batch confirmed it needs a smaller pass (see DECISIONS.md and
src.sae.causal_ranking's module docstring for why this doesn't change the
actual result, just how it's computed).
"""

from __future__ import annotations

from typing import Callable

from src.sae import gemma_scope, llama_scope, qwen_scope

SAE_PROVIDERS: dict[str, tuple[Callable[..., object], list[int], int | None]] = {
    "Qwen/Qwen3-8B": (qwen_scope.load_sae, [23, 25, 24], None),
    "meta-llama/Llama-3.1-8B-Instruct": (llama_scope.load_sae, [27, 26, 21], None),
    "google/gemma-2-9b-it": (gemma_scope.load_sae, [34, 35, 33], 2),
}
