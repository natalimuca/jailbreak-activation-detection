# Design decisions and open items

## Pretrained SAEs, not trained-from-scratch

The planned SAE-feature detector will use **pretrained sparse autoencoders**
rather than training our own. Training a good SAE from scratch needs a large
multi-position activation corpus (typically hundreds of thousands to millions
of vectors) and significant compute -- not a good use of solo-researcher time
on a 6GB local GPU when high-quality pretrained options already exist for
most of the target models:

- **Gemma-2-9B**: [Gemma Scope](https://arxiv.org/html/2408.05147v2) -- residual
  stream, MLP, and attention SAEs at every layer.
- **Llama-3.1-8B**: [Llama Scope](https://arxiv.org/pdf/2410.20526) (OpenMOSS)
  and [Goodfire's Llama-3 SAEs](https://www.goodfire.ai/research/understanding-and-steering-llama-3).
- **Qwen3 family**: [Qwen-Scope](https://www.marktechpost.com/2026/05/01/qwen-ai-releases-qwen-scope-an-open-source-sparse-autoencoders-sae-suite-that-turns-llm-internal-features-into-practical-development-tools/) --
  covers Qwen3-1.7B/8B and Qwen3.5, **not Qwen2.5**.

**Decided (2026-07-08): Phase 3's first model is Qwen3-8B, using
[Qwen-Scope](https://arxiv.org/pdf/2605.11887)'s `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`
checkpoint.** Rationale:

- **Ungated** -- verified via `huggingface_hub.model_info`: `google/gemma-2-2b-it`
  and `meta-llama/Llama-3.1-8B-Instruct` both require manual HF approval
  (an action only the user can do, with unknown turnaround time); `Qwen/Qwen3-8B`
  does not. This unblocks Phase 3 immediately rather than waiting on external
  approval.
- **8B fits the local 6GB GPU** with 4-bit quantization (`bitsandbytes`,
  already a dependency; `load_model(..., load_in_4bit=True)` already
  supported in `src/activations/extract.py`, just unused until now).
- **Known caveat, not a blocker**: Qwen-Scope's SAEs are trained on
  **Qwen3-8B-Base** activations, not the instruct/chat model's. Applying a
  base-trained SAE to instruct-model activations is standard, published
  practice in this exact literature -- "Understanding Refusal in Language
  Models with Sparse Autoencoders" (arXiv:2505.23556, see
  [LITERATURE.md](LITERATURE.md)) does the same thing with GemmaScope/LlamaScope
  and reports it as a limitation, not a disqualifying flaw. We do the same:
  proceed, document it, don't pretend it's not a compromise.

This replaces the original plan's Qwen2.5-7B-Instruct for the Qwen leg.
Whether to also add Gemma-2-9B / Llama-3.1-8B later (once/if HF gating is
resolved) for the full three-family cross-model set is still open --
doesn't block starting Phase 3 on Qwen3-8B alone.

## Phase 3 methodology (adapted from arXiv:2505.23556)

Following "Understanding Refusal in Language Models with Sparse
Autoencoders" (close-read in full, see LITERATURE.md), not the more ad hoc
single-feature approach in arXiv:2411.11296 (which caused severe capability
collapse -- MMLU 68.8%->36.0% at their strongest steering setting -- from
amplifying one hand-picked feature with no systematic selection):

1. Compute the difference-in-means refusal direction per layer (reuse
   `src/direction/compute.py`, already built for Phase 1).
2. Restrict the SAE's full feature space to the top K0=10 features per layer
   by cosine similarity to that direction.
3. Rank those by causal effect (paper uses Attribution Patching /
   integrated gradients; our first pass may use a cheaper ablation-sweep
   proxy given compute constraints, upgrading to AP if time allows), keep
   the top K*=20.
4. Validate causally via suppression (scale the feature down, measure
   refusal-rate drop) the same way Phase 1 validated the single direction --
   this makes the two phases' causal-validation methodology directly
   comparable.
5. **Any activation-addition/steering experiment must report a capability
   check (e.g. coherence on a held-out benign-completion set) alongside the
   induced-refusal rate** -- arXiv:2411.11296's single-feature approach
   looked like a safety win by refusal-rate alone while quietly destroying
   general capability. Refusal rate alone is not sufficient evidence of a
   working intervention.

## Ethics / department sign-off

Not yet requested. This is calendar time, not engineering time -- worth
starting the conversation with the advisor/department early rather than
leaving it for later, even though the actual risk is low (defensive research
on standard, already-public safety benchmarks, no novel harmful content
sourced or generated for release).
