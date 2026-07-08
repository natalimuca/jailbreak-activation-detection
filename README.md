# Jailbreak Detection via Internal Activations

Detects jailbreak / harmful-intent prompts using an LLM's internal activations
instead of surface-level (keyword or perplexity) filtering.

## Core idea

Builds on Arditi et al., "Refusal in Language Models Is Mediated by a Single
Direction" (NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)):
a single direction in residual-stream activation space causally controls refusal
across open chat models. Surface-level detectors can be fooled by disguising a
harmful request as fiction/roleplay; an activation-based detector reads intent
rather than wording and should be more robust to that disguise.

This project extends that finding in two directions:

1. **Cross-model generalization** — does a detector trained on one model's
   internals transfer to a different model family/size, or does it need
   per-model retraining?
2. **Engineering rigor** — calibration, decision-curve / operating-point
   tuning, false-positive cost analysis, latency benchmarking, and an honest
   head-to-head comparison against baselines on a held-out adversarial set.

## Tooling

- **nnsight** for activation extraction (chosen over TransformerLens: traces
  native HF models directly, so 4-bit quantization and multi-architecture
  support work without a model-conversion step — important given a 6GB local
  GPU and multiple model families in scope).
- PyTorch, HF Transformers/Accelerate/bitsandbytes, HF Datasets.

## Datasets

- [AdvBench](https://github.com/llm-attacks/llm-attacks) — harmful instructions
- [HarmBench](https://github.com/centerforaisafety/HarmBench) — standardized red-teaming eval set
- [JailbreakBench / JBB-Behaviors](https://github.com/JailbreakBench/jailbreakbench) — curated behaviors + real jailbreak artifacts
- [XSTest](https://github.com/paul-rottger/xstest) — safe-but-superficially-harmful prompts, for measuring over-refusal

## Models

Open-weight, via Hugging Face: `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`
for fast iteration, plus a larger cross-model set (`Llama-3-8B-Instruct`,
`Gemma-2-9B-it`, `Qwen2.5-7B-Instruct`) run 4-bit quantized to fit local VRAM,
or via nnsight remote execution for larger runs.

## Repo layout

```
src/
  data/          dataset loading + harmful/harmless prompt pairing
  activations/   activation extraction and caching via nnsight
  direction/     refusal-direction computation, ablation, activation addition
scripts/         standalone runnable pipeline stages
notebooks/       exploratory analysis
results/         metrics, figures, activation caches (gitignored)
tests/
```
