# Jailbreak Detection via Internal Activations



## Core idea

Builds on Arditi et al., "Refusal in Language Models Is Mediated by a Single
Direction" (NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)):
a single direction in residual-stream activation space causally controls refusal
across open chat models. Surface-level detectors can be fooled by disguising a
harmful request as fiction/roleplay; an activation-based detector reads intent
rather than wording and should be more robust to that disguise.

Since "activation-based jailbreak detector" alone is now a busy 2025-2026 research
area (SAE-feature detectors beating the single-direction baseline under adversarial
paraphrase), this thesis's contribution is one or both of:

1. **Cross-model generalization** — does a detector trained on one model's
   internals transfer to a different model family/size, or does it need
   per-model retraining?
2. **The engineering-rigor layer papers skip** — calibration, decision-curve /
   operating-point tuning, false-positive cost analysis, latency benchmarking,
   honest head-to-head comparison against baselines on a held-out adversarial set.

## Tooling

- **nnsight** for activation extraction (chosen over TransformerLens: traces
  native HF models directly, so 4-bit quantization and multi-architecture
  support work without a model-conversion step — important given local GPU is
  a 6GB RTX 4050 and the plan requires 3 different 7-9B model families).
- PyTorch, HF Transformers/Accelerate/bitsandbytes, HF Datasets.

## Datasets

- [AdvBench](https://github.com/llm-attacks/llm-attacks) — harmful instructions
- [HarmBench](https://github.com/centerforaisafety/HarmBench) — standardized red-teaming eval set
- [JailbreakBench / JBB-Behaviors](https://github.com/JailbreakBench/jailbreakbench) — curated behaviors + real jailbreak artifacts
- [XSTest](https://github.com/paul-rottger/xstest) — safe-but-superficially-harmful prompts, for measuring over-refusal

## Models

Open-weight, via Hugging Face. Small model for pipeline sanity checks:
`Qwen2.5-1.5B-Instruct`. Main cross-model set (later phases):
`Llama-3-8B-Instruct`, `Gemma-2-9B-it`, `Qwen2.5-7B-Instruct` (4-bit quantized
to fit local VRAM; larger runs via nnsight remote execution if needed).

## Repo layout

```
src/
  data/          dataset loading + harmful/harmless prompt pairing
  activations/   activation extraction via nnsight
  direction/     refusal-direction computation, ablation, activation addition
scripts/         standalone runnable pipeline stages
notebooks/       exploratory analysis
results/         metrics, figures (gitignored for large artifacts)
tests/
```

