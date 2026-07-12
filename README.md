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

This project extends that finding in three directions:

1. **Sparse, interpretable features over one dense direction** — rather than
   a single residual-stream direction, a pretrained sparse autoencoder (SAE)
   is used to find the specific sparse features (out of tens of thousands)
   causally responsible for refusal, selected via cosine similarity to the
   refusal direction and ranked by attribution-patching causal effect, then
   validated by suppressing them and measuring the effect on real generated
   completions.
2. **Cross-model generalization** — does a detector trained on one model's
   internals transfer to a different model family/size, or does it need
   per-model retraining?
3. **Engineering rigor** — calibration, formal paired significance testing
   (McNemar's, DeLong's, Cochran's Q) rather than eyeballing confidence
   intervals, an honest head-to-head comparison against baselines (keyword
   filter, Olmo-3-1025-7B perplexity filter) on a held-out adversarial
   paraphrase set built from real, published JailbreakBench attack
   artifacts, and false-positive ("safety tax") measurement on XSTest's
   safe-but-scary-looking prompts.

Methodology and results for the single-direction reproduction, the
SAE-feature detector, and the baseline/adversarial detector comparison are
in [METHODOLOGY.md](METHODOLOGY.md) and [RESULTS.md](RESULTS.md); design
decisions and open questions in [DECISIONS.md](DECISIONS.md).

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
for fast iteration on the single-direction reproduction; `Qwen3-8B` (4-bit
quantized to fit a 6GB local GPU) for the SAE-feature detector, paired with
[Qwen-Scope](https://arxiv.org/pdf/2605.11887)'s pretrained sparse
autoencoders. A larger cross-model set (`Llama-3-8B-Instruct`, `Gemma-2-9B-it`)
is planned for the cross-model generalization question above, pending HF
gating approval.

## Repo layout

```
src/
  data/          dataset loading + harmful/harmless prompt pairing
  activations/   activation extraction and caching via nnsight
  direction/     refusal-direction computation, ablation, activation addition,
                 SAE feature-suppression generation, refusal metric
  sae/           pretrained SAE loading, feature selection, causal ranking
  baselines/     keyword-filter and perplexity-filter prompt classifiers
  detectors/     dense-direction and SAE-feature prompt classifiers
  eval/          shared detector metrics, adversarial paraphrase set
scripts/         standalone runnable pipeline stages
notebooks/       exploratory analysis
results/         metrics, figures, activation caches (gitignored)
tests/
```
