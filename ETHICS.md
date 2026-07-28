# Research Ethics Self-Assessment

**Student:** Natalia Muça
**Programme:** MSc Artificial Intelligence (cybersecurity profile)
**Thesis title:** Jailbreak detection via internal model activations
**Repository:** github.com/natalimuca/jailbreak-activation-detection
**Date:** [submission date]

## 1. Summary of the research

I am researching whether a jailbreak / harmful-intent prompt detector built on
an LLM's internal activations is more robust than surface-level (keyword or
perplexity-based) filtering. Surface filters can be evaded by disguising a
harmful request as fiction, roleplay, or a hypothetical -- an
activation-based detector, reading the model's internal state rather than the
prompt's surface wording, is intended to be harder to evade this way.

This is **defensive security research**. The goal is a better detector, not a
new attack technique. I am not developing, publishing, or disclosing any
novel jailbreak method as part of this work -- every adversarial technique I
use (PAIR, GCG) is an already-published attack I am reusing exactly as its
original authors intended, purely to test detector robustness.

## 2. Justification and expected contribution

The approach builds on a real, published finding (Arditi et al., NeurIPS
2024, "Refusal in Language Models Is Mediated by a Single Direction") showing
that refusal behaviour in open-weight chat models is controlled by a single
activation-space direction. My contribution is extending this into an
evaluated, calibrated detector -- with cross-model generalisation testing,
adversarial-robustness testing, and statistical significance testing -- which
the literature I surveyed had not yet done rigorously at the time I started
(see LITERATURE.md).

## 3. Data sources

All harmful-instruction prompts used in this project come from four
established, publicly published AI-safety benchmarks, used exactly as their
authors intend and cited throughout my write-up:

- AdvBench (Zou et al. 2023, github.com/llm-attacks/llm-attacks)
- HarmBench (Mazeika et al. 2024, github.com/centerforaisafety/HarmBench)
- JailbreakBench / JBB-Behaviors (Chao et al. 2024, NeurIPS Datasets and Benchmarks track)
- XSTest (Röttger et al. 2023), used for the over-refusal / false-positive side of evaluation

I have not authored, sourced, or scraped any harmful content from elsewhere.
Every harmful prompt in this project already exists in a public, citable
safety-research benchmark that is itself used by numerous other published
papers in this literature.

## 4. Methodology that generates harmful content locally

Testing whether a candidate activation direction *causally* controls refusal
(rather than merely correlating with it) requires running the benchmark
prompts through open-weight models with the refusal mechanism deliberately
disabled, and observing whether the model complies. This step does produce
real harmful completions, generated and stored temporarily on my own local
machine during experiments.

**How I handle this:**
- Raw model completions are never committed to the project's git repository
  or shared anywhere (`results/` is gitignored).
- Only aggregate statistics (refusal rates and confidence intervals, no
  actual harmful text) are recorded in the project's version-controlled
  results documentation.
- Generated harmful completions are not used for any purpose beyond
  confirming the causal effect exists for this research; they are not
  published, distributed, or repurposed.

## 5. Models used

All models are open-weight, instruction-tuned checkpoints already publicly
released and downloadable by anyone (models from the Qwen, Llama, Gemma,
SmolLM2, and DeepSeek families -- see DECISIONS.md for the full list and the
reasoning behind each addition). This research does not create any new
capability: these models' ability to produce harmful content once their
safety training is bypassed is already a known, publicly documented property
of each model, and is precisely the property the underlying published
research (Arditi et al. 2024) demonstrates and that I am building on.

## 6. Risk assessment

I assess this project as **low risk**:

- It uses only established public benchmarks and already-public open-weight
  models -- no new harmful capability or novel attack is created.
- No content generated during this research is intended for, or released
  for, any use outside verifying the detector's own effectiveness.
- The end goal is protective (a better jailbreak detector), not harmful.

Two points I want to flag explicitly to my advisor/department for review:

1. **Local storage of model-generated harmful text during experiments.** I
   mitigate this via `.gitignore` (never committed) and by not sharing raw
   completions with anyone; only aggregate statistics leave my machine.
2. **Eventual publication of a working detector's methodology.** This is
   standard practice in this research area -- every paper I surveyed in
   LITERATURE.md publishes comparable methodology -- but I want this
   explicitly acknowledged as part of my ethics review, not assumed.

## 7. Dissemination plan

I intend to publish this thesis and its supporting repository as a portfolio
piece documenting a defensive security methodology, following the same
disclosure norms as the published papers I cite and build on (results and
methodology published; no novel attack methodology; no raw harmful
completions released).

## 8. Declaration

I confirm that the above accurately describes my research as conducted, that
all harmful-instruction data used is drawn from established public
benchmarks rather than authored by me, and that I have not disclosed or
intend to disclose any novel jailbreak technique as part of this work.

**Signature:** ______________________ **Date:** ______________
