# Jailbreak Detection via Internal Activations

**Detecting harmful-intent prompts from an LLM's internal activations instead of
surface-level (keyword or perplexity) filtering, with a rigorous, honestly-hedged
cross-model comparison across 6 open-weight chat models (1.5B to 9B parameters).**

## Technology Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-4--bit_quant-EE4C2C)
![nnsight](https://img.shields.io/badge/nnsight-activation_patching-8E44AD)
![Transformers](https://img.shields.io/badge/HF_Transformers-model_loading-FFB000)
![Accelerate](https://img.shields.io/badge/Accelerate-device_mapping-FFB000)
![bitsandbytes](https://img.shields.io/badge/bitsandbytes-4--bit_quant-FFB000)
![NumPy](https://img.shields.io/badge/NumPy-latest-013243)
![pandas](https://img.shields.io/badge/pandas-latest-150458)
![SciPy](https://img.shields.io/badge/SciPy-latest-0054A6)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E)
![Statistics](https://img.shields.io/badge/Statistics-Wilson_%2F_McNemar_%2F_DeLong_%2F_Cochran-2E86C1)
![FastAPI](https://img.shields.io/badge/FastAPI-live_inference-009688)
![uvicorn](https://img.shields.io/badge/uvicorn-ASGI_server-009688)
![pytest](https://img.shields.io/badge/pytest-175_passing-0A9EDC)
![matplotlib](https://img.shields.io/badge/matplotlib-figures-11557C)
![seaborn](https://img.shields.io/badge/seaborn-figures-4C72B0)

| Area | Tools |
|---|---|
| Language & core | Python 3, PyTorch, NumPy, pandas, scikit-learn, SciPy |
| Model internals | [nnsight](https://nnsight.net/) (activation extraction/patching), HF Transformers, Accelerate, bitsandbytes (4-bit quantization) |
| Sparse autoencoders | Qwen-Scope, LlamaScope, GemmaScope, EleutherAI's suite (pretrained, per-model) |
| Statistics | Wilson score intervals, McNemar's exact test, DeLong's test, Cochran's Q, exact/permutation tests |
| API | FastAPI, uvicorn |
| Testing | pytest (156 fast tests + 19 real-GPU regression tests) |
| Visualization | matplotlib, seaborn |

Logic lives in tested `src/` modules; `scripts/` are standalone, runnable pipeline
stages, one per experiment, so every number in this README traces back to a script
and a saved result file, not a notebook cell. `notebooks/mechanism.ipynb` and
`notebooks/transfer.ipynb` turn a subset of those results into real, executed
figures with formulas and interpretation, for a faster read than the full reports.

## Core idea

Builds on Arditi et al., "Refusal in Language Models Is Mediated by a Single
Direction" (NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)):
a single direction in residual-stream activation space causally controls refusal
across open chat models. Surface-level detectors can be fooled by disguising a
harmful request as fiction or roleplay, since they only ever read the prompt's
wording. An activation-based detector instead reads whatever internal state the
model itself uses to decide "this is harmful," which should be harder to fool
with surface-level disguise since the disguise never has to change what the
request actually means.

## Research question

Does a single dense refusal direction hold up as a detector and as a causal
mechanism across model families and scales? Does a sparse-autoencoder (SAE)
feature basis do better where one is available? And does either approach
transfer across models?

Headline findings so far:

- **Both activation-based detectors crush surface-level baselines on clean
  prompts**: dense-direction and SAE-feature detectors hit **AUROC 0.91–0.99**
  across all 6 models tested, versus keyword filtering (~0.60) and perplexity
  filtering (~0.52, essentially chance).
- **Llama-3.1-8B-Instruct has the best classifier accuracy of any model tested
  (93.1%, AUROC 0.989)**, yet is one of the weakest causal mechanisms: its full
  dense direction barely moves refusal when ablated (92%→88%, not significant).
  **This is now resolved, not just documented.** Decomposing the direction into
  the component aligned with its own top SAE feature and the orthogonal
  remainder shows the small aligned piece alone crashes refusal far more
  (92%→38%, p<0.0001) than the full direction does, while the orthogonal
  remainder (98% of the vector's magnitude) does nothing at all (92%→92%,
  identical to baseline). Qwen3-8B shows no such asymmetry: both components
  independently crash its refusal, matching the full direction.
- **A single SAE feature reproduces almost all of Llama's refusal-ablation
  effect** (baseline 92–98%→10% from one feature alone, zero degenerate
  completions), whereas the same test on Qwen3-8B needs the *full* top-15
  feature set (top-1 alone: no effect at all, 82%→84%). This
  concentration-vs-distribution asymmetry is the throughline behind both the
  causal-gap result above and the paraphrase-robustness mechanism below.
- **PAIR-paraphrase robustness has a real, statistically confirmed 6-model
  spread** (Cochran's Q = 34.44, df = 5, p = 1.94e-6): SmolLM2 (90.5%) >
  Llama-3.1-8B (66.7%) > gemma-2-9b-it (47.6%) > Qwen3-8B (42.9%) >
  Qwen2.5-1.5B (38.1%) > **DeepSeek-R1-Distill-Qwen-1.5B (9.5%)**, a dramatic
  new low. A **controlled wrapper-swap factorial** (real core-request ×
  wrapper-framing design, not just observational PAIR transcripts) explains
  *why* for the two models examined in full depth: Qwen3-8B's dominant feature
  tracks wrapper framing (η²=0.656, p=0.0001), Llama's tracks the request's
  actual content (η²=0.407→0.366 once replicated, p=0.015/<0.0001). A same-day
  replicated design with real per-cell phrasing variants confirms Llama's large
  residual term is a genuine core-by-category interaction (42.4% of variance,
  p<0.0001 by both an F-test and an independent permutation check), not just
  measurement noise.
- **A sixth model, DeepSeek-R1-Distill-Qwen-1.5B, is diffuse across every
  measurement approach used in this project**: a genuinely null
  activation-addition result, the weakest dense-direction detector of all six
  (84.7%, AUROC 0.911), by far the worst PAIR robustness, and an SAE-feature
  detector that barely fires at all (4.4% VAL recall). Four independent angles
  converging on the same description is itself the finding.
- **Cross-model direction transfer, now tested on two architecture-matched
  pairs, shows no evidence of transfer on either one.** Qwen3-8B↔Llama-3.1-8B
  (both 4096-d) gives a clean no-transfer result for Qwen3-8B on both necessity
  and sufficiency; a second pair, Qwen2.5-1.5B↔DeepSeek (both 1536-d, found
  already sitting in this project's cached models), replicates the same clean
  no-transfer pattern on its well-powered side (Qwen2.5-1.5B: 96%→4% own vs.
  96%→90% foreign, p=0.0 vs. p=0.25).
- **Necessity (ablation) generalizes across model scale far more robustly than
  sufficiency (activation addition)**: a pattern first seen at 1.5–1.7B scale,
  confirmed again at 8–9B scale, and most extreme for DeepSeek (a genuine
  zero-effect sufficiency null against a real, if floor-constrained, necessity
  signal).

## How It Works

Two different detectors are built on top of the same idea: a model's own
internal activations encode whether it "thinks" a request is harmful, before
it ever writes a word of its response.

**Dense-direction detector.** For each model, harmful and harmless prompts are
formatted with the model's own chat template and passed through it; the
residual-stream activation at the last prompt token is captured at the output
of every decoder layer. For a candidate layer, the refusal direction is simply
the difference of means: `direction = mean(harmful_activations) -
mean(harmless_activations)`. As a classifier, a new prompt's activation is
projected onto this direction; the projection is thresholded against a value
calibrated on a held-out validation split. As a causal mechanism, two
interventions are tested: **ablation** projects the direction's contribution
out of the residual stream at every generated token (does removing it
suppress refusal, i.e. necessity?), and **activation addition** adds the raw
direction, scaled by a calibrated coefficient, into the residual stream (does
adding it induce refusal, i.e. sufficiency?). Both are validated by generating
real completions and scoring them, not by trusting the intervention's
attribution score alone.

**SAE-feature detector.** Where the dense direction finds one linear axis,
this asks a finer question: which specific sparse, interpretable components,
out of tens of thousands per layer, are causally responsible for refusal.
Using each model's own pretrained sparse autoencoder (SAE), candidate features
are first screened by cosine similarity to the dense direction (the top 10 per
layer), then causally ranked via attribution patching (integrated gradients on
a differentiable refusal-vs-compliance logit-difference metric), keeping the
top 20 overall. Causal validation ablates each ranked feature's own
contribution to the residual stream and measures the resulting refusal-rate
change through real generation. As a classifier, a prompt's score is the sum
of its encoded activations across the top-15 causally-ranked (layer, feature)
pairs.

Both detectors are then stress-tested the same way: a held-out clean test
split for accuracy/AUROC, an XSTest-derived over-refusal check, and a real
adversarial paraphrase set built from published JailbreakBench attack
artifacts (not self-authored jailbreak templates), split into PAIR (fluent
roleplay/fictional-framing paraphrase) and GCG (gibberish adversarial suffix),
since they represent genuinely different failure modes. Every close
comparison is backed by a formal paired significance test (McNemar's exact
test, DeLong's test, Cochran's Q), not an eyeballed confidence-interval
overlap.

## Key Results

### Dense-direction detector: cross-model comparison (6 models)

Layer selected and threshold calibrated on VAL, final metrics on held-out TEST
(n=288: 158 harmful, 130 harmless), same 35-prompt real-JailbreakBench adversarial
paraphrase set reused across all six models.

| model | layer | TEST accuracy | TEST AUROC | XSTest-safe correctly-not-flagged | PAIR detection |
|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 7 | 84.7% | 0.911 | **100.0%** | **9.5%** |
| Qwen2.5-1.5B-Instruct | 20 | 89.6% | 0.970 | 75.7% | 38.1% |
| SmolLM2-1.7B-Instruct | 14 | 87.8% | 0.945 | **100.0%** | **90.5%** |
| Qwen3-8B | 23 | 88.9% | 0.983 | 94.6% | 42.9% |
| **Llama-3.1-8B-Instruct** | 27 | **93.1%** | **0.989** | 97.3% | 66.7% |
| Gemma-2-9B-it | 34 | 93.1% | 0.984 | 89.2% | 47.6% |

Five of six models land in a comparably strong 87.8–93.1% accuracy / 0.94–0.99
AUROC band on clean, in-distribution prompts; DeepSeek is the outlier, weakest of
all six but still a working classifier. The PAIR-robustness spread is where
models genuinely diverge, DeepSeek most dramatically (Cochran's Q = 34.44,
df = 5, **p = 1.94e-6**).

### Are the differences real? (formal significance testing)

A raw metric-gap table invites over-reading small differences, so every
model-vs-model or detector-vs-detector claim here is backed by a paired test on
the same items, not two independently-eyeballed confidence intervals:

- **Dense-direction vs. SAE-feature, TEST AUROC** (DeLong's test, paired): not
  significant for Qwen3-8B (p=0.068), significant favoring dense-direction for
  both Llama-3.1-8B (p=0.024) and gemma-2-9b-it (p=0.0063). A genuinely
  different outcome per model, not smoothed into one headline.
- **Dense-direction vs. SAE-feature on the adversarial set** (McNemar's exact):
  not significant for Qwen3-8B (p=0.5) or Llama-3.1-8B (p=0.25, though
  SAE-feature numerically *beats* dense-direction here, 80.9% vs. 66.7%, the
  one case in this project matching the source paper's original claim), except
  gemma-2-9b-it (p=0.0156, all 7 discordant pairs favor dense).
- **PAIR-robustness across all 6 models** (Cochran's Q, generalizes McNemar's to
  *k* related classifiers on the same items): Q=34.44, df=5, **p=1.94e-6**.
- **Llama's own dense-direction causal necessity, independently replicated at
  n=75**: baseline 96.0% vs. own-ablation 94.7%, McNemar p=1.0. This does not
  confirm the original n=50 reading (92%→88%, thought to be "real but small");
  the honest read is that the original observation was sample noise.

### SAE-feature detector: causal validation and cross-model extension

Feature suppression validated via real generation (greedy decoding), not just
attribution scores. K=15 (the original default) independently reproduced as
each model's own true minimum.

| model | baseline refusal | top-1 feature alone | best (top-K) | concentration |
|---|---|---|---|---|
| Qwen3-8B | 82.0–84.0% | 84.0% (no effect) | 24.0% @ top-15 | **distributed** across the set |
| Llama-3.1-8B-Instruct | 92.0–98.0%\* | **10.0%** | 0.0% @ top-15 | **concentrated** in 1 feature |
| gemma-2-9b-it | 96.0% | 94.0% | 82.0% @ top-15/20 | modest, gradual decline |
| DeepSeek-R1-Distill-Qwen-1.5B | 14.3% | — | — | detector barely fires at all (4.4% VAL recall) |

\*Llama's baseline appears at several values across this project's history after
a real curly-apostrophe classifier bug was found and fixed (see reports/DECISIONS.md);
every reading is reported honestly where it appears rather than silently
normalized.

TEST AUROC for the SAE-feature detector itself: Qwen3-8B 0.975, Llama-3.1-8B
0.978, gemma-2-9b-it 0.966, all in the same high-0.9x band as the dense
direction. DeepSeek's SAE-feature detector is a genuine architectural failure,
not a calibration bug: its SAE's hard top-32-of-65536 sparsity means only a
handful of the 15 hand-selected features ever fire on a given prompt at all.

### Baseline detectors and adversarial evaluation (activation rows: Qwen3-8B)

| detector | TEST accuracy | TEST AUROC | adversarial pooled (n=35) | GCG (n=14) | PAIR (n=21) |
|---|---|---|---|---|---|
| keyword filter | 56.6% | 0.603 | 17.1% | 7.1% | 23.8% |
| perplexity filter (Olmo-3-1025-7B) | 54.9% | 0.520 | 40.0% | **100.0%** | **0.0%** |
| LLM judge (Llama-3.3-70B, text only) | **94.1%** | 0.954 | 71.4% | 92.9% | 57.1% |
| **dense-direction** | 88.9% | **0.983** | 62.9% | 92.9% | 42.9% |
| **SAE-feature (top-15)** | 87.8% | 0.975 | 57.1% | 92.9% | 33.3% |

**GCG (gibberish-suffix) detection is a perfect 100% across five independent
perplexity backbones tried**: strong, convergent evidence this is a real,
model-agnostic property of the attack, not an artifact of any one scorer.
**PAIR (fluent paraphrase) detection is 0.0% for every backbone except the
original, weakest one**: perplexity filtering structurally cannot catch fluent
paraphrase attacks. Every detector, including the activation-based ones,
degrades sharply on PAIR relative to clean TEST performance.

**A frontier LLM prompted as a classifier is a genuinely strong text-only
baseline, and the comparison splits -- tested on three models.**
Dense-direction wins threshold-independent ranking on all three, always
significantly (AUROC 0.983/0.989/0.984 vs the judge's 0.954; DeLong
p=0.0041/0.0015/0.0053). The judge is more accurate at its operating point on
two of the three (Qwen3-8B +5.2pp, Llama +1.0pp) and indistinguishable on
gemma (p=1.0). **On PAIR no difference is significant on any model**,
including Llama, where dense-direction's 66.7% and SAE-feature's 81.0% against
the judge's 57.1% still gives McNemar p=0.73 at n=21. Better ranking alongside
worse thresholded accuracy points at threshold selection, not signal quality:
the judge's scores are bimodal (0 or 100, threshold 100.0), so it is already a
binary classifier with little left to rank, while the activation detectors'
continuous scores rank better but sit on a suboptimal Youden-J operating point.
The judge also over-refuses: 13.5% false positives on XSTest's
harmless-but-scary prompts, against dense-direction's 2.7%.

### Cross-model direction transfer

Does a direction fit on one model do anything applied to a *different* model's
activations? Tested on both architecture-matched pairs currently available,
in both causal directions: necessity (ablation) and sufficiency (activation
addition).

**Pair 1: Qwen3-8B ↔ Llama-3.1-8B-Instruct (both 4096-d), necessity.**

| | baseline | own-ablation | foreign-ablation |
|---|---|---|---|
| Qwen3-8B (foreign = Llama's direction) | 84% | **8%** | 84% (no effect) |
| Llama-3.1-8B (foreign = Qwen's direction) | 92% | 88% (n.s., p=0.5; 96.0%→94.7% at n=75) | 92% |

**Pair 1, sufficiency**, using each model's already-generated baseline/own-addition
completions plus a freshly-calibrated alpha for the foreign direction:

| | baseline | own addition | foreign addition |
|---|---|---|---|
| Qwen3-8B (foreign = Llama's direction) | 6.0% | **70.0%** | 6.0% (p=1.0 vs. baseline) |
| Llama-3.1-8B (foreign = Qwen's direction) | 10.0% | 34.0% (p=0.0005) | 6.0% (p=0.5 vs. baseline) |

**Qwen3-8B: a clean, unambiguous no-transfer result on both axes.** **Llama-3.1-8B:
necessity is underpowered** (own-direction effect too weak to distinguish from
noise), **but sufficiency resolves it**: own vs. foreign addition is clearly,
significantly different (p<0.001), and foreign has no detectable effect at all.

**Pair 2: Qwen2.5-1.5B-Instruct ↔ DeepSeek-R1-Distill-Qwen-1.5B (both 1536-d),
necessity only** (found already sitting in this project's cached models, no new
downloads; an asymmetric design: Qwen2.5-1.5B's standard 40-token budget vs.
DeepSeek's 2048-token whole-generation budget with truncation-aware McNemar
comparisons):

| | baseline | own-ablation | foreign-ablation |
|---|---|---|---|
| Qwen2.5-1.5B (foreign = DeepSeek's direction) | 96.0% | **4.0%** | 90.0% (p=0.25, n.s.) |
| DeepSeek (foreign = Qwen2.5-1.5B's direction) | 5.9% (n=17/30) | 5.0% (n=20/30) | 0.0% (n=13/30) |

Qwen2.5-1.5B replicates Pair 1's clean no-transfer pattern exactly. DeepSeek as
the target is genuinely underpowered (near-floor baseline, heavy truncation
shrinking paired samples to single digits); reported honestly as inconclusive,
not oversold as a second no-transfer data point.

### The Llama causal-gap investigation: resolved

Why is Llama-3.1-8B's dense-direction approach weak on both necessity and
sufficiency, while its SAE-feature approach ablates refusal dramatically from a
single feature? Two candidate mechanical explanations were tested and one ruled
out (`scripts/analyze_llama_causal_gap.py`) before a follow-up experiment closed
the gap for real:

- **Ruled out**: raw direction magnitude. Llama's direction has the smallest raw
  norm of any model tested, but normalized by its own ambient activation scale
  at that layer, it's actually the *largest* fraction (70%) of the models
  checked, the opposite of what a magnitude-dilution story predicts.
- **Inconclusive on its own**: cosine similarity between the dense direction and
  the top causal SAE feature is ~0.20 for *both* Llama and Qwen3-8B: real
  alignment above a random baseline, but identical across the model where the
  dense direction works and the one where it doesn't.
- **Resolved by component decomposition** (`scripts/decompose_llama_causal_gap.py`):
  splitting Llama's unit-normalized dense direction into the component aligned
  with its own top causal SAE feature and the orthogonal remainder, then
  ablating each separately (identical 50-prompt held-out set as the transfer
  experiment above):

  | Llama-3.1-8B-Instruct | refusal rate | vs. baseline |
  |---|---|---|
  | baseline | 92.0% | — |
  | full-direction ablation | 88.0% | p=0.5 (n.s.) |
  | parallel-component ablation alone | **38.0%** | p<0.0001 |
  | orthogonal-component ablation alone | 92.0% | p=1.0 (identical) |

  The small feature-aligned piece (only ~20% of the unit direction's weight by
  cosine) carries essentially all of the causal effect; the large orthogonal
  remainder (98% of the vector's magnitude) does nothing at all. Run as a
  contrast, Qwen3-8B shows no such asymmetry: both components independently
  crash its refusal to near-zero, matching the full direction. This directly
  explains Llama's best-classifier/weakest-causal-mechanism anomaly: the dense
  direction is dominated by norm from a causally-inert axis, correlating well
  enough with the refusal label to classify accurately without being a good
  causal handle on the mechanism that actually produces refusal.

### Wrapper-swap variance decomposition: what does a dominant feature track?

A controlled factorial (10 real core requests × wrapper framing conditions,
each model's own top causal SAE feature read via one forward pass, no
generation) disentangles content from framing directly, closing a qualitative
token-attribution finding with real statistical power:

| model | η² (core) | η² (wrapper/category) | η² (interaction/residual) |
|---|---|---|---|
| Qwen3-8B | 0.227 (p=0.0001) | 0.656 (p=0.0001) | 0.117 |
| Llama-3.1-8B-Instruct | 0.407 (p=0.0148) | 0.029 (n.s.) | 0.564 |

Qwen3-8B's dominant feature is driven by wrapper/framing identity; Llama's by
the underlying request's content, exactly the asymmetry a qualitative
token-level reading suggested, now backed by a within-block permutation test.
Llama's large residual was flagged as unexplained and closed the same day with
a **replicated design** (real per-cell phrasing variants, not just repeated
readings):

| model | η² core | η² category | η² interaction | interaction F-test p | interaction permutation p |
|---|---|---|---|---|---|
| Qwen3-8B | 0.349 | 0.217 | 0.086 | 0.813 (n.s.) | 0.680 (n.s.) |
| Llama-3.1-8B-Instruct | 0.366 | 0.004 (n.s.) | **0.424** | **<0.0001** | **0.0001** |

Llama's large residual really was substantial genuine interaction (core-request
and wrapper framing interact non-additively for this model), not primarily
measurement idiosyncrasy: both the classical F-test (now valid with a real
per-cell error term) and an independent Freedman-Lane permutation check agree.
Qwen3-8B's core and category effects remain additive, no real interaction.

Five further rounds dug into *why* Llama's interaction concentrates in
specific requests rather than spreading evenly (per-cell decomposition, a
task-type hypothesis, a blind multi-feature search, a literature-motivated
salience test, and a token-level attribution read); see the Roadmap below for
what held up and what didn't.

### What this adds up to

Passive classification (does this prompt look harmful) generalizes across
model family and scale far more cleanly than causal mechanism (does this
direction actually *cause* refusal) or cross-model transfer do. Llama's
best-classifier/weakest-mechanism split, DeepSeek's diffuseness across every
angle tried, and the total absence of transfer on two independent
architecture-matched pairs are three separate illustrations of the same
underlying point: a detector that works well is not the same claim as a
detector that has found the model's real causal mechanism, and this project's
own results keep landing on the side of "these are genuinely different
properties," not "close enough in practice."

## Data

- [AdvBench](https://github.com/llm-attacks/llm-attacks): harmful instructions
- [HarmBench](https://github.com/centerforaisafety/HarmBench): standardized red-teaming eval set
- [JailbreakBench / JBB-Behaviors](https://github.com/JailbreakBench/jailbreakbench): curated behaviors + real jailbreak artifacts
- [XSTest](https://github.com/paul-rottger/xstest): safe-but-superficially-harmful prompts, for measuring over-refusal
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca): general-purpose harmless instructions

Full source/usage table in [DATASETS.md](DATASETS.md). 1990 raw prompts → 1922
after deduplication, stratified train/val/test split via a reproducible
hash-based manifest (`data/splits/corpus_split_v1.json`). Activations cached per
model (`results/activations/*.pt`, gitignored, multi-GB).

## Models

Open-weight, via Hugging Face: `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`
for fast iteration; `Qwen3-8B`, `Llama-3.1-8B-Instruct`, `gemma-2-9b-it` (all
4-bit quantized to fit a 6GB local GPU) for the full cross-model comparison;
`DeepSeek-R1-Distill-Qwen-1.5B` added last, cheap to run (1.5B, no quantization
needed) but requiring a genuine methodology adaptation: its chat template
always reasons inside a `<think>` block with no disable switch, so every
downstream measurement resolves through the full reasoning trace first rather
than reading the wrong position (see reports/METHODOLOGY.md). The
dense-direction detector runs on all 6 models; the SAE-feature detector runs on
the 4 models with a pretrained sparse-autoencoder suite (Qwen-Scope, LlamaScope,
GemmaScope, EleutherAI's suite for DeepSeek). Qwen2.5-1.5B and SmolLM2 have no
published SAE suite, so the SAE-feature detector isn't available for them
(reported as an honest gap, not hidden).

## Repo Structure

```
src/
  data/          dataset loading + harmful/harmless prompt pairing
  activations/   activation extraction and caching via nnsight
  direction/     refusal-direction computation, ablation, activation addition,
                 SAE feature-suppression generation, refusal metric
  sae/           pretrained SAE loading, feature selection, causal ranking
  baselines/     keyword-filter, perplexity-filter, and LLM-judge prompt classifiers
  detectors/     dense-direction and SAE-feature prompt classifiers
  eval/          shared detector metrics, adversarial paraphrase set
  api/           FastAPI backend serving live per-prompt detector inference
scripts/         standalone runnable pipeline stages (see scripts/README.md)
notebooks/       mechanism.ipynb (within-model causal mechanism), transfer.ipynb
                 (cross-model transfer): real executed figures over saved results
reports/         DECISIONS.md, RESULTS.md, METHODOLOGY.md, ETHICS.md, figures/
results/         metrics, figures, activation caches (gitignored)
tests/           156 fast tests (CI) + 19 real-GPU regression tests
```

Full rationale for every methodology choice is in [DECISIONS.md](reports/DECISIONS.md);
results are in [RESULTS.md](reports/RESULTS.md); how each technique works is in
[METHODOLOGY.md](reports/METHODOLOGY.md); ethics/safety handling is in
[ETHICS.md](reports/ETHICS.md).

## Reproduce

```bash
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cu130
pytest -m "not model"          # 156 fast tests, no GPU needed
```

The pipeline runs as standalone scripts in a fixed order, not one orchestrated
job; each stage reads what an earlier stage already saved to `results/`
(gitignored) and writes its own new file there. The full script-to-experiment
mapping lives in `scripts/README.md`; this is the short version:

```bash
# 1. Reproduce the foundational finding on two small models (fast, ~minutes)
python scripts/reproduce_direction.py

# 2. Extract and cache activations for a target model (GPU, one pass per model;
#    the dataset split is built and applied automatically the first time)
python scripts/extract_activations.py Qwen/Qwen3-8B --4bit

# 3. SAE-feature causal ranking and validation (Qwen3-8B; needs its own SAE suite)
python scripts/rank_sae_features.py
python scripts/validate_sae_features.py

# 4. Head-to-head against baselines on clean + adversarial prompts
python scripts/compare_detectors.py
```

Most scripts are cheap: activation extraction and SAE-feature ranking are
single forward passes, done in minutes even on a 6GB GPU. Causal-validation
scripts that generate real completions (ablation, activation addition,
suppression) are the slow ones, since they run the model's own generation
loop rather than one forward pass; budget real GPU time for those, not just a
few minutes. Every model up to 9B fits 4-bit quantized on a 6GB card; nothing
in this project needs cloud compute. Each script writes its aggregate results
to `results/<name>.json`; raw model completions are never committed (see
`.gitignore`), only aggregate statistics are.

## Statistical Rigor

- **Greedy (`do_sample=False`) decoding** for every causal-validation generation:
  every model's default `GenerationConfig` otherwise samples, which would
  conflate the intervention's true effect with sampling noise. Discovered mid-way
  through, then every earlier result was re-run on the same deterministic
  footing rather than left inconsistent.
- **Wilson score intervals** on every reported refusal-rate proportion.
- **Paired tests, not independent-CI eyeballing**: McNemar's exact test for two
  classifiers/conditions on the same items, DeLong's test for paired AUROC,
  Cochran's Q for *k* related classifiers on the same items, each adopted after
  catching an earlier informal comparison that used the wrong tool.
- **Exact/permutation-based p-values at small sample sizes**: used for
  rank-correlation tests at n=5–6 models and for the wrapper-swap factorial's
  interaction tests, where a conventional asymptotic test's own assumptions
  don't hold at these sample sizes.
- **Split discipline**: layer selection and threshold calibration on VAL only,
  TEST untouched until final reporting; a mild leakage pattern in an early run
  was found, checked (made no practical difference), and fixed going forward.
- **Independent replication before trusting a small effect**: Llama's original
  92%→88% ablation reading was re-tested at n=75 on a fresh sample rather than
  taken as settled; it did not replicate. The wrapper-swap factorial's
  unreplicated residual term was similarly re-tested with a genuine per-cell
  replicated design before being trusted as real interaction.
- **Real generation, not just attribution scores**, for every causal-validation
  claim (SAE suppression, dense-direction ablation/addition, component
  decomposition): attribution patching only ranks candidates, the actual
  result is measured by generating real completions and scoring them.

## Testing & Validation

156 fast unit/integration tests (`pytest -m "not model"`, CI-required, no GPU
needed) covering direction computation, ablation/addition math, refusal
classification, detector metrics (Wilson CI, McNemar, DeLong, Cochran's Q), SAE
loading per provider, and the FastAPI backend (stubbed model/tokenizer, matching
this project's own established stub-testing convention). 19 additional
`@pytest.mark.model`-tagged tests run against real small models as a
lower-frequency hardware regression check, not part of the CI-required set.

## Deliverables

**Live-inference API backend** (`src/api/`, done, merged): a FastAPI service
that loads at most one model at a time on a 6GB GPU (evicting/reloading as
requests switch targets) and serves all four detectors (keyword, perplexity,
dense-direction, SAE-feature) for arbitrary prompts, not just the pre-collected
corpus.

**Interactive detector UI** (`webapp/`, merged into `master`): a plain
HTML/JS/CSS page over the API above, plus a Findings dashboard built from this
project's own reports/RESULTS.md numbers, verified live in-browser against
the real GPU. Current with all 6 models. Scoped to detector performance for
the live demo, not a browser for every research finding in this README, so
the newer transfer/wrapper-swap/mechanistic results aren't in it by design.

## Roadmap

**Done:**
- Single-direction reproduction (Qwen2.5-1.5B, SmolLM2): necessity and
  sufficiency both confirmed, non-overlapping CIs vs. baseline.
- Dataset pipeline: 5 datasets unified, deduplicated, stratified split,
  activations cached and cross-model-consistency-checked.
- Dense-direction and SAE-feature detectors extended across all applicable
  models (dense: 6 of 6; SAE-feature: 4 of 6, the ones with a pretrained SAE
  suite): causal ranking, real-generation causal validation, and the
  prompt-classifier reframing, each backed by formal significance testing.
- Baseline detectors + adversarial evaluation: keyword/perplexity baselines,
  head-to-head against both activation-based detectors, real JailbreakBench
  adversarial paraphrase set, XSTest false-positive ("safety tax") measurement.
- DeepSeek-R1-Distill-Qwen-1.5B onboarded as a 6th model, including the
  reasoning-trace methodology adaptation its `<think>`-block chat template
  required. Diffuse across every measurement approach used in this project.
- Cross-model direction transfer, necessity and sufficiency, on **two**
  architecture-matched pairs (Qwen3-8B↔Llama-3.1-8B, Qwen2.5-1.5B↔DeepSeek):
  clean no-transfer on every well-powered comparison.
- **The Llama causal-gap investigation, resolved**: component decomposition
  isolates the exact axis carrying Llama's (otherwise weak) causal effect,
  explaining its best-classifier/weakest-causal-mechanism anomaly directly
  rather than leaving it as an open correlational finding.
- **Wrapper-swap variance decomposition, closed and replicated**: a controlled
  factorial establishes what each model's dominant SAE feature tracks (content
  vs. framing) with real statistical power, then a same-day replicated design
  confirms Llama's large residual term is genuine interaction, not noise.
- PAIR-robustness spread: matched-pair mechanistic chain (margin correlation →
  signal-decay analysis → token-level attribution → the wrapper-swap factorial
  above), from an aggregate correlation to a real, causally-grounded account
  for two of six models.
- **Why Llama's core-by-category interaction concentrates in specific
  requests, investigated across five rounds** (per-cell decomposition, a
  task-type hypothesis, a blind multi-feature search, a literature-motivated
  salience test, and a token-level attribution read). The first four were
  independently rejected as formal hypotheses (task type, word count,
  average word length, keyword-lexicon score, source dataset,
  salience-via-perplexity all fail to predict it, while the interaction
  itself keeps replicating cleanly, strongest yet at η²=0.286, n=48). The
  fifth round found a real qualitative signature at the mechanism level
  instead of another failed surface property: low-interaction requests keep
  the same top-attributed token locked across every wrapper condition, while
  high-interaction requests show the top token swinging between content and
  framing language, most consistently under the fiction wrapper specifically.
  Genuinely inconclusive on the underlying *why*, same as this project's
  standing practice elsewhere (see the Llama dense-vs-SAE PAIR flip above);
  investigated thoroughly and reported honestly rather than left untouched or
  forced into an answer.
- Live-inference API backend, with full test coverage and a real-GPU smoke test.
- **LLM-as-judge baseline** (Llama-3.3-70B, text-only): a strong third
  baseline replacing comparison against weak strawmen alone, evaluated on
  three models with paired significance tests. Activation-based detection
  wins threshold-independent ranking on all three (DeLong p=0.0041/0.0015/
  0.0053); the judge is more accurate at its operating point on two of three;
  no PAIR difference is significant on any model.
- **Threshold reselection experiment**: tests the calibration diagnosis the
  judge comparison produced. Changing only the VAL-fitted threshold rule lifts
  Qwen3-8B's TEST accuracy 88.9% -> 92.0% (McNemar p=0.0002) and is a no-op on
  Llama and gemma, exactly where the diagnosis predicted.
- Interactive frontend: live probe across five detectors, published-attack
  presets from the real JailbreakBench artifacts, token-level attribution
  view over the completed leave-one-out run, corpus composition panel, and
  four evidence charts.
- Ethics documentation split: a public safety-handling document in this repo,
  with the signed institutional submission kept outside it (still only
  submittable by the project owner, not something this repo automates).

**Open decisions:**
- **Whether to adopt the reselected threshold as default.** It is a
  significant improvement on Qwen3-8B and a no-op elsewhere, but adopting it
  means re-running the head-to-head and updating every downstream number, and
  applying it only to the model it helps invites the obvious objection. The
  result is recorded; the decision is deliberately left open.

**Remaining, deliberately not pursued right now:**
- **A reusable automated moralize-vs-comply classifier**: three candidate
  local judge models tried, all failed validation, each a different failure
  mode. A fourth attempt is a plausible next step but carries real risk of a
  fourth distinct failure mode rather than a guaranteed fix, so it isn't
  attempted just to close the gap.
- **Comparing DeepSeek's diffuse representation against other distilled
  models**: would need models genuinely outside this project's current scope
  (nothing to swap in from the existing six), a real scope expansion rather
  than a quick follow-up.
