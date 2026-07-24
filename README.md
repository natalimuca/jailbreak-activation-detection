# Jailbreak Detection via Internal Activations

**Detecting harmful-intent prompts from an LLM's internal activations instead of
surface-level (keyword or perplexity) filtering, with a rigorous, honestly-hedged
cross-model comparison across 5 open-weight chat models (1.5B to 9B parameters).**

## Technology Stack

| Area | Tools |
|---|---|
| Language & core | Python 3, PyTorch, NumPy, pandas, scikit-learn, SciPy |
| Model internals | [nnsight](https://nnsight.net/) (activation extraction/patching), HF Transformers, Accelerate, bitsandbytes (4-bit quantization) |
| Sparse autoencoders | Qwen-Scope, LlamaScope, GemmaScope (pretrained, per-model) |
| Statistics | Wilson score intervals, McNemar's exact test, DeLong's test, Cochran's Q |
| API | FastAPI, uvicorn |
| Testing | pytest (147 fast tests + 11 real-GPU regression tests) |
| Visualization | matplotlib, seaborn |

Logic lives in tested `src/` modules; `scripts/` are standalone, runnable pipeline
stages, one per experiment, so every number in this README traces back to a script
and a saved result file, not a notebook cell.

## Core idea

Builds on Arditi et al., "Refusal in Language Models Is Mediated by a Single
Direction" (NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)):
a single direction in residual-stream activation space causally controls refusal
across open chat models. Surface-level detectors can be fooled by disguising a
harmful request as fiction/roleplay; an activation-based detector reads intent
rather than wording and should be more robust to that disguise.

## Research question

Does a single dense refusal direction hold up as a detector and as a causal
mechanism across model families and scales, and where a sparse-autoencoder (SAE)
feature basis instead, and does either approach transfer across models?

Headline findings so far:

- **Both activation-based detectors crush surface-level baselines on clean prompts**
  — dense-direction and SAE-feature detectors hit **AUROC 0.94–0.99** across all 5
  models tested, versus keyword filtering (~0.60) and perplexity filtering
  (~0.52, essentially chance).
- **Llama-3.1-8B-Instruct has the best classifier accuracy of any model tested
  (93.1%, AUROC 0.989)**, yet is the *only* model where the dense direction's own
  causal necessity (ablation) fails to replicate (96.0%→94.7%, McNemar p=1.0,
  n=75) — a genuine, unresolved tension between "best classifier" and "cleanest
  causal story."
- **A single SAE feature reproduces almost all of Llama's refusal-ablation
  effect** (baseline 86%→10% from one feature alone, zero degenerate
  completions) — where the same test on Qwen3-8B needs the *full* top-15 feature
  set (top-1 alone: no effect at all, 82%→84%). Two candidate mechanical
  explanations for this gap were tested directly against cached data and one was
  ruled out (see "The Llama causal-gap investigation" below) rather than forced
  into a tidy story.
- **PAIR-paraphrase robustness has a real, statistically confirmed 5-model
  spread** (Cochran's Q p=0.0006): SmolLM2 (90.5%) > Llama-3.1-8B (66.7%) >
  Gemma-2-9B (47.6%) > Qwen3-8B (42.9%) > Qwen2.5-1.5B (38.1%) — flagged as an
  open question, not forced into an explanation the data doesn't support.
- **Necessity (ablation) generalizes across model scale far more robustly than
  sufficiency (activation addition)** — a pattern first seen at 1.5–1.7B scale,
  confirmed again independently at 8–9B scale with a different model pair.

## Key Results

### Dense-direction detector — cross-model comparison (5 models)

Layer selected and threshold calibrated on VAL, final metrics on held-out TEST
(n=288: 158 harmful, 130 harmless), same 35-prompt real-JailbreakBench adversarial
paraphrase set reused across all five models.

| model | layer | TEST accuracy | TEST AUROC | XSTest-safe correctly-not-flagged | PAIR detection |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 20 | 89.6% | 0.970 | 75.7% | 38.1% |
| SmolLM2-1.7B-Instruct | 14 | 87.8% | 0.945 | **100.0%** | **90.5%** |
| Qwen3-8B | 23 | 88.9% | 0.983 | 94.6% | 42.9% |
| **Llama-3.1-8B-Instruct** | 27 | **93.1%** | **0.989** | 97.3% | 66.7% |
| Gemma-2-9B-it | 34 | 93.1% | 0.984 | 89.2% | 47.6% |

All five models land in a comparably strong 87.8–93.1% accuracy / 0.94–0.99 AUROC
band on clean, in-distribution prompts — the dense-direction approach isn't
fragile to model choice there. The PAIR-robustness spread above is where models
genuinely diverge (Cochran's Q = 19.52, df = 4, **p = 0.0006**).

### Are the differences real? (formal significance testing)

A raw metric-gap table invites over-reading small differences, so every
model-vs-model or detector-vs-detector claim here is backed by a paired test on
the same items, not two independently-eyeballed confidence intervals:

- **Dense-direction vs. SAE-feature, TEST AUROC** (DeLong's test, paired): not
  significant for Qwen3-8B (p=0.068) or Llama-3.1-8B (p=0.024 is significant,
  actually — dense wins), clearly significant favoring dense-direction for
  gemma-2-9b-it (p=0.0063). A genuinely different outcome per model, not
  smoothed into one headline.
- **Dense-direction vs. SAE-feature on the adversarial set** (McNemar's exact):
  not significant for any of the 3 models with an SAE (Qwen3-8B p=0.5,
  Llama-3.1-8B p=0.25), *except* gemma-2-9b-it (p=0.0156, all 7 discordant pairs
  favor dense).
- **PAIR-robustness across all 5 models** (Cochran's Q, generalizes McNemar's to
  *k* related classifiers on the same items): Q=19.52, df=4, **p=0.0006**.
- **Llama's own dense-direction causal necessity, independently replicated at
  n=75**: baseline 96.0% vs. own-ablation 94.7%, McNemar p=1.0 — does not
  confirm the original n=50 reading (92%→88%, thought to be "real but small");
  the honest read is that the original observation was sample noise.

### SAE-feature detector — causal validation and cross-model extension

Feature suppression validated via real generation (40 tokens, greedy decoding),
not just attribution scores. K=15 (the original default) independently reproduced
as each model's own true minimum:

| model | baseline refusal | top-1 feature alone | best (top-K) | concentration |
|---|---|---|---|---|
| Qwen3-8B | 82.0% | 84.0% (no effect) | 24.0% @ top-15 | **distributed** across the set |
| Llama-3.1-8B-Instruct | 86.0–98.0%\* | **10.0%** | 0.0% @ top-15 | **concentrated** in 1 feature |
| gemma-2-9b-it | 96.0% | 94.0% | 82.0% @ top-15/20 | modest, gradual decline |

\*Llama's baseline was corrected from an original 86.0% after a real
curly-apostrophe classifier bug was found and fixed (see DECISIONS.md);
both readings are reported honestly where they appear historically.

TEST AUROC for the SAE-feature detector itself: Qwen3-8B 0.975, Llama-3.1-8B
0.978, gemma-2-9b-it 0.966 — all in the same high-0.9x band as the dense
direction, confirming nothing broke in the generalization before looking at the
finer, genuinely-different comparisons above.

### Baseline detectors and adversarial evaluation (Qwen3-8B)

| detector | TEST accuracy | TEST AUROC | adversarial pooled (n=35) | GCG (n=14) | PAIR (n=21) |
|---|---|---|---|---|---|
| keyword filter | 56.6% | 0.603 | 17.1% | 7.1% | 23.8% |
| perplexity filter (Olmo-3-1025-7B) | 54.9% | 0.520 | 40.0% | **100.0%** | **0.0%** |
| **dense-direction** | **88.9%** | **0.983** | 62.9% | 92.9% | 42.9% |
| **SAE-feature (top-15)** | **87.8%** | **0.975** | 57.1% | 92.9% | 33.3% |

**GCG (gibberish-suffix) detection is a perfect 100% across five independent
perplexity backbones tried** — strong, convergent evidence this is a real,
model-agnostic property of the attack, not an artifact of any one scorer.
**PAIR (fluent paraphrase) detection is 0.0% for every backbone except the
original, weakest one** — perplexity filtering structurally cannot catch fluent
paraphrase attacks. All four detectors, including the activation-based ones,
degrade sharply on PAIR relative to clean TEST performance.

### Cross-model direction transfer

Does a direction fit on one model do anything applied to a *different* model's
activations? Scoped to the only `d_model`-matched pair (Qwen3-8B ↔
Llama-3.1-8B-Instruct, both 4096). Tested both directions of causal
intervention: necessity (ablation) and sufficiency (activation addition).

**Necessity:**

| | baseline | own-ablation | foreign-ablation |
|---|---|---|---|
| Qwen3-8B (foreign = Llama's direction) | 84% | **8%** | 84% (no effect) |
| Llama-3.1-8B (foreign = Qwen's direction) | 96.0% | 94.7% (n=75, p=1.0) | 92% |

**Qwen3-8B: a clean, unambiguous no-transfer result** — own-direction ablation
works dramatically, the foreign direction does nothing at all. **Llama-3.1-8B:
inconclusive** — its own dense-direction causal necessity doesn't clearly
separate from baseline at this sample size either way, so "own vs. foreign" isn't
a meaningful contrast here yet.

**Sufficiency**, using each model's already-generated baseline/own-addition
completions plus a freshly-calibrated alpha for the foreign direction:

| | baseline | own addition | foreign addition |
|---|---|---|---|
| Qwen3-8B (foreign = Llama's direction) | 6.0% | **70.0%** | 6.0% (p=1.0 vs. baseline) |
| Llama-3.1-8B (foreign = Qwen's direction) | 10.0% | 34.0% | 6.0% (p=0.5 vs. baseline) |

**Foreign-direction addition induces zero refusal above baseline in both
models, at every alpha tested** — not a weak effect, an absent one, with a
clean paired McNemar verdict this time (own vs. foreign p<0.001 for both).
This resolves what the necessity side left inconclusive for Llama: on the
sufficiency side, own and foreign are clearly, significantly different, and
foreign has no detectable effect at all. Both necessity and sufficiency now
point the same way for this one model pair — not generalized further than
that.

### The Llama causal-gap investigation

Why is Llama-3.1-8B's dense-direction approach weak on both necessity (above)
and sufficiency (10%→34%, degenerating from alpha≥2.0), while its SAE-feature
approach ablates refusal dramatically from a single feature? Re-analyzed
entirely from data already on disk (`scripts/analyze_llama_causal_gap.py`, no new
GPU generation) — two candidate explanations tested, one ruled out, one left
genuinely open:

- **Ruled out**: raw direction magnitude. Llama's direction has the smallest raw
  norm of any model tested (19.1), but normalized by its own ambient activation
  scale at that layer, it's actually the *largest* fraction (70%) of the models
  checked — the opposite of what a magnitude-dilution story predicts. Extended
  to gemma-2-9b-it: its ratio (0.525) is unremarkable, right in the middle of
  the pack — confirms Llama specifically is the outlier, not "8-9B models" in
  general.
- **Inconclusive, not differentiating**: cosine similarity between the dense
  direction and the top causal SAE feature is ~0.20 for *both* Llama and
  Qwen3-8B — real alignment above a random-feature baseline (~0.013), but
  identical across the model where the dense direction works and the one where
  it doesn't, so it can't be what separates them. gemma's is notably higher
  (0.367) — a genuine third data point, but gemma has no own-direction
  dense-ablation causal test in this project to correlate it against, and its
  own SAE causal-effect shape (modest, gradual decline, no standout feature) is
  a third pattern entirely, not a blend of Llama's and Qwen3-8B's.
- **What the data does support**: a genuine concentration-vs-distribution
  asymmetry (Key Results above) that co-occurs with which model's dense
  direction ablates/adds cleanly — reported as a real, multi-pronged
  correlational finding, not a proven mechanism.

## Data

- [AdvBench](https://github.com/llm-attacks/llm-attacks) — harmful instructions
- [HarmBench](https://github.com/centerforaisafety/HarmBench) — standardized red-teaming eval set
- [JailbreakBench / JBB-Behaviors](https://github.com/JailbreakBench/jailbreakbench) — curated behaviors + real jailbreak artifacts
- [XSTest](https://github.com/paul-rottger/xstest) — safe-but-superficially-harmful prompts, for measuring over-refusal

1990 raw prompts → 1922 after deduplication, stratified train/val/test split via
a reproducible hash-based manifest (`data/splits/corpus_split_v1.json`).
Activations cached per model (`results/activations/*.pt`, gitignored — multi-GB).

## Models

Open-weight, via Hugging Face: `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`
for fast iteration; `Qwen3-8B`, `Llama-3.1-8B-Instruct`, and `gemma-2-9b-it` (all
4-bit quantized to fit a 6GB local GPU) for the full cross-model comparison. The
dense-direction detector runs on all 5 models; the SAE-feature detector runs on
the 3 8–9B models, each paired with its own pretrained sparse-autoencoder suite
(Qwen-Scope, LlamaScope, GemmaScope) — Qwen2.5-1.5B and SmolLM2 have no published
SAE suite, so the SAE-feature detector isn't available for them (reported as an
honest gap, not hidden).

## Repo Structure

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
  api/           FastAPI backend serving live per-prompt detector inference
scripts/         standalone runnable pipeline stages (see scripts/README.md)
notebooks/       exploratory analysis
results/         metrics, figures, activation caches (gitignored)
tests/           147 fast tests (CI) + 11 real-GPU regression tests
```

Full rationale for every methodology choice is in [DECISIONS.md](DECISIONS.md);
results are in [RESULTS.md](RESULTS.md); how each technique works is in
[METHODOLOGY.md](METHODOLOGY.md).

## Reproduce

```bash
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cu130
pytest -m "not model"          # 147 fast tests, no GPU needed
python scripts/reproduce_direction.py       # single-direction reproduction
python scripts/rank_sae_features.py         # SAE causal ranking (Qwen3-8B)
python scripts/validate_sae_features.py     # SAE causal validation
python scripts/compare_detectors.py         # head-to-head vs. baselines
```

Each script writes its aggregate results to `results/<name>.json`; raw model
completions are never committed (see `.gitignore`) — only aggregate statistics.

## Statistical Rigor

- **Greedy (`do_sample=False`) decoding** for every causal-validation generation
  — every model's default `GenerationConfig` otherwise samples, which would
  conflate the intervention's true effect with sampling noise. Discovered mid-way
  through, then every earlier result was re-run on the same deterministic
  footing rather than left inconsistent.
- **Wilson score intervals** on every reported refusal-rate proportion.
- **Paired tests, not independent-CI eyeballing**: McNemar's exact test for two
  classifiers/conditions on the same items, DeLong's test for paired AUROC,
  Cochran's Q for *k* related classifiers on the same items — each adopted after
  catching an earlier informal comparison that used the wrong tool.
- **Split discipline**: layer selection and threshold calibration on VAL only,
  TEST untouched until final reporting; a mild leakage pattern in an early run
  was found, checked (made no practical difference), and fixed going forward.
- **Independent replication before trusting a small effect**: Llama's original
  92%→88% ablation reading was re-tested at n=75 on a fresh sample rather than
  taken as settled — it did not replicate.
- **Real generation, not just attribution scores**, for every causal-validation
  claim (SAE suppression, dense-direction ablation/addition) — attribution
  patching only ranks candidates; the actual result is measured by generating
  real completions and scoring them.

## Testing & Validation

147 fast unit/integration tests (`pytest -m "not model"`, CI-required, no GPU
needed) covering direction computation, ablation/addition math, refusal
classification, detector metrics (Wilson CI, McNemar, DeLong, Cochran's Q), SAE
loading per provider, and the FastAPI backend (stubbed model/tokenizer, matching
this project's own established stub-testing convention). 11 additional
`@pytest.mark.model`-tagged tests run against real small models as a
lower-frequency hardware regression check, not part of the CI-required set.

## Deliverables

**Live-inference API backend** (`src/api/`, done, merged) — FastAPI service that
loads at most one model at a time on a 6GB GPU (evicting/reloading as requests
switch targets) and serves all four detectors (keyword, perplexity,
dense-direction, SAE-feature) for arbitrary prompts, not just the pre-collected
corpus.

**Interactive detector UI** (frontend, in progress, not yet merged) — plain
HTML/JS/CSS page over the API above, plus a Findings dashboard built from this
project's own RESULTS.md numbers (TEST AUROC by model, PAIR-robustness ranking,
dense-vs-SAE head-to-head), verified live in-browser against the real GPU.

## Roadmap

**Done:**
- Single-direction reproduction (Qwen2.5-1.5B, SmolLM2) — necessity and
  sufficiency both confirmed, non-overlapping CIs vs. baseline.
- Dataset pipeline — 4 datasets unified, deduplicated, stratified split,
  activations cached and cross-model-consistency-checked.
- SAE-feature detector (Qwen3-8B) — causal ranking via attribution patching,
  causal validation via real-generation suppression, a clean monotonic
  dose-response curve after fixing a stochastic-sampling bug.
- Baseline detectors + adversarial evaluation — keyword/perplexity baselines,
  head-to-head against both activation-based detectors, real JailbreakBench
  adversarial paraphrase set, XSTest false-positive ("safety tax") measurement.
- Dense-direction detector extended to all 5 models — comparably strong clean-
  prompt accuracy everywhere, a real and statistically confirmed PAIR-robustness
  spread across models.
- SAE-feature detector extended to Llama-3.1-8B and gemma-2-9b-it — causal
  ranking, real-generation validation, and the prompt-classifier reframing, all
  three models now covered.
- Cross-model direction transfer (necessity) — a clean no-transfer result for
  Qwen3-8B, an inconclusive one for Llama-3.1-8B, independently replicated at
  larger N.
- Cross-model direction transfer (sufficiency) — zero detectable effect from
  either model's foreign direction at any alpha, a clean paired-McNemar verdict
  resolving what the necessity side left inconclusive for Llama.
- Live-inference API backend, with full test coverage and a real-GPU smoke test.
- The Llama causal-gap investigation (see Key Results above).

**Remaining:**
- Interactive detector UI frontend — built and live-verified, pending final
  sign-off before merge.
- A reusable automated moralize-vs-comply classifier — two candidate local judge
  models both failed validation on this specific distinction; extending
  true-harmful-compliance labeling elsewhere still needs manual effort per
  completion set.
- A handful of statistically-confirmed cross-model differences with no
  mechanistic explanation (PAIR-robustness spread, XSTest false-positive
  non-monotonicity, SAE causal-effect-concentration spread) — reported honestly
  as open questions rather than forced into an explanation, consistent with this
  project's standing practice throughout.
