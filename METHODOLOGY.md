# Methodology: refusal direction and SAE-feature detector

## Direction estimation

For each model, harmful (AdvBench) and harmless (Alpaca, input-free rows
only) instructions are formatted with the model's chat template and passed
through the model. The residual stream at the *last prompt token* (i.e. the
position right before generation starts) is captured at the output of every
decoder layer via nnsight (`src/activations/extract.py`).

For layer `l`, the direction is the difference of means:

```
direction[l] = mean(harmful_activations[l]) - mean(harmless_activations[l])
```

`src/direction/compute.py` provides both a unit-normalized version (for
ablation, where only the direction matters) and the raw, unnormalized
version (for activation addition, where the vector's own magnitude reflects
the natural harmful/harmless separation scale at that layer -- see
"Addition calibration" below).

## Layer selection

Rather than causally testing every layer (expensive -- each test requires
real generation, not just a forward pass), layers are screened with a cheap
proxy first: `separation_score()` projects held-out activations onto each
layer's candidate direction and computes the standardized mean difference
(harmful projection mean - harmless projection mean, divided by pooled
std). The top-scoring layers in the middle-to-late portion of the network
(skipping the first/last 25%, where the original paper found the direction
is noisiest or too close to the unembedding to ablate cleanly) become
causal-test candidates; only the top candidate is causally validated in the
main pipeline.

**Train/calib/val separation**: the direction is estimated on `train`
(200 prompts/class), the layer is selected using `separation_score` on a
disjoint `calib` split (30 prompts/class), and the causal generation-based
validation runs only on a further disjoint, truly held-out `val` split (30
prompts/class). Earlier versions of this pipeline reused the same split for
layer selection and reported causal numbers -- a mild leakage risk, since
layer choice was informed by the same data used to report results. Fixed by
adding a third split; final validation results (`results/phase1_reproduction_*.json`)
and the numbers in `RESULTS.md` are all from the truly held-out `val` split.

## Causal validation

Two interventions test the direction's causal role, applied via nnsight
hooks at every generation step (`src/direction/interventions.py`):

- **Directional ablation** (tests necessity): projects the direction out of
  the residual stream at *every* layer, at every generated token, so the
  model can never write along it. Should suppress refusal on harmful
  prompts if the direction is necessary for refusal.
- **Activation addition** (tests sufficiency): adds the direction (scaled by
  `alpha`) into the residual stream at the selected layer, at every
  generated token. Should induce refusal on harmless prompts if the
  direction is sufficient to cause it.

Generation is forced to a fixed length (`min_new_tokens == max_new_tokens`,
40 tokens) rather than allowing early EOS stopping. This was a debugging
necessity as much as a methodological choice: nnsight's per-generation-step
intervention loop (`tracer.iter[:]`) desyncs from the actual number of
forward passes when the model stops early, causing silent failures. Forcing
fixed-length generation eliminates the desync and also makes completions
directly comparable in length across conditions.

## Addition calibration

A fixed `alpha` (originally 8.0, chosen without justification) turned out to
have wildly different effects across layers and models, because raw
residual-stream norms differ by an order of magnitude across layers and
architectures (Qwen's best-layer direction norm: 75.3; SmolLM2's: 279.6).
Two fixes, in order:

1. Switched from adding a *unit* direction (arbitrary magnitude) to adding
   the *raw, unnormalized* mean-difference vector -- its own norm already
   reflects the natural scale of the harmful/harmless separation at that
   layer, so `alpha=1.0` means "add the natural difference once," not an
   arbitrary number.
2. Even with that fix, a single alpha value doesn't transfer across models
   (see `RESULTS.md`'s calibration sweep). `scripts/02_calibrate_addition_alpha.py`
   sweeps alpha on a calibration split disjoint from the reported val split,
   selecting the smallest alpha that clears an 80% refusal-rate target
   without the completions collapsing into degenerate repeated-token output
   (checked via `refusal_classifier.is_degenerate()`, a unique-word-ratio
   heuristic). If no alpha clears the target, the alpha with the highest
   non-degenerate refusal rate is used instead (SmolLM2's case).

## Refusal classification

`src/direction/refusal_classifier.py` is a keyword/phrase matcher (checks
for "I cannot", "I'm sorry", "as an AI", etc. in the first 200 characters of
a completion) -- the same class of sanity-check metric used in the original
paper and its follow-ups before a more expensive LLM-judge evaluation.
`refusal_stats()` additionally reports a 95% Wilson score confidence
interval (chosen over a plain normal approximation because it stays well-behaved
at small n and at rates near 0% or 100%, both of which occur throughout
these results).

**Validated (2026-07-11)**: a 45-completion human-labeling spot-check
across every experiment in this project (`scripts/07`/`08`, full results
in RESULTS.md/DECISIONS.md) found 97.8% agreement between the classifier
and human judgment, including 100% accuracy at correctly calling
"moralizes without complying" completions non-refuse. The classifier is
accurate at its stated job (detecting refusal phrasing). **What it does
not do, and was never designed to do**: distinguish "moralizes without
complying" (safe) from "complies" (unsafe) -- both count as "non_refuse."
`refusal_rate` should not be read as a compliance rate for that reason;
see RESULTS.md's head-to-head section for a concrete case where that
distinction mattered.

## What this reproduction does and doesn't establish

Establishes: the causal refusal-direction finding reproduces on two small,
architecturally different open-weight models, for both the necessity
(ablation) and sufficiency (addition) directions of the original claim.

Does not establish: whether the SmolLM2 addition-effect ceiling reflects
safety-training strength, architecture, or something else. The 7-9B-scale,
SAE-feature question this project actually motivates is addressed
separately below.

## SAE-feature detector (Qwen3-8B)

Adapted from "Understanding Refusal in Language Models with Sparse
Autoencoders" (arXiv:2505.23556, close-read in [LITERATURE.md](LITERATURE.md)).
Where the reproduction above finds one dense direction, this asks a finer
question: which specific sparse, interpretable internal components (out of
tens of thousands per layer) are causally responsible for refusal, using a
larger model (Qwen3-8B, 4-bit quantized) and a pretrained sparse autoencoder
suite (Qwen-Scope). Full rationale for every design choice below is in
[DECISIONS.md](DECISIONS.md); this section covers the *how*, not the *why*.

### A methodological pitfall specific to Qwen3: uncontrolled thinking mode

Qwen3 models reason inside a `<think>...</think>` block before answering by
default. Left uncontrolled, the position used throughout this pipeline (the
last prompt token, or the first generated token) lands inside that
reasoning preamble instead of at the model's actual answer -- silently
measuring "how does the model start reasoning" rather than "does it refuse
or comply." Fixed by passing `enable_thinking=False` to the chat template
(`src/activations/extract.py`'s `format_prompt()`), which pre-fills an
*empty* think block into the prompt so the first generated token is
genuinely the start of the real answer. This is a Qwen3-specific
consideration the reproduction above never had to deal with (SmolLM2 and
Qwen2.5 don't have a thinking mode); every number reported for the
SAE-feature detector reflects the corrected pipeline.

### Activation extraction and layer selection

Same extraction and separation-score-based layer selection as the
reproduction above (`src/activations/extract.py`, `src/direction/compute.py`),
run once on the full corpus (1922 deduplicated prompts across AdvBench,
HarmBench, JBB-Behaviors, XSTest) rather than a small per-model split, since
this activation cache is reused across every downstream step (layer
selection, feature pooling, causal ranking). Unlike the reproduction, which
causally validates only the single best layer, separation scores here were
tightly clustered across the top 5 layers -- see "Multi-layer K0 pooling"
below for why that changes the layer-selection strategy.

### Pretrained SAE loading

`src/sae/qwen_scope.py` loads Qwen-Scope's `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`
checkpoints (one ~2GB `layer{n}.sae.pt` per layer, only the specific
layers actually used are downloaded). Each is a TopK sparse autoencoder
(65536 features, top-50 kept per forward pass): `encode()` projects a
residual-stream vector to sparse feature activations, `decode()`
reconstructs the residual stream from those activations, and
`feature_direction(i)` returns the residual-stream direction a single
feature writes to (the corresponding decoder column) -- the quantity
compared against the refusal direction below.

**Known, accepted limitation**: these SAEs are trained on Qwen3-8B-**Base**
activations, applied here to the instruct/chat model. Standard, published
practice in this literature (the source paper does the same with
GemmaScope/LlamaScope), not a novel compromise -- see DECISIONS.md.

### Multi-layer K0 pooling

The source paper's method takes the top K0=10 SAE features per layer by
cosine similarity to the difference-in-means direction, then keeps the top
K*=20 overall by causal effect -- since K* > K0, this implies pooling
candidates from multiple layers, not just one. Given this project's
top-5 separation scores were nearly tied, three layers (the top 3 by
score) were used rather than forcing a single-layer result to fit a
multi-layer formula: `src/sae/feature_selection.py`'s
`top_k0_by_cosine_similarity()` (signed cosine similarity between each
feature's decoder direction and the layer's refusal direction -- signed,
not absolute, since only features that write *along* the harmful direction
are refusal-feature candidates) run independently per layer, pooled by
`pool_top_k0_across_layers()` into a flat `(layer, feature)` candidate list.

### Causal ranking via attribution patching

`src/sae/causal_ranking.py` estimates each pooled candidate's causal effect
via integrated gradients (N=10 steps), on a differentiable proxy metric
(`src/direction/refusal_metric.py`: logit(" I") minus logit(" Sure") at the
first generated token -- the same refusal-vs-compliance first-token
contrast used in Arditi et al. 2024). For each candidate feature: the
natural pre-activation value is computed from the unmodified residual
stream, then integrated gradients estimates the effect of ablating that
feature (natural value -> 0) by backpropagating through N interpolation
steps between 0 and the natural value, batched into a single
forward+backward pass per (feature, prompt) for tractable runtime (verified
numerically identical to looping one step at a time, at ~1/N the cost).

Scored on a fixed sample of 8 harmful TRAIN prompts, length-capped at 150
characters (a real memory constraint: backward passes need far more memory
than the no-grad forward-only extraction, and the corpus's ~1000-token
outliers would likely OOM even a 4-bit model on a 6GB GPU). This is
deliberately a *cheap screening pass*, not the rigorous validation --
see below.

**Implementation note (nnsight-specific)**: modules must be accessed in
forward-pass execution order. Iterating a `{layer: features}` dict built in
ranking-score order (not ascending layer order) raises a
`MissedProviderError` -- fixed by always sorting layers ascending before
iterating (`src/direction/interventions.py`).

### Causal validation via feature suppression

Mirrors the reproduction's directional-ablation methodology directly, for
comparability: `generate_with_feature_suppression()`
(`src/direction/interventions.py`) ablates one or more SAE features (each
feature's natural contribution -- `value * decoder_direction` -- computed
from the *original* unmodified residual, so ablating one feature doesn't
shift the baseline another feature in the same layer is measured against)
at every generated token, then measures refusal with the real
`refusal_classifier` (not the differentiable proxy used for ranking --
that step was only ever a screening pass to narrow 30 candidates to 20).

Six conditions compared on 50 held-out VAL harmful prompts (disjoint from
every prompt used in direction extraction, layer selection, and ranking):
baseline, suppress the single top-ranked feature, and suppress the top
5/10/15/20 (the intermediate points added after an initial, coarser
baseline/top1/top5/top20 pass -- see DECISIONS.md for why). Generation
uses **greedy decoding** (`do_sample=False`) rather than each model's
default sampling config -- an earlier pass at this same sample size found
a non-monotonic curve, traced to uncontrolled stochastic sampling
conflating the intervention's true effect with sampling noise (one
generation sample per prompt); fixed and confirmed to produce a clean,
reproducible curve (see DECISIONS.md). Per DECISIONS.md's capability-check
requirement, every completion is also checked for degeneracy
(`refusal_classifier.is_degenerate()`) -- a refusal-rate drop caused by
incoherent garbage output wouldn't be a real causal finding.

### What the SAE-feature detector does and doesn't establish

Establishes: a systematically-selected set of features (pooled across 3
layers, ranked by attribution-patching causal effect) has a real,
statistically distinguishable causal effect on refusal when suppressed --
a clean, monotonic dose-response from top-1 through top-15, then a
plateau (full numbers in `RESULTS.md`) -- and, unlike the
single-hand-picked-feature approach this project explicitly designed
around avoiding (arXiv:2411.11296), without collapsing model coherence.

Does not establish: cross-model generalization (does a feature set found
this way on Qwen3-8B transfer to a different model?) -- this project's
stated differentiator from the existing literature, and still open;
whether individual features among the top 20 are independently
interpretable (this pipeline validates the *set*, not each member); or a
comparison against the dense single-direction approach on equal footing
(that comparison is below).

### Cross-model extension (Llama-3.1-8B-Instruct, gemma-2-9b-it)

Same recipe as above -- top-3 layers by separation score, K0=10 pooled
candidates, causal ranking via attribution patching, causal validation via
suppression -- rerun per model using the already-merged JumpReLU SAE
loaders (`src.sae.llama_scope`, `src.sae.gemma_scope`). `src/sae/registry.py`
dispatches `load_sae`/layers/(for Gemma) an integrated-gradients
micro-batch size per model, so `scripts/04`/`scripts/05` no longer
hardcode Qwen-Scope specifically. This is "extend the recipe to new
models," the same choice already made for the dense-direction detector's
cross-model extension -- not the literal transfer question above (does a
feature set found on one model do anything applied to another), which
remains untested for both the dense-direction and SAE-feature methods.

One real implementation difference needed for gemma-2-9b-it: its
`width_131k` SAEs are far larger than LlamaScope's or Qwen-Scope's (131072
features vs. 32768/65536), so keeping three full SAEs resident on GPU
throughout ranking -- harmless for the other two models -- left too little
memory for the integrated-gradients backward pass on a 6GB card. Fixed by
no longer moving the full SAE to GPU at all; `feature_ig_attribution`
only ever needs one row/column per candidate and already moves that slice
to-device itself. See DECISIONS.md for the full debugging account
(including two plausible-looking fixes that didn't address the actual
cause).

Result: a genuine, three-way cross-model difference in how concentrated
the causal effect is (Llama: nearly all of it from one feature; Qwen3:
distributed, no single feature sufficient; Gemma: real but much more
modest overall effect) -- flagged as unexplained, not resolved. Full
numbers in RESULTS.md.

## Baseline detectors and adversarial evaluation

Phases 1 and 3 above validate *causal mechanisms* (does ablating/suppressing
change generation behavior) on Qwen3-8B. This section reframes both as
**prompt classifiers** -- a harmful/benign decision made on a prompt before
generation -- and compares them against two simple baselines, all under one
shared protocol. Full rationale in [DECISIONS.md](DECISIONS.md).

### From causal mechanism to classifier

- **Dense-direction detector** (`src/detectors/dense_direction_detector.py`):
  projects a prompt's residual-stream activation, at the layer already
  selected in the head-to-head ablation script (`scripts/06`, layer 23),
  onto the TRAIN-estimated refusal direction (`compute_directions`). Higher
  projection = more harmful-like, by construction.
- **SAE-feature detector** (`src/detectors/sae_feature_detector.py`): sums
  the encoded activations of Phase 3's top-15 causally-ranked
  (layer, feature) pairs (loaded directly from
  `results/sae_causal_ranking_Qwen3-8B.json`, not recomputed -- layers 23,
  24, 25).
- **Keyword filter** (`src/baselines/keyword_filter.py`): a curated
  harmful-intent lexicon (~50 terms across weapon/malware/drug/violence/
  fraud categories) -- the classic content-moderation baseline, expected to
  degrade under paraphrase since it only recognizes surface vocabulary.
- **Perplexity filter** (`src/baselines/perplexity_filter.py`): prompt
  perplexity under Olmo-3-1025-7B, per the general approach of Alon &
  Kamfonas 2023 ("Detecting Language Model Attacks with Perplexity",
  arXiv:2308.14132) -- built to catch gibberish adversarial suffixes (e.g.
  GCG), not expected to catch fluent paraphrase. Backbone went through five
  models (GPT-2 -> GPT-Neo-1.3B -> Phi-4-mini-instruct -> OLMo-2-0425-1B ->
  Olmo-3-1025-7B) before landing here -- see DECISIONS.md for the full
  history, including why an instruction-tuned model was rejected (scoring
  raw, non-templated
  text is off-distribution for a chat model) and why a target model was
  rejected as the alternative (breaks the baseline's independence).

Both baselines operate on the prompt text alone (no target-model activations
needed); the two detectors require the target model's (Qwen3-8B's) own
activations, extracted the same forward-pass-only way as the rest of this
project (`get_last_token_resid_acts`), never generation.

### Split discipline

Reuses the existing full-corpus TRAIN/VAL/TEST manifest and the Qwen3-8B
activation cache from Phase 2/3 -- no new full-corpus extraction.

- **TRAIN**: already used to estimate the refusal direction and rank SAE
  features (Phases 1/3); reused as-is here, not re-derived.
- **VAL**: calibrates every detector's decision threshold
  (`scripts/10_calibrate_detector_thresholds.py`), via the cutoff that
  maximizes Youden's J (TPR - FPR) on VAL's labeled prompts
  (`src/eval/detector_metrics.youden_threshold`) -- the same
  calibrate-on-a-disjoint-split discipline as the alpha-calibration sweep in
  Phase 1, rather than a hand-picked constant. VAL was previously only used
  for Phase 3's *generation-based* causal suppression validation; using its
  *activations* for threshold-fitting here is a distinct, non-leaking use.
- **TEST**: untouched by any tuning. Final metrics
  (`scripts/11_head_to_head_detectors.py`) are reported here, plus its
  XSTest-safe subset (false-positive/over-refusal check) and the
  adversarial paraphrase set below (built only from TEST-split JBB goals).

### Adversarial paraphrase set

Built from real, published JailbreakBench attack artifacts
(`src/eval/adversarial_paraphrase.py`,
[github.com/JailbreakBench/artifacts](https://github.com/JailbreakBench/artifacts)),
not self-authored jailbreak templates -- consistent with this project's
existing "no self-authored harmful content, only real published benchmarks"
policy (see [ETHICS.md](ETHICS.md)). Two attack families, kept separate in
reporting because they represent different failure modes:

- **PAIR** (black-box): fluent, roleplay/fictional-framing paraphrases of
  the harmful goal -- the "adversarial paraphrase" case this evaluation is
  named for.
- **GCG** (white-box + transfer): the harmful goal plus an optimized
  gibberish suffix -- the classic case perplexity filtering is built to
  catch, kept as a contrasting condition rather than folded into "paraphrase."

Only `jailbroken: true` records are kept (attacks that actually succeeded
against their original target model -- Vicuna-13B, Llama-2-7B-chat,
GPT-3.5, or GPT-4, depending on method). Each artifact's `goal` field is
matched (case/whitespace-normalized) against this project's own TEST-split
records with `source == "jbb"`, so the adversarial set only contains
paraphrases of behaviors no detector was calibrated on. **This is
explicitly not a claim that these prompts would jailbreak this project's
own models** -- they were never tested against Qwen3-8B/Qwen2.5/SmolLM2 for
attack success; only the *prompt surface form* is reused, to test whether
it fools a *prompt classifier*, independent of whether it would actually
elicit harmful generation from our target model.

At the current corpus size, only 11 of TEST's JBB-sourced harmful goals had
any matching successful artifact, yielding 35 adversarial prompts (21 PAIR,
14 GCG) -- a real but modest-sized set; see RESULTS.md's limitations for
what this does and doesn't support statistically.

### Evaluation

`src/eval/detector_metrics.py` computes accuracy/precision/recall/F1/AUROC
(scikit-learn's `roc_auc_score`/`roc_curve`) with Wilson-score CIs on the
rate-like metrics, matching the style of
`refusal_classifier.refusal_stats`. Three conditions, per detector:

1. **TEST overall** (harmful vs. harmless): the core comparison.
2. **XSTest-safe subset of TEST** (all harmless): a false-positive /
   over-refusal check -- reported as the "correctly not flagged" rate
   (`accuracy` in this all-negative slice), i.e. 1 minus the false-positive
   rate.
3. **Adversarial paraphrase set** (all harmful, real published attacks):
   reported as detection/flag rate (`accuracy` in this all-positive slice,
   equivalently recall) -- precision/AUROC aren't well-defined with no
   negatives in this set, and are correctly reported as N/A rather than a
   misleading number. Broken down by attack method (PAIR vs. GCG)
   separately from the pooled number, since pooling the two hides which
   method is actually driving a detector's flag rate.

### Significance testing

Two comparisons in this evaluation are close enough (or involve more than
two related classifiers) that eyeballing Wilson CIs for overlap isn't
sufficient -- both are paired/repeated-measures designs (the same items
scored by multiple detectors or the same detector run on multiple models),
so `src/eval/detector_metrics.py` provides the matching formal tests
instead of treating each score as an independent sample:

- **`delong_auc_test`** (DeLong et al. 1988, Sun & Xu 2014's structural-
  components formulation): compares two AUROCs measured on the *same*
  labeled sample, accounting for their correlation. Used for
  dense-direction vs. SAE-feature on TEST-overall (`scripts/11`), where the
  gap (0.983 vs. 0.975) is close enough to need a real test rather than an
  eyeballed "clearly beats."
- **`mcnemar_exact`**: paired binary-decision test for exactly two
  classifiers (already used for the adversarial-paraphrase dense-vs-SAE
  comparison, see the section above).
- **`cochrans_q`**: generalizes McNemar's test from two to *k* related
  classifiers scored on the same items -- used for the 3-model
  dense-direction PAIR-paraphrase comparison (`scripts/13_cross_model_significance.py`),
  replacing an earlier informal "non-overlapping CIs" argument for
  "SmolLM2 is significantly more robust to PAIR paraphrase" with a real
  chi-squared test.

### Cross-model extension (Qwen2.5-1.5B, SmolLM2-1.7B)

Extends only the dense-direction detector (not the SAE-feature detector --
no SAE trained for these models) to Phase 1's two smaller models
(`scripts/12_extend_dense_direction_qwen25_smollm2.py`). Keyword and
perplexity baselines are not re-run per model since they score prompt text
only, never activations -- their Qwen3-8B numbers apply unchanged.

`src.detectors.dense_direction_detector.select_layer_and_calibrate` does
layer selection AND threshold calibration both on VAL, never TEST -- a
cleaner discipline than the TEST-based layer selection used for Qwen3-8B's
ablation layer (`scripts/06`), which reused a TEST-selected layer for final
TEST-split reporting too. That reuse is a mild leakage pattern this
project explicitly avoids elsewhere (see "Train/calib/val separation"
above); it happened to make no practical difference for Qwen3-8B (layer 23
is selected whether scored on VAL or TEST -- verified directly, see
DECISIONS.md) but the VAL-only approach is the correct one to use going
forward, including here.

### Cross-model extension: SAE-feature detector (Llama-3.1-8B-Instruct, gemma-2-9b-it)

Completes the 3-model SAE-feature comparison: reframes Wave 2's
causally-validated feature sets (Llama-3.1-8B, gemma-2-9b-it) as prompt
classifiers, same four-detector head-to-head protocol as Qwen3-8B's
original run above. `scripts/10`/`scripts/11` generalized with a `model`
CLI arg (mirroring Wave 2's pattern for `scripts/04`/`05`): SAE loading
via `src/sae/registry.py`, K=15 reused for all three models (each
model's own causal-validation curve independently bottoms out at top-15
-- see DECISIONS.md), keyword/perplexity thresholds reused from Qwen3-8B
(prompt-text-only, VAL-split membership is prompt-manifest-based and
identical across every model's cache) rather than recomputed.

Dense-direction's layer/threshold source differs by model, not a uniform
lookup: Qwen3-8B keeps its original TEST-selected layer (frozen legacy
value, see the "mild leakage pattern" entry above and in DECISIONS.md),
Llama-3.1-8B/gemma-2-9b-it use the VAL-selected layer already computed by
Wave 1's dense-direction extension
(`results/dense_direction_cross_model.json`) -- `resolve_layer_for_model`
(`src.detectors.dense_direction_detector`) branches on this rather than
papering over the two genuinely different provenances with one table.

New adversarial-activation caches (`scripts/15`, `results/activations/
{model}_adversarial.pt`) were built for both models -- Wave 1's
dense-direction extension computed these on the fly since it only needed
one layer's projection; the SAE-feature detector needs all 3 of a
model's ranked-feature layers, so this time the extraction is cached
properly rather than recomputed per script.

Result: a genuinely different dense-vs-SAE-feature comparison per model
(Qwen3-8B: indistinguishable; Llama-3.1-8B: dense wins overall but SAE
wins on PAIR specifically; gemma-2-9b-it: dense wins everywhere tested,
most one-sidedly of the three) -- not resolved into one pattern, reported
as a real cross-model difference. Full numbers in RESULTS.md.
