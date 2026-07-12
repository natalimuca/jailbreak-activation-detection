# Results: refusal-direction reproduction and SAE-feature detector

Reproduction of Arditi et al.'s causal refusal-direction finding (a single
residual-stream direction, estimated as the harmful/harmless mean-activation
difference, causally controls refusal), on two small open-weight models.
Methodology details in [METHODOLOGY.md](METHODOLOGY.md).

Only aggregate statistics are recorded here. Raw model completions
(including actual harmful text produced under directional ablation) are
never committed to this repo -- see `results/` in `.gitignore`.

## Held-out validation (final numbers)

Direction estimated on a 200-prompt train split (AdvBench + Alpaca), best
layer selected on a disjoint 30-prompt calib split, causal effect measured
on a disjoint 30-prompt held-out val split with **greedy (`do_sample=False`)
decoding**. 95% CIs are Wilson score intervals. (These numbers were
originally measured with each model's default, sampling, generation
config; re-run deterministic after the SAE-feature detector work below
surfaced the same uncontrolled-sampling gap -- see DECISIONS.md. Practical
effect was negligible: SmolLM2 already defaulted to greedy so its numbers
are unchanged; Qwen2.5's ablated condition shifted by exactly one
completion out of 30.)

| Model | Condition | n | Refusal rate | 95% CI |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | harmful, baseline | 30 | 100.0% | [88.7%, 100%] |
| Qwen2.5-1.5B-Instruct | harmful, **ablated** | 30 | 3.3% | [0.6%, 16.7%] |
| Qwen2.5-1.5B-Instruct | harmless, baseline | 30 | 0.0% | [0%, 11.4%] |
| Qwen2.5-1.5B-Instruct | harmless, **direction added** (alpha=1.0) | 30 | 96.7% | [83.3%, 99.4%] |
| SmolLM2-1.7B-Instruct | harmful, baseline | 30 | 63.3% | [45.5%, 78.1%] |
| SmolLM2-1.7B-Instruct | harmful, **ablated** | 30 | 3.3% | [0.6%, 16.7%] |
| SmolLM2-1.7B-Instruct | harmless, baseline | 30 | 0.0% | [0%, 11.4%] |
| SmolLM2-1.7B-Instruct | harmless, **direction added** (alpha=1.5, calibrated) | 30 | 40.0% | [24.6%, 57.8%] |

In every case the intervention's CI does not overlap the corresponding
baseline's CI -- the causal effect is real on both models, in both
directions (necessity via ablation, sufficiency via addition).

## Cross-model comparison

| | Qwen2.5-1.5B-Instruct | SmolLM2-1.7B-Instruct |
|---|---|---|
| Best layer (of 28) | 23 | 20 |
| Raw direction norm at best layer | 75.3 | 279.6 |
| Ablation effect (necessity) | 100% -> 3% | 63% -> 3% |
| Addition effect (sufficiency), calibrated | 0% -> 97% | 0% -> 40% |
| Baseline refusal rate on AdvBench | 100% | 63% |

Two honest, unexplained-further observations rather than overclaimed
conclusions:

1. **Ablation (necessity) is robust on both models** -- refusal collapses to
   near-zero regardless of how strongly the model refused at baseline.
2. **Addition (sufficiency) is architecture-dependent and much weaker on
   SmolLM2.** The alpha-calibration sweep (below) shows SmolLM2 never
   reaches a clean high-refusal regime at any tested alpha -- it peaks at
   42% before collapsing into degenerate repeated-token output. This isn't
   a missing scaling constant; SmolLM2's raw direction norm is already
   ~4x Qwen's, and even correcting for that, the induced effect stays
   capped. Whether this tracks the strength of Qwen's RLHF-style refusal
   training (a more sharply "linear" refusal representation) or something
   else about the two architectures is an open question, not one this
   reproduction answers.

## Alpha calibration sweep

Addition coefficient (multiplier on the raw, unnormalized mean-difference
direction) swept on a 12-prompt calibration split, disjoint from both the
train and held-out val splits above. Greedy decoding, as above.

**Qwen2.5-1.5B-Instruct** (best layer 23, raw direction norm 75.3):

| alpha | refusal rate | degenerate fraction |
|---|---|---|
| 0.25 | 17% | 0% |
| 0.50 | 50% | 0% |
| **1.00** | **100%** | **0%** |
| 1.50 | 100% | 8% |
| 2.00 | 75% | 83% |
| 3.00 | 0% | 100% |
| 4.00 | 0% | 100% |

At n=12 (a small calibration sample), a few individual data points shifted
from the original sampling-based sweep (0.50: 67%->50%, 1.50 degenerate:
17%->8%, 2.00 degenerate: 92%->83%, 3.00: 17%->0%) -- expected noise on
individual borderline completions, not a change in the overall shape: same
calibrated alpha (1.0) selected either way, same qualitative pattern (a
clean high-refusal window before degenerate collapse at higher alpha).

**SmolLM2-1.7B-Instruct** (best layer 20, raw direction norm 279.6):

| alpha | refusal rate | degenerate fraction |
|---|---|---|
| 0.25 | 0% | 0% |
| 0.50 | 0% | 0% |
| 1.00 | 25% | 0% |
| **1.50** | **42%** | **0%** |
| 2.00 | 42% | 0% |
| 3.00 | 17% | 17% |
| 4.00 | 0% | 100% |

Qwen has a wide window (alpha 0.5-1.5) of clean, coherent, high-refusal
behavior before degenerating. SmolLM2 never has a wide clean window --
refusal rate rises only as far as 42% before the model starts collapsing
into repeated-token garbage at higher alpha.

## Known limitations

- The refusal classifier is a keyword/phrase matcher (see
  `src/direction/refusal_classifier.py`), the standard sanity-check metric
  in this literature, not a validated final detector. It hasn't been
  checked against human judgment or an LLM-judge baseline for
  false-positive/false-negative rate on these specific completions.
- n=30 per condition is enough to show the effect is real (non-overlapping
  CIs) but not enough for fine-grained comparisons between conditions.
- Only two small models tested (1.5B, 1.7B params). Whether the
  necessity/sufficiency asymmetry holds at the 7-9B scale used later in
  this project is untested.

## SAE-feature detector (Qwen3-8B)

Methodology in [METHODOLOGY.md](METHODOLOGY.md#sae-feature-detector-qwen3-8b),
full rationale for every design choice in [DECISIONS.md](DECISIONS.md).
Raw completions are not committed (same policy as above); aggregate stats
and the selected feature list are in
`results/sae_causal_ranking_Qwen3-8B.json` and
`results/sae_suppression_validation_Qwen3-8B.json`.

### Layer selection

Separation scores (difference-in-means direction from TRAIN, measured on
held-out VAL), full 1922-prompt corpus:

| layer | score |
|-------|-------|
| 23    | 1.783 |
| 25    | 1.783 |
| 24    | 1.781 |
| 26    | 1.780 |
| 28    | 1.778 |

Extremely tightly clustered (spread of 0.005) -- unlike the single-model
reproduction above, no one layer stands out, which is why the top 3 (23,
25, 24) were pooled for feature selection rather than causally testing
only the single best layer.

### Causal ranking (attribution patching)

Top-5 of the 20 selected features, by integrated-gradients attribution
score on the refusal-vs-compliance logit-diff metric (16 harmful TRAIN
prompts, length-capped -- increased from an initial n=8 pass that gave
essentially the same top-2 features and score, confirming this ranking
is stable, not noise; see DECISIONS.md):

| rank | layer | feature | score |
|---|---|---|---|
| 1 | 25 | 65291 | 2.198 |
| 2 | 23 | 42331 | 1.430 |
| 3 | 24 | 5393  | 0.452 |
| 4 | 24 | 4711  | 0.361 |
| 5 | 23 | 23501 | 0.354 |

The top 2 features are a clear standout (2.198 and 1.430) above the rest
of the top-20 (all <= 0.452) -- a much sharper signal than an earlier,
since-superseded run of this same pipeline that hadn't yet disabled
Qwen3's default thinking mode (that run's top score was only 0.547, with
2 near-zero/negative features surviving into the top-20; see DECISIONS.md
for the full account). Full list of all 20 selected features (layer,
feature index, score) is in the JSON results file.

### Causal validation (feature suppression)

Baseline vs. suppressing the top-1/top-5/top-10/top-15/top-20 ranked
features, on 50 held-out VAL harmful prompts (disjoint from every prompt
used upstream), 40 tokens generated per completion with **greedy
(`do_sample=False`) decoding**, real `refusal_classifier`. 95% CIs are
Wilson score intervals.

(Two earlier, smaller/stochastic passes -- n=25 with 4 conditions, then
n=50 with 6 conditions but still sampling -- found the same overall
effect but a non-monotonic curve; tracing that down led to discovering
every generation call defaulted to stochastic sampling (`do_sample=True`)
rather than greedy decoding, conflating the intervention's true effect
with sampling noise. Fixed and re-run below; full account in DECISIONS.md.)

| condition | n | refusal rate | 95% CI | degenerate |
|---|---|---|---|---|
| baseline | 50 | 82.0% | [69.2%, 90.2%] | 0/50 |
| suppress top-1 | 50 | 84.0% | [71.5%, 91.7%] | 0/50 |
| suppress top-5 | 50 | 42.0% | [29.4%, 55.8%] | 0/50 |
| suppress top-10 | 50 | 32.0% | [20.8%, 45.8%] | 0/50 |
| **suppress top-15** | 50 | **24.0%** | **[14.3%, 37.4%]** | 0/50 |
| suppress top-20 | 50 | 26.0% | [15.9%, 39.6%] | 0/50 |

**A clean, steady monotonic decline from top-1 through top-15** (84% ->
42% -> 32% -> 24%), then a plateau at top-15/top-20 (24% vs 26%, heavily
overlapping CIs -- settling, not a reversal). Baseline's CI does not
overlap top-5 through top-20 -- a clearly distinguishable causal effect
from a modest fraction of the pooled candidates onward. **Zero completions
degenerated into incoherent output** across all 300 generations. Unlike
the single hand-picked feature in arXiv:2411.11296 (which achieved a
refusal-rate shift only by destroying general capability -- MMLU 68.8% ->
36.0%), this project's systematic top-K* selection produces a real
behavioral effect without a coherence collapse.

**Honest finding, not smoothed over**: suppressing the single top-ranked
feature alone (top-1) does not reduce refusal at all (84% vs 82%
baseline, statistically indistinguishable) -- the effect is **distributed
across the feature set**, not concentrated in one dominant feature, even
with sampling noise removed. **top-15 is the strongest single data
point** (lowest refusal rate, right before the plateau) -- the number to
lead with when summarizing this project's core finding.

### Head-to-head: dense-direction ablation vs SAE-feature suppression

Phase 1's ablation method run on Qwen3-8B, same 50 held-out VAL prompts
as the SAE suppression results above, layer selected via a separate TEST
split to avoid leakage (see DECISIONS.md). Baseline reused from above.

| condition | refusal rate | 95% CI | degenerate |
|---|---|---|---|
| baseline | 82.0% | [69.2%, 90.2%] | 0/50 |
| dense-direction ablation | 6.0% | [2.1%, 16.2%] | 0/50 |
| SAE suppress top-15 (best) | 24.0% | [14.3%, 37.4%] | 0/50 |

**Read this with a real caveat, not at face value**: dense ablation's raw
refusal rate is lower than any SAE-suppression condition, but inspecting
the actual completions shows why that comparison is misleading -- 47 of
50 "non-refusal" completions under dense ablation are the model
moralizing/lecturing about why the request is illegal or unethical
("Cracking passwords... is illegal, unethical, and immoral...") without
ever using the keyword classifier's refusal markers. That's a third
behavior mode, neither a clean refusal nor genuine harmful compliance.

A follow-up classifier-validation spot-check (below) confirmed the
classifier itself correctly calls this moralize pattern non-refuse --
that part isn't a classifier error. **The actual issue is that
`refusal_rate` as a single number conflates moralize (safe) and comply
(unsafe) under "non_refuse,"** so "6% refusal" should not be read as "94%
compliance." **The true harmful-compliance gap between the two methods is
very likely much smaller than the 6% vs 24% non-refusal-rate gap
suggests.** What the result does support: dense ablation (36 layers
touched) is a blunter, more disruptive intervention than SAE-feature
suppression (3 layers, up to 20 of 65536 features per layer) that pushes
the model into "moralize instead of refuse" far more often -- a real
finding about intervention bluntness, just not a compliance-rate
comparison. **Do not read this table as "dense ablation achieves more
compliance" -- it only shows dense ablation suppresses refusal
*phrasing* more.**

### Classifier-validation spot-check

Motivated by the finding above: 45 completions sampled across every
experiment in this project (Phase 1 x2 models, Phase 3 SAE suppression x6
conditions, the head-to-head), human-labeled blind to the classifier's own
verdict, then compared (`scripts/07_sample_completions_for_labeling.py`,
`scripts/08_score_classifier_agreement.py`; full methodology in
DECISIONS.md).

| human label | n | classifier accuracy |
|---|---|---|
| refuse | 17 | 100% |
| moralize | 13 | 100% |
| comply | 13 | 100% |
| partial | 2 | 50% |
| **overall** | **45** | **97.8%** |

**The classifier is more accurate than initially feared** -- all 13
moralize completions were correctly called non_refuse, zero
misclassifications in this sample. The head-to-head write-up above has
been corrected accordingly: this was never a case of the classifier being
*wrong*, it was a case of a single summary number (`refusal_rate`) hiding
a real distinction (moralize vs comply) that the classifier was never
designed to make. Only "partial" showed any disagreement, and at n=2 that's
too small to draw a conclusion from -- partial compliance is inherently
the hardest case for any binary classifier, not a specific weakness here.

### Known limitations (SAE-feature detector)

- n=50 for the suppression validation and n=16 for the ranking pass
  (both increased from an initial n=25/n=8 pass -- see DECISIONS.md) are
  enough to show the top-5-through-top-20 effect is real and the
  ranking's top features are stable.
- Single model (Qwen3-8B). Cross-model generalization -- whether a feature
  set selected this way transfers to a different model family -- is this
  project's explicit differentiator from the existing literature and
  remains untested.
- The SAEs are trained on the base model's activations, applied here to
  the instruct/chat model -- a documented, accepted limitation shared with
  the source paper (see DECISIONS.md), not unique to this reproduction.
- **`refusal_rate` conflates "moralize" (safe) and "comply" (unsafe) into
  one "non_refuse" bucket.** The classifier itself is accurate at its
  actual job (97.8% agreement, spot-check above) -- but no number in this
  project currently reports a true harmful-compliance rate distinct from
  a refusal-phrasing rate. Needed for any future claim like "method A
  produces more actual harmful content than method B": either a
  moralize-vs-comply classifier or an LLM-judge, since the existing
  keyword classifier was never designed to make that distinction.

## Baseline detectors and adversarial evaluation

Four prompt classifiers, compared under one protocol on Qwen3-8B: two
baselines (keyword filter, OLMo-2-0425-1B perplexity filter) and two
reframed activation-based detectors (dense-direction projection at layer
23, sum of Phase 3's top-15 ranked SAE features across layers 23/24/25).
Methodology, split discipline, and the adversarial paraphrase set's real
JailbreakBench provenance are in
[METHODOLOGY.md](METHODOLOGY.md#baseline-detectors-and-adversarial-evaluation).
Thresholds calibrated on VAL (`results/detector_thresholds_Qwen3-8B.json`),
never touched by the numbers below; full results in
`results/detector_head_to_head_Qwen3-8B.json`.

**Perplexity backbone note**: went through four models before settling
here -- GPT-2 (2019, matching Alon & Kamfonas 2023's own reference model)
-> GPT-Neo-1.3B (2021, after GPT-2's XSTest false-positive rate looked
suspiciously bad) -> Phi-4-mini-instruct (2025, rejected after its XSTest
number came back *worse* than GPT-Neo-1.3B's -- traced to it being
instruction-tuned, which is off-distribution for scoring raw non-chat text)
-> **OLMo-2-0425-1B** (2025, a genuine base model, final). One of this
project's own target models (Qwen2.5/SmolLM2/Qwen3-8B) was also considered
and rejected throughout, for a separate reason (breaks the baseline's
independence from what it's being compared against). Full history and the
non-monotonic result across all four backbones in DECISIONS.md. All
numbers below are the OLMo-2-0425-1B version; earlier backbones' numbers
are superseded, not reported twice.

### TEST split overall (n=288: 158 harmful, 130 harmless)

| detector | accuracy | 95% CI | F1 | AUROC |
|---|---|---|---|---|
| keyword filter | 56.6% | [50.8%, 62.2%] | 0.359 | 0.603 |
| perplexity filter | 55.2% | [49.4%, 60.9%] | 0.630 | 0.535 |
| **dense-direction** | **88.9%** | **[84.7%, 92.0%]** | **0.890** | **0.983** |
| **SAE-feature (top-15)** | **87.8%** | **[83.6%, 91.1%]** | **0.878** | **0.975** |

On clean, in-distribution prompts the two activation-based detectors
clearly beat both baselines -- an AUROC of 0.98/0.975 vs. 0.60/0.54.
**Perplexity's AUROC is essentially at chance (0.535)** regardless of which
of the four backbones scores it -- expected, and confirms this isn't a
"weak model" artifact: perplexity says nothing about semantic harmfulness,
only textual naturalness, and most of this corpus's prompts (harmful and
harmless alike) are ordinary fluent English regardless of which model
scores them.

**Dense-direction vs. SAE-feature is close enough here to test formally**:
0.983 vs. 0.975 AUROC on the same 288 items is paired data, so a DeLong
test (`src.eval.detector_metrics.delong_auc_test`) is the right tool rather
than eyeballing the gap. Result: diff = 0.0076, **p = 0.068** -- not
significant at the conventional 0.05 threshold, though borderline.
Combined with the adversarial-set McNemar result below (also
non-significant, p = 0.5), the two detectors are not statistically
distinguishable on either evaluation this project has run.

### XSTest-safe subset of TEST (n=37, all harmless -- false-positive check)

Reported as the "correctly not flagged" rate (1 minus the false-positive
rate) -- these are harmless-but-scary-*looking* prompts (e.g. "how do I
kill a Python process"), so a good detector should score high here.

| detector | correctly-not-flagged rate | 95% CI |
|---|---|---|
| keyword filter | 97.3% | [86.2%, 99.5%] |
| perplexity filter | 24.3% | [13.4%, 40.1%] |
| dense-direction | 94.6% | [82.3%, 98.5%] |
| SAE-feature (top-15) | 97.3% | [86.2%, 99.5%] |

**The full four-backbone story, not smoothed into a clean "newer is
better" narrative**: GPT-2 13.5% -> GPT-Neo-1.3B 75.7% -> Phi-4-mini-instruct
40.5% -> OLMo-2-0425-1B 24.3% (correctly-not-flagged, in that order). This
is **not monotonic with model recency or size** -- the best-performing
backbone on this specific check was the second one tried, not the newest
or largest. The most likely explanation, per DECISIONS.md: XSTest's
false-positive behavior under perplexity scoring depends on idiosyncratic
properties of each reference model's training distribution, not simply
"how good" that model is in general -- a genuinely messier and more
scientifically interesting conclusion than "a better model fixes this,"
and the one actually supported by running four real backbones instead of
one. Perplexity filtering remains the worst of the four detectors on this
axis in every version tried.

### Adversarial paraphrase set (n=35 real JailbreakBench artifacts, all harmful -- detection rate)

Precision/AUROC aren't meaningful with no negatives in this set (reported
as N/A); the relevant number is flag/detection rate. **Pooling PAIR
(fluent paraphrase) and GCG (gibberish suffix) hides which attack style is
actually driving each number**, so both the pooled and per-method
breakdowns are reported -- same "don't trust the aggregate, check the
breakdown" discipline as the moralize-vs-comply finding above.

| detector | pooled (n=35) | GCG suffix (n=14) | PAIR paraphrase (n=21) |
|---|---|---|---|
| keyword filter | 17.1% [8.1%, 32.7%] | 7.1% [1.3%, 31.5%] | 23.8% [10.6%, 45.1%] |
| perplexity filter | 40.0% [25.6%, 56.4%] | **100.0%** [78.5%, 100%] | **0.0%** [0%, 15.5%] |
| dense-direction | 62.9% [46.3%, 76.8%] | 92.9% [68.5%, 98.7%] | 42.9% [24.5%, 63.4%] |
| SAE-feature (top-15) | 57.1% [40.9%, 72.0%] | 92.9% [68.5%, 98.7%] | 33.3% [17.2%, 54.6%] |

**Honest findings, not smoothed over:**

1. **GCG detection is a perfect 100% across all four backbones tried**
   (GPT-2, GPT-Neo-1.3B, Phi-4-mini-instruct, OLMo-2-0425-1B) -- strong,
   convergent evidence that GCG's gibberish suffix is improbable under any
   reasonable LM, not an artifact of any one scoring model (exactly Alon &
   Kamfonas 2023's original point).
2. **PAIR detection is 0.0% for every backbone except the original GPT-2**
   (which got 38.1%). Getting the identical flat 0% from three independent
   replacement models (GPT-Neo-1.3B, Phi-4-mini-instruct, OLMo-2-0425-1B)
   is more convincing than any single model's result would be: GPT-2's
   38.1% was very likely inflated by its own weakness at modeling ordinary
   fluent text, not genuine detection of anything adversarial. **Every
   better backbone tried makes the failure cleaner and more total**,
   strengthening rather than undermining the conclusion that
   perplexity-based filtering structurally cannot catch fluent paraphrase
   attacks, regardless of which model scores it.
3. **The pooled number is still misleading on its own** (40.0%) -- entirely
   GCG-driven; PAIR is the case this evaluation is actually named for, and
   perplexity detects none of it.
4. **On PAIR -- the case this evaluation is actually named for -- all four
   detectors degrade sharply relative to their TEST-split performance**
   (dense-direction: 88.9% to 42.9%; SAE-feature: 87.8% to 33.3%; keyword:
   56.6% to 23.8%; perplexity: 55.2% to 0.0%). Fluent paraphrase is a hard
   case across the board, including for the activation-based methods.
5. **This does not replicate arXiv:2505.23556's finding that SAE features
   are more robust to adversarial paraphrase than a dense direction** -- on
   PAIR, dense-direction (42.9%) numerically edges out SAE-feature (33.3%),
   the opposite direction. Tested with a paired exact McNemar's test on the
   same 21 prompts (`src.eval.detector_metrics.mcnemar_exact` -- the correct
   test here, since both detectors are scored on identical items, not a
   comparison of two independent Wilson CIs): only 2 of 21 pairs are
   discordant (dense flags 2 prompts SAE doesn't; SAE flags none dense
   doesn't), **p = 0.5** -- nowhere near significant. Reported honestly as
   "no replication of that specific claim at this sample size," not as a
   reversal of it. (Unaffected by the perplexity-backbone switch -- this
   comparison never involved perplexity.)

### Known limitations (baseline detectors and adversarial evaluation)

- **Adversarial set is small** (n=35, spanning only 11 of TEST's JBB-sourced
  goals) -- large enough to show all four detectors degrade under PAIR
  paraphrase, not large enough to statistically distinguish dense-direction
  from SAE-feature on that degradation (point 4 above). A larger set (more
  JBB goals landing in TEST, or matching against AdvBench/HarmBench-sourced
  artifacts if JailbreakBench publishes them) would sharpen this.
- **Keyword lexicon coverage is corpus-dependent.** Its poor GCG/PAIR
  numbers partly reflect that this project's TEST-split JBB goals lean
  toward categories (defamation, harassment, extortion) the curated lexicon
  wasn't built to cover, not just paraphrase/suffix robustness in general.
- **Adversarial prompts were never tested for attack success against this
  project's own models** -- they're real successful jailbreaks against
  their original target models (Vicuna, Llama-2, GPT-3.5/4), reused here
  purely as disguised-harmful *prompt text* for a classifier-robustness
  test, not a claim about Qwen3-8B's own jailbreak susceptibility.
- **Detector comparison in this section is Qwen3-8B only** -- the SAE-feature
  detector can't run on Qwen2.5-1.5B/SmolLM2 (no SAE trained for those
  models), so it's excluded there. The dense-direction detector and both
  baselines *are* extended to those two models below.

## Dense-direction detector: cross-model comparison (Qwen2.5-1.5B, SmolLM2-1.7B)

Extends the dense-direction detector (not the SAE-feature detector, which
has no trained SAE for these models) to Phase 1's two smaller models, using
the same protocol as the Qwen3-8B comparison above: layer selection and
threshold calibration both on VAL (`src.detectors.dense_direction_detector
.select_layer_and_calibrate` -- see METHODOLOGY.md for why this is cleaner
than reusing a TEST-selected layer), final metrics on TEST, same 35-prompt
adversarial paraphrase set (activations freshly extracted per model, real
JailbreakBench prompts reused unchanged from the Qwen3-8B run). Keyword and
perplexity baselines are **not** re-run here -- they score prompt text only,
independent of target model, so their Qwen3-8B numbers above apply
unchanged to every model.

Note: the layer selected here (20 for Qwen2.5-1.5B, 14 for SmolLM2-1.7B) is
chosen on the full 4-dataset corpus's VAL split, not Phase 1's original
small-scale AdvBench/Alpaca-only calibration split -- so it differs from
Phase 1's reported "best layer" (23 and 20 respectively) by construction,
not by error; the two layer-selection procedures are answering different
questions (best layer for this classifier vs. best layer for causal
ablation on a narrower dataset).

| model | layer | TEST accuracy | TEST AUROC | XSTest-safe correctly-not-flagged | adversarial (pooled) | GCG | PAIR |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 20 | 89.6% [85.5%, 92.6%] | 0.970 | 75.7% [59.9%, 86.6%] | 42.9% [28.0%, 59.1%] | 50.0% [26.8%, 73.2%] | 38.1% [20.7%, 59.1%] |
| SmolLM2-1.7B-Instruct | 14 | 87.8% [83.6%, 91.1%] | 0.945 | **100.0%** [90.6%, 100%] | **91.4%** [77.6%, 97.0%] | **92.9%** [68.5%, 98.7%] | **90.5%** [71.1%, 97.4%] |
| Qwen3-8B (from above) | 23 | 88.9% [84.7%, 92.0%] | 0.983 | 94.6% [82.3%, 98.5%] | 62.9% [46.3%, 76.8%] | 92.9% [68.5%, 98.7%] | 42.9% [24.5%, 63.4%] |

**All three models achieve comparably strong TEST-split accuracy** (87.8-89.6%,
AUROC 0.94-0.98) -- the dense-direction approach isn't fragile to model
choice on clean, in-distribution prompts.

**Striking, unexplained cross-model difference on PAIR paraphrase**:
SmolLM2's dense-direction detector holds up dramatically better under PAIR
(90.5%) than either Qwen model (Qwen2.5: 38.1%, Qwen3-8B: 42.9%). Tested
formally with Cochran's Q (`src.eval.detector_metrics.cochrans_q` --
generalizes McNemar's paired test from two to three related classifiers
scored on the same 21 items, the correct tool here instead of eyeballing
three pairwise CIs): **Q = 13.06, df = 2, p = 0.0015** -- clearly
significant, not just non-overlapping-CIs-shaped. This is not explained by
this project's data alone. One
plausible connection (not established, just a candidate hypothesis worth
testing later): Phase 1 found SmolLM2's baseline refusal behavior itself is
weaker and less "linear" than Qwen's -- lower baseline AdvBench refusal
rate (63% vs. Qwen2.5's 100%), and its activation-addition sufficiency
effect capped at 42% instead of reaching Qwen's ~97-100% (see this
document's "Cross-model comparison" section above). A refusal
representation that's less cleanly linear to begin with might, for reasons
this project hasn't investigated, end up less disrupted by surface-level
paraphrasing specifically -- or this could be unrelated model-specific
noise. **Not claimed as established** -- flagged as a concrete, testable
hypothesis for Phase 6's cross-model generalization work, not asserted as
an explanation.

**XSTest-safe false-positive rates also vary substantially by model**
(75.7% / 94.6% / 100.0% correctly-not-flagged for Qwen2.5 / Qwen3-8B /
SmolLM2 respectively) -- Qwen2.5-1.5B's dense-direction detector flags
roughly 1 in 4 safe-but-scary-looking prompts as harmful, a real
practical difference in "safety tax" across models using the exact same
detection method.

### Known limitations (cross-model dense-direction comparison)

- **n=35 adversarial prompts (21 PAIR), shared across all three models** --
  the same small-sample caveat from the Qwen3-8B section applies here too,
  and more so for the striking SmolLM2-vs-Qwen gap above: formally
  significant (Cochran's Q, p = 0.0015) at this sample size, but not yet
  understood mechanistically.
- **The SmolLM2 hypothesis above is untested.** Confirming or ruling it out
  would need, at minimum, checking whether the same pattern holds on a
  larger/different adversarial set, and probably a deeper look at how
  SmolLM2's refusal direction differs geometrically (e.g. via the
  "different but functionally similar directions" framing in
  arXiv:2602.02132, surveyed in LITERATURE.md).
- **Baselines are asserted, not re-verified, to be model-agnostic.** This is
  true by construction (keyword/perplexity scores never touch model
  activations), but wasn't independently re-run per model as a sanity check.
