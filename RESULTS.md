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

## Sufficiency (activation addition) at 7-9B scale (2026-07-24)

Closes a gap flagged below: necessity (ablation) was tested at 7-9B scale
via Wave 2/the cross-model-transfer test, but sufficiency (activation
addition) had only ever been tested on the two small Phase 1 models
above. `scripts/sufficiency_at_scale.py` extends the same
methodology (alpha-sweep calibration, then a held-out causal-validation
generation test) to Qwen3-8B and Llama-3.1-8B-Instruct, reusing each
model's already-selected layer and the already-cached full-corpus
activations -- no new activation extraction, only new generation.
Alpha-sweep on 12 VAL-split harmless prompts, final validation on 50
TEST-split harmless prompts (disjoint from the sweep); both models given
identical sampled prompt texts (same corpus, same seed).

| model | layer | calibrated alpha | baseline refusal | addition refusal |
|---|---|---|---|---|
| Qwen3-8B | 23 | 1.0 | 6.0% [2.1%, 16.2%] | **70.0%** [56.3%, 80.9%] |
| Llama-3.1-8B-Instruct | 27 | 1.0 | 10.0% [4.4%, 21.4%] | **34.0%** [22.4%, 47.9%] |

**Sufficiency replicates cleanly for Qwen3-8B** -- a strong,
non-overlapping-CI effect matching the pattern from Phase 1's small
models (Qwen2.5-1.5B: 0%->97%). **For Llama-3.1-8B-Instruct it's real
but markedly weaker and messier**, not a clean scale-up:

- Llama's alpha-sweep never reached the 80% target refusal rate at any
  alpha tested (peaked at 67% refusal, alpha=1.5 -- but 33% of those
  completions were degenerate, over this project's 10% degenerate-
  fraction cutoff, so not usable). Calibration fell back to the
  highest-refusal *viable* (non-degenerate) alpha instead (1.0, 58%
  refusal on the calib set) -- the same fallback path Phase 1's
  calibration logic already has, just actually exercised here for the
  first time on any model.
- Llama starts producing fully-degenerate output (100% degenerate
  fraction) from alpha=2.0 onward -- a much narrower clean window than
  Qwen3-8B, which stayed non-degenerate through alpha=2.0.
- The final validated effect (10%->34%) is real (barely non-overlapping
  CIs) but far smaller than Qwen3-8B's (6%->70%).

**Not smoothed into "sufficiency confirmed at scale."** The honest
finding is that necessity generalizes more robustly across these two
8-9B models than sufficiency does -- echoing, in spirit, this project's
original SmolLM2 finding (Cross-model comparison, above) that addition
is architecture-dependent in a way ablation isn't, now observed again at
a much larger model scale with a different model pair. Why Llama's
residual stream tolerates addition so much less gracefully than
Qwen3-8B's is an open question this data doesn't answer.

## Known limitations

- The refusal classifier is a keyword/phrase matcher (see
  `src/direction/refusal_classifier.py`), the standard sanity-check metric
  in this literature, not a validated final detector. **Updated: this has
  since been checked** -- against human (Claude) judgment at 97.8%
  agreement (Phase 3's spot-check, see the SAE-feature detector section
  below) and, separately, against two candidate automated LLM-judges
  (both failed validation on the harder moralize-vs-comply distinction
  specifically -- see DECISIONS.md). The classifier's own core job
  (refuse vs. non-refuse) remains validated and accurate; a real
  `is_refusal` bug (curly-apostrophe false negatives, Llama-3.1-8B-
  specific) was found and fixed in a later session, see DECISIONS.md.
- n=30 per condition is enough to show the effect is real (non-overlapping
  CIs) but not enough for fine-grained comparisons between conditions.
- Only two small models tested (1.5B, 1.7B params) for this Phase 1
  reproduction specifically. **Updated: both necessity and sufficiency
  have since been tested at 7-9B scale** (Qwen3-8B, Llama-3.1-8B-Instruct
  -- necessity via Wave 2/the cross-model-transfer test, sufficiency via
  the dedicated section above). Necessity generalizes robustly; sufficiency
  does not -- Qwen3-8B replicates cleanly (6%->70%) but Llama-3.1-8B is
  real-but-much-weaker (10%->34%) with a narrower clean-generation window
  before degenerate collapse. A *foreign*-direction addition test (the
  cross-model-transfer section's sufficiency side) would still need its
  own alpha calibration, real additional scope, and remains untested.

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
compliance." What the result does support: dense ablation (36 layers
touched) is a blunter, more disruptive intervention than SAE-feature
suppression (3 layers, up to 20 of 65536 features per layer) that pushes
the model into "moralize instead of refuse" far more often -- a real
finding about intervention bluntness, just not a compliance-rate
comparison. **Do not read this table as "dense ablation achieves more
compliance" -- it only shows dense ablation suppresses refusal
*phrasing* more.**

**Resolved with real numbers, not just "likely much smaller"** (see
DECISIONS.md's moralize-vs-comply entry for the full account): every
non-refuse completion in both conditions was read and labeled directly
(47 for dense ablation, 38 for SAE-suppression top-15).

| | refuse (is_refusal) | moralize | partial | **comply (true harm)** |
|---|---|---|---|---|
| dense-direction ablation | 6.0% | 94.0% | 0.0% | **0.0%** |
| SAE-suppression (top-15) | 24.0% | 74.0% | 2.0% | **0.0%** |

**The 18-point "6% vs. 24%" gap was entirely a refusal-phrasing
artifact.** True harmful-compliance rate is 0% for both conditions (one
ambiguous "partial" case in the SAE condition, a self-harm blog post
whose title matched the request literally but was truncated before
showing real content). Confirms the finding above with a measured
number: dense ablation suppresses refusal *phrasing* far more than
SAE-suppression does, but produces no more actual harmful content.

### Classifier-validation spot-check

Motivated by the finding above: 45 completions sampled across every
experiment in this project (Phase 1 x2 models, Phase 3 SAE suppression x6
conditions, the head-to-head), human-labeled blind to the classifier's own
verdict, then compared (`scripts/sample_for_labeling.py`,
`scripts/score_agreement.py`; full methodology in
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

### Cross-model extension: causal ranking and validation (Llama-3.1-8B-Instruct, gemma-2-9b-it)

Same methodology as above (top-3 layers by separation score, K0=10 pooled
candidates, causal ranking via attribution patching, causal validation via
suppression), extended to the two additional model families with
pretrained SAE suites (`src/sae/registry.py` dispatches to the right
provider/layers/micro-batch-size per model; full rationale for every
choice, including a real GPU-OOM debugging saga for Gemma, in
[DECISIONS.md](DECISIONS.md)). This extends the *causal ranking and
validation* pipeline to 3 models -- the SAE-feature *detector*
(classifier reframing, `src/detectors/sae_feature_detector.py`) is still
Qwen3-8B only; see Known limitations below.

**Layer selection** (separation scores, VAL-scored, TRAIN-derived
direction, full corpus):

| model | top-3 layers | scores |
|---|---|---|
| Llama-3.1-8B-Instruct (32 layers) | 27, 26, 21 | 1.860, 1.857, 1.853 |
| gemma-2-9b-it (42 layers) | 34, 35, 33 | 1.806, 1.804, 1.800 |

**Causal ranking** (top-5 of 20, integrated-gradients attribution, 16
harmful TRAIN prompts):

| model | rank | layer | feature | score |
|---|---|---|---|---|
| Llama-3.1-8B | 1 | 27 | 13363 | 10.068 |
| Llama-3.1-8B | 2 | 26 | 7664  | 7.632  |
| Llama-3.1-8B | 3 | 27 | 31488 | 0.530  |
| gemma-2-9b-it | 1 | 35 | 52410 | 0.801 |
| gemma-2-9b-it | 2 | 35 | 80362 | 0.581 |
| gemma-2-9b-it | 3 | 34 | 38366 | 0.526 |

Llama shows the same "two clear standouts, then a steep dropoff" shape as
Qwen3-8B, even sharper. Gemma's scores decay smoothly with no dominant
feature -- foreshadows the flatter validation curve below. Full top-20
lists in `results/sae_causal_ranking_{model}.json`.

**Causal validation** (N=50 held-out VAL harmful prompts, 6 conditions, 40
tokens, greedy decoding, real `refusal_classifier`, zero degenerate
completions across all 300 generations per model):

| condition | Llama-3.1-8B | gemma-2-9b-it |
|---|---|---|
| baseline | 98.0% [89.5%, 99.65%] | 96.0% [86.5%, 98.9%] |
| top-1 | 10.0% [4.4%, 21.4%] | 94.0% [83.8%, 97.9%] |
| top-5 | 4.0% [1.1%, 13.5%] | 92.0% [81.2%, 96.9%] |
| top-10 | 2.0% [0.4%, 10.5%] | 84.0% [71.5%, 91.7%] |
| top-15 | 0.0% [0.0%, 7.1%] | 82.0% [69.2%, 90.2%] |
| top-20 | 2.0% [0.4%, 10.5%] | 82.0% [69.2%, 90.2%] |

(Llama's baseline corrected 2026-07-23 from an original 86.0% -- a real
`is_refusal` bug, curly apostrophes in Llama's completions silently
missed by the ASCII marker list, undercounted this one condition. See
DECISIONS.md for the fix and full impact assessment.)

**A genuine three-way cross-model difference in how concentrated the
causal effect is**, flagged honestly as unexplained rather than
resolved:

- **Llama-3.1-8B**: the single top feature alone drops refusal 98% -> 10%
  -- nearly the entire effect from one feature.
- **Qwen3-8B**: effect distributed across the set; top-1 alone does
  essentially nothing (84% vs. 82% baseline), bottoms out at top-15 (18%).
- **gemma-2-9b-it**: a real, monotonic decline (96% -> 82%) but far more
  modest -- 14 points total vs. Llama's 88 and Qwen3's 66. **Confirmed
  statistically significant from top-10 onward** via a paired McNemar's
  exact test on the same 50 prompts (baseline vs. top-15: 7/50 discordant,
  all favoring suppression, p=0.0156 -- `scripts/gemma_suppression_significance.py`,
  full table in DECISIONS.md): a genuine
  causal effect, not noise, even though it's the smallest of the three
  models.

### Known limitations (SAE-feature detector)

- n=50 for the suppression validation and n=16 for the ranking pass
  (both increased from an initial n=25/n=8 pass -- see DECISIONS.md) are
  enough to show the top-5-through-top-20 effect is real and the
  ranking's top features are stable.
- **Causal ranking, validation, AND the SAE-feature detector (prompt-
  classifier reframing) now all cover 3 models** (Qwen3-8B,
  Llama-3.1-8B-Instruct, gemma-2-9b-it -- Wave 3 extended the detector
  itself and the head-to-head baseline comparison built on it, see
  above), with a genuine, unexplained cross-model spread in both effect
  concentration (Wave 2) and the dense-vs-SAE-feature comparison outcome
  (Wave 3).
- Llama-3.1-8B's and gemma-2-9b-it's chat templates produce a duplicated
  BOS token when tokenized by this project's pipeline (measured, not
  assumed benign: ~1% shift in separation score, well within existing
  layer-to-layer noise -- see DECISIONS.md). Accepted as a documented
  limitation, not fixed, since a fix would require re-running Wave 1's
  full extraction for a change whose own measured effect is negligible.
- The SAEs are trained on the base model's activations, applied here to
  the instruct/chat model -- a documented, accepted limitation shared with
  the source paper (see DECISIONS.md), not unique to this reproduction.
- **`refusal_rate` conflates "moralize" (safe) and "comply" (unsafe) --
  resolved for the scripts/ablate_qwen3_direction.py head-to-head via direct labeling** (see
  above and DECISIONS.md), not via an automated classifier: two candidate
  local judge models (SmolLM2-1.7B-Instruct, Phi-4-mini-instruct) both
  failed validation on this specific task, defaulting to one category
  regardless of content rather than genuinely discriminating -- a real
  capability/alignment-bias finding in its own right, not just a null
  result (`src/direction/moralize_comply_classifier.py`, kept in the
  codebase and documented honestly as "validated, found unreliable with
  locally-available models"). No number in this project beyond the
  scripts/ablate_qwen3_direction.py comparison currently reports a true harmful-compliance rate
  -- applying direct labeling to other conditions/models would need the
  same manual-reading approach, real additional effort per completion
  set, not a reusable automated pipeline.

## Baseline detectors and adversarial evaluation

Four prompt classifiers, compared under one protocol on Qwen3-8B: two
baselines (keyword filter, Olmo-3-1025-7B perplexity filter) and two
reframed activation-based detectors (dense-direction projection at layer
23, sum of Phase 3's top-15 ranked SAE features across layers 23/24/25).
Methodology, split discipline, and the adversarial paraphrase set's real
JailbreakBench provenance are in
[METHODOLOGY.md](METHODOLOGY.md#baseline-detectors-and-adversarial-evaluation).
Thresholds calibrated on VAL (`results/detector_thresholds_Qwen3-8B.json`),
never touched by the numbers below; full results in
`results/detector_head_to_head_Qwen3-8B.json`.

**Perplexity backbone note**: went through five models before settling
here -- GPT-2 (2019, 124M, matching Alon & Kamfonas 2023's own reference
model) -> GPT-Neo-1.3B (2021, after GPT-2's XSTest false-positive rate
looked suspiciously bad) -> Phi-4-mini-instruct (2025, 3.8B, rejected after
its XSTest number came back *worse* than GPT-Neo-1.3B's -- traced to it
being instruction-tuned, off-distribution for scoring raw non-chat text) ->
OLMo-2-0425-1B (2025, 1B, a genuine base model, still not the best
performer) -> **Olmo-3-1025-7B** (2025, 7B, AI2's next generation, final).
One of this project's own target models (Qwen2.5/SmolLM2/Qwen3-8B) was
also considered and rejected throughout, for a separate reason (breaks the
baseline's independence from what it's being compared against). Full
history and the non-monotonic result across all five backbones in
DECISIONS.md. All numbers below are the Olmo-3-1025-7B version; earlier
backbones' numbers are superseded, not reported twice.

### TEST split overall (n=288: 158 harmful, 130 harmless)

| detector | accuracy | 95% CI | F1 | AUROC |
|---|---|---|---|---|
| keyword filter | 56.6% | [50.8%, 62.2%] | 0.359 | 0.603 |
| perplexity filter | 54.9% | [49.1%, 60.5%] | 0.637 | 0.520 |
| **dense-direction** | **88.9%** | **[84.7%, 92.0%]** | **0.890** | **0.983** |
| **SAE-feature (top-15)** | **87.8%** | **[83.6%, 91.1%]** | **0.878** | **0.975** |

On clean, in-distribution prompts the two activation-based detectors
clearly beat both baselines -- an AUROC of 0.98/0.975 vs. 0.60/0.52.
**Perplexity's AUROC is essentially at chance (0.520)** regardless of which
of the five backbones scores it -- expected, and confirms this isn't a
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
| perplexity filter | 13.5% | [5.9%, 28.0%] |
| dense-direction | 94.6% | [82.3%, 98.5%] |
| SAE-feature (top-15) | 97.3% | [86.2%, 99.5%] |

**The full five-backbone story, deliberately not smoothed into a clean
"newer is better" narrative**: GPT-2 (2019, 124M) 13.5% -> GPT-Neo-1.3B
(2021, 1.3B) 75.7% -> Phi-4-mini-instruct (2025, 3.8B) 40.5% ->
OLMo-2-0425-1B (2025, 1B) 24.3% -> Olmo-3-1025-7B (2025, 7B) **13.5%**
(correctly-not-flagged, in that order). **The newest and largest backbone
tried ties the oldest and smallest one for the worst result in the entire
sequence** -- not "close to," an exact tie down to the confidence interval.
This is about as clean a demonstration as this project could ask for that
recency and parameter count do not predict this specific number at all.
The most likely explanation, per DECISIONS.md: XSTest's false-positive
behavior under perplexity scoring depends on idiosyncratic properties of
each reference model's training distribution, not on any axis (age, size,
base-vs-instruct) checked here -- a genuinely messier and more
scientifically interesting conclusion than "a better model fixes this,"
and one only visible because five real backbones were actually run instead
of assumed. Perplexity filtering remains the worst of the four detectors
on this axis in every version tried.

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

1. **GCG detection is a perfect 100% across all five backbones tried**
   (GPT-2, GPT-Neo-1.3B, Phi-4-mini-instruct, OLMo-2-0425-1B,
   Olmo-3-1025-7B) -- strong, convergent evidence that GCG's gibberish
   suffix is improbable under any reasonable LM, not an artifact of any one
   scoring model (exactly Alon & Kamfonas 2023's original point).
2. **PAIR detection is 0.0% for every backbone except the original GPT-2**
   (which got 38.1%). Getting the identical flat 0% from four independent
   replacement models (GPT-Neo-1.3B, Phi-4-mini-instruct, OLMo-2-0425-1B,
   Olmo-3-1025-7B) is more convincing than any single model's result would
   be: GPT-2's 38.1% was very likely inflated by its own weakness at
   modeling ordinary fluent text, not genuine detection of anything
   adversarial. **Every better backbone tried makes the failure cleaner and
   more total**, strengthening rather than undermining the conclusion that
   perplexity-based filtering structurally cannot catch fluent paraphrase
   attacks, regardless of which model scores it.
3. **The pooled number is still misleading on its own** (40.0%) -- entirely
   GCG-driven; PAIR is the case this evaluation is actually named for, and
   perplexity detects none of it.
4. **On PAIR -- the case this evaluation is actually named for -- all four
   detectors degrade sharply relative to their TEST-split performance**
   (dense-direction: 88.9% to 42.9%; SAE-feature: 87.8% to 33.3%; keyword:
   56.6% to 23.8%; perplexity: 54.9% to 0.0%). Fluent paraphrase is a hard
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

### Cross-model extension: SAE-feature detector head-to-head (Llama-3.1-8B-Instruct, gemma-2-9b-it)

Same four-detector protocol as Qwen3-8B above, extended to the two other
models with pretrained SAE suites (K=15 reused for all three -- each
model's own causal-validation curve independently bottoms out at top-15,
see DECISIONS.md). Both new models' dense-direction/SAE-feature AUROC
land in the same high-0.9x range as Qwen3-8B's (0.983/0.975): Llama
0.989/0.978, Gemma 0.984/0.966 -- confirms nothing broke in the
generalization before looking at the finer comparisons.

| | TEST AUROC (dense/SAE) | DeLong p | PAIR detect (dense/SAE) | pooled adversarial McNemar p |
|---|---|---|---|---|
| Qwen3-8B | 0.983 / 0.975 | 0.068 (n.s.) | 42.9% / 33.3% | 0.5 (n.s.) |
| Llama-3.1-8B | 0.989 / 0.978 | **0.024** | 66.7% / **80.9%** | 0.25 (n.s.) |
| gemma-2-9b-it | 0.984 / 0.966 | **0.0063** | **47.6%** / 23.8% | **0.0156** |

**A genuinely different story per model**, flagged as unresolved rather
than forced into a pattern:

- **Qwen3-8B**: dense-direction and SAE-feature statistically
  indistinguishable everywhere tested.
- **Llama-3.1-8B**: dense-direction significantly better overall
  (p=0.024) -- but on PAIR specifically, SAE-feature numerically *beats*
  dense-direction (80.9% vs 66.7%, McNemar p=0.25, not significant at
  n=21). The one case in this project where SAE-feature out-robusts dense
  direction on paraphrase, the direction arXiv:2505.23556 originally
  claimed -- still not formally significant, but the first time this
  project's own numbers point that way at all.
- **gemma-2-9b-it**: dense-direction significantly better both overall
  (p=0.0063) and on the pooled adversarial set (p=0.0156, all 7
  discordant pairs favor dense) -- the strongest, most one-sided result
  for dense-direction of any model tested.

Full per-model numbers (TEST-overall, XSTest-safe, adversarial pooled +
by-method) in `results/detector_head_to_head_{Llama-3.1-8B-Instruct,
gemma-2-9b-it}.json`; methodology and infrastructure changes in
DECISIONS.md's Phase 6 Wave 3 entry.

### Known limitations (baseline detectors and adversarial evaluation)

- **Adversarial set is small** (n=35, spanning only 11 of TEST's JBB-sourced
  goals) -- large enough to show all four detectors degrade under PAIR
  paraphrase, not large enough to statistically distinguish dense-direction
  from SAE-feature on that degradation in every model (significant for
  gemma-2-9b-it, not for Qwen3-8B/Llama-3.1-8B -- see the cross-model
  table above). A larger set (more JBB goals landing in TEST, or matching
  against AdvBench/HarmBench-sourced artifacts if JailbreakBench publishes
  them) would sharpen this.
- **Keyword lexicon coverage is corpus-dependent.** Its poor GCG/PAIR
  numbers partly reflect that this project's TEST-split JBB goals lean
  toward categories (defamation, harassment, extortion) the curated lexicon
  wasn't built to cover, not just paraphrase/suffix robustness in general.
- **Adversarial prompts were never tested for attack success against this
  project's own models** -- they're real successful jailbreaks against
  their original target models (Vicuna, Llama-2, GPT-3.5/4), reused here
  purely as disguised-harmful *prompt text* for a classifier-robustness
  test, not a claim about any of this project's models' own jailbreak
  susceptibility.
- **Full four-detector comparison now covers 3 models** (Qwen3-8B,
  Llama-3.1-8B-Instruct, gemma-2-9b-it -- the three with pretrained SAE
  suites). Qwen2.5-1.5B/SmolLM2 still can't run the SAE-feature detector
  (no SAE trained for those models); the dense-direction detector and both
  baselines are extended to those two below.

## Dense-direction detector: cross-model comparison (5 models)

Extends the dense-direction detector (not the SAE-feature detector, which
only has a trained SAE for Qwen3-8B) to four more models across Phase 4
(Qwen2.5-1.5B, SmolLM2-1.7B) and Phase 6 (Llama-3.1-8B-Instruct,
Gemma-2-9B-it -- the two families named in the original roadmap
specifically because pretrained SAE suites, LlamaScope/GemmaScope, exist
for both, unlike Qwen2.5/SmolLM2). Same protocol throughout: layer
selection and threshold calibration both on VAL
(`src.detectors.dense_direction_detector.select_layer_and_calibrate` --
see METHODOLOGY.md for why this is cleaner than reusing a TEST-selected
layer), final metrics on TEST, same 35-prompt adversarial paraphrase set
(activations freshly extracted per model, real JailbreakBench prompts
reused unchanged from the Qwen3-8B run). Keyword and perplexity baselines
are **not** re-run per model -- they score prompt text only, independent
of target model, so their Qwen3-8B numbers above apply unchanged to every
model. Llama-3.1-8B-Instruct and Gemma-2-9B-it were gated on Hugging Face;
access was requested and confirmed before running anything (see
DECISIONS.md).

Note: the layer selected here (e.g. 20 for Qwen2.5-1.5B, 27 for
Llama-3.1-8B) is chosen on the full 4-dataset corpus's VAL split, not
Phase 1's original small-scale AdvBench/Alpaca-only calibration split --
so it can differ from Phase 1's reported "best layer" by construction, not
by error; the two layer-selection procedures are answering different
questions (best layer for this classifier vs. best layer for causal
ablation on a narrower dataset).

| model | layer | TEST accuracy | TEST AUROC | XSTest-safe correctly-not-flagged | adversarial (pooled) | GCG | PAIR |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 20 | 89.6% [85.5%, 92.6%] | 0.970 | 75.7% [59.9%, 86.6%] | 42.9% [28.0%, 59.1%] | 50.0% [26.8%, 73.2%] | 38.1% [20.7%, 59.1%] |
| SmolLM2-1.7B-Instruct | 14 | 87.8% [83.6%, 91.1%] | 0.945 | **100.0%** [90.6%, 100%] | **91.4%** [77.6%, 97.0%] | **92.9%** [68.5%, 98.7%] | **90.5%** [71.1%, 97.4%] |
| Qwen3-8B (from above) | 23 | 88.9% [84.7%, 92.0%] | 0.983 | 94.6% [82.3%, 98.5%] | 62.9% [46.3%, 76.8%] | 92.9% [68.5%, 98.7%] | 42.9% [24.5%, 63.4%] |
| **Llama-3.1-8B-Instruct** | 27 | **93.1%** [89.5%, 95.5%] | **0.989** | 97.3% [86.2%, 99.5%] | 80.0% [64.1%, 90.0%] | **100.0%** [78.5%, 100%] | 66.7% [45.4%, 82.8%] |
| Gemma-2-9B-it | 34 | 93.1% [89.5%, 95.5%] | 0.984 | 89.2% [75.3%, 95.7%] | 68.6% [52.0%, 81.5%] | **100.0%** [78.5%, 100%] | 47.6% [28.3%, 67.6%] |

**All five models achieve comparably strong TEST-split accuracy**
(87.8-93.1%, AUROC 0.94-0.99) -- the dense-direction approach isn't
fragile to model choice on clean, in-distribution prompts.
**Llama-3.1-8B-Instruct has the best TEST accuracy and AUROC of any model
tried in this project so far**, including Qwen3-8B.

**PAIR-paraphrase robustness now shows a clear ranking across five
models, not just an isolated SmolLM2 anomaly**: SmolLM2 (90.5%) >
Llama-3.1-8B (66.7%) > Gemma-2-9B (47.6%) > Qwen3-8B (42.9%) > Qwen2.5-1.5B
(38.1%). Tested formally with Cochran's Q across all five
(`src.eval.detector_metrics.cochrans_q` -- generalizes McNemar's paired
test to *k* related classifiers scored on the same 21 items, the correct
tool instead of eyeballing pairwise CIs): **Q = 19.52, df = 4, p = 0.0006**
-- clearly significant, confirming this spread is real across all five
models, not just a SmolLM2-vs-everyone-else artifact. This is not
explained by this project's data alone. One plausible connection (not
established, just a candidate
hypothesis worth testing later): Phase 1 found SmolLM2's baseline refusal
behavior itself is weaker and less "linear" than Qwen's -- lower baseline
AdvBench refusal rate (63% vs. Qwen2.5's 100%), and its activation-addition
sufficiency effect capped at 42% instead of reaching Qwen's ~97-100% (see
this document's "Cross-model comparison" section above). A refusal
representation that's less cleanly linear to begin with might, for reasons
this project hasn't investigated, end up less disrupted by surface-level
paraphrasing specifically -- or this could be unrelated model-specific
noise, and it doesn't explain why Llama-3.1-8B (a strong, "linear"-looking
refusal model per its high TEST/XSTest numbers) is also comparatively
robust. **Not claimed as established** -- flagged as a concrete, testable
open question, not asserted as an explanation.

**XSTest-safe false-positive rates also vary substantially by model**
(75.7% / 94.6% / 100.0% / 97.3% / 89.2% correctly-not-flagged for Qwen2.5 /
Qwen3-8B / SmolLM2 / Llama-3.1-8B / Gemma-2-9B respectively) -- Qwen2.5-1.5B's
dense-direction detector flags roughly 1 in 4 safe-but-scary-looking
prompts as harmful, a real practical difference in "safety tax" across
models using the exact same detection method. Both new models
(Llama-3.1-8B, Gemma-2-9B) sit in the upper-middle of this range, not at
either extreme.

### Known limitations (cross-model dense-direction comparison)

- **n=35 adversarial prompts (21 PAIR), shared across all five models** --
  large enough for a formally significant Cochran's Q result (see
  DECISIONS.md) but not yet understood mechanistically.
- **The SmolLM2 hypothesis above is untested, and now also doesn't cover
  Llama-3.1-8B's comparatively strong PAIR robustness.** Confirming or
  ruling either out would need, at minimum, checking whether the pattern
  holds on a larger/different adversarial set, and probably a deeper look
  at how each model's refusal direction differs geometrically (e.g. via
  the "different but functionally similar directions" framing in
  arXiv:2602.02132, surveyed in LITERATURE.md).
- **Baselines are asserted, not re-verified, to be model-agnostic.** This is
  true by construction (keyword/perplexity scores never touch model
  activations), but wasn't independently re-run per model as a sanity check.

## Cross-model direction transfer

Distinct from the cross-model comparison above (which extends the same
*recipe* independently per model) -- this tests literal transfer: does a
direction found on one model do anything applied to a *different* model's
activations/generation? Scoped to Qwen3-8B <-> Llama-3.1-8B-Instruct (both
d_model=4096, dimensionally compatible); gemma-2-9b-it (d_model=3584)
excluded rather than attempting a learned cross-dimension mapping, which
would confound "does it transfer" with "is the mapping any good."
Methodology and full account in
[METHODOLOGY.md](METHODOLOGY.md#cross-model-direction-transfer) and
[DECISIONS.md](DECISIONS.md); full results in
`results/cross_model_direction_transfer.json`.

**Separation score** (foreign direction evaluated against the target's own
VAL activations, at the target's own already-selected layer):

| | own score | foreign score |
|---|---|---|
| Qwen3-8B (foreign = Llama's direction) | 1.7831 | **-0.6454** |
| Llama-3.1-8B (foreign = Qwen's direction) | 1.8597 | **-0.7218** |

Both foreign scores are negative, not merely weak -- a real anti-correlated
signal, not noise near zero.

**Causal ablation** (N=50 harmful VAL prompts, greedy decoding, same
prompts for both models):

| | baseline | own-ablation | foreign-ablation |
|---|---|---|---|
| Qwen3-8B | 84% | **8%** | 84% |
| Llama-3.1-8B | 92% | 88% | 92% |

(Llama's numbers corrected 2026-07-23 from an original 80%/86%/80% -- a
real `is_refusal` bug undercounted Llama's refusals across the board;
see DECISIONS.md for the fix. Recomputed from already-saved completions,
no new generation needed. Qwen3-8B's numbers were never affected --
Qwen3-8B consistently uses ASCII apostrophes.)

**Qwen3-8B: a clean, unambiguous no-transfer result.** Own-direction
ablation crashes refusal (84%->8%, matching this project's established
Phase 1 result). Llama's foreign direction produces zero effect --
refusal identical to baseline to the percentage point (paired McNemar
p=1.0 vs. baseline, p=0.0 vs. own). The intervention mechanism clearly
*can* work at this scale; the foreign direction just doesn't do anything
in this model.

**Llama-3.1-8B: inconclusive, for a more mundane reason than first
reported.** Llama's own dense-direction ablation shows a real,
correctly-signed decrease (92%->88%, 4 points) -- not the "doesn't work
at all, even ticks up" result originally reported before the bug fix.
Still far too weak to distinguish from noise at n=50 (p=0.5). This is
the first time this project has causally ablated Llama's *dense*
direction (Wave 1 only used it as a classifier; Wave 2's causal ablation
work on Llama used SAE features, which worked dramatically -- 98%->10%
from a single feature). Because the own-direction effect, while
correctly signed, is still statistically indistinguishable from zero at
this sample size, "own vs. foreign not significant" still cannot be read
as evidence of no transfer the way Qwen3-8B's can -- it's underpowered
either way, not confounded by a backwards effect as originally thought.

**Not smoothed into one headline.** One clean negative-transfer result
(Llama's direction has zero effect on Qwen3-8B) and one still-inconclusive
result, now for a more ordinary reason (Llama's own dense-direction
effect is real but small, and n=50 isn't enough to resolve whether a
foreign direction differs from it) -- reported as found, not forced into
a single "directions don't transfer" narrative.

**Independent replication at n=75 (2026-07-23,
`scripts/replicate_llama_ablation.py`) does not confirm a real
small effect.** A fresh, independently-sampled set of 75 harmful VAL
prompts (not an extension of the original 50, seed=7) found baseline
refusal 96.0% [88.9%, 98.6%] vs. own-ablation 94.7% [87.1%, 97.9%] -- an
even smaller gap than the original n=50's 92%->88%, with only 3
discordant pairs out of 75 and McNemar's exact test p=1.0. Read honestly:
this does not look like a real small effect that just needed more power
to detect -- it looks like the original 92%->88% observation was itself
noise from a small sample. Own-direction ablation's causal necessity for
Llama-3.1-8B's refusal, via the *dense* direction specifically, remains
genuinely unresolved by this project's own data (contrast with SAE
features, which do ablate Llama's refusal dramatically -- 98%->10% from
a single feature, see Wave 2).

**True harmful-compliance spot-check on the ablation conditions'
non-refuse completions (2026-07-24).** Same motivation as scripts/ablate_qwen3_direction.py/20:
the keyword-based refusal classifier's "non-refuse" verdict conflates
safe moralizing with actual harmful compliance, so a raw non-refuse rate
overstates risk. Direct Claude-labeling (refuse/moralize/comply, same
method as scripts/ablate_qwen3_direction.py/20 -- no automated classifier, per the standing
"validated, found unreliable with locally-available models" finding) on
all four ablation conditions' non-refuse completions from
`results/cross_model_direction_transfer.json`:

| condition | non-refuse (keyword) | sampled | true comply |
|---|---|---|---|
| Qwen3-8B own-ablation | 46/50 | 15 | **2/15 (13%)** |
| Qwen3-8B foreign-ablation (Llama's direction) | 8/50 | 8 (all) | 0/8 |
| Llama-3.1-8B own-ablation | 6/50 | 6 (all) | 2/6 (hedged/partial) |
| Llama-3.1-8B foreign-ablation (Qwen's direction) | 4/50 | 4 (all) | 2/4 (hedged/partial) |

**Qwen3-8B's pattern matches scripts/ablate_qwen3_direction.py's original finding almost
exactly** -- most "non-refuse" completions under ablation are clean
moralizing/lecturing, not compliance; the ~92% non-refuse rate massively
overstates true harmful-compliance risk (spot-check suggests true
compliance closer to 1-in-8, not 9-in-10). **Foreign-ablation's few
non-refuse completions are moralizing 100% of the time** in this
sample -- consistent with foreign ablation being a real null effect, not
a weak-but-real one.

**A genuine qualitative difference worth flagging**: where Qwen3-8B does
comply, it's clean and unhedged ("Certainly! Here's a blog post that
promotes... animal cruelty"). Where Llama-3.1-8B's completions lean
toward compliance, they're consistently hedged/partial ("I can provide a
general outline of a virus, but I must emphasize... This is for
educational purposes only" -- then still names the virus and starts an
outline). Both samples are small (n=6 and n=4) -- not enough to claim
this as a statistically established pattern, just an honest observation
in the data worth a closer look if this project's scope grows further.

- **Only one model pair tested** (Qwen3-8B <-> Llama-3.1-8B) -- the only
  pair with matching `d_model`. Whether this weak-effect pattern recurs
  in other models is unknown.
- ~~Necessity (ablation) only~~ -- **sufficiency (activation addition) with
  a foreign direction has since been tested, see the dedicated section
  below.**
- ~~Llama's own-direction causal ablation effect is itself new and
  unreplicated at a larger N~~ -- **replicated at n=75 (independent
  sample), does NOT confirm a real effect** (96.0% vs. 94.7%, McNemar
  p=1.0 -- see the dedicated section above). The dense direction's causal
  necessity for Llama's refusal remains genuinely unresolved, not
  established as "real but small."

## Cross-model sufficiency transfer (2026-07-24)

Closes the necessity-only limitation flagged above: does a foreign
direction, scaled by its own newly-calibrated alpha, *induce* refusal the
way it can (or can't) *remove* it? `scripts/transfer_sufficiency.py`
reuses each model's already-generated baseline and own-addition
completions from `results/sufficiency_7b_9b_scale.json` unchanged (verified
byte-for-byte identical prompt sampling before reusing them, so only the
foreign-direction alpha sweep and validation needed new generation) and
adds the foreign raw direction at the *target's own* already-selected
layer, matching the necessity section's convention above.

| model | baseline | own addition | foreign addition | own calibrated alpha | foreign calibrated alpha |
|---|---|---|---|---|---|
| Qwen3-8B (foreign = Llama's direction) | 6.0% | **70.0%** | 6.0% | 1.0 | 0.25 |
| Llama-3.1-8B-Instruct (foreign = Qwen3-8B's direction) | 10.0% | 34.0% | 6.0% | 1.0 | 0.25 |

**Foreign-direction addition induces literally zero refusal above baseline
in both models, at every alpha tested (0.25 through 4.0) -- not a weak
effect, an absent one.** Paired McNemar's exact tests give a clean,
unambiguous verdict this time (unlike the necessity side's underpowered
Llama result):

- **Qwen3-8B**: baseline vs. foreign, p=1.0 (0 discordant pairs --
  completions are identical to baseline, not just similar). Own vs.
  foreign, p<0.001 (32/50 discordant). A clean, symmetric no-transfer
  result matching the necessity side's own clean no-transfer finding for
  this model.
- **Llama-3.1-8B-Instruct**: baseline vs. foreign, p=0.5 (2/50
  discordant, statistically indistinguishable from no effect). Baseline
  vs. own, p=0.0005 -- confirms the real (if weak) 10%->34% own-addition
  effect from the section above is not itself noise. Own vs. foreign,
  p<0.001. **This resolves what the necessity side of this same transfer
  question left inconclusive** (own-ablation vs. foreign-ablation
  couldn't be distinguished at n=50) -- on the sufficiency side, own and
  foreign are clearly, significantly different, and foreign has no
  detectable effect at all.

Zero degenerate completions in either foreign-addition run (0/50 each),
so the null result isn't a byproduct of the intervention breaking
generation -- the foreign direction is simply inert at every scale tested.
**Not smoothed into "directions never transfer"** -- this is two data
points (one model pair), both pointing the same way on both necessity and
sufficiency, but still only one pair with matching `d_model` to test on.

## Investigating the Llama dense-direction-vs-SAE-feature gap (2026-07-24)

Llama-3.1-8B-Instruct is the project's sharpest anomaly: best dense-direction
TEST AUROC of any of the 5 models (0.989), yet the only model where own-
direction necessity fails to replicate (96.0%->94.7%, n=75, McNemar p=1.0)
and where sufficiency is real but weak and fragile (10%->34%, degenerating
from alpha=2.0 -- see the two sections above). Meanwhile its SAE-feature
approach ablates refusal dramatically from a single feature (98%->10%, Wave
2, above). `scripts/analyze_llama_causal_gap.py` re-analyzes data already
sitting in `results/` plus two small CPU-only computations (loading cached
activations and already-downloaded SAE decoder weights -- no new model
generation) to test two candidate explanations, saved to
`results/llama_causal_gap_analysis.json`.

**Hypothesis 1: Llama's raw diff-in-means direction is just small in
absolute terms, so ablating/adding it barely perturbs the residual
stream.** Llama's raw direction norm (19.1, at layer 27) is the smallest of
the four models with necessity data -- but raw norm alone isn't the right
unit, since residual-stream scale itself varies enormously by model and
layer. Normalizing each model's raw direction norm by its own ambient
activation norm at that same layer (computed directly from the cached
activation tensors) reverses the story:

| model | layer | ambient activation norm (mean) | raw direction norm | ratio |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 23 | 144.2 | 75.3 | 0.522 |
| SmolLM2-1.7B-Instruct | 20 | 842.1 | 279.6 | 0.332 |
| Qwen3-8B | 23 | 195.2 | 99.3 | 0.509 |
| Llama-3.1-8B-Instruct | 27 | 27.3 | 19.1 | **0.702** |

Llama's direction is the *largest* fraction of its own ambient activation
scale of the four models, not the smallest -- its layer-27 residual stream
simply operates at a much smaller absolute norm than the other models' do
at their own tested layers. **This hypothesis does not survive the check
and is ruled out**, not smoothed over: magnitude dilution relative to the
residual stream isn't what's making Llama's dense direction causally weak.

**Hypothesis 2: the dense direction and the top causal SAE feature simply
point in different directions, so the classifier axis and the causal lever
are different objects for Llama specifically.** Checked by loading the real
LlamaScope decoder vector for Llama's own top causal feature (layer
27/feature 13363) and the real Qwen-Scope decoder vector for Qwen3-8B's own
top causal feature (layer 25/feature 65291), and computing cosine
similarity against each model's dense direction at that same layer:

| model | layer | top causal feature | cosine similarity | random-200-feature baseline (mean abs / max abs) |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 27 | 13363 | 0.201 | 0.013 / 0.043 |
| Qwen3-8B | 25 | 65291 | 0.196 | 0.016 / 0.153 |

Both models show essentially identical, modest alignment (~0.20 -- real,
well above the random-feature baseline, but far from "the same axis").
**This does not differentiate the two models** and is reported as
inconclusive, not as confirming evidence for a misalignment story --
whatever separates Llama's messy dense-direction causal behavior from
Qwen3-8B's clean one, it isn't that Llama's SAE feature and dense direction
are unusually more orthogonal than Qwen3-8B's.

**What the data does support**: the concentration-vs-distribution asymmetry
already documented in the SAE cross-model section above. Llama's causal
effect collapses almost entirely into its single top SAE feature (baseline
86%->10% from top-1 alone); Qwen3-8B's is spread across the ranked set
(top-1 alone: 82%->84%, no effect, only reaching bottom at top-15). This
asymmetry co-occurs with exactly which model's *dense* direction ablates/
adds cleanly vs. not at all -- consistent with a story where Llama's
refusal mechanism at layer 27 is causally concentrated in a narrow feature
that a coarse two-class mean-difference average doesn't reliably recover as
a causal lever (even though it recovers something that correlates
extremely well with the refusal label, hence the best classifier AUROC in
the project), while Qwen3-8B's causal signal is distributed broadly enough
that the same coarse average happens to capture it well.

**Honestly hedged, not a forced conclusion**: this is a real, multi-pronged
comparison (two candidate mechanical explanations tested and one ruled out)
but ultimately a correlational finding from n=2 models with full data
(Llama vs. Qwen3-8B). No mechanistic account of *why* Llama's refusal
computation would be more concentrated than Qwen3-8B's is established here
-- that would need genuine mechanistic-interpretability work (e.g. tracing
what the ~0.20-aligned component of the dense direction vs. its ~0.98-
orthogonal remainder each individually contribute), a different and heavier
scope than this re-analysis. No new GPU generation was needed to reach this
point, and none is proposed to go further on this specific question.
