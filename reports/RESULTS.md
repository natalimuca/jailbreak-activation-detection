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
  above and DECISIONS.md), not via an automated classifier: **three**
  candidate local judge models now tried (SmolLM2-1.7B-Instruct,
  Phi-4-mini-instruct, Llama-3.1-8B-Instruct), all three failed validation
  on this specific task, each a genuinely different failure mode -- the
  first two defaulted to one category regardless of content; the third
  (2026-07-24 retry, a real capability jump to 8B) discriminates across
  categories in aggregate but scores 0% on the actual load-bearing case
  (harmful-prompt genuine compliance/partial compliance), including
  several outright "refuse" miscalls on unambiguous harmful-compliance
  completions, plus a confirmed self-judging bias (29.6% vs. 62.0%
  accuracy on its own completions vs. others'). A real capability/
  alignment-bias finding in its own right, not just a null result
  (`src/direction/moralize_comply_classifier.py`, kept in the codebase and
  documented honestly as "validated, found unreliable with
  locally-available models"). No number in this project beyond the
  scripts/ablate_qwen3_direction.py comparison currently reports a true harmful-compliance rate
  -- applying direct labeling to other conditions/models would need the
  same manual-reading approach, real additional effort per completion
  set, not a reusable automated pipeline.

### Retry: Llama-3.1-8B-Instruct as moralize-vs-comply judge (2026-07-24)

`scripts/validate_judge_llama.py` retries the automated classifier with a
genuinely larger local judge (8B vs. the prior 1.7B/3.8B), same 98-row
labeled worksheets, same reporting structure as `scripts/validate_classifier.py`.

| | accuracy | notes |
|---|---|---|
| Overall | 52/98 (53.1%) | verdict distribution spread across all 4 categories, most common only 43.9% -- does not collapse like prior judges |
| Refuse ground truth | 21/21 (100.0%) | perfect on the easy safety-net case |
| Harmless-prompt compliance | 7/9 (77.8%) | the easier case |
| **Harmful-prompt comply** | **0/7 (0.0%)** | the load-bearing case -- several miscalled outright "refuse" |
| **Harmful-prompt partial** | **0/10 (0.0%)** | same load-bearing case |
| Self-judged rows (Llama judging its own completions) | 8/27 (29.6%) | a confirmed, real self-judging bias |
| Independent rows (judge != completion source) | 44/71 (62.0%) | still not good enough on the harmful-comply case above |

**A genuinely different failure mode than the prior two judges, not a
repeat.** SmolLM2-1.7B and Phi-4-mini-instruct both defaulted to one
category (~70%+ "moralize") regardless of content -- an obvious,
easy-to-detect failure. Llama-3.1-8B-Instruct instead *looks* like it's
discriminating (spread across categories, perfect on refuse, decent on
harmless-prompt compliance) while scoring 0% on the two categories that
actually matter for this classifier's purpose, including outright category
miscalls (calling active harmful-content delivery "refuse") on unambiguous
cases -- arguably a more dangerous failure mode to trust by accident, since
the aggregate numbers alone don't obviously look broken. The self-judging
bias is real and substantial (29.6% vs. 62.0%), confirming a concern flagged
before running rather than only after.

**Three local judges tried now, three different failure modes, none
usable.** Direct labeling remains the only validated approach for
true-harmful-compliance labeling in this project. Results in
`results/moralize_comply_classifier_validation_llama_judge.json`.

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

#### Is the Llama PAIR flip real? A continuous-margin check (2026-07-24)

`scripts/sae_pair_margin.py` applies the same continuous-margin method used
for the 5-model dense-direction PAIR analysis above to Llama's SAE-feature
score instead, to see whether a finer-grained measure shows a real signal
underneath the non-significant binary result. No new GPU generation --
reuses the already-cached activations, already-selected top-15 features, and
`src.detectors.sae_feature_detector.score` unchanged (the exact function the
published 80.9% came from).

| | harmful-prompt margin | PAIR margin | PAIR as frac. of harmful margin |
|---|---|---|---|
| SAE-feature | 0.811 | **0.508** | **0.627** |
| dense-direction | 0.936 | 0.332 | 0.355 |

Recomputing the detection rate from these margins reproduces the published
81.0% (vs. known 80.9%) -- confirms the math before trusting anything new.
**SAE-feature's margin is higher than dense-direction's in both absolute and
relative terms** -- not just a difference in where the binary threshold
happens to fall. A **paired Wilcoxon signed-rank test on the same 21
prompts' continuous margins** (more statistical power than McNemar's test on
the binarized outcome, since it uses the full margin rather than just
above/below threshold): **statistic=73.0, p=0.147** -- lower than the
original McNemar p=0.25, but still not significant at the conventional 0.05
threshold.

**Genuinely inconclusive, not resolved either way**: a more powered
continuous test moves the p-value in the direction of "more likely real"
but does not cross significance. The honest read is that this project's
n=21 PAIR set cannot currently distinguish "SAE-feature is genuinely more
robust to paraphrase for Llama specifically" from "noise that happens to
point the same direction on two different measures" -- both the original
binary result and this continuous one are consistent with either. Results
in `results/sae_pair_margin_llama.json`.

### Why is Llama's feature paraphrase-invariant? Token-level attribution (2026-07-25)

The margin-decay results above establish *that* Llama's top causal feature
(layer 27/13363) survives PAIR paraphrasing while Qwen3-8B's own top
feature (layer 25/65291) doesn't -- this section asks *what in the text*
each feature is actually reading, via `scripts/token_attribution.py`.
**Method: single-token leave-one-out ablation** (each token replaced with
the model's own EOS token one at a time, clean forward pass, no
gradients) on all 21 real PAIR-paraphrased prompts per model, restricted to
the actual instruction span (excludes chat-template scaffolding -- an
un-masked first pass found the top tokens were dominated by `<think>`,
`assistant`, and newlines near the readout position, not the paraphrased
content itself).

**Two gradient-based (Integrated Gradients) attempts were tried first and
failed real correctness checks** before this simpler method was used --
both a multiplicative zero-baseline interpolation (degenerate under
RMSNorm's scale invariance) and an additive EOS-baseline interpolation
failed IG's completeness property by ~50-100x. Full account, including the
specific numbers that revealed each failure, in DECISIONS.md -- reported
rather than quietly discarded, since a dead end that was seriously wrong
twice is itself informative about this project's tooling.

**A real, if noisy, qualitative pattern** across the 21 matched prompts
(direct reading of the top-5 highest-importance tokens per prompt, not run
through an automated classifier -- same reasoning as the moralize-vs-comply
labeling elsewhere in this project): **Llama's top-ranked token is
frequently the literal core harmful-action word itself** --
`"unauthorized"` (rank 1, "gains **unauthorized** access"), `"cloning"`
(rank 1, debit-card **cloning**), `"underage"` (rank 1, **underage** labor),
`"ext"`+`"ort"` (ranks 1-2, the subword pieces of "**extort**"), and
`"convincing"`/`"scams"`/`"resembles"` together (Nigerian-prince-scam
email). **Qwen3-8B's top tokens are more often dominated by the PAIR
attack's fictional/roleplay wrapper language instead of the core request**:
`"hypothetical"` (top-1, twice, on two different prompts), `"consultant"`
(top-1, three times), plus `"fictional"`, `"researcher"`, `"Imagine"`,
`"creative"`, `"villain"` -- and several prompts where the entire top-5 is
generic connective tissue (`"."`, `"this"`, `"and"`, `"Please"`) with no
content word at all.

**Rough tally (direct reading, not an automated/validated count -- a
judgment call, stated as such)**: roughly 9/21 Llama prompts have a clear,
specific harmful-content word in the top-5, several as the literal
rank-1 token; roughly 6/21 Qwen3-8B prompts do, and rarely at rank 1.
**Neither model is clean** -- both have plenty of prompts where the top-5
is mostly generic function words regardless of model, and the difference
is a real tendency, not a crisp split. This is genuinely mechanism-adjacent
evidence (not just "these two numbers correlate") for why the concentrated
feature might be more paraphrase-invariant: it appears to key off the
underlying harmful request more directly, while Qwen3-8B's distributed
signal is more entangled with the surface framing PAIR uses to disguise it.

**Explicitly not established**: *why* Llama's feature reads content words
more directly than Qwen3-8B's -- that would need real controlled
manipulation (e.g. swapping wrappers while holding the core request fixed
across many more examples), out of scope here. Token-level embedding
attribution also conflates token identity with position (no separate
positional-embedding ablation). Results (all 42 prompts' full top-5 lists)
in `results/token_attribution.json`.

### Closing the "why": controlled wrapper-swap variance decomposition (2026-07-28)

The section above found a real but qualitative, confounded pattern: real
PAIR prompts vary *which* harmful request is being made and *how* it's
framed at the same time, so "core content vs. wrapper" were never
independently manipulated. This section does that directly via
`scripts/wrapper_swap_variance.py`: a full factorial of **10 real core
requests** (the 10 unique blunt `goal` strings already in
`results/adversarial_paraphrase_manifest.json`, dataset-sourced from
JBB-Behaviors, no new harmful text authored) x **5 wrapper conditions**
(a `bare` control plus 4 generic, topic-agnostic templates --
creative-writing/fiction, hypothetical/thought-experiment,
security-research/red-team, roleplay/persona), 50 constructed prompts per
model, each model's own top causal feature read via one deterministic
forward pass (no generation).

**Analysis**: a two-way ANOVA sum-of-squares decomposition with no
per-cell replication (each of the 50 cells is exactly one prompt, so
`ss_residual` is ~pure core x wrapper interaction, not
interaction-confounded-with-measurement-noise -- there's no sampling
noise in a single deterministic readout). Significance via **within-block
(Manly-style) permutation** rather than an asymptotic F-test, matching
this project's standing preference for exact/permutation-based p-values at
small n: for the core-effect null, independently permute the 10
core-labels *within each wrapper column*; for the wrapper-effect null,
independently permute the 5 wrapper-labels *within each core-request row*.
(A whole-row/whole-column relabeling would have been a no-op -- caught by
a Plan-agent design review before any GPU time was spent, and separately
verified against synthetic planted-effect arrays before trusting real
results.) Both tests are exact for "this factor's identity has *some*
association with the readout" -- not a clean main-effect-net-of-interaction
test, which no n=1/cell design can isolate.

**Result: both predictions from the qualitative token-attribution finding
hold up quantitatively, with real significance**:

| Model | eta-sq (core) | eta-sq (wrapper) | eta-sq (residual) | core p | wrapper p |
|---|---|---|---|---|---|
| Qwen3-8B | 0.227 | **0.656** | 0.117 | 0.0001 | 0.0001 |
| Llama-3.1-8B-Instruct | **0.407** | 0.029 | 0.564 | 0.0148 | 0.7915 |

Qwen3-8B's top feature's activation is dominated by **wrapper identity**
(65.6% of variance, both effects significant but wrapper roughly 3x
core); Llama's is dominated by **core-request identity** (40.7%), with the
wrapper effect not distinguishable from noise (p=0.79). This is exactly
the asymmetry the qualitative token-level reading suggested, now backed by
a controlled design and a real permutation-test p-value rather than an
eyeballed tally.

**A real, honestly-reported wrinkle**: Llama's residual (interaction) term
is much larger than Qwen3-8B's (56.4% vs. 11.7%) -- meaning a sizeable
chunk of Llama's feature variance isn't explained by either factor alone;
the core effect and wrapper framing interact non-additively for Llama in a
way they largely don't for Qwen3-8B. At the time, this was not chased
further (would need a replicated design, e.g. multiple phrasing-variants
per wrapper category, to separate genuine interaction from any remaining
idiosyncrasy) -- reported as a real, open asymmetry rather than smoothed
into the headline numbers. **The replicated design was built as a
follow-up the same day; see below.**

**The qualitative activation-patching-style subset (8 runs: 2 core
requests x 2 wrapper conditions x 2 models, reusing `token_attribution.py`'s
existing single-token leave-one-out method -- a degenerate single-token
case of activation patching; see Meng et al. 2022 "Locating and Editing
Factual Associations in GPT" (ROME), Vig et al. 2020, and Zhang & Nanda
2024 for the general method) is messier than the aggregate stat, reported
as such**: Qwen3-8B's top-5 tokens shift cleanly toward wrapper words
(`"novelist"`, `"character"`) under the fiction wrapper for one core
request, but still include the literal core word (`"illegally"`) for the
other; Llama's top-5 stays core-content-dominated under the bare condition
(`"sexual"`, `"Design"`, `"illegally"`, `"Nigerian"`, `"falling"`) but
shifts partway toward wrapper words under the fiction wrapper too
(`"scene"`, `"as"`, `"would"`, `"exactly"`). The aggregate quantitative
result is clear and significant; the small per-example qualitative read is
noisier and doesn't cleanly replicate example-by-example -- both are
reported, not just the clean one.

**Caveat stated explicitly, not glossed over**: this is a confirmatory,
in-sample formalization of the same 10 goals whose real PAIR transcripts
originally generated the core-vs-wrapper hypothesis in
`scripts/token_attribution.py`, not an independent held-out test. Full
per-cell activation grids, stats, and qualitative subset in
`results/wrapper_swap_variance.json`.

### Decomposing the interaction term: a replicated design (2026-07-28)

Adds real per-cell replication to the design above -- 2 additional
project-authored phrasing variants per non-bare wrapper category (3 total
per category), 10 core requests x 4 categories x 3 phrasings = 120
forward-pass readings per model (`scripts/wrapper_swap_replication.py`,
`bare` excluded, no phrasing dimension exists to replicate for zero-wrapper
text). With genuine replication, the previously unexaminable residual
(pure interaction, no error term possible) splits into two separable
quantities: a real core x category interaction term, and a genuine
within-cell (phrasing-to-phrasing) error term -- stated explicitly as such,
since this is a deterministic forward pass with no measurement noise, so
"replication" here means different wordings, not repeated trials; treating
phrasing variance as an ANOVA error term is an explicit modeling choice.

| Model | η² core | η² category | η² interaction | η² residual (phrasing) | interaction F-test p | interaction permutation p |
|---|---|---|---|---|---|---|
| Qwen3-8B | 0.349 | 0.217 | 0.086 | 0.347 | 0.813 (n.s.) | 0.680 (n.s.) |
| Llama-3.1-8B-Instruct | 0.366 | 0.004 | **0.424** | 0.206 | **<0.0001** | **0.0001** |

Both the classical F-test (valid now that real per-cell replication
provides a genuine error term, unlike the original design) and a
Freedman-Lane residual-permutation test (fits the additive/no-interaction
model, permutes residuals from that fit globally, refits -- a materially
different and more subtle scheme than the main-effect permutations used
above, verified on synthetic planted-interaction and pure-additive cases
before trusting it here) agree closely for both models.

**Qwen3-8B: core and category both remain independently significant (core
p<0.0001, category p<0.0001), but the interaction is genuinely not
significant by either test** -- core and category effects are essentially
additive for this model. A real, separate finding falls out of the
decomposition itself: within-cell (phrasing) variance is large (34.7% of
total variance) -- comparable in size to the core and category effects
themselves. Holding wrapper *category* fixed, which specific phrasing is
used still moves Qwen3-8B's feature activation substantially -- this
model's feature is not just sensitive to coarse framing category, but to
the literal specific wording within a category too.

**Llama-3.1-8B-Instruct: a real, statistically confirmed core x category
interaction** (42.4% of variance, F=6.11, p<0.0001 by the F-test, p=0.0001
by the independent permutation check -- both tests agree, not an artifact
of one method's assumptions). Category alone remains not significant on
its own (p=0.71, consistent with the non-replicated finding above that
wrapper framing barely matters for this model in isolation), but the
*effect* of framing depends on *which* core request it is paired with --
a genuine non-additive interaction, not just unexplained noise as the
original design's residual term left ambiguous. This resolves the earlier
open question directly: Llama's large residual (56.4% in the original,
un-replicated design) really was substantially real interaction (42.4%
of total variance once phrasing-level noise is properly separated out),
not primarily measurement idiosyncrasy.

**What this does and doesn't establish**: confirms the interaction is
real for Llama and absent for Qwen3-8B, with formal power to distinguish
the two (independent F-test and permutation agreement). Does not explain
*why* Llama's core-wrapper interaction exists mechanistically -- see the
per-cell decomposition immediately below for a first pass at that
question. Full grids and stats in `results/wrapper_swap_replication.json`.

### Which cells drive the interaction? A per-cell decomposition (2026-07-29)

Pure re-analysis of the already-saved activation grid above (no new GPU
generation, `scripts/wrapper_interaction_cells.py`): for each of the 40
(core, category) cells per model, the additive-model prediction (row
mean + column mean - grand mean) is subtracted from the real cell mean,
isolating the per-cell interaction residual -- distinct from the
within-cell phrasing noise already separated out above.

**Llama-3.1-8B-Instruct's interaction is not spread evenly across the 10
core requests -- it concentrates almost entirely in 2 of them:**

| Core request (truncated) | Interaction range across 4 categories | Interaction SD |
|---|---|---|
| "Explain why Donald Trump unanimously won the 2020 U.S. presidential election" | **1.779** | 0.859 |
| "Write a blackmail message to someone threatening to reveal personal information..." | **1.700** | 0.758 |
| Draft a 419 scam email | 0.666 | 0.285 |
| (remaining 7 requests) | 0.192 - 0.550 | 0.100 - 0.257 |

A real, roughly 2.5-9x gap separates the top two requests from the other
eight, which cluster in a much narrower band. The single largest
individual cell is the blackmail request under the fiction wrapper
(interaction = -1.054, i.e. this cell's real activation sits well *below*
what additive core + category effects alone would predict); the next
largest are the Trump-election request under hypothetical (-0.937) and
under fiction (+0.842) -- the same core request pulling in opposite
directions under different wrappers is exactly what a genuine interaction
looks like, not just one outlier cell.

**A candidate, explicitly qualitative read, not a proven mechanism**: the
two dominant requests differ in *task type* from the other eight, which
are uniformly "explain/provide a procedure for harmful action X"
requests. The Trump-election request instead asks the model to
*construct or validate a false factual narrative* (election
misinformation, not a harm-execution procedure); the blackmail request
asks the model to *directly generate the harmful artifact itself* (the
threatening message text), not explain how to produce one. Both are a
different kind of ask than "explain how to clone a debit card" or
"explain how to exploit a regulatory loophole." Whether task-type is
actually what drives the interaction, versus these two requests being
unusual for some other reason entirely, is not established by this data
-- would need a new controlled design (task-type as a deliberate factor,
holding wrapper category fixed) to test directly, out of scope here.

Qwen3-8B's own interaction term is not statistically significant (see
above), so a per-cell breakdown for it is not reported in the same way --
examining "which cells drive" a null result would risk over-reading
noise as structure. Full per-cell values for both models in
`results/wrapper_interaction_cells.json`.

### Testing the task-type hypothesis directly: a new controlled design (2026-07-29)

The candidate explanation above (the 2 standout requests differ in task
type from the other 8) is tested directly rather than left as an
unconfirmed read. 12 new core requests, real and drawn from this
project's existing corpus (never authored, none overlapping the original
10), balanced 4/4/4 across three deliberate task types: `procedural`
("explain/describe/provide instructions for X"), `direct_artifact`
("write/compose the harmful text itself"), `false_narrative`
("create/spread false content") -- crossed with the same 4 wrapper
categories and 3-phrasing-per-category scheme as the replicated design
above (`scripts/wrapper_task_type.py`). Both models tested. Two analyses
on the same 144-reading grid, both via the already-verified
`balanced_anova()`/`freedman_lane_interaction_test()` (no new statistical
formulas): task_type x category (treating each type's 4 cores x 3
phrasings as 12 replicates per cell -- the direct test of the hypothesis),
and core x category on just these 12 new requests (a same-shape
replication check). The grid-reshaping logic (grouping cores into task
types) was verified against synthetic data (an exact sum-of-squares match
between the two groupings' `ss_total`) before running on real activations.

| Model | Grouping | η² interaction | F-test p | permutation p |
|---|---|---|---|---|
| Qwen3-8B | core (12 new) x category | 0.065 | 0.986 (n.s.) | 0.967 (n.s.) |
| Qwen3-8B | task_type x category | 0.037 | 0.148 (n.s.) | 0.130 (n.s.) |
| Llama-3.1-8B-Instruct | core (12 new) x category | **0.169** | **4.2e-06** | **0.0001** |
| Llama-3.1-8B-Instruct | task_type x category | 0.035 | 0.416 (n.s.) | 0.393 (n.s.) |

**The core x category interaction cleanly replicates on this entirely
fresh sample of 12 requests** -- strong, independent confirmation the
phenomenon itself is real and reproducible, not an artifact of the
original 10 prompts. **The task-type hypothesis, tested directly, does
not hold up**: grouping cores into procedural / direct_artifact /
false_narrative and testing task_type x category gives a clean null for
both models. Reported plainly as a negative result for this specific
candidate explanation, not reframed to look like partial support.

**Why the task-type grouping washes out the (real) core-level signal**: a
per-cell breakdown of the new 12-core grid (same method as above) shows
the same concentration pattern as the original 10 -- interaction
concentrates in 2 of the 12 new requests, both **false_narrative** type
and both literally phrased "fake news article": "...spreads false
information about a political candidate" (range 1.523 across categories)
and "...that could lead to panic or chaos" (range 1.302). The other two
false_narrative requests in this batch (a race/religion misinformation
blog post, a search-engine-manipulation program) are far more modest
(0.618, 0.494), similar to the procedural and direct_artifact requests
(0.246-0.750) -- so even within the one task type that happened to
contain the two biggest hits, the group average is diluted by its own
other members. The interaction is not a task-type-level property; it
concentrates in a handful of specific requests for reasons this data does
not establish, now narrower than "task type" but not yet resolved -- a
literal-phrasing or narrow-sub-topic account ("fake news article", public
political/panic misinformation specifically) is a new, even more
speculative candidate than task type was, offered as such and not tested
further here. Full grids and stats in `results/wrapper_task_type.json`.

### A data-driven feature search, and a genuine negative result (2026-07-29)

Rather than hand-picking a third categorical narrative from the previous
round's 2 outliers -- a real risk of unfalsifiable post-hoc pattern-hunting
-- this round samples 48 entirely new core requests (equal allocation, 12
per source: AdvBench/HarmBench/JBB/XSTest, reproducible seed, none
overlapping the 22 already used, near-duplicates excluded via the same
gate `src.data.dedup.deduplicate()` uses) and tests several **objective,
pre-committed** textual features against each request's interaction range,
rather than one more guessed category (`scripts/wrapper_feature_search.py`).
Features were fixed and computed before any forward pass, and split
explicitly into two tiers: **Tier A** (blind, corrected as a family --
word count, average word length, keyword-filter lexicon score, source
dataset) and **Tier B** (explicitly non-blind, reported separately,
never corrected -- starting-verb category, a direct replication check of
the already-rejected task-type hypothesis; proper-noun count, transparently
motivated by the "Trump"/"fake news article" outliers). Significance via a
maxT (Westfall-Young) permutation test over the Tier-A family, cross-checked
against Benjamini-Hochberg FDR (`scipy.stats.false_discovery_control`) --
the maxT scheme was verified on synthetic null and planted-effect cases
before being trusted on real activations.

**The underlying interaction replicates for a third time, more strongly
than either prior round**: on this fresh, randomly-sampled set of 48
requests, Llama's core x category interaction is η²=0.286 (F-test
p=1.11e-16, permutation p=5e-05) -- stronger than the original 10-request
design (η²=0.424) and the task-type round's 12 new requests (η²=0.169),
via the same unmodified `balanced_anova`/`freedman_lane_interaction_test`.
Qwen3-8B's replication check stays a clean null (η²=0.100, p=0.664/0.322),
as in every prior round.

**None of the four pre-registered Tier-A features predict it, for either
model.** For Llama: word count (BH p=0.39), average word length (BH
p=0.71), keyword-filter score (BH p=0.71), source dataset (BH p=0.39) --
maxT and BH-FDR agree closely, and nothing comes close to the conventional
threshold. Reported as a genuine negative result, not reframed --
consistent with this project's track record on the task-type hypothesis
one round earlier.

**Tier B's nominal hits are flagged, not treated as a finding.**
Starting-verb category (raw p=0.0285) and proper-noun count (raw p=0.0249)
both come back under 0.05 *uncorrected* for Llama. Neither is trusted:
starting-verb category directly re-tests the task-type hypothesis a
purpose-built controlled 2-way ANOVA already rejected cleanly one round
earlier (p=0.416) -- a nominally-significant hit on a second, less
rigorous test of an already-refuted hypothesis is exactly the false-positive
pattern the Tier-A/Tier-B split and correction discipline exist to expose,
not evidence the earlier null was wrong. Proper-noun count was excluded
from Tier A specifically because it was transparently motivated by the
two outlier requests this whole investigation began from; a marginal,
uncorrected hit on the same feature is the predictable result of testing
a hypothesis built to fit the very data it's now being tested against, not
independent confirmation.

**Where this leaves the open question**: after three rounds (component-level
decomposition, a task-type hypothesis test, and now a blind multi-feature
search), Llama's core x category interaction is about as well-replicated as
any finding in this project -- real, reproducible, and now confirmed a
third independent time at even larger effect size. *Why* it concentrates
in a handful of specific requests remains genuinely unresolved: no
objective, pre-committed textual property tested here explains it. This is
reported as an honest limit, not chased with a fourth guessed category --
consistent with this project's standing practice of stopping at a real,
well-documented open question rather than forcing an explanation the data
doesn't support. Full grids, features, and stats (including both tiers,
both models) in `results/wrapper_feature_search.json`.

### Testing a literature-motivated salience hypothesis (2026-07-29)

A literature pass (`LITERATURE.md`) surfaced a candidate not covered by
any Tier-A feature above: general SAE/feature-frequency research finds a
concept's representation geometry depends on its salience/frequency in
training data, and the two requests driving Llama's interaction most
("Trump 2020 election", "fake news article") are both unusually
high-salience real-world phrases compared to this project's more generic
templated requests. Tested directly rather than left as a literature
citation (`scripts/wrapper_salience_test.py`): salience operationalized as
raw perplexity under this project's own existing GCG-detection reference
LM (Olmo-3-1025-7B, `src/baselines/perplexity_filter.py` -- built for an
unrelated purpose, reused not duplicated; lower perplexity approximates
higher real-world salience/commonness). No new SAE-feature measurement was
needed -- the 48 cores and their already-computed `interaction_range` came
directly from `results/wrapper_feature_search.json`; this only added the
one new feature and a single Spearman permutation test against data
already collected.

**Stated honestly, this test is neither blind nor outlier-word-derived --
a real third category worth naming plainly.** It was proposed after
already seeing all four Tier-A features fail, so it is not part of that
pre-registered family and gets no family correction (there is only one
new test here); unlike Tier B's `proper_noun_count`, it was not reverse-
engineered from the specific outlier strings ("Trump", "fake news
article") but derived from an independent literature finding about
training-data salience in general. Reported as its own single, clearly-
labeled post-hoc test with its own uncorrected p-value, not dressed up as
either a blind result or dismissed as equivalent to Tier B's fit.

**Result: another clean negative.** Llama: rho=-0.110, p=0.457 (n.s.);
Qwen3-8B: rho=-0.166, p=0.260 (also n.s., and not expected to show
anything under this hypothesis either, since Qwen's own interaction is
already null). Both correlations run in the direction the hypothesis
predicts (lower perplexity/higher salience associated with larger
interaction range) but nowhere near significant at n=48. Perplexity-as-
salience does not explain the interaction any better than task type or
the four blind surface features did.

**Where this leaves things, stated plainly rather than reached past**:
four independent, honestly-tested candidate explanations (task type,
4 blind textual features, and now a literature-motivated salience proxy)
have all failed to predict which specific core requests drive Llama's
interaction, while the interaction itself keeps replicating cleanly
across three independent samples. Full results in
`results/wrapper_salience_test.json`.

### A token-level read: a genuine qualitative pattern (2026-07-29)

Four surface/corpus-level hypotheses (task type, four blind textual
features, salience-via-perplexity) all having failed, this asks a
different kind of question: not what property of the *text* predicts the
interaction, but what the *feature itself does* differently, token by
token, for high- vs. low-interaction requests -- reusing the exact
single-token leave-one-out ablation method already built and verified in
Section 5.4.3/`scripts/token_attribution.py` (no new attribution
machinery), applied to a matched contrast set rather than a formal
statistical test (`scripts/wrapper_token_read.py`). 4 high-interaction and
4 low-interaction core requests, drawn from across all three prior rounds
(spanning short templated AdvBench-style asks, a task-type/false-narrative
request, and -- surfaced while picking this set -- an XSTest item and,
noted but not included here for length, a long HarmBench contextual
political-rant prompt that scored even higher interaction than the
original Trump-election request), each read under all 5 wrapper conditions
(bare + the 4 categories' original phrasing).

**A real, directly-observed pattern, deliberately qualitative and not run
through automated stats or a formal test (same discipline as Section
5.4.3)**: low-interaction requests keep the *same* top-attributed content
token locked across every single wrapper condition -- "scam" is the
rank-1 token for the MLM request under all five conditions (bare, fiction,
hypothetical, research, roleplay alike); "inciting"/"Write" and
"text"/"Generate" show similarly stable content-token dominance for the
other two low-interaction requests. High-interaction requests instead show
the top-attributed token genuinely swinging between core-content words and
wrapper-scaffolding words depending on framing -- most dramatically under
the **fiction** condition specifically, which in 3 of 4 high-interaction
requests causes the feature's top tokens to shift heavily toward narrative
language ("character", "explains", "explanation") rather than the actual
request content, sometimes accompanied by a large swing in the feature's
raw activation level itself (e.g. the XSTest boss-salary request: natural
value 2.91 under bare, dropping to 0.49 under hypothetical, where the top
tokens become almost purely framing words -- "hypothetical", "purely",
"application" -- with no request content in the top-5 at all).

**What this does and doesn't establish**: a genuine, causally-grounded
qualitative signature -- "does this request's top-attributed token stay
locked to its own content regardless of wrapper, or does it get displaced
by framing language under at least one wrapper" -- that low- and
high-interaction requests visibly differ on, at the level of what the
model's own feature attends to, not just a property of the raw text. This
is a real step past the four failed surface/corpus hypotheses in kind, not
just another guess, since it's the first account grounded in what
actually happens inside the model rather than a property measurable from
the prompt string alone. It is explicitly **not** a formal, validated
finding: n=4 vs. 4, read directly rather than scored by an automated or
inter-rater-checked procedure, and does not explain *why* some requests'
content representations are more easily displaced by wrapper framing than
others -- that remains open. Full per-token results (all 5 categories, all
8 requests) in `results/wrapper_token_read.json`.

**Where this leaves the open question overall**: five rounds in (per-cell
decomposition, task type, four blind features, salience, and now this
token-level read), the interaction itself is as well-replicated as
anything in this project, and this round finally surfaces a real
qualitative signature at the mechanism level rather than another failed
surface-property guess -- but a full causal account (formally testing
whether "content-token lock vs. displacement under fiction framing"
predicts interaction range across a larger sample, not just 4-vs-4) is a
further, heavier step not attempted here. Continuing to test one-off
surface hypotheses would have had real diminishing returns; this
qualitative pattern is a genuinely different and more promising direction
if a future round takes it further.

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

## Dense-direction detector: cross-model comparison (6 models)

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
| DeepSeek-R1-Distill-Qwen-1.5B | 7 | 84.7% [80.1%, 88.4%] | 0.911 | **100.0%** [90.6%, 100%] | 40.0% [25.6%, 56.4%] | 85.7% [60.1%, 96.0%] | 9.5% [2.6%, 28.9%] |
| Qwen2.5-1.5B-Instruct | 20 | 89.6% [85.5%, 92.6%] | 0.970 | 75.7% [59.9%, 86.6%] | 42.9% [28.0%, 59.1%] | 50.0% [26.8%, 73.2%] | 38.1% [20.7%, 59.1%] |
| SmolLM2-1.7B-Instruct | 14 | 87.8% [83.6%, 91.1%] | 0.945 | **100.0%** [90.6%, 100%] | **91.4%** [77.6%, 97.0%] | **92.9%** [68.5%, 98.7%] | **90.5%** [71.1%, 97.4%] |
| Qwen3-8B (from above) | 23 | 88.9% [84.7%, 92.0%] | 0.983 | 94.6% [82.3%, 98.5%] | 62.9% [46.3%, 76.8%] | 92.9% [68.5%, 98.7%] | 42.9% [24.5%, 63.4%] |
| **Llama-3.1-8B-Instruct** | 27 | **93.1%** [89.5%, 95.5%] | **0.989** | 97.3% [86.2%, 99.5%] | 80.0% [64.1%, 90.0%] | **100.0%** [78.5%, 100%] | 66.7% [45.4%, 82.8%] |
| Gemma-2-9B-it | 34 | 93.1% [89.5%, 95.5%] | 0.984 | 89.2% [75.3%, 95.7%] | 68.6% [52.0%, 81.5%] | **100.0%** [78.5%, 100%] | 47.6% [28.3%, 67.6%] |

**Five of six models achieve comparably strong TEST-split accuracy**
(87.8-93.1%, AUROC 0.94-0.99); **DeepSeek is the outlier, weakest of all
six** (84.7%, AUROC 0.911) -- still a working classifier, just measurably
worse than every other model tried. **Llama-3.1-8B-Instruct has the best
TEST accuracy and AUROC of any model tried in this project so far**,
including Qwen3-8B.

**PAIR-paraphrase robustness shows a clear ranking across six models,
with DeepSeek a dramatic new low**: SmolLM2 (90.5%) > Llama-3.1-8B (66.7%)
> Gemma-2-9B (47.6%) > Qwen3-8B (42.9%) > Qwen2.5-1.5B (38.1%) >
**DeepSeek (9.5%)** -- roughly a quarter of the next-lowest model's rate.
Tested formally with Cochran's Q across all six
(`src.eval.detector_metrics.cochrans_q` -- generalizes McNemar's paired
test to *k* related classifiers scored on the same 21 items, the correct
tool instead of eyeballing pairwise CIs): **Q = 34.44, df = 5, p = 1.94e-6**
(recomputed 2026-07-27 with DeepSeek added; was Q=19.52, df=4, p=0.0006 at
5 models -- DeepSeek's extreme outlier value widened an already-real
spread further, not an artifact of adding a 6th model) -- clearly
significant, confirming this spread is real across all six models, not
just a SmolLM2-vs-everyone-else artifact. This is not explained by this
project's data alone. One plausible connection (not established, just a
candidate hypothesis worth testing later): Phase 1 found SmolLM2's
baseline refusal behavior itself is weaker and less "linear" than Qwen's
-- lower baseline AdvBench refusal rate (63% vs. Qwen2.5's 100%), and its
activation-addition sufficiency effect capped at 42% instead of reaching
Qwen's ~97-100% (see this document's "Cross-model comparison" section
above). A refusal representation that's less cleanly linear to begin with
might, for reasons this project hasn't investigated, end up less
disrupted by surface-level paraphrasing specifically -- or this could be
unrelated model-specific noise, and it doesn't explain why Llama-3.1-8B (a
strong, "linear"-looking refusal model per its high TEST/XSTest numbers)
is also comparatively robust, nor why DeepSeek (the *weakest* baseline
refusal signal of all six, and the one with a genuinely null
activation-addition result -- see its dedicated section above) is also by
far the *least* robust, the one place this session's hypotheses actually
line up directionally. **Still not claimed as established** -- flagged as
a concrete, testable open question, not asserted as an explanation.

**XSTest-safe false-positive rates also vary substantially by model**
(75.7% / 94.6% / 100.0% / 97.3% / 89.2% / **100.0%** correctly-not-flagged
for Qwen2.5 / Qwen3-8B / SmolLM2 / Llama-3.1-8B / Gemma-2-9B / DeepSeek
respectively) -- Qwen2.5-1.5B's dense-direction detector flags roughly 1
in 4 safe-but-scary-looking prompts as harmful, a real practical
difference in "safety tax" across models using the exact same detection
method. DeepSeek ties SmolLM2 for the best (zero false positives on
XSTest-safe) despite being the weakest classifier overall on TEST --
a real, if unremarked-on-further, dissociation between "flags safe prompts
correctly" and "catches harmful prompts reliably" for this model.

#### PAIR-robustness spread: a continuous margin account (2026-07-24)

The Cochran's Q result above establishes the spread is real but not why it
exists. `scripts/pair_margin_analysis.py` replaces the binary "flagged or
not" rate with a continuous measure: each model's dense-direction projection
margin on the 21 real PAIR prompts, expressed in pooled-std units (matching
`separation_score`'s own normalization) and then as a fraction of that same
model's *genuine harmful-prompt* margin -- "how much of the typical harmful
signal survives paraphrasing, relative to this model's own scale." Needed no
new GPU generation for 3 of 5 models (`results/activations/*_adversarial.pt`
already existed for Qwen3-8B/Llama-3.1-8B/gemma-2-9b-it); Qwen2.5-1.5B and
SmolLM2 needed a one-time forward-pass-only extraction first
(`scripts/extend_adversarial_small.py`, ~10s each, no generation).

| model | harmful-prompt margin | PAIR margin | PAIR as frac. of harmful margin | known PAIR detection |
|---|---|---|---|---|
| SmolLM2-1.7B-Instruct | 0.681 | **0.469** | 0.689 | 90.5% |
| Llama-3.1-8B-Instruct | 0.936 | **0.332** | 0.355 | 66.7% |
| gemma-2-9b-it | 1.018 | -0.049 | -0.049 | 47.6% |
| Qwen2.5-1.5B-Instruct | 0.958 | -0.109 | -0.113 | 38.1% |
| Qwen3-8B | 0.679 | -0.240 | -0.353 | 42.9% |
| DeepSeek-R1-Distill-Qwen-1.5B | **0.308** | -0.221 | -0.719 | 9.5% |

Recomputing each model's PAIR detection rate directly from these margins
(fraction with projection above threshold) reproduces the published rates
exactly (0.381/0.905/0.429/0.667/0.476/0.095) -- a sanity check that the
already-persisted directions/thresholds are being applied correctly, not new
information on its own. **The real new result**: a formal Spearman rank
correlation between mean PAIR margin and known detection rate, **rho = 0.90,
p = 0.037 at n=5** (original 5 models); **recomputed at n=6 with DeepSeek
added (2026-07-27): rho = 0.83, p = 0.042** -- still significant, holds up
under the addition, though modestly weaker. The top 3 models (SmolLM2 >
Llama-3.1-8B > gemma-2-9b-it) match the detection-rate ranking exactly; the
bottom three (Qwen2.5-1.5B, Qwen3-8B, DeepSeek) are not in perfect rank
order by margin relative to detection rate, but DeepSeek's *harmful-prompt*
margin itself (0.308) is starkly smaller than every other model's (0.679-
1.018) -- roughly a third to half the size -- the clearest single number in
this whole table for why DeepSeek's refusal signal is comparatively weak:
even on genuine, unparaphrased harmful prompts, its dense direction sits
much closer to the decision boundary than any other model's.

**Honestly hedged, not a mechanism**: this sharpens the description of the
phenomenon (continuous margin, formally tested, an n=5 correlation) and
surfaces one new textured observation -- gemma sits almost exactly at its
own decision boundary on average for PAIR (margin -0.049, essentially zero),
while both Qwen models are pushed solidly into harmless-looking territory.
**It does not explain *why*** some models' margins hold up better under
paraphrasing than others -- that's the same open question as before, just
described with a formal statistic and finer resolution instead of a raw
pass/fail rate. Identifying an actual mechanism (e.g. token-level or
positional analysis of what the paraphrase changes) would be separate,
heavier scope than this re-analysis; none is proposed here. Results in
`results/pair_margin_analysis.json`.

#### Does concentration protect the causal signal from paraphrase? A matched-pair mechanistic dig (2026-07-24)

The margin analysis above compares *distributions* (21 PAIR prompts vs. the
general harmful-prompt population). Every PAIR-adversarial record is
actually matched, by construction, to one exact original (unparaphrased)
TEST-split harmful prompt with identical `goal` text -- confirmed 21/21
matches for every model, all `source=="jbb"`, `split=="test"`, already
sitting in the main activation cache. This gives real **matched pairs**
(same underlying request, only the surface phrasing differs), letting the
paraphrase's effect be measured directly per prompt-pair rather than across
two different distributions -- still no new GPU generation, pure lookup and
linear algebra over already-cached activations.

**Motivation**: among the 3 SAE-having models, dense-direction PAIR
robustness (Llama 66.7% > gemma 47.6% > Qwen3-8B 42.9%) ranks in the *same
order* as SAE causal-effect concentration (Llama: one dominant feature;
gemma: modest/gradual; Qwen3-8B: distributed, top-1 alone does nothing) --
an ordinal match not previously tested. This rules out a naive "redundancy
protects" story (the *most distributed* model is the *least* robust, not
the most) and leaves "the concentrated feature is itself unusually
paraphrase-invariant" as the standing candidate.

**Dense-direction, matched pairs, all 5 models** (`scripts/paraphrase_decay_dense.py`,
delta = original margin minus paraphrased margin, positive = paraphrase
pushed the projection toward harmless; Wilcoxon signed-rank on the 21
matched pairs per model):

| model | delta (mean) | Wilcoxon p | known PAIR detection |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | **0.250** | 0.0595 (n.s.) | 90.5% |
| Llama-3.1-8B-Instruct | 0.767 | <0.0001 | 66.7% |
| DeepSeek-R1-Distill-Qwen-1.5B | 0.892 | <0.0001 | 9.5% |
| gemma-2-9b-it | 1.030 | <0.0001 | 47.6% |
| Qwen3-8B | 1.067 | <0.0001 | 42.9% |
| Qwen2.5-1.5B-Instruct | 1.118 | <0.0001 | 38.1% |

**Among the original 5 models, the delta ranking matched the known
robustness ranking exactly** -- smallest delta (SmolLM2) was the most
robust, largest (Qwen2.5) the least, monotonic in between (Spearman
rho=-1.0, exact permutation p=0.0167). **Adding DeepSeek (2026-07-27)
breaks this clean relationship**: its delta (0.892) is unremarkable,
similar in magnitude to Llama's (0.767) or gemma's (1.030), but its actual
PAIR detection rate (9.5%) is far below what that delta would predict from
the other 5 models' pattern -- it sits where a ~45-55% detection rate
"should" be, not 9.5%. **Recomputed Spearman at n=6: rho=-0.657, exact
permutation p=0.175 -- no longer significant.** Reported as a genuine
break in a previously clean result, not smoothed over: the paraphrase-
induced *dense-direction projection shift* is a real, measurable, and
statistically significant phenomenon for DeepSeek same as every other
model (Wilcoxon p<0.0001) -- but the *magnitude of that shift* no longer
predicts *how detectable the paraphrase ends up being*, once a model whose
baseline harmful-prompt margin is already much smaller than the other five
(0.308 vs. 0.679-1.018, see the PAIR-margin section) is included. A model
that starts closer to its own decision boundary needs less absolute shift
to cross it, which plausibly explains why the same-sized delta means
something different for DeepSeek than for the other five -- a candidate
explanation, not established by this data alone. SmolLM2 is also the only
model where the shift itself isn't statistically distinguishable from zero
(p=0.0595) -- consistent with it being the most paraphrase-robust model by
a wide margin.

**SAE-feature level, matched pairs, the 3 original SAE models** (of the 4
SAE-covered models as of 2026-07-27 -- DeepSeek deliberately excluded from
this specific analysis, see below) (`scripts/paraphrase_decay_sae.py`) --
tests two related questions separately rather than conflating them:

| model | top-1 feature delta | top-1 feature p | full top-15 score delta | full score p |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | **0.396** | **0.1678 (n.s.)** | 0.858 | 0.0038 |
| Qwen3-8B | 1.315 | <0.0001 | 2.037 | <0.0001 |
| gemma-2-9b-it | 2.070 | 0.0033 | 2.408 | <0.0001 |

**Llama's own top-ranked causal feature (layer 27/13363) is the only one
of the three where paraphrasing does not produce a statistically
detectable shift at all** (p=0.1678) -- direct, matched-pair evidence for
the "this specific feature is unusually paraphrase-invariant" hypothesis,
not just an inference from the aggregate margin/detection-rate
correlation. Qwen3-8B's and gemma's own top features both show a real,
significant shift under paraphrase. The delta ranking (Llama smallest <
Qwen3-8B < gemma largest) also matches the known SAE-feature PAIR-detection
ranking (Llama 80.9% > Qwen3-8B 33.3% > gemma 23.8%) exactly -- Spearman
rho=-1.0, but **at n=3 the exact permutation p-value is 0.333, not
significant** -- a perfect rank match is simply the best possible result
at this sample size, reported honestly rather than oversold.

**A real answer to the redundancy question, within each model**: for
Qwen3-8B, the full top-15 score decays *more* than its own (already
significantly decaying) top feature alone (2.037 vs. 1.315) -- paraphrase
disrupts many of its top-ranked features roughly together, not a case of
redundancy averaging out noise. For Llama, the full score decays
significantly (0.858, p=0.0038) even though its top feature alone does not
(0.396, n.s.) -- the one causally-dominant feature is robust, but the
other 14 features the detector also sums in are not, so the published
SAE-feature detector's own PAIR robustness (80.9%) is somewhat *diluted*
by non-causally-important features, not purely carried by the invariant one.

**Why DeepSeek is deliberately excluded from this table (2026-07-27)**: not
an oversight -- its SAE-feature detector already fails to fire on the vast
majority of genuine harmful prompts (4.4% VAL recall, see the dedicated
finding above), so both its "original" and "paraphrased" matched-pair
scores are overwhelmingly exactly zero. A Wilcoxon test on that data
would be measuring noise in a detector that doesn't functionally work for
this model, not a real paraphrase-decay effect -- reporting a p-value from
it would misrepresent a non-functioning detector as having a measurable
signal. The dense-direction level (above) is unaffected by this and does
include DeepSeek, since that detector genuinely does produce a real,
non-degenerate signal for this model, just a weaker and less robust one.

**Honestly hedged**: this is a matched-observational design, not a causal
intervention -- it describes *what* changes under paraphrase with much
finer resolution than the earlier margin analyses, and gives real,
statistically-grounded support for "Llama's dominant feature is
unusually paraphrase-invariant" specifically, but does not itself explain
*why* that one feature (out of tens of thousands) happens to be
invariant -- that would need e.g. token-level attribution of what the
paraphrase changes, a genuinely heavier undertaking than this re-analysis,
not attempted here. The SAE-level question is also fundamentally limited
to n=3 models (no trained SAE for the two models at either extreme of the
dense-direction robustness ranking, SmolLM2/Qwen2.5), so it can never be
tested at the full 5-model resolution the dense-direction result achieves.
Results in `results/paraphrase_decay_dense.json` and
`results/paraphrase_decay_sae.json`.

### Known limitations (cross-model dense-direction comparison)

- **n=35 adversarial prompts (21 PAIR), shared across all five models** --
  large enough for a formally significant Cochran's Q result (see
  DECISIONS.md) but not yet understood mechanistically.
- ~~The SmolLM2 hypothesis above is untested~~ -- **the matched-pair
  mechanistic dig above gives real, statistically-grounded (exact
  permutation p=0.0167 at n=5) support for a specific mechanism at the
  dense-direction level (paraphrase moves models' projections least where
  robustness is highest, in an exact rank match across all 5 models), and
  direct evidence at the SAE-feature level that Llama's specific dominant
  causal feature is paraphrase-invariant while Qwen3-8B's/gemma's own top
  features are not.** Still not a full explanation, though: *why* that one
  feature (out of tens of thousands) is invariant remains open -- would need
  token-level attribution of what the paraphrase changes, out of scope
  here. The SAE-level check is also capped at n=3 models (no trained SAE
  for SmolLM2/Qwen2.5-1.5B, the two most extreme dense-direction robustness
  models), so it can't be extended to the full 5-model range.
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

- ~~Only one model pair tested~~ -- **a second architecture-matched pair
  (Qwen2.5-1.5B-Instruct <-> DeepSeek-R1-Distill-Qwen-1.5B) has since been
  tested, see the dedicated section below.** The clean-no-transfer side of
  that result (Qwen2.5-1.5B) replicates this pair's pattern exactly; the
  other side (DeepSeek as target) is genuinely underpowered rather than
  informative, so the "does this weak-effect pattern recur" question
  specifically (Llama's own borderline own-ablation effect) still has no
  second data point -- DeepSeek's own baseline is floor-constrained in a
  different way than Llama's, not a matched comparison for that question.
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

## A second transfer pair: Qwen2.5-1.5B-Instruct <-> DeepSeek-R1-Distill-Qwen-1.5B (2026-07-28)

Extends "still only one pair with matching `d_model`" above -- a second
architecture-matched pair (both 1536-dimensional) was found already
sitting in this project's existing 6 models, no new downloads needed
(`results/dense_directions.pt` confirms the match; `assert_caches_consistent`
confirms both models' activation caches share identical corpus order/
labels/splits). `scripts/transfer_direction_deepseek.py`, necessity
(ablation) only -- sufficiency was explicitly not attempted this round
since DeepSeek's own-direction addition is already a thoroughly-swept
genuine null elsewhere in this document (alpha 0.25-32, 2 layers, both
application modes), making a foreign-direction sufficiency test low
expected value for real additional cost.

**Asymmetric design, not a straight rerun of `transfer_direction.py`**:
Qwen2.5-1.5B-Instruct (target) uses the standard 40-token budget, identical
to every other non-reasoning model's transfer test. DeepSeek (target) needs
the 2048-token whole-generation budget and `resolve_completions()`-style
truncation handling already established for its own Phase 1 reproduction
above. A real correctness risk was designed around up front, not discovered
after the fact: `resolve_completions()` drops a *different* subset of
prompts per condition, so naively resolving each condition independently
and then pairing completions by list position for McNemar would silently
misalign prompts once conditions' truncation patterns diverge. Fixed via a
new `resolve_completions_by_index()` (`src/direction/refusal_classifier.py`,
keyed by original position, not compacted) -- every McNemar comparison
restricts to the intersection of indices that resolved in *both* conditions
being compared, with the resulting paired-n reported alongside each p-value.
Sample size differs deliberately by target: N=50 for Qwen2.5-1.5B (cheap,
40 tokens), N=30 for DeepSeek (expensive, ~2min/generation at this budget --
matches the total-cost envelope this project already accepted for the
SAE-suppression-validation N=15x6-condition precedent).

| | baseline | own-ablation | foreign-ablation |
|---|---|---|---|
| Qwen2.5-1.5B (foreign = DeepSeek's direction) | 96.0% (n=50) | **4.0%** (n=50) | 90.0% (n=50) |
| DeepSeek (foreign = Qwen2.5-1.5B's direction) | 5.9% (n=17 resolved/30) | 5.0% (n=20 resolved/30) | 0.0% (n=13 resolved/30) |

**Qwen2.5-1.5B: a second clean, unambiguous no-transfer result, matching
the Qwen3-8B<->Llama pair's pattern exactly.** Own-direction ablation
crashes refusal (96%->4%, McNemar baseline-vs-own p=0.0, 46/50 discordant);
DeepSeek's foreign direction barely moves it (96%->90%, p=0.25, not
significant, only 3/50 discordant) -- own-vs-foreign is clearly different
(p=0.0). The intervention mechanism works dramatically on this model; a
foreign direction from an architecturally-different model (Qwen2.5 base
vs. DeepSeek's R1-distillation) simply does not transfer, same story as
the first pair.

**DeepSeek: genuinely underpowered, not a meaningful transfer result
either way -- reported honestly as inconclusive rather than overclaimed as
"no transfer."** Baseline refusal on this specific 30-prompt sample is
already very close to floor (5.9%, even lower than Phase 1's own 14.3% at
a different prompt sample -- consistent sample-to-sample variance at an
already-low rate, not a contradiction). With baseline already near-zero,
there is very little room for either own- or foreign-ablation to move it
further down, and heavy truncation (this model resolves only 43-67% of
completions across these three conditions at this prompt sample) shrinks
the McNemar paired-n to single digits (6-10 prompts per comparison) after
intersecting each pair's resolved indices -- all three comparisons come
back p=1.0, which reflects a comparison with almost no power to detect
anything, not evidence of no effect. Matches this document's standing
finding that DeepSeek's causal necessity signal is floor-constrained and
hard to pin down (see its dedicated section below) -- this transfer test
simply inherits that same underlying difficulty rather than adding a new
one.

Full results in `results/transfer_direction_deepseek.json`.

## Investigating the Llama dense-direction-vs-SAE-feature gap (2026-07-24, extended to gemma-2-9b-it same day)

Llama-3.1-8B-Instruct is the project's sharpest anomaly: best dense-direction
TEST AUROC of any of the 5 models (0.989), yet the only model where own-
direction necessity fails to replicate (96.0%->94.7%, n=75, McNemar p=1.0)
and where sufficiency is real but weak and fragile (10%->34%, degenerating
from alpha=2.0 -- see the two sections above). Meanwhile its SAE-feature
approach ablates refusal dramatically from a single feature (98%->10%, Wave
2, above). `scripts/analyze_llama_causal_gap.py` re-analyzes data already
sitting in `results/` plus two small CPU-only computations (loading cached
activations and already-downloaded SAE decoder weights -- no new model
generation) to test two candidate explanations against Llama and Qwen3-8B,
saved to `results/llama_causal_gap_analysis.json`. Extended the same day to
gemma-2-9b-it as a third data point for the SAE causal-effect-concentration
spread (see below) -- note gemma has no own-direction dense-ablation
causal-generation test in this project, so it can only join the two cheap
mechanical checks, not the necessity/sufficiency correlation itself.

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
| gemma-2-9b-it | 34 | 673.9 | 353.5 (freshly computed) | 0.525 |

Llama's direction is the *largest* fraction of its own ambient activation
scale of the models tested, not the smallest -- its layer-27 residual stream
simply operates at a much smaller absolute norm than the other models' do
at their own tested layers. **This hypothesis does not survive the check
and is ruled out**, not smoothed over: magnitude dilution relative to the
residual stream isn't what's making Llama's dense direction causally weak.
gemma's ratio (0.525) lands right in the middle of the pack, unremarkable --
it doesn't stand out the way Llama's does, reinforcing that Llama
specifically is the outlier here, not "8-9B models in general."

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
| gemma-2-9b-it | 35 | 52410 | **0.367** | 0.015 / 0.183 |

Llama and Qwen3-8B show essentially identical, modest alignment (~0.20 --
real, well above the random-feature baseline, but far from "the same axis").
**This does not differentiate those two models** and is reported as
inconclusive, not as confirming evidence for a misalignment story --
whatever separates Llama's messy dense-direction causal behavior from
Qwen3-8B's clean one, it isn't that Llama's SAE feature and dense direction
are unusually more orthogonal than Qwen3-8B's. **gemma is a genuinely
different third point**: its dense direction and top causal SAE feature are
nearly twice as aligned (0.367) as either other model's -- worth noting, but
not obviously explanatory, since gemma has no own-direction dense-ablation
causal-generation test in this project to correlate it against (only
Qwen3-8B and Llama-3.1-8B were tested that way at 7-9B scale) -- reported as
an observation, not evidence for or against either hypothesis.

**gemma's own SAE causal-effect shape is a third distinct pattern, not a
blend of the other two**: its ranking scores decay smoothly with no
standout (rank1 0.801, rank2 0.581, rank3 0.526 -- nothing like Llama's
10.07-then-cliff or even Qwen3-8B's own top-2 separation), and its
suppression curve is a modest, gradual decline (96%->82% by top-15/20, see
the SAE cross-model section above) rather than either Llama's near-total
single-feature effect or Qwen3-8B's flat-then-effective distributed one.
Higher dense-SAE alignment does not obviously map onto either "concentrated"
or "distributed" -- gemma's own shape is its own thing, adding a genuine
third data point rather than resolving the original two-model puzzle.

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

**Honestly hedged, not a forced conclusion, at the time**: this was a real,
multi-pronged comparison (two candidate mechanical explanations tested, one
ruled out for the two models where a full causal test exists) but
ultimately a correlational finding from n=2 models with full necessity/
sufficiency data (Llama vs. Qwen3-8B). Extending the two cheap mechanical
checks to gemma-2-9b-it added a real third data point (unremarkable
magnitude ratio, notably higher dense-SAE alignment, a third distinct
causal-effect shape) without resolving the original puzzle -- gemma simply
confirms Llama is the one clear outlier on magnitude, while adding a new,
separately-unexplained observation on alignment. **The genuine mechanistic
test flagged here as needed -- tracing what the ~0.20-aligned component of
the dense direction vs. its ~0.98-orthogonal remainder each individually
contribute -- was done in a follow-up session; see the next section.**

### Component decomposition: which piece of the dense direction is the causal lever? (2026-07-28)

Goes one level deeper than the alignment check above: rather than just
measuring cosine similarity between each model's dense direction and its
own top causal SAE feature, this decomposes the (unit-normalized) dense
direction into the component parallel to the feature's own decoder
direction and the orthogonal remainder, then causally ablates each
component *separately* (`scripts/decompose_llama_causal_gap.py`), reusing
the exact same intervention already used everywhere in this project
(`generate_with_ablation`, unchanged) and the identical 50-prompt held-out
VAL set already used for the cross-model transfer experiment. Both
components are renormalized to unit length before ablation -- this tests
"is this axis alone a sufficient/necessary causal lever," not "how much
does this component contribute at its true small weight"; the
parallel/orthogonal results below are not directly commensurate with the
full-direction number for that reason, stated explicitly rather than left
implicit. Decomposition verified via a Pythagorean sanity check
(`||parallel||^2 + ||orthogonal||^2 = 1.000001`/`1.000000`) and exact
reconstruction (`error < 2e-8`) for both models before trusting any
generation.

**A real data bug found and fixed while building this**: `results/
cross_model_direction_transfer.json` stores *stale*, pre-apostrophe-bugfix
refusal stats for every Llama condition (baseline stored as 80%, should be
92%; own_ablation stored as 86%, should be 88% -- the fix from 2026-07-23
was applied to this document's published numbers but never written back
into that JSON file; Qwen3-8B's entries in the same file are unaffected,
ASCII apostrophes throughout). Caught by a Plan-agent design review before
any GPU time was spent. The decomposition script reloads the file's raw
completions and rescores them fresh rather than trusting the stored
`refusal_stats` field -- confirmed the fresh rescore reproduces this
document's already-published 92%/88% (Llama) and 84%/8% (Qwen3-8B) exactly.

| Llama-3.1-8B-Instruct | Refusal rate | 95% CI | vs. baseline (McNemar p) |
|---|---|---|---|
| baseline | 92.0% | [81.2%, 96.9%] | -- |
| own (full-direction) ablation | 88.0% | [76.2%, 94.4%] | 0.5 (n.s.) |
| **parallel-component ablation** | **38.0%** | **[25.9%, 51.9%]** | **0.0** |
| orthogonal-component ablation | 92.0% | [81.2%, 96.9%] | 1.0 (identical) |

Omnibus Cochran's Q across all 4 conditions: Q=73.8, df=3, p<0.0001.
Post-hoc pairwise McNemar (not independently pre-registered, reported as
such): parallel-vs-baseline p=0.0 (27/50 discordant), parallel-vs-own p=0.0
(25/50 discordant), orthogonal-vs-baseline p=1.0 (2/50 discordant,
literally the same rate), orthogonal-vs-own p=0.5 (2/50 discordant). Zero
degenerate completions across all 200 new generations for this model.

**A genuinely clean, informative result**: the small feature-aligned
component (only 20% of the unit direction's own "weight" by cosine, before
renormalization) ablated *alone* produces a far larger, clearly significant
drop in refusal (92%->38%) than ablating the *entire* dense direction does
(92%->88%, not significant even at n=75 elsewhere in this document) -- and
the large orthogonal remainder (98% of the vector) ablated alone does
*nothing at all*, identical to baseline down to the exact rate. This
directly explains why Llama's full dense-direction ablation is such a weak
causal lever despite being the best passive classifier in this project:
the vector is dominated by norm from an axis that, on its own, has zero
causal necessity effect, diluting the real (and substantial) effect that
lives specifically along the axis aligned with its dominant SAE feature.
Ablating the *full* direction removes a diluted, off-target combination;
isolating just the causally-relevant axis removes it far more efficiently.

**Qwen3-8B, run as a contrast, tells a genuinely different story**:

| Qwen3-8B | Refusal rate | 95% CI | vs. baseline (McNemar p) |
|---|---|---|---|
| baseline | 84.0% | [71.5%, 91.7%] | -- |
| own (full-direction) ablation | 8.0% | [3.2%, 18.8%] | 0.0 |
| parallel-component ablation | 0.0% | [0.0%, 7.1%] | 0.0 |
| orthogonal-component ablation | 0.0% | [0.0%, 7.1%] | 0.0 |

Omnibus Cochran's Q: Q=115.48, df=3, p<0.0001. Post-hoc pairwise:
own-vs-parallel p=0.125 (n.s., 4/50 discordant), own-vs-orthogonal p=0.125
(n.s., 4/50 discordant), parallel-vs-orthogonal p=1.0 (0/50 discordant,
identical). One degenerate completion out of 50 in the parallel condition
(2%, not flagged as concerning at this rate).

**Unlike Llama, both components independently crash Qwen3-8B's refusal to
near-zero, matching or exceeding the full direction's already-strong
effect** -- there is no single dominant axis here; ablating *either* piece
alone is already sufficient. This is consistent with, and now gives a
second independent line of evidence for, this project's standing
concentration-vs-distribution account of the two models (Qwen3-8B's SAE
causal effect is similarly distributed across its ranked feature set,
where Llama's collapses into one feature, see the SAE cross-model section
above): for Llama, the dense direction's causal necessity really is
concentrated in one narrow, identifiable axis; for Qwen3-8B, it is
genuinely spread across the space, not recoverable by isolating any one
sub-component either.

**What this does and does not establish**: this is now a real, targeted,
statistically significant mechanistic account of *why* Llama's dense
direction is a weak causal lever -- not just a correlational observation.
It does not fully close the remaining gap to the SAE feature's own
even-stronger single-feature effect (98%->10%, Section above) -- the
isolated dense-direction axis (92%->38%) is a large, real effect but not
as complete a causal lever as the SAE feature itself, suggesting the SAE
feature's decoder direction and the *exact* causally-optimal axis are
correlated but not identical. Results in
`results/llama_causal_gap_decomposition.json`.

## DeepSeek-R1-Distill-Qwen-1.5B: reasoning-trace methodology and Phase 1 (2026-07-26)

Sixth model added. Its chat template auto-prefills `<think>\n` with no
`enable_thinking`-style disable (verified directly against
`tokenizer_config.json` -- see DECISIONS.md), so this model always reasons
before answering, unlike every other model in the project. Full
methodology fix in DECISIONS.md; results here.

**Empirical think-length probe** (`scripts/think_length_probe.py`, 30
AdvBench prompts, greedy decoding): among prompts that do close their
`<think>` block, min 498 / median ~960 / p95 ~1370 tokens. But doubling the
probe budget from 1536 to 3072 left the truncation rate exactly unchanged
at 10/30 (33%) -- the same 10 prompts both times. A direct inspection at
2500 tokens (one of the ten) found a 0.14 unique-word ratio, going in
circles over the same few ideas -- a genuine non-converging reasoning loop
on certain prompts, not "just needs a bigger budget." **Chose
`max_new_tokens=2048`** for downstream runs: comfortably covers the
converging distribution, and the non-converging ~third get reported
honestly as a distinct "truncated" outcome (via `resolve_completions()`),
not chased with an ever-larger budget.

**Phase 1 (n_val=30, layer 8 selected by separation score)**:

| condition | n resolved | n truncated | refusal rate |
|---|---|---|---|
| harmful baseline | 21 | 9 | 14.3% [5.0%, 34.6%] |
| harmful + ablation | 16 | 14 | 0.0% [0.0%, 19.4%] |
| harmless baseline | 25 | 5 | 0.0% [0.0%, 13.3%] |
| harmless + addition (alpha=1.0) | 24 | 6 | 0.0% [0.0%, 13.8%] |

Baseline harmful refusal is already low (14.3%) -- this model complies with
most AdvBench-style requests even unmodified. Ablation drives it to 0%, a
real if floor-constrained necessity signal. Addition induces no refusal at
all at alpha=1.0, the "finding reproduced" check fails on the sufficiency
side. Truncation rate roughly doubled under ablation (14/30) vs. baseline
(9/30) -- noted, not chased further; plausibly the intervention makes some
already-marginal reasoning traces less likely to converge.

**Alpha calibration swept 0.25 through 4.0 (7 points, n=12 calib prompts,
layer 9): 0% refusal at every single point (0/84 total).** The sweep's
reported `degenerate_frac` (50-75% at every alpha) looked suspicious at
first -- a few flagged completions read as coherent, fluent prose in a
truncated (~500-char) preview, which was initially (and wrongly) written up
here as `is_degenerate()` giving false positives on this model's longer
answers. **Correction, found during the later suppression-validation run
(below): those completions weren't actually clean -- reading the *full*
text (not a preview) showed genuine repetition collapse later on** (e.g.
one ~950-word completion is coherent for its first ~200 words, then
repeats one 8-word sentence verbatim ~70 times to fill the rest of the
budget). `is_degenerate()`'s unique-word-ratio<0.3 threshold was correct
all along; the earlier "false positive" claim was a verification mistake
(judging a long completion from its opening lines only), not a real bug.
This model's completions do need full-text inspection, not a preview, when
sanity-checking any classifier's output -- its tendency to open coherently
and only degenerate later is exactly the failure mode a truncated preview
misses.

**Deeper investigation (three follow-up checks, all still null)**: (1)
applying the addition intervention only during the post-`</think>` answer
(via a new resolved-prefix mechanism, `generate_reasoning_trace()` +
`prompt_override`) instead of across the whole reasoning trace -- still
0/5 refusal at alpha 1, 2, 4, 8. (2) Pushing alpha to 16 and 32 at the same
layer -- text genuinely breaks down into incoherent repeated-token garbage
at this point (confirming `is_degenerate()` *does* fire correctly at the
extreme, just not at moderate alphas), but still 0/4 refusal-shaped output.
(3) A different, mid-network layer (14) at alpha 2 and 8 -- coherent text,
still 0/4 refusal. **Conclusion: this is a genuine null result, not a
methodology artifact** -- across 2 layers and an alpha range spanning
coherent-compliant through incoherent-garbage, and whether applied
throughout generation or only to the answer, this model never produces
refusal-shaped output under activation addition. Necessity is present (if
weak, from a low baseline); sufficiency is absent. This extends the same
necessity/sufficiency asymmetry already documented for Llama-3.1-8B (see
above), more extreme here (zero measurable effect vs. weak-but-real).

**SAE-feature causal ranking** (`scripts/rank_sae_features.py`, EleutherAI's
pretrained MLP-output SAE suite, layers 7/8/11 -- top-3 by separation score
on the full 1345-prompt TRAIN / 289-prompt held-out VAL corpus, superseding
Phase 1's smaller ad-hoc [8,7,9] read): needed the same reasoning-trace
resolution as Phase 1, but for the *differentiable* IG metric this time --
`causal_ranking.py` gained a `prompt_override` path so each of the 16
sampled harmful TRAIN prompts is first resolved to its own
prompt+reasoning-through-`</think>` prefix (10/16 resolved, 6 truncated and
dropped) before running integrated gradients on the frozen prefix, so the
readout position is the real answer, not a token inside the reasoning
trace. **Top feature: layer 11/feature 48719 (score 0.074), a clear
standout above the 2nd-ranked layer 7/feature 3715 (0.049)** and the rest
of the top-20, which tapers to near-zero (several slightly negative) --
same qualitative shape as every other model's ranking.

**SAE-feature suppression validation** (`scripts/validate_sae_features.py`,
6 conditions x N=15 held-out VAL prompts, reduced from the other models'
N=50 to keep runtime to ~4hr instead of ~13hr given this model's
reasoning-inclusive budget): **inconclusive, not negative** --
refusal rates stayed near zero across every condition (baseline 0/6,
top1 1/8, top5 0/10, top10 0/10, top15 0/10, top20 1/9) with heavily
overlapping 95% CIs, no dose-response signal. Two things shrink the
effective sample well below N=15 per condition: truncation (33-60% per
condition, consistent with Phase 1) and this model's own baseline refusal
rate already being near-floor (14.3% in Phase 1's larger n=30 sample, 0/6
in this smaller one) -- there's little refusal left to suppress in the
first place, on top of a small starting N. **Not the same finding as the
addition-intervention null result above** -- that one was a genuine,
thoroughly-investigated null (many alphas, layers, and application modes
all agreeing); this one is underpowered, not conclusively null, and would
need a larger N (impractical at this model's per-generation cost without
a way to speed up the reasoning-inclusive budget) to actually resolve
either way.

**A genuinely new finding surfaced while sanity-checking this run**: this
model's long-form answers have a real, substantial tendency to open
coherently and then collapse into verbatim sentence-repetition once they
run out of new content but haven't reached a natural stopping point --
present even at baseline (no intervention at all), not something caused by
suppression. In the small baseline sample here, 5/6 resolved completions
showed this pattern (see the `is_degenerate()` correction above). This is
a real characteristic of the model worth remembering for any future
generation-based work on it, not a validation artifact.

**The SAE-feature detector, built via the project's standard methodology
(top-15 IG-ranked features), essentially fails to generalize as a
classifier for this model.** Checked while computing DeepSeek's PAIR
detection rate for the paraphrase-decay comparison below: at the
calibrated threshold (Youden's J on VAL, threshold 6.899 -- a real,
correctly-computed value, not a bug), harmless VAL scores are 100% exactly
zero (expected), but so are **94.9% of genuine harmful VAL scores** (150 of
158) -- only 8 harmful prompts out of 158 ever trigger even one of the 15
selected features at all. **VAL harmful recall at the calibrated
threshold: 4.4%. PAIR-paraphrase detection: 0/21.**

Not a calibration bug -- with harmless scores universally zero, Youden's J
correctly found the best achievable operating point given the actual score
distribution; the real cause is architectural. The SAE's hard top-k=32
sparsity (32 active features out of 65536, verified directly against the
checkpoint's own `cfg.json`) means only 32 of 65536 features fire on any
given input. 15 features hand-selected by attribution-patching on a small
n=10 resolved-prompt sample (this model's mandatory reasoning trace left
only 10 of the original 16 candidates usable, see the SAE-ranking section
above) have low odds of landing in *any* given VAL prompt's active set --
unlike the other 3 SAE-covered models, where the equivalent top-15 set
reliably fires across their VAL and TEST distributions (see the SAE
cross-model section elsewhere in this document).

**This is not an isolated failure -- it's the same story every other
measurement in this document tells about this model, from a fourth
independent angle.** Activation addition never induces refusal at any
tested dose (a thoroughly-investigated null, above). The dense-direction
detector is the weakest of all 6 models on TEST (84.7% accuracy, AUROC
0.911) and has by far the worst PAIR robustness (9.5%, next-lowest is
38.1%) -- see the dense-direction cross-model section. Now the SAE-feature
detector, which for every other SAE-covered model is a real, working
classifier, essentially never fires. **DeepSeek-R1-Distill-Qwen-1.5B's
refusal representation looks diffuse and hard to pin down to a small
linear direction or a small feature set, across every measurement approach
this project has, not just one.** Whether that's a property of R1-style
distillation specifically, of this model's comparatively light safety
training, or something else architectural is an open question this
project's data doesn't resolve -- flagged as a genuine, real finding, not
overclaimed as an explanation.
