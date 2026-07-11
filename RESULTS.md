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
on a disjoint 30-prompt held-out val split. 95% CIs are Wilson score
intervals.

| Model | Condition | n | Refusal rate | 95% CI |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | harmful, baseline | 30 | 100.0% | [88.7%, 100%] |
| Qwen2.5-1.5B-Instruct | harmful, **ablated** | 30 | 0.0% | [0%, 11.4%] |
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
| Ablation effect (necessity) | 100% -> 0% | 63% -> 3% |
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
train and held-out val splits above.

**Qwen2.5-1.5B-Instruct** (best layer 23, raw direction norm 75.3):

| alpha | refusal rate | degenerate fraction |
|---|---|---|
| 0.25 | 17% | 0% |
| 0.50 | 67% | 0% |
| **1.00** | **100%** | **0%** |
| 1.50 | 100% | 17% |
| 2.00 | 75% | 92% |
| 3.00 | 17% | 100% |
| 4.00 | 0% | 100% |

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
- **Generation was not forced to greedy decoding** -- Qwen2.5-1.5B-Instruct's
  default `GenerationConfig` samples (`do_sample=True`, temperature=0.7),
  so these numbers include uncontrolled sampling noise on top of the
  intervention's true effect (SmolLM2 defaults to greedy, so its numbers
  aren't affected the same way). Discovered and fixed for the SAE-feature
  detector below (`do_sample=False` now set in
  `src/direction/interventions.py`); not retroactively fixed here since
  these results are already stable/published -- see DECISIONS.md.

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
used upstream), 40 tokens generated per completion, real
`refusal_classifier`. 95% CIs are Wilson score intervals. (An initial pass
at n=25 with only baseline/top1/top5/top20 found the same overall effect
but with wider, partly-overlapping CIs -- superseded by this larger,
finer-grained run; both stages are documented in DECISIONS.md.)

| condition | n | refusal rate | 95% CI | degenerate |
|---|---|---|---|---|
| baseline | 50 | 84.0% | [71.5%, 91.7%] | 0/50 |
| suppress top-1 | 50 | 78.0% | [64.8%, 87.3%] | 0/50 |
| suppress top-5 | 50 | 44.0% | [31.2%, 57.7%] | 0/50 |
| suppress top-10 | 50 | 44.0% | [31.2%, 57.7%] | 0/50 |
| **suppress top-15** | 50 | **18.0%** | **[9.8%, 30.8%]** | 0/50 |
| suppress top-20 | 50 | 26.0% | [15.9%, 39.6%] | 0/50 |

**Suppressing the top-15 features gives the strongest effect** (84% -> 18%
refusal), with baseline's CI not overlapping top-5 through top-20 at all --
a clearly distinguishable causal effect from a modest fraction of the
pooled candidates onward, not just the full top-20 as the smaller first
pass suggested. **Zero completions degenerated into incoherent output**
across all 300 generations in this run. Unlike the single hand-picked
feature in arXiv:2411.11296 (which achieved a refusal-rate shift only by
destroying general capability -- MMLU 68.8% -> 36.0%), this project's
systematic top-K* selection produces a real behavioral effect without a
coherence collapse.

**Honest findings, not smoothed over**:
- Suppressing the single top-ranked feature alone (top-1) barely moved
  refusal (78% vs 84% baseline) -- the effect is **distributed across the
  feature set**, not concentrated in one dominant feature, confirmed again
  at this larger sample size.
- **Not a clean monotonic dose-response curve**: refusal bottoms out at
  top-15 (18%) then rises to 26% at top-20. top-15 and top-20's CIs
  overlap (9.8-30.8% vs 15.9-39.6%), so this uptick isn't necessarily a
  real reversal -- plausibly noise from the 5 lowest-ranked features
  (IG scores 0.02-0.08, close to the noise floor in the ranking pass),
  one of which may mildly counteract the others' effect. top-5 and top-10
  landed on the exact same refused-count (22/50) -- features ranked 6-10
  contributed no net additional suppression on this sample.

### Known limitations (SAE-feature detector)

- n=50 for the suppression validation and n=16 for the ranking pass (both
  increased from an initial n=25/n=8 pass -- see DECISIONS.md) are enough
  to show the top-15/top-20 effect is real and the ranking's top features
  are stable, but still not enough to fully explain the non-monotonic
  dip at top-15 vs top-20 -- that would need either more prompts or a
  repeated run with a different random seed to distinguish real signal
  from sampling noise.
- Single model (Qwen3-8B). Cross-model generalization -- whether a feature
  set selected this way transfers to a different model family -- is this
  project's explicit differentiator from the existing literature and
  remains untested.
- The SAEs are trained on the base model's activations, applied here to
  the instruct/chat model -- a documented, accepted limitation shared with
  the source paper (see DECISIONS.md), not unique to this reproduction.
- No comparison yet against the single-dense-direction approach (Phase 1
  above) on equal footing -- that head-to-head is separate, later work.
