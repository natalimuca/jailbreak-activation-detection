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
score on the refusal-vs-compliance logit-diff metric (8 harmful TRAIN
prompts, length-capped):

| rank | layer | feature | score |
|---|---|---|---|
| 1 | 25 | 65291 | 2.371 |
| 2 | 23 | 42331 | 1.461 |
| 3 | 23 | 23501 | 0.501 |
| 4 | 24 | 4711  | 0.474 |
| 5 | 24 | 5393  | 0.421 |

The top 2 features are a clear standout (2.371 and 1.461) above the rest
of the top-20 (all <= 0.501) -- a much sharper signal than an earlier,
since-superseded run of this same pipeline that hadn't yet disabled
Qwen3's default thinking mode (that run's top score was only 0.547, with
2 near-zero/negative features surviving into the top-20; see DECISIONS.md
for the full account). Full list of all 20 selected features (layer,
feature index, score) is in the JSON results file.

### Causal validation (feature suppression)

Baseline vs. suppressing the top-1/top-5/top-20 ranked features, on 25
held-out VAL harmful prompts (disjoint from every prompt used upstream),
40 tokens generated per completion, real `refusal_classifier`. 95% CIs are
Wilson score intervals.

| condition | n | refusal rate | 95% CI | degenerate |
|---|---|---|---|---|
| baseline | 25 | 72.0% | [52.4%, 85.7%] | 0/25 |
| suppress top-1 | 25 | 84.0% | [65.4%, 93.6%] | 0/25 |
| suppress top-5 | 25 | 44.0% | [26.7%, 62.9%] | 0/25 |
| **suppress top-20** | 25 | **20.0%** | **[8.9%, 39.1%]** | 0/25 |

The top-20 condition's CI does not overlap baseline's -- suppressing this
systematically-selected feature set causes a real, large drop in refusal
(72% -> 20%), with **zero completions degenerating into incoherent
output** across all 100 generations in this run. Unlike the single
hand-picked feature in arXiv:2411.11296 (which achieved a refusal-rate
shift only by destroying general capability -- MMLU 68.8% -> 36.0%), this
project's systematic top-K* selection produces a real behavioral effect
without a coherence collapse.

**Honest finding, not smoothed over**: suppressing the single top-ranked
feature alone (top-1) did *not* reduce refusal -- it rose slightly (84% vs
72% baseline), though the CIs overlap heavily enough that this likely
isn't a real reversal, just noise at n=25. The refusal-suppressing effect
is **distributed across the feature set**: it only becomes distinguishable
from baseline once enough of the top-20 are suppressed together (5 is not
enough on its own; the CI at top-5 still marginally overlaps baseline's).

### Known limitations (SAE-feature detector)

- n=25 for the suppression validation and n=8 for the ranking pass are
  enough to show the top-20 effect is real, not enough for fine-grained
  per-feature confidence -- the bottom few of the 20 ranked features carry
  more uncertainty than the top 2.
- Single model (Qwen3-8B). Cross-model generalization -- whether a feature
  set selected this way transfers to a different model family -- is this
  project's explicit differentiator from the existing literature and
  remains untested.
- The SAEs are trained on the base model's activations, applied here to
  the instruct/chat model -- a documented, accepted limitation shared with
  the source paper (see DECISIONS.md), not unique to this reproduction.
- No comparison yet against the single-dense-direction approach (Phase 1
  above) on equal footing -- that head-to-head is separate, later work.
