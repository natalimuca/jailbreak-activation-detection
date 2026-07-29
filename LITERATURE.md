# Related work

Note on scope: most entries below are an abstract-level survey (read via
search results and arXiv abstracts, not full close reading), intended to
situate this project's contribution and confirm it isn't duplicating
existing work. The two papers Phase 3 draws its methodology from
("Understanding Refusal..." and "Steering Language Model Refusal...") have
been close-read in full -- see their entries below.

## Foundational finding

**Arditi et al., "Refusal in Language Models Is Mediated by a Single
Direction"** (NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)).
Shows a single residual-stream direction, estimated as a harmful/harmless
activation mean-difference, causally controls refusal across 13 open chat
models up to 72B params -- ablating it suppresses refusal, adding it induces
refusal. This is the finding reproduced in Phase 1 (see
[METHODOLOGY.md](reports/METHODOLOGY.md), [RESULTS.md](reports/RESULTS.md)).

## Direct challenge to the foundational finding

**Joad, Hawasly, Boughorbel, Durrani, Sencar, "There Is More to Refusal in
Large Language Models than a Single Direction"** (Feb 2026,
[arXiv:2602.02132](https://arxiv.org/abs/2602.02132)). Across eleven
categories of refusal/non-compliance (safety refusals, incomplete requests,
anthropomorphization, over-refusal, etc.), the refusal behaviors correspond
to *geometrically distinct* directions -- not one single direction.
**However**, the paper also finds that linear steering along any of these
different refusal-related directions produces nearly identical
refusal-vs-over-refusal trade-off curves, i.e. they act as a shared
one-dimensional control knob regardless of which specific direction is used.
The different directions mainly affect *how* the model refuses, not whether
it refuses.

**Relevance to this project**: this doesn't undermine Phase 1's
reproduction so much as sharpen what it actually shows -- *a* refusal
direction causally controls refusal, consistent with Joad et al.'s finding
that any refusal-related direction does. It's a caution against overclaiming
that AdvBench-derived direction is *the* unique refusal mechanism, which
matters directly for the cross-model generalization question this project
is asking: if there are multiple geometrically distinct but functionally
similar directions, a detector trained on one model's "the" direction might
transfer poorly not because refusal representations differ across models,
but because different (but equally valid) directions get picked up by the
same estimation procedure on different models.

## Activation-based jailbreak detectors (already-busy area)

Confirmed via this search pass: activation-based jailbreak detection is an
active 2025-2026 research area, not an open niche -- consistent with the
decision (made during project scoping) to differentiate via cross-model
generalization and engineering rigor rather than "detector beats keyword
filter" alone.

- **ALERT** ([arXiv:2601.03600](https://arxiv.org/abs/2601.03600), Jan
  2026): zero-shot jailbreak detection via layer-wise, module-wise, and
  token-wise amplification of internal representation discrepancies between
  benign and jailbreak prompts. Reports beating the next-best baseline by
  10-40% in accuracy/F1 across datasets and attack strategies. Directly
  competitive with what this project's baseline comparison (Phase 4) will
  need to measure against.
- **"Refusal Before Decoding: Detecting and Exploiting Refusal Signals in
  Intermediate LLM Activations"** ([arXiv:2605.28553](https://arxiv.org/abs/2605.28553)):
  refusal signals are detectable via layer-specific classifiers on
  intermediate activations, tested on Llama-3.2-3B-Instruct,
  Qwen3Guard-Gen-4B, Qwen-3.6-27B. Notably dual-use -- also demonstrates
  these signals can be exploited for white-box-to-black-box attack
  transfer, which is directly relevant to this project's adversarial test
  set (Phase 4).
- **"Understanding Refusal in Language Models with Sparse Autoencoders"**
  ([arXiv:2505.23556](https://arxiv.org/abs/2505.23556), ACL Findings
  EMNLP 2025) -- **close-read in full; Phase 3's methodology is adapted from
  this paper.** Models: Gemma-2-2B and Llama-3.1-8B, using GemmaScope
  (expansion 32) and LlamaScope (expansion 8), both **base-model SAEs
  applied to the instruct/chat model's activations** -- the same
  methodological choice this project makes with Qwen-Scope (see
  [DECISIONS.md](reports/DECISIONS.md)), which the paper itself flags as a
  limitation rather than treating as disqualifying.
  Method: (1) compute a difference-in-means refusal direction exactly like
  Phase 1's `compute_directions()`; (2) restrict the SAE's full feature
  space to the top K0=10 features per layer by cosine similarity to that
  direction; (3) rank those by causal effect via Attribution Patching
  (integrated gradients, N=10 steps), keep the top K*=20; (4) split into
  refusal features F_R (common across all harmful categories) vs harm
  features F_H (category-specific). Causal validation: scale the selected
  feature's activation by a coefficient (c=-3 Gemma, c=-1/-3 Llama) and
  measure suppression via HarmBench's classifier, not string-matching.
  **Results worth replicating**: suppressing F_H gave 48% (Gemma) / 51%
  (Llama) refusal suppression vs a 5-6% random-feature baseline. Most
  relevant to this project's motivation: **sparse SAE-feature probes were
  far more robust to adversarial paraphrasing than dense single-direction
  probes** -- accuracy gap between vanilla and adversarial prompts was only
  0.03-0.17 for sparse features vs 0.52-0.93 for the dense direction probe.
  Stated limitations: small K* may omit relevant features; the
  restrict-to-refusal-direction step inherits whatever bias the direction
  itself has; cross-model transfer was not systematically tested (explicitly
  left open -- this project's differentiator).
- **"Steering Language Model Refusal with Sparse Autoencoders"**
  ([arXiv:2411.11296](https://arxiv.org/abs/2411.11296)) -- **close-read in
  full; cautionary methodology, not directly adapted.** Model: Phi-3 Mini,
  a Top-k SAE (k=32, expansion 8, 24,576 features) trained on layer 6.
  Feature selection was manual/ad hoc: one archetypal refusal completion,
  grid search over ~100 candidate features on 250 samples, picked a single
  best feature by hand. At their strongest steering coefficient (clamp=12):
  Wild Guard unsafe-prompt refusal rose 58%->96% and jailbreak success fell
  55.9%->32.6%, but **capability collapsed** -- MMLU 68.8%->36.0%, GSM8K
  82.5%->35.6% -- with no corresponding over-refusal increase, meaning the
  damage was general capability loss, not a safety/helpfulness tradeoff.
  Their own conclusion: amplifying a single feature causes broad collateral
  damage because SAE features aren't as modular/independent as assumed.
  **Why this project won't repeat their approach**: single hand-picked
  feature + no capability check is exactly the failure mode. Phase 3 uses
  the systematic top-K* causal-ranking approach from the paper above
  instead, and any activation-addition experiment must report a capability/
  coherence check alongside the induced-refusal rate, not refusal rate
  alone.
- **"Graph-Regularized Sparse Autoencoders for Robust LLM Safety Steering"
  (GSAE)** ([arXiv:2512.06655](https://arxiv.org/abs/2512.06655)):
  graph-regularization for more robust SAE-based safety steering --
  relevant prior art for whatever robustness techniques the SAE-detector
  phase ends up needing against adversarial paraphrase.

## Why does Llama's core-request x wrapper-framing interaction concentrate in specific requests? (literature context, 2026-07-29)

Three rounds of this project's own experiments (per-cell decomposition,
a rejected task-type hypothesis, a blind multi-feature search with proper
multiple-comparison correction -- see [RESULTS.md](reports/RESULTS.md),
[DECISIONS.md](reports/DECISIONS.md)) confirmed Llama-3.1-8B-Instruct's
core x category interaction is real and highly reproducible (η²=0.286 at
n=48, the strongest replication yet), but left *why* it concentrates in a
handful of specific requests unresolved -- no objective feature tested
predicts it. A literature pass (abstract-level, not close-read) looking
for external context on this specific question found three candidate
angles, none of which resolve the question but each gives it real
grounding rather than leaving it purely speculative:

- **Khorramrouz & Levy, "Characterizing Selective Refusal Bias in Large
  Language Models"** (ACL Findings 2026,
  [arXiv:2510.27087](https://arxiv.org/abs/2510.27087)). Finds refusal
  rates and refusal-response characteristics (length, type) vary by
  specific demographic target/topic in ways not explained by a uniform
  harm-level function -- i.e. topic-specific idiosyncrasy in refusal
  behavior is an independently documented phenomenon in the field, not
  unique to this project's data. **Relevance**: supports treating "which
  specific requests interact unusually with framing" as a real, recognized
  class of effect rather than assuming it must reduce to some clean
  category (task type, source dataset, etc.) this project simply hasn't
  found yet -- the literature's own finding is that this kind of
  topic-specificity often does *not* reduce to a clean predictor.
- **"Can't hide behind the frame: Disentangling goal & framing for
  detecting LLM jailbreaks"** ([OpenReview](https://openreview.net/forum?id=VY9hVzmxg6),
  submitted ICLR 2026, **later withdrawn** -- treated with appropriate
  extra skepticism given that status, not close-read). Its stated premise
  is a self-supervised disentanglement of "goal" (the underlying request)
  and "framing" (surface wrapper) as separate signals in LLM
  representations, reporting "distinct profiles for goal and framing
  signals" per input. **Relevance**: this is essentially the same
  goal-vs-framing distinction this project's wrapper-swap design already
  makes, from an independent group, on the general premise that some
  inputs are goal-dominant and others framing-dominant -- external
  validation that the *paradigm* (separating core-request identity from
  wrapper-framing identity) is a real, recognized axis of variation in
  LLM representations, even though it doesn't say why *these two* specific
  requests land where they do.
- **General SAE/feature-frequency literature** (surveyed abstract-level,
  not one specific paper): recurring finding that a concept's representation
  geometry depends on its frequency/salience in training data -- rarer
  features/co-occurrences tend toward more superposed (entangled)
  representations, while linear relational representations are more likely
  to form once a concept's co-occurrence frequency clears some threshold.
  **Relevance, tested directly rather than left speculative**: "Trump 2020
  election" and "fake news article" are both unusually high-salience,
  heavily-discussed real-world phrases compared to this project's more
  generic templated harmful requests, so if a request's real-world
  discourse salience shapes how strongly its "goal" representation
  competes with wrapper "framing" for the same feature, that would predict
  exactly the request-specific interaction this project keeps finding.
  **Tested via perplexity under this project's own existing reference LM
  as a salience proxy** (`scripts/wrapper_salience_test.py`,
  2026-07-29) against the already-collected 48-core interaction data --
  **result: a clean negative** (Llama rho=-0.110, p=0.457; Qwen3-8B
  rho=-0.166, p=0.260), the correct direction but nowhere near
  significant. See [RESULTS.md](reports/RESULTS.md) and
  [DECISIONS.md](reports/DECISIONS.md) for the full account, including
  this project's now four-hypotheses-deep honest track record on this
  specific open question.

## Where this project's contribution sits

Given the above, "activation-based jailbreak detector" alone is not novel.
This project's differentiation (per [DECISIONS.md](reports/DECISIONS.md) and the
original scoping) is:

1. **Cross-model generalization**: none of the papers found so far directly
   report training a detector on one model and testing transfer to a
   different model family/size -- this remains a genuinely open question,
   sharpened rather than answered by the Joad et al. finding above.
2. **Engineering rigor**: calibration, decision-curve analysis, honest
   baseline comparison on a held-out adversarial set -- the kind of
   evaluation infrastructure most of these papers report results without
   fully engineering (e.g. this project's own Phase 1 mistake of using an
   uncalibrated, untested alpha=8 constant before the calibration sweep in
   `scripts/calibrate_alpha.py` is exactly the kind of gap this
   angle is meant to close).
