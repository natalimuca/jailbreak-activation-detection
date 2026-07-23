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
[METHODOLOGY.md](METHODOLOGY.md), [RESULTS.md](RESULTS.md)).

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
  [DECISIONS.md](DECISIONS.md)), which the paper itself flags as a
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

## Where this project's contribution sits

Given the above, "activation-based jailbreak detector" alone is not novel.
This project's differentiation (per [DECISIONS.md](DECISIONS.md) and the
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
