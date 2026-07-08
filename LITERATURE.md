# Related work

Note on scope: this is an abstract-level survey (read via search results and
arXiv abstracts, not full close reading of every paper) intended to situate
this project's contribution and confirm it isn't duplicating existing work.
A deeper close-reading pass belongs in a later phase once the SAE-detector
work starts drawing directly on these papers' methods.

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
  EMNLP 2025): harm and refusal are encoded as *separate* SAE feature sets,
  with harmful features causally affecting refusal features; adversarial
  jailbreaks work by suppressing specific refusal-related features. This is
  close to the planned SAE-feature detector's core premise and should be
  read closely (not just at abstract level) before finalizing that phase's
  design.
- **"Steering Language Model Refusal with Sparse Autoencoders"**
  ([arXiv:2411.11296](https://arxiv.org/abs/2411.11296)): SAE-based
  steering, precedent for using pretrained SAE features (rather than the
  raw single direction) for refusal control.
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
   `scripts/02_calibrate_addition_alpha.py` is exactly the kind of gap this
   angle is meant to close).
