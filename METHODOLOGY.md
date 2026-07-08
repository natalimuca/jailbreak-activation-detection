# Methodology: refusal-direction reproduction

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

**Known limitation**: this classifier has not been validated against human
judgment or an LLM-judge baseline on these specific completions. It is
adequate for confirming a causal intervention has *some* effect (which is
all Phase 1 needs), not for precise rate estimation.

## What this reproduction does and doesn't establish

Establishes: the causal refusal-direction finding reproduces on two small,
architecturally different open-weight models, for both the necessity
(ablation) and sufficiency (addition) directions of the original claim.

Does not establish: whether this holds at the 7-9B parameter scale used
elsewhere in this project: whether the SmolLM2 addition-effect ceiling
reflects safety-training strength, architecture, or something else; or
anything about the SAE-feature detectors that motivate this project's
actual contribution (see [DECISIONS.md](DECISIONS.md) and
[LITERATURE.md](LITERATURE.md)) -- those are separate, later questions.
