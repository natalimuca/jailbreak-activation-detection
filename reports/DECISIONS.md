# Design decisions and open items

## Pretrained SAEs, not trained-from-scratch

The planned SAE-feature detector will use **pretrained sparse autoencoders**
rather than training our own. Training a good SAE from scratch needs a large
multi-position activation corpus (typically hundreds of thousands to millions
of vectors) and significant compute -- not a good use of solo-researcher time
on a 6GB local GPU when high-quality pretrained options already exist for
most of the target models:

- **Gemma-2-9B**: [Gemma Scope](https://arxiv.org/html/2408.05147v2) -- residual
  stream, MLP, and attention SAEs at every layer.
- **Llama-3.1-8B**: [Llama Scope](https://arxiv.org/pdf/2410.20526) (OpenMOSS)
  and [Goodfire's Llama-3 SAEs](https://www.goodfire.ai/research/understanding-and-steering-llama-3).
- **Qwen3 family**: [Qwen-Scope](https://www.marktechpost.com/2026/05/01/qwen-ai-releases-qwen-scope-an-open-source-sparse-autoencoders-sae-suite-that-turns-llm-internal-features-into-practical-development-tools/) --
  covers Qwen3-1.7B/8B and Qwen3.5, **not Qwen2.5**.

**Decided (2026-07-08): Phase 3's first model is Qwen3-8B, using
[Qwen-Scope](https://arxiv.org/pdf/2605.11887)'s `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`
checkpoint.** Rationale:

- **Ungated** -- verified via `huggingface_hub.model_info`: `google/gemma-2-2b-it`
  and `meta-llama/Llama-3.1-8B-Instruct` both require manual HF approval
  (an action only the user can do, with unknown turnaround time); `Qwen/Qwen3-8B`
  does not. This unblocks Phase 3 immediately rather than waiting on external
  approval.
- **8B fits the local 6GB GPU** with 4-bit quantization (`bitsandbytes`,
  already a dependency; `load_model(..., load_in_4bit=True)` already
  supported in `src/activations/extract.py`, just unused until now).
- **Known caveat, not a blocker**: Qwen-Scope's SAEs are trained on
  **Qwen3-8B-Base** activations, not the instruct/chat model's. Applying a
  base-trained SAE to instruct-model activations is standard, published
  practice in this exact literature -- "Understanding Refusal in Language
  Models with Sparse Autoencoders" (arXiv:2505.23556, see
  [LITERATURE.md](../LITERATURE.md)) does the same thing with GemmaScope/LlamaScope
  and reports it as a limitation, not a disqualifying flaw. We do the same:
  proceed, document it, don't pretend it's not a compromise.

This replaces the original plan's Qwen2.5-7B-Instruct for the Qwen leg.

## Qwen3-8B 4-bit loading on a 6GB GPU (2026-07-09)

Getting `load_model("Qwen/Qwen3-8B", load_in_4bit=True)` to actually run
required three fixes, in `src/activations/extract.py`:

1. **`transformers>=5` removed the `load_in_4bit=True` shorthand kwarg.**
   Quantization must go through `quantization_config=BitsAndBytesConfig(...)`
   instead -- the old shorthand raises `TypeError: ...__init__() got an
   unexpected keyword argument 'load_in_4bit'`.
2. **`device_map="auto"` refuses to fit the model at all**, even though it
   does fit: accelerate's memory estimator decides a few modules need
   CPU/disk offload, and bnb then refuses that combination unless
   `llm_int8_enable_fp32_cpu_offload=True` is also set. Tested that path --
   bnb's CPU-backend int4 ops are extremely slow (confirmed via a stalled
   probe run), not viable. **Fix: force everything onto the single GPU with
   `device_map={"": 0}`, bypassing accelerate's conservative auto-balancer
   entirely.**
3. **Peak GPU memory for the corpus's longest prompts (~1000 tokens, from
   HarmBench's contextual behaviors) exceeds the card's 6GB** even with
   4-bit weights (`bnb_4bit_compute_dtype=torch.float16` +
   `bnb_4bit_use_double_quant=True` to trim overhead, `use_cache=False`
   since only one forward pass per prompt is needed, no generation).
   Measured peak: ~6.7-6.9GB allocated on the worst-case (1001-token) prompt.
   This only works via Windows' CUDA shared-memory spillover into system
   RAM (confirmed via `nvidia-smi` showing less physical VRAM used than
   PyTorch reports allocated) -- slow (~45-55s) for that one prompt, but
   **only ~60-94 of the corpus's 1990 prompts exceed 1000 characters**
   (median prompt is 63 chars); the rest run in well under a second. Total
   extraction time estimated at 1.5-2 hours, acceptable for a one-off batch
   job. Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and added
   a periodic `torch.cuda.empty_cache()` every 50 prompts to reduce
   allocator fragmentation risk from alternating short/long prompts so close
   to the memory ceiling.
   **Known fragility, accepted rather than engineered around further:**
   this relies on driver-level spillover, so running another GPU-heavy
   process concurrently during the ~2hr extraction could cause an OOM that
   a purely in-VRAM setup wouldn't hit. Documented rather than silently
   risked.

## Top-K0 SAE feature selection: signed, not absolute, cosine similarity

`src/sae/feature_selection.py`'s `top_k0_by_cosine_similarity()` ranks SAE
features by **signed** cosine similarity between each feature's decoder
direction and the harmful-minus-harmless direction, not absolute value.
Rationale: we want features that write *along* the harmful direction
(candidate refusal/harm features), not features whose axis merely
correlates with it regardless of sign (a strongly anti-aligned feature is
evidence *against* refusal, not a refusal-feature candidate). LITERATURE.md's
source paper (arXiv:2505.23556) doesn't spell out signed-vs-absolute
explicitly; this is our resolution of that ambiguity, made explicit here
rather than silently picking one.

## SAE layer selection: layer 22 (2026-07-09) -- SUPERSEDED, see below

**Superseded (2026-07-10)**: this run predates the `enable_thinking=False`
fix (see "Qwen3's default thinking mode was uncontrolled" below). Kept for
history, not for use.

Ran the full Qwen3-8B activation extraction (1922-prompt corpus) and computed
per-layer separation scores (difference-of-means direction from train,
measured on held-out val -- same method as Phase 1). Top-5:

| layer | score |
|-------|-------|
| 22    | 1.742 |
| 21    | 1.739 |
| 20    | 1.738 |
| 23    | 1.720 |
| 19    | 1.717 |

Scores are tightly clustered (19-23), consistent with Phase 1's finding that
refusal-relevant layers sit in the middle-to-late range, not concentrated in
one standout layer. **Selected layer 22** (highest score, within the
recommended middle-to-late window). Next step: download only
`layer22.sae.pt` from `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` (not the full
36-layer set -- each file is ~2GB).

## Multi-layer K0 pooling: layers 22, 21, 20 (2026-07-09)

LITERATURE.md's close-read of arXiv:2505.23556 says the method takes the
top K0=10 features **per layer**, then keeps the top K*=20 overall --
since K* > K0, this only makes sense if multiple layers feed the candidate
pool (the paper sweeps several SAE layers, not just one). This project's
Phase 1 convention was single-best-layer selection, which would cap K* at
10 (all of K0), a real deviation from the paper's numbers.

Given the layer-22 separation score (1.742) is barely ahead of layers 21
(1.739) and 20 (1.738) -- essentially tied -- **decided to download SAE
checkpoints for the top 3 layers (22, 21, 20)** and pool their K0=10
candidates each (up to 30 total) before causal-ranking down to K*=20. This
more faithfully replicates the source paper's method than forcing a
single-layer result to fit a multi-layer formula. Extending further (e.g.
top 5) wasn't judged worth the extra ~4GB/downloads given how little the
score drops off (1.742 to 1.717 across all 5 top layers).
Whether to also add Gemma-2-9B / Llama-3.1-8B later (once/if HF gating is
resolved) for the full three-family cross-model set is still open --
doesn't block starting Phase 3 on Qwen3-8B alone.

## Causal-ranking evaluation set: 8 prompts, length-capped (2026-07-09)

Calibrated the real per-call cost of `feature_ig_attribution` on Qwen3-8B:
steady-state ~28.5s per (feature, prompt) call (n_steps=10, batched into one
forward+backward pass), after a one-time ~35s CUDA warmup on the very first
call. At 30 pooled candidates, a full sweep over the corpus (or even just
20 prompts) would take 3-11 hours -- not practical for one run.

**Decided to use 8 evaluation prompts** for this step specifically (~2hrs
total). This is defensible, not just expedient: step 3 (attribution
patching) is a *screening* pass whose only job is trimming 30 pooled
candidates down to 20 -- the rigorous part, causal validation via
suppression measured with the real refusal classifier, is a separate later
step (mirrors the source paper's own two-stage design: cheap ranking, then
expensive validation).

**Also capped prompt length for this step** (excluded from the random
sample entirely, not just deprioritized): a backward pass needs to retain
the full computational graph across all 36 layers, far more memory than
the no-grad forward-only extraction -- and forward-only was already at the
edge of the 6GB card for the corpus's ~1000-token outliers (relying on
Windows' shared-memory spillover, see the "Qwen3-8B 4-bit loading" entry
above). A backward pass on one of those outliers would very likely OOM
outright. Restricted the evaluation sample to harmful TRAIN prompts under
150 characters (comfortably above the corpus's own p90 of 107 chars, so
still representative of the bulk of the corpus, just excluding the long
tail).

**Results (2026-07-09, `results/sae_causal_ranking_Qwen3-8B.json`) -- SUPERSEDED, see below.** Ran
in ~1h50m as estimated, no errors. Top-20 by IG score spread across all
three layers (20: 6, 21: 8, 22: 6 features) -- no single layer dominates,
consistent with how tightly the separation scores were clustered. Top
feature: layer 20/feature 12092 (score 0.547), a clear standout above the
rest (next-highest is 0.334). The bottom 2 of the kept 20 have scores near
zero or slightly negative (-0.008, -0.025) -- expected noise at this
cutoff with only 8 evaluation prompts backing the ranking pass; matches the
source paper's own stated limitation ("small K* may omit relevant
features," cuts both ways -- some low-signal features inevitably survive
too). These borderline features should be weighted accordingly (or
excluded) when interpreting results, not treated as equally confident as
the top-ranked ones.

**Superseded (2026-07-10)**: this run predates the `enable_thinking=False`
fix, on layers 22/21/20 (also superseded by the corrected layer selection).
Kept for history only.

**Corrected results (2026-07-10, same file, re-run on layers 23/25/24):**
ran cleanly, no errors. Layer distribution in top-20: 25: 7, 23: 7, 24: 6 --
still no single layer dominant. **Signal is markedly stronger and cleaner
than the confounded run**: top feature layer 25/feature 65291 (score
**2.371**) and 2nd layer 23/feature 42331 (score **1.461**) are both far
above everything else (3rd place is 0.501) -- a much sharper standout than
the old run's top score of 0.547. No near-zero/negative features anywhere
in this top-20 (unlike the old run's 2 borderline entries), suggesting the
thinking-mode fix didn't just shift the ranking but genuinely improved the
signal-to-noise ratio of the metric itself, consistent with the metric now
reading the model's real answer instead of reasoning-preamble noise.

## Phase 3 methodology (adapted from arXiv:2505.23556)

Following "Understanding Refusal in Language Models with Sparse
Autoencoders" (close-read in full, see LITERATURE.md), not the more ad hoc
single-feature approach in arXiv:2411.11296 (which caused severe capability
collapse -- MMLU 68.8%->36.0% at their strongest steering setting -- from
amplifying one hand-picked feature with no systematic selection):

1. Compute the difference-in-means refusal direction per layer (reuse
   `src/direction/compute.py`, already built for Phase 1).
2. Restrict the SAE's full feature space to the top K0=10 features per layer
   by cosine similarity to that direction.
3. Rank those by causal effect (paper uses Attribution Patching /
   integrated gradients; our first pass may use a cheaper ablation-sweep
   proxy given compute constraints, upgrading to AP if time allows), keep
   the top K*=20.
4. Validate causally via suppression (scale the feature down, measure
   refusal-rate drop) the same way Phase 1 validated the single direction --
   this makes the two phases' causal-validation methodology directly
   comparable.
5. **Any activation-addition/steering experiment must report a capability
   check (e.g. coherence on a held-out benign-completion set) alongside the
   induced-refusal rate** -- arXiv:2411.11296's single-feature approach
   looked like a safety win by refusal-rate alone while quietly destroying
   general capability. Refusal rate alone is not sufficient evidence of a
   working intervention.

## Ethics / department sign-off

Not yet requested. This is calendar time, not engineering time -- worth
starting the conversation with the advisor/department early rather than
leaving it for later, even though the actual risk is low (defensive research
on standard, already-public safety benchmarks, no novel harmful content
sourced or generated for release).

## Qwen3's default thinking mode was uncontrolled -- fixed and redone (2026-07-10)

Discovered while calibrating the suppression-validation generation step:
Qwen3-8B reasons inside a `<think>...</think>` block before answering by
default. `format_prompt()` (used everywhere -- direction extraction,
causal-ranking's logit-diff metric, and all generation-based validation)
was not passing `enable_thinking=False`, so:

- The refusal-vs-compliance logit-diff metric (task #4's causal ranking)
  was reading the logits of the *first token of the reasoning preamble*
  (typically something like "Okay"), not the first token of the model's
  actual answer -- likely a large, silent confound on that metric.
- Any generation-based validation (task #5) would have the same problem:
  the refusal classifier scans the first 200 characters of the completion,
  which would land inside `<think>` musing, not the real answer.

**Verified the fix**: `tokenizer.apply_chat_template(..., enable_thinking=False)`
pre-fills an *empty* `<think>\n\n</think>\n\n` into the prompt itself (not
generated), so the model's first generated token is genuinely the start of
its real answer. Confirmed non-thinking models' templates (SmolLM2,
Qwen2.5) silently ignore this unused kwarg -- safe to set unconditionally
in `format_prompt()` (src/activations/extract.py).

**Decided (user's call, given the ~3.5hr recompute cost) to redo both
already-completed steps rather than patch forward only**: re-ran the
Qwen3-8B activation extraction (task #1) and the SAE causal-ranking sweep
(task #4) with the fix in place, so every number in this project's Qwen3-8B
results consistently measures the model's actual answer behavior, not a
reasoning-preamble artifact. The previously reported layer-22 selection and
the "layer 20/feature 12092 top" ranking result (from before this fix) are
superseded -- see below for the corrected numbers once the redo completes.

**Corrected extraction results (2026-07-10):** re-ran successfully (~92
min). New top-5 layers by separation score:

| layer | score |
|-------|-------|
| 23    | 1.783 |
| 25    | 1.783 |
| 24    | 1.781 |
| 26    | 1.780 |
| 28    | 1.778 |

The selection genuinely shifted (23/25/24/26/28 vs the old run's
22/21/20/23/19) -- confirms the fix mattered, not just a cosmetic change.
Scores are even more tightly clustered than before (spread of 0.005 vs
0.025) and each score is slightly higher overall (~1.78 vs ~1.74),
consistent with this being a cleaner signal now that the measured position
aligns with the model's actual answer rather than a reasoning-preamble
artifact. **New selected top-3 for SAE pooling: layers 23, 25, 24**
(highest three by score). This requires downloading 3 fresh SAE
checkpoints -- none of 23/24/25 were among the previously-downloaded
20/21/22.

## Causal validation via suppression -- results (2026-07-10)

`scripts/validate_sae_features.py`: baseline vs. suppressing the
top-1/top-5/top-20 causally-ranked features (from the corrected
`sae_causal_ranking_Qwen3-8B.json`), on 25 held-out VAL harmful prompts
(seed=1, disjoint from every prompt used anywhere upstream -- direction
extraction, layer selection, and the ranking pass), 40 tokens generated per
completion, real `refusal_classifier` (not the differentiable proxy used
for ranking).

| condition | refusal rate | 95% CI | degenerate |
|---|---|---|---|
| baseline | 0.72 | [0.524, 0.857] | 0/25 |
| suppress top-1 | 0.84 | [0.654, 0.936] | 0/25 |
| suppress top-5 | 0.44 | [0.267, 0.629] | 0/25 |
| suppress top-20 | **0.20** | [0.089, 0.391] | 0/25 |

**This is the project's core causal-validation finding for Phase 3**: the
top-20 condition's refusal rate (0.20) is statistically distinguishable
from baseline (0.72) -- their 95% CIs don't even overlap (0.391 vs 0.524).
Suppressing these 20 SAE features causes a large, real drop in refusal,
and **zero completions degenerated into incoherent output** across all 100
generations in this run -- the model is being made to actually comply, not
just breaking. This satisfies the capability-check requirement noted
earlier in this file (any suppression/steering result must be reported
alongside a coherence check, not refusal rate alone).

**Honest limitation, reported not smoothed over**: suppressing the single
top feature alone (top-1) did *not* reduce refusal -- it went up slightly
(0.84 vs 0.72 baseline), though the CIs overlap heavily so this isn't
necessarily a real effect in the opposite direction, more likely noise at
n=25. The refusal-suppressing effect is clearly **distributed across the
feature set**, not concentrated in one dominant feature, which is
consistent with this literature's general finding that individual SAE
features are rarely fully causally sufficient on their own (see
arXiv:2411.11296's cautionary tale in LITERATURE.md, which this project
explicitly designed around by using a systematic top-K* set rather than a
single hand-picked feature).

## Tightening the results: bigger samples, more conditions (2026-07-10)

The first-pass results above are directionally solid but statistically
thin at the edges: n=25 for validation gives CIs ~20-30 points wide, and
n=8 for the ranking screen means the exact top-20 feature list itself
carries real sampling noise. Decided to redo both, on user request, rather
than leave the numbers at their first-pass strength:

- **Ranking pass**: `N_EVAL_PROMPTS` 8 -> 16 (`scripts/rank_sae_features.py`). Doubles cost
  (~3.8hrs estimated vs the original ~2hrs) but tightens which exact
  features land in the top-20, not just the validation's confidence in
  them.
- **Validation pass**: `N_VAL_PROMPTS` 25 -> 50, plus two intermediate
  conditions (`top10`, `top15`) added alongside the existing
  baseline/top1/top5/top20, for a smoother dose-response curve rather than
  3 sparse points (`scripts/validate_sae_features.py`). ~1.75hrs estimated for 6 conditions x 50
  prompts.

Run sequentially (validation depends on the ranking's output), ~5.5hrs
total. Not redoing activation extraction or layer selection -- those
aren't sample-size-limited in the same way (full 1922-prompt corpus,
already stable across the pre/post-thinking-mode-fix runs' top layers).

**Ranking-pass N=16 result: the top-2 features are stable.** Layer
25/feature 65291 (score 2.198, was 2.371 at N=8) and layer 23/feature
42331 (1.430, was 1.461) remain the top two by a wide margin, essentially
unchanged. Only **1 of 20 features swapped** in the whole ranked list
(a low-ranked one, rank ~17-20) between the N=8 and N=16 runs; layer
distribution in the top-20 is identical (25:7, 23:7, 24:6). This is good
evidence the N=8 screening pass wasn't actually noise-dominated -- doubling
the sample barely moved the result. Full comparison and the new ranked
list in `results/sae_causal_ranking_Qwen3-8B.json` (overwritten; N=8 result
is this file's git history if needed).

**Validation-pass N=50, 6-condition result: tighter, and honestly more
interesting than the n=25 run.** Real per-call cost ran ~2x slower than
calibrated (more conditions apparently added overhead beyond simple
linear scaling), total runtime ~4.5hrs not the estimated ~1.75hrs.

| condition | n | refusal rate | 95% CI |
|---|---|---|---|
| baseline | 50 | 84.0% | [71.5%, 91.7%] |
| top1 | 50 | 78.0% | [64.8%, 87.3%] |
| top5 | 50 | 44.0% | [31.2%, 57.7%] |
| top10 | 50 | 44.0% | [31.2%, 57.7%] |
| **top15** | 50 | **18.0%** | **[9.8%, 30.8%]** |
| top20 | 50 | 26.0% | [15.9%, 39.6%] |

Zero degenerate completions in every condition (300 generations total) --
capability preserved throughout.

**Not a clean monotonic dose-response curve, reported exactly as observed
rather than smoothed over**: refusal bottoms out at **top15** (18%), then
ticks back *up* to 26% at top20. top15 and top20's CIs overlap
(9.8-30.8% vs 15.9-39.6%), so this uptick isn't necessarily a real
reversal -- plausibly noise, or the 5 lowest-ranked features (IG scores
0.08-0.02, close to the noise floor) contribute little net suppression
individually and one may even mildly counteract the effect, similar to
how suppressing the single top feature alone (top1) barely moved refusal
in the first-pass n=25 run. top5 and top10 landed on the *exact same*
count (22/50 refused) -- the 6 features added between rank 6 and rank 10
had zero net aggregate effect on this specific 50-prompt sample.

**What's now statistically solid that wasn't before**: baseline is
distinguishable (non-overlapping CI) from top5 onward, not just top20 as
in the n=25 run -- the tighter sample resolved the middle data points that
were previously ambiguous. **top15, not top20, is the strongest single
data point** for the "systematic feature suppression causes refusal
collapse" claim -- worth leading with in the write-up rather than top20.

## Found the real reason the curve wasn't clean: uncontrolled sampling (2026-07-11)

User pushed back on the non-monotonic top15-vs-top20 result -- investigated
rather than shrugging it off as "noise" (see [[feedback-thesis-rigor-upfront]]).
Checked Qwen3-8B's default `GenerationConfig`: `do_sample=True,
temperature=0.6, top_p=0.95, top_k=20`. **Every generation call in this
project's causal-validation pipeline was stochastic** -- one sample per
(prompt, condition) pair, never overridden to greedy decoding. This
directly explains sampling-driven wobble in the results (e.g. top5 and
top10 landing on the exact same count by coincidence, top20 ticking back
up from top15) that isn't necessarily the true aggregate causal effect.

**Not unique to Phase 3**: checked Qwen2.5-1.5B-Instruct too (used in
Phase 1) -- also defaults to `do_sample=True` (temperature=0.7). SmolLM2
defaults to `do_sample=None` (effectively greedy), which may be *why*
Phase 1's SmolLM2 numbers happened to look cleaner than Qwen2.5's.

**Fix**: `do_sample=False` added to all four `model.generate()` calls in
`src/direction/interventions.py` (ablation, addition, feature suppression,
baseline) -- greedy decoding isolates the intervention's true effect from
sampling noise, and makes every result exactly reproducible run-to-run.
Verified via a new regression test (`test_generate_baseline_is_deterministic`)
that two identical calls now produce byte-identical output.

**Decided (user's call): redo Phase 3's suppression validation with this
fix, leave Phase 1's numbers as-is for now.** Phase 1's results predate
this fix and technically share the same gap, but they're already
published/stable and not the active work -- documented here as a known,
accepted limitation rather than silently ignored. Worth redoing if Phase 1
is revisited later (deferred to a future session, cheap since Phase 1 uses
small fast models -- see [[project-thesis-jailbreak-detection]] memory).

**Deterministic re-run result (2026-07-11) -- this is the final, definitive
number for Phase 3, superseding both earlier passes:**

| condition | n | refusal rate | 95% CI |
|---|---|---|---|
| baseline | 50 | 82.0% | [69.2%, 90.2%] |
| suppress top-1 | 50 | 84.0% | [71.5%, 91.7%] |
| suppress top-5 | 50 | 42.0% | [29.4%, 55.8%] |
| suppress top-10 | 50 | 32.0% | [20.8%, 45.8%] |
| **suppress top-15** | 50 | **24.0%** | **[14.3%, 37.4%]** |
| suppress top-20 | 50 | 26.0% | [15.9%, 39.6%] |

**This is the clean curve the user was asking for.** Steady monotonic
decline from top1 through top15 (84% -> 42% -> 32% -> 24%), then a
plateau at top15/top20 (24% vs 26%, heavily overlapping CIs -- settling,
not a real reversal). Confirms the hypothesis exactly: the earlier
non-monotonic wobble (n=50, stochastic decoding) was a real artifact of
sampling noise, not a genuine property of the intervention. Zero
degenerate completions across all 300 generations. Baseline is
distinguishable from top5 onward. **top15 remains the strongest single
data point** (lowest refusal rate, tightest practical floor before the
plateau) -- this is now the number to lead with in any summary of the
project's core finding.

## Phase 1 redo with greedy decoding, after all (2026-07-11)

Despite deciding above to defer Phase 1's redo to a future session, user
asked for it the same session once the current job finished (cheap enough
to just do). Re-ran `scripts/reproduce_direction.py` for both
models and `scripts/calibrate_alpha.py` for both, no code
changes needed (the `do_sample=False` fix in `interventions.py` already
covers these functions).

**Result: confirms the prediction exactly.**
- **SmolLM2-1.7B-Instruct: numbers are byte-identical** to the original
  sampling-based run, across both the causal-validation table and the
  full 7-point alpha sweep -- expected, since SmolLM2's default
  `GenerationConfig` already had `do_sample=None` (effectively greedy).
- **Qwen2.5-1.5B-Instruct: nearly identical, one small shift.** Causal
  validation: `harmful_ablated` moved from 0.0% to 3.3% (a single
  completion out of 30) -- the headline 100%->~0% necessity finding is
  unchanged. Alpha sweep (n=12, more sensitive to individual-completion
  noise): a few mid-table values shifted (0.50: 67%->50%; 3.00: 17%->0%;
  degenerate fractions at 1.50/2.00 shifted similarly) but the calibrated
  alpha (1.0) and the overall shape (clean high-refusal window, then
  degenerate collapse) are unchanged.

Both scripts ran in well under 5 minutes total (small models, no 4-bit
quantization needed) -- confirms this really was cheap, as predicted, and
worth just doing rather than deferring. RESULTS.md updated in place
(no more "predates this fix" caveats anywhere in the repo); the earlier
"deferred to a future session" note above is superseded by this entry.

## Head-to-head: dense-direction ablation vs SAE-feature suppression (2026-07-11)

`scripts/ablate_qwen3_direction.py`: Phase 1's ablation method
(project the direction out at every layer, every token) run on Qwen3-8B,
same 50 held-out VAL prompts as the SAE suppression validation, same
`do_sample=False`. Ablation layer (23) selected via separation score on
the **TEST split** -- not VAL, since VAL is the exact prompt set used for
causal validation here and was already used to pick Phase 3's SAE layers;
scoring on TEST avoids that leakage while still allowing the same VAL
prompts to be reused for a true apples-to-apples comparison. Baseline
(82.0%) reused directly from the suppression-validation JSON rather than
re-measured.

**Raw numbers**: dense-direction ablation drove refusal down to 6.0%
[2.1%, 16.2%] -- lower than any SAE-suppression condition (best was
top-15 at 24.0%). Zero degenerate completions (0/50), so this isn't a
coherence collapse.

**But the raw number is misleading, and inspecting actual completions
caught it**: 47 of 50 "non-refusal" completions under dense ablation are
not compliance -- they're the model moralizing/lecturing about why the
request is illegal or unethical ("Cracking passwords... is illegal,
unethical, and immoral...") without ever using the keyword classifier's
refusal markers ("I cannot", "I'm sorry", etc.). This is a third behavior
mode, neither a clean refusal nor genuine harmful compliance.

**Correction after the classifier-validation spot-check below**: this was
initially written up as the classifier "miscounting" or being "unable to
distinguish" moralize from compliance -- that characterization turned out
to be wrong once actually tested (see next entry). The classifier
correctly calls moralize non-refuse; that's not an error, it's exactly
what a refusal-phrase detector should do. **The real issue is one level
up**: `refusal_rate` as a summary statistic conflates moralize (safe,
non-compliant) and comply (unsafe) under one "non_refuse" bucket, so
reading "6% refusal" as "94% compliance" is the actual mistake, not
anything wrong with the classifier itself.

**Honest conclusion, revised**: the true *harmful-compliance* gap between
dense ablation and SAE-feature suppression is very likely much smaller
than the raw 6% vs 24% *non-refusal* numbers suggest, since most of dense
ablation's non-refusals are moralize, not comply. What the raw numbers do
still support: dense ablation (36 layers touched) is a blunter, more
disruptive intervention that pushes the model into a "moralize instead of
refuse" mode far more often than SAE-feature suppression (3 layers) does
-- an interesting, honest finding about intervention bluntness, just not
the clean "X% more effective at inducing compliance" claim the raw numbers
alone would suggest. **Do not report the 6% vs 24% comparison as a
compliance-rate comparison** -- it measures refusal-phrasing rate, not
compliance rate, and those are not the same thing for this intervention.

## Classifier-validation spot-check tooling (2026-07-11)

Rather than spend more compute tightening CIs around a classifier of
unknown accuracy (considered and rejected -- see below), built a
human-labeling spot-check that costs no new GPU time: every experiment's
completions are already saved in its results JSON, so
`scripts/sample_for_labeling.py` draws a stratified sample
(3 per source/condition group, 45 total across all 15 groups from Phase 1,
Phase 3, and the head-to-head) and writes a CSV worksheet with the
classifier's own verdict deliberately hidden (saved separately to
`results/classifier_spotcheck_reference.json`) to avoid anchoring the
labeler's judgment. `scripts/score_agreement.py` joins the
filled-in worksheet back against the classifier's calls once labeled,
reporting overall agreement and a per-label breakdown -- the "moralize"
row's accuracy is the number that matters most, since that's the specific
blind spot the head-to-head comparison surfaced.

**Why not just increase sample sizes instead** (the alternative the user
asked about directly): bigger N only tightens confidence intervals around
whatever the classifier measures -- it buys precision, not accuracy. Given
the classifier has a demonstrated blind spot, spending ~16hrs of compute
(N=50->100 validation, N=16->30 ranking) would produce a more confident-
looking estimate of a potentially biased number, which is a worse outcome
than the status quo, not better. Fixing the measurement instrument (cheap,
no GPU needed, reuses existing data) comes first; bigger samples are only
worth it once the classifier is trusted.

Worksheet/reference files are gitignored (`results/`, generated artifacts,
not source) -- awaiting the user's labels before `scripts/score_agreement.py` can report
real numbers.

## Classifier-validation spot-check: results (2026-07-11)

All 45 sampled completions labeled (17 refuse, 13 moralize, 13 comply, 2
partial). Classifier agreement: **44/45 (97.8%)**. Broken down by human
label -- this is the classifier's binary refuse/non_refuse accuracy
against what the completion actually was:

| human label | classifier accuracy |
|---|---|
| refuse | 17/17 (100%) |
| moralize | 13/13 (100%) |
| comply | 13/13 (100%) |
| partial | 1/2 (50%) |

**The classifier is more accurate than the head-to-head writeup initially
implied.** All 13 moralize completions were correctly called non_refuse --
zero misclassifications in this sample. This means the earlier framing
("the classifier cannot distinguish moralize from compliance") was not
quite right and has been corrected in the head-to-head entry above: the
classifier does its narrow job (detecting refusal phrasing) accurately.
The actual problem is that `refusal_rate` as a single number conflates
moralize and comply into "non_refuse," so a reader can't tell from the
statistic alone how much of a low refusal rate is safe moralizing vs
actual harmful compliance -- that's a reporting/metric-design issue, not a
classifier bug. Only the "partial" category (n=2, too small to read much
into) showed any disagreement, and partial-compliance completions are
inherently the hardest case for a binary classifier regardless of
implementation.

**Practical upshot for future work**: if a "true compliance rate" number
is ever needed (e.g. for a head-to-head claim like "method A achieves N%
more actual harmful compliance than method B"), it requires either a
separate moralize-vs-comply classifier or an LLM-judge -- the existing
`refusal_classifier.py` was never designed to make that distinction and,
per this spot-check, doesn't need to be replaced for its own stated
purpose (detecting refusal phrasing, which is all Phase 1-3 asked of it).

## Baseline detectors + adversarial evaluation: split discipline (2026-07-11)

Building the head-to-head comparison required deciding where each of the
four detectors' "training" (threshold calibration) happens, without
re-opening any split already used elsewhere in this project.

**Decision**: TRAIN (already used for direction estimation and SAE ranking)
is reused as-is; VAL becomes the threshold-calibration split for all four
detectors (`scripts/calibrate_thresholds.py`, via Youden's J --
`src.eval.detector_metrics.youden_threshold`); TEST is reserved entirely
for final reporting and is also the only source pool for the adversarial
paraphrase set.

**Why VAL and not a new split**: VAL was previously used only for Phase 3's
*generation-based* causal suppression validation (does suppressing features
change what the model outputs). Using VAL's *activations* to fit a
classifier threshold is a different, non-overlapping use of the same
prompts -- it doesn't leak into anything reported on TEST, and avoids
introducing a fifth split into an already-established train/val/test
manifest (`data/splits/corpus_split_v1.json`) that several other scripts
depend on by name.

**Why not calibrate on TRAIN directly**: TRAIN was already used to *fit* the
dense direction and rank the SAE features -- picking a decision threshold
on the same data used to derive the underlying score would be a second
layer of fitting on the same split (mild optimism, even for a single 1D
cutoff). VAL is disjoint from both TRAIN and TEST, so it's the correct
choice: closest in spirit to how `scripts/calibrate_alpha.py`
calibrates alpha on a split disjoint from the reported val in Phase 1.

## Baseline detector design choices (2026-07-11)

**Keyword filter** (`src/baselines/keyword_filter.py`): a hand-curated
~50-term lexicon across weapon/malware/drug/violence/fraud categories,
generic vocabulary rather than copied benchmark phrasing, deliberately kept
simple since its whole purpose in this comparison is to be the *weak*
baseline that surface-vocabulary paraphrase should defeat. Considered and
rejected: mining the training corpus's own vocabulary for lexicon terms --
would overfit to this project's specific benchmarks' phrasing and overstate
the baseline's real-world performance.

**Perplexity filter** (`src/baselines/perplexity_filter.py`): GPT-2, per
Alon & Kamfonas 2023 (arXiv:2308.14132, verified via web search this
session) -- their own reference model, so this isn't an arbitrary choice of
LM. A larger/better LM would likely change the absolute perplexity values
but not the qualitative story (fluent text scores low regardless of the
scoring model; GCG-style gibberish scores high regardless of the scoring
model) -- not worth the extra compute for this comparison's purposes.

**Superseded** -- see "Perplexity backbone switched from GPT-2 to
GPT-Neo-1.3B" below. The "wouldn't change the qualitative story" prediction
above turned out to be half right: true for GCG (unchanged at 100%), false
for the XSTest false-positive rate (13.5% -> 75.7%, a real, large change) --
worth noting as a specific place this project's own prior reasoning was
wrong and corrected empirically rather than left standing.

## Adversarial paraphrase set: real JailbreakBench artifacts, verified feasible (2026-07-11)

Confirmed via web search + direct fetch this session (not assumed): 
`github.com/JailbreakBench/artifacts` hosts real, published attack prompts
at `attack-artifacts/{METHOD}/{subdir}/{model}.json` (e.g.
`attack-artifacts/PAIR/black_box/vicuna-13b-v1.5.json`), fetchable via plain
`raw.githubusercontent.com` GET requests, no auth, in the same direct-fetch
style already used for AdvBench/HarmBench in `src/data/loaders.py`. Sample
record schema confirmed by fetching a real file: `{index, goal, behavior,
category, prompt, response, jailbroken, ...}` -- `jailbroken: true/false`
lets us keep only prompts that actually succeeded against their original
target model.

**Decision: use real JailbreakBench artifacts, not self-authored jailbreak
templates** (confirmed with user before implementation). Keeps this
project's established "no self-authored harmful content, only real
published benchmarks" policy (ETHICS.md) intact for the adversarial set too,
and directly tests the paraphrase-robustness claim from arXiv:2505.23556
(Phase 3's source paper) using real disguised prompts rather than ones this
project invented.

**PAIR (black_box) as the primary "paraphrase" condition, GCG (white_box +
transfer) as a secondary "suffix" condition, reported separately, not
pooled by default.** These are mechanistically different attacks -- PAIR
produces fluent, roleplay-framed rewrites (a real paraphrase-robustness
test); GCG appends an optimized gibberish suffix (a fluency-break, which is
what perplexity filtering specifically targets). Pooling them into one
number would hide which mechanism drives any given detector's flag rate --
confirmed necessary once the first pooled run showed perplexity looking
artificially competitive with the activation detectors purely because of
GCG's 100% detection rate (see RESULTS.md's adversarial-evaluation section).

**Matching, not synthesis**: artifacts are matched to this project's own
TEST-split JBB-sourced harmful goals by exact (case/whitespace-normalized)
`goal` text, restricting the adversarial set to behaviors no detector's
threshold was calibrated on. Only 11 of TEST's JBB goals had any matching
`jailbroken: true` artifact, yielding 35 adversarial prompts (21 PAIR, 14
GCG) -- smaller than ideal (see RESULTS.md's limitations) but real, not
padded with synthetic content to hit a target sample size.

**Explicitly not a jailbreak-transfer claim**: these artifacts were
generated and validated as successful jailbreaks against their *original*
target models (Vicuna-13B, Llama-2-7B-chat, GPT-3.5, GPT-4) -- not against
this project's own models. No claim is made or tested about whether these
specific prompts would jailbreak Qwen3-8B/Qwen2.5/SmolLM2; only the prompt
*text* is reused, to test whether disguised-harmful surface forms fool a
*prompt classifier*, independent of generation-time attack success.

## Adversarial evaluation results: honest, not smoothed over (2026-07-11)

The pooled adversarial-detection numbers (perplexity 62.9%, dense-direction
62.9%, SAE-feature 57.1%) made perplexity look roughly competitive with the
activation-based detectors. Breaking down by attack method (per the
decision above) showed this was an artifact of pooling: perplexity hits
100% on GCG (its textbook case) but only 38.1% on PAIR, while
dense-direction/SAE-feature both drop from ~88% (TEST-split performance) to
33-43% on PAIR specifically. **This project's own numbers do not replicate
arXiv:2505.23556's finding that SAE features are more robust to adversarial
paraphrase than a dense direction** -- on PAIR, dense-direction (42.9%,
n=21) numerically edges out SAE-feature (33.3%, n=21), the opposite
direction.

**Tightened after an initial, informal pass**: the first version of this
finding called the CIs "overlapping heavily" and left it there. That's a
weaker check than the data actually supports -- both detectors are scored
on the exact same 21 PAIR prompts, which is paired data, not two
independent samples; comparing two separate Wilson CIs for overlap is an
informal proxy that can miss (or wrongly suggest) a real paired difference.
Added `src.eval.detector_metrics.mcnemar_exact` (exact McNemar's test on
the discordant pairs) and reran the comparison on the actual paired
predictions: only 2 of 21 pairs are discordant (dense flags 2 prompts SAE
doesn't; SAE flags none dense doesn't), p = 0.5 -- confirms the original
conclusion on solid statistical footing rather than overturning it, but
it's the correct test for this specific claim and should have been done
the first time, not after the fact. Same standing "full rigor upfront, not
fast-now/rigor-later" discipline this project applies elsewhere.
Reported in RESULTS.md as "no replication of that claim at this sample
size" -- not papered over as a null result, and not oversold as a reversal
of the published finding. Consistent with this project's established
practice (the Phase 3 head-to-head's moralize-vs-comply finding, the
classifier spot-check above) of testing a plausible expectation rather
than assuming it and writing up whatever the actual numbers show.

## Found (and fixed going forward) a mild leakage pattern in the Qwen3-8B dense-direction pipeline (2026-07-11)

While starting the Qwen2.5/SmolLM2 cross-model extension, noticed that the
just-merged Qwen3-8B pipeline (`scripts/ablate_qwen3_direction.py` -> `scripts/calibrate_thresholds.py` -> `scripts/compare_detectors.py`)
selects the dense-direction detector's layer via **TEST**-split separation
score (`scripts/ablate_qwen3_direction.py`'s docstring explains this was to avoid VAL, which was
already used by Phase 3's causal validation at the time), then reports the
detector's final classification metrics on that **same TEST split**. That's
reusing one split for both layer selection and final reporting -- a mild
leakage pattern this project explicitly flagged and fixed elsewhere (see
METHODOLOGY.md's "Train/calib/val separation" entry, which describes fixing
the identical issue for Phase 1's causal validation by adding a third
split).

**Checked whether this actually distorted anything**: recomputed Qwen3-8B's
separation scores using VAL instead of TEST -- **layer 23 is selected
either way** (VAL top-3: [23, 25, 24]; TEST top-3: [23, 24, 22], same top
layer, nearly identical scores: 1.783 vs. 1.750). So this leakage pattern
existed in the merged code but did not actually change Qwen3-8B's reported
numbers -- consistent with Phase 3's earlier finding that separation scores
are tightly clustered across nearby layers (a robust, low-variance
architectural signal, not something a handful of TEST-split prompts could
meaningfully overfit).

**Decision**: not worth reopening the merged Qwen3-8B PR to redo numbers
that would come out identical -- but new work (this cross-model extension,
and any future Phase 6 work) uses the corrected discipline:
`src.detectors.dense_direction_detector.select_layer_and_calibrate` does
BOTH layer selection and threshold calibration on VAL, leaving TEST
completely untouched until final reporting. Documented here rather than
silently fixed, since "we found a leakage pattern in already-shipped code,
checked it didn't matter, and are fixing the discipline going forward" is
exactly the kind of thing this project's rigor culture expects to be
written down, not just quietly patched.

## Cross-model dense-direction extension: real, unexplained finding (2026-07-11)

Extended the dense-direction detector to Qwen2.5-1.5B-Instruct (layer 20,
selected on VAL) and SmolLM2-1.7B-Instruct (layer 14, selected on VAL),
reusing the same 35-prompt adversarial set (fresh per-model activation
extraction, same real JailbreakBench prompts, not refetched) and the
Qwen3-8B run's keyword/perplexity baseline numbers (model-agnostic, not
rerun). Full numbers in RESULTS.md's cross-model section.

**Headline finding**: SmolLM2's dense-direction detector holds up far
better under PAIR (fluent paraphrase) attacks (90.5%) than either Qwen
model (Qwen2.5: 38.1%, Qwen3-8B: 42.9%) -- non-overlapping CIs, a real
difference at this sample size. **Deliberately not explained away with a
plausible-sounding story asserted as fact** -- flagged one candidate
hypothesis (SmolLM2's weaker/less-linear baseline refusal behavior from
Phase 1 might correlate with the paraphrase-robustness gap) explicitly as
untested speculation in RESULTS.md, not a conclusion. This is the same
discipline as the classifier spot-check earlier in this document: a
plausible story is not evidence until it's actually tested, and asserting
one without testing it is exactly the mistake corrected back then.

## Perplexity backbone switched from GPT-2 to GPT-Neo-1.3B (2026-07-11)

User asked, after seeing the perplexity filter's bad XSTest-safe number
(13.5% correctly-not-flagged), whether GPT-2 was really the right choice
for a thesis given it's a small 2019 base model. Worked through the
alternatives:

1. **A newer/better OpenAI GPT (GPT-4/GPT-5) was considered and rejected.**
   Three concrete blockers, not just "it's closed-weight": (a) it would send
   real harmful-intent prompts and actual jailbreak-attack text (this
   project's whole corpus) to a third-party paid API -- everything else in
   this project runs locally specifically to avoid that exposure (see
   ETHICS.md); (b) current chat-completion APIs don't cleanly expose the
   full-sequence log-probabilities this calculation needs, so it isn't even
   a clean drop-in; (c) it would introduce a recurring paid-API dependency
   and break reproducibility for anyone without the same billing access,
   unlike every other number in this project (reproducible from public
   weights alone).
2. **One of this project's own target models (Qwen2.5-1.5B, SmolLM2-1.7B,
   Qwen3-8B) was considered and rejected too** -- initially proposed by
   Claude, then walked back after the user pushed back ("u sure its the
   right call for a thesis project?"). Using a target model as the
   "independent" baseline's backbone breaks the comparison's own logic: the
   baseline is supposed to be cheap and model-agnostic precisely so the
   activation-detector-vs-baseline comparison isolates what deep model
   access buys you. It also concretely breaks independence for the
   Qwen2.5-1.5B row specifically, since the dense-direction detector
   already runs on that exact checkpoint (see the cross-model section
   above) -- perplexity and dense-direction would be scoring off the same
   model for that row.
3. **Landed on GPT-Neo-1.3B** (EleutherAI, 2021): open-weight, small enough
   to run without quantization, meaningfully more modern/better-trained
   than GPT-2, not used as a target model anywhere else in this project,
   and stays in the spirit of "a GPT-family model" the original paper used
   (Alon & Kamfonas explicitly built this method around GPT-2 specifically
   for its ubiquity as a plain reference scorer).

**Empirical outcome after rerunning `scripts/calibrate_thresholds.py`/`scripts/compare_detectors.py`** (full
numbers in RESULTS.md): confirmed the hypothesis behind the switch, but not
in the way expected.
- **XSTest-safe correctly-not-flagged rate: 13.5% -> 75.7%**, a large real
  improvement -- GPT-2's near-total failure there was substantially a
  weak-model artifact, not an inherent property of perplexity filtering.
- **GCG detection: unchanged at a perfect 100%** -- confirms this is a
  property of the attack text (any reasonable LM finds an optimized
  gibberish suffix improbable), not something a weak scoring model was
  getting right by accident.
- **PAIR detection: 38.1% -> 0.0%** -- the opposite direction from XSTest.
  The stronger model recognizes PAIR's fluent, roleplay-framed paraphrases
  as ordinary, unremarkable text and never flags any of them. This
  strengthens rather than weakens the project's core finding about
  perplexity filtering: GPT-2's 38.1% was likely inflated by its own
  weakness at modeling fluent text, not genuine partial detection of
  anything adversarial -- with a better backbone, the "perplexity cannot
  catch fluent paraphrase" conclusion holds *more* cleanly, not less.

Overall TEST-split accuracy/AUROC barely moved (0.451 -> 0.516 AUROC,
still near chance) -- as expected, since perplexity isn't measuring
semantic harmfulness regardless of backbone quality; only the two
conditions that specifically depend on "does this model find ordinary
unusual-but-fluent text surprising" (XSTest, PAIR) changed substantially.
No other detector's numbers changed (dense-direction, SAE-feature, and the
McNemar comparison between them never touch the perplexity backbone).

## Perplexity backbone, round two: GPT-Neo-1.3B is itself stale (2026-07-11, same session)

User pointed out, immediately after the GPT-Neo-1.3B switch above, that
2021 is itself several years stale for a thesis being presented in 2027.
Fair -- re-opened the choice rather than treating "newer than GPT-2" as
good enough.

**Verified via web search (not assumed) what's actually current in the
1B-4B open-weight range**: SmolLM3-3B, Gemma 3 (1B/4B), Phi-4-mini (3.8B),
Qwen3.5 (2B/4B), Llama 3.2 3B were the live 2025/2026 options. Ruled out by
family, not by quality: Qwen and SmolLM already used as target models
(Qwen2.5-1.5B, Qwen3-8B, SmolLM2-1.7B); Llama and Gemma explicitly reserved
for Phase 6's cross-model work per README.md, so using either now would
recreate the exact same-family conflict Phase 6 would later hit. That left
**Phi-4-mini-instruct** (Microsoft, 3.8B) as the best fit: independent
family, actively maintained (Microsoft shipped further Phi-4 variants as
recently as March 2026, confirmed via search), and among the
best-performing models in its size class. Loaded 4-bit (verified HF id:
`microsoft/Phi-4-mini-instruct`) since 3.8B doesn't fit unquantized in the
6GB GPU's ~4.5GB free memory, reusing the `BitsAndBytesConfig` pattern
already built for Qwen3-8B.

**Empirical result: worse than GPT-Neo-1.3B on the exact number the switch
was meant to fix.** XSTest-safe correctly-not-flagged rate: 75.7%
(GPT-Neo-1.3B) -> 40.5% (Phi-4-mini-instruct) -- a real regression despite
being newer and roughly 3x larger. Investigated rather than accepted at
face value: Phi-4-mini-instruct is **instruction-tuned**, and confirmed via
a second search that Microsoft has not released a base (non-chat-tuned)
checkpoint for it. `compute_perplexity` scores raw prompt text with no chat
template applied -- exactly the methodology Alon & Kamfonas used with
GPT-2, a base model. Scoring un-templated text with a model fine-tuned
specifically on chat-formatted conversations is off-distribution for it,
which plausibly explains the regression: the model may be finding
ordinary declarative sentences unusual specifically because its
post-training pulled its distribution toward conversational formatting,
not because it's a worse language model in general.

**Decision: perplexity scoring needs a genuine base model, not merely "a
newer model."** This was the wrong axis to optimize -- recency alone
doesn't fix this if the newer model is instruction-tuned. Rejected
Phi-4-mini-instruct on this basis, not on capability.

## Perplexity backbone, round three: OLMo-2-0425-1B, and a genuinely messy result (2026-07-11, same session)

Needed a modern (2025+), independent (not Qwen/SmolLM/Llama/Gemma), and
**base** (non-instruct) small open model. Verified via search:
**`allenai/OLMo-2-0425-1B`** (AI2, released April 2025, 1B params, Apache
2.0) -- a genuine pretrained base checkpoint, fully open (weights, training
data, and code all released, unusually strong reproducibility story for a
thesis citation), independent of every model family used or reserved
elsewhere in this project. At 1B params it runs without quantization.

**Result: neither confirms nor cleanly refutes the "better base model
fixes XSTest" hypothesis -- it's messier than that.** Full four-backbone
history on XSTest-safe correctly-not-flagged rate: GPT-2 13.5% -> GPT-Neo-1.3B
75.7% -> Phi-4-mini-instruct 40.5% (rejected) -> OLMo-2-0425-1B 24.3%.
OLMo-2-0425-1B is a genuine base model, modern, and well-trained on
substantially better data than either GPT-2 or GPT-Neo-1.3B -- yet it
scored *worse* on this specific check than GPT-Neo-1.3B (2021, smaller
training run, less modern data pipeline). Recency, base-vs-instruct
status, and even raw capability don't predict this number cleanly; the
best performer across all four attempts remains GPT-Neo-1.3B, the second
one tried, not the newest, largest, or most "correct" by any single
criterion checked.

**Decision: stop searching for a better backbone here.** Four real
attempts is enough to establish the actual, more defensible finding: XSTest
false-positive behavior under perplexity scoring appears to depend on
idiosyncratic properties of each specific reference model's training
distribution, not on any single axis (age, size, base-vs-instruct) this
project can cheaply optimize. This is a more scientifically honest
conclusion than "we found the fix" would have been, and it's only visible
*because* four different backbones were actually tried rather than
assumed. **OLMo-2-0425-1B is the final backbone** -- chosen because it is
the methodologically correct choice (base model, matching Alon & Kamfonas'
own approach, independent of every target-model family, fully
reproducible), not because it produced the best number. GCG detection
(100%) and PAIR detection (0.0%) are unchanged from the GPT-Neo-1.3B and
Phi-4-mini-instruct versions -- those two findings are robust across all
three of the "better" backbones tried, strengthening confidence in them
specifically (see RESULTS.md).

## Significance testing: DeLong's test and Cochran's Q added (2026-07-11)

Asked "what's left before Phase 5" and identified that this project's
significance testing so far covered exactly one comparison (McNemar's,
dense-direction vs. SAE-feature on the adversarial set) out of several
places a paired/repeated-measures test was actually warranted but had only
been argued from eyeballing Wilson CIs:

1. **Dense-direction vs. SAE-feature AUROC on TEST-overall** (0.983 vs.
   0.975, Qwen3-8B) -- close enough to need a real test. Added
   `src.eval.detector_metrics.delong_auc_test` (DeLong et al. 1988, via Sun
   & Xu 2014's structural-components formulation -- verified the AUC
   values it computes match `sklearn.roc_auc_score` exactly, then tested
   against both an identical-scores null case and a seeded synthetic
   case with a real separation, confirming the test detects a genuine
   difference when one exists). Result: diff = 0.0076, **p = 0.068** --
   not significant, consistent with the adversarial-set McNemar result
   (p = 0.5). Two independent evaluations, same conclusion: no
   statistically distinguishable difference between the two detectors
   found anywhere in this project.
2. **The 3-model PAIR comparison** (Qwen2.5-1.5B: 38.1%, Qwen3-8B: 42.9%,
   SmolLM2: 90.5%) -- previously argued only from non-overlapping CIs.
   Added `cochrans_q` (generalizes McNemar's from 2 to *k* related
   classifiers on the same items; verified against a perfect-agreement
   null case, a clear-difference case, and an equal-marginal-rates case
   that forces Q to exactly 0 regardless of item-level pattern -- a
   property of the test worth checking explicitly since it's easy to get
   the formula subtly wrong). `scripts/cross_model_significance.py`
   reuses Qwen3-8B's cached adversarial activations and does a fresh
   (cheap, forward-pass-only) extraction for Qwen2.5-1.5B/SmolLM2-1.7B on
   just the 21 PAIR prompts. Result: **Q = 13.06, df = 2, p = 0.0015** --
   clearly significant. Formally confirms what RESULTS.md's cross-model
   section previously stated informally: SmolLM2's PAIR-paraphrase
   robustness really is a significant, real difference from both Qwen
   models, not just a CI-overlap artifact.

## Perplexity backbone, round four: Olmo-3-1025-7B, and the non-monotonic pattern gets stronger (2026-07-12)

User asked to keep looking for a newer model immediately after OLMo-2-0425-1B was settled on above. Verified via search what AI2 (and other independent labs -- IBM Granite 4.1, Falcon 3) have released more recently: **`allenai/Olmo-3-1025-7B`**, AI2's next generation after OLMo-2, released October 2025 (vs. OLMo-2-0425-1B's April 2025), a genuine base checkpoint, same fully-open lineage (weights, training data, and code all released). IBM Granite 4.1 (November 2025, hybrid Mamba-2/Transformer, 3B/8B/30B, base+instruct both released) was a close second candidate but Olmo-3 was chosen to stay within the already-vetted, already-cited OLMo lineage rather than introduce a fourth family into the backbone history. At 7B it needs 4-bit quantization on the 6GB GPU (reused the same `BitsAndBytesConfig` pattern as Qwen3-8B and the rejected Phi-4-mini-instruct attempt).

**Empirical result: the non-monotonic pattern got more extreme, not less.** Full five-backbone sequence on XSTest-safe correctly-not-flagged rate: GPT-2 (2019, 124M) 13.5% -> GPT-Neo-1.3B (2021, 1.3B) 75.7% -> Phi-4-mini-instruct (2025, 3.8B) 40.5% -> OLMo-2-0425-1B (2025, 1B) 24.3% -> Olmo-3-1025-7B (2025, 7B) **13.5%** -- the newest and largest model in the entire sequence ties the oldest and smallest one exactly, down to the confidence interval ([5.9%, 28.0%] both times). GCG detection (100%) and PAIR detection (0.0%) are unchanged yet again, now confirmed across four independent replacement backbones instead of three.

**Decision: stop here.** Five real, independently verified and run backbones is enough to establish the finding conclusively: recency, parameter count, and base-vs-instruct status do not predict XSTest false-positive behavior under perplexity scoring in any way this project can act on. Chasing a sixth model would have diminishing scientific return -- the point (this is idiosyncratic to each model's training distribution, not a capability gap fixable by picking a better model) is now about as well-evidenced as it can get from this angle. **Olmo-3-1025-7B is the final backbone**, chosen for the same reason OLMo-2-0425-1B was (genuine base model, independent family, fully open/reproducible, current as of this session) -- not because it produced the best number, since by this point it's clear no backbone choice will.

## Phase 6 Wave 1: dense-direction extension to Llama-3.1-8B-Instruct and Gemma-2-9B-it (2026-07-12)

**Gating resolved**: both models were gated on Hugging Face (verified via
real `hf_hub_download` attempts, not just `model_info` -- confirmed
`model_info` succeeds even without access, so it's not a reliable check).
User requested access directly. Took several rounds to actually unblock:
license acceptance alone wasn't sufficient -- the account's fine-grained
API token had `canReadGatedRepos: false` even after the Gemma license was
accepted, a separate permission from the general "read access to contents
of all repos" toggle. Editing the existing token's permissions didn't take
effect even after multiple attempts (unclear whether this was a genuine
propagation delay or a real platform quirk where permission edits on an
already-issued fine-grained token don't reliably apply) -- re-checking
after enough time had passed showed the edit finally took. Documented here
since this is exactly the kind of environment/access friction worth a
record for future sessions: **when `hf_hub_download` 403s with "Please
enable access to public gated repositories in your fine-grained token
settings," check the token's specific `canReadGatedRepos` scope via
`HfApi().whoami()["auth"]["accessToken"]["fineGrained"]`, not just whether
the model's license was accepted -- these are two independent gates.**

**Wave 1 execution**: reused `scripts/extract_activations.py`
unchanged (fully generic, no new extraction code needed) for both models,
`--4bit` (8-9B params on a 6GB GPU). Full-corpus extraction took ~1h45m
(Llama-3.1-8B-Instruct) and ~2h08m (Gemma-2-9B-it). New
`scripts/extend_llama_gemma.py` mirrors
`scripts/extend_qwen_smollm.py`'s pattern exactly (same `select_layer_and_calibrate`,
`detector_stats`, adversarial-set reuse), merging results into the
existing `results/dense_direction_cross_model.json` rather than
overwriting Phase 4's Qwen2.5/SmolLM2 entries.

**Real numbers** (full table in RESULTS.md): Llama-3.1-8B-Instruct has the
best TEST-split accuracy/AUROC of any model tried in this project so far
(93.1%, AUROC 0.989) -- even better than Qwen3-8B. Both new models land in
the upper-middle of the XSTest false-positive range (97.3%/89.2%
correctly-not-flagged for Llama/Gemma respectively). On PAIR paraphrase,
adding two more models sharpens the cross-model story from "one anomaly"
to a real spread: SmolLM2 (90.5%) > Llama-3.1-8B (66.7%) > Gemma-2-9B
(47.6%) > Qwen3-8B (42.9%) > Qwen2.5-1.5B (38.1%).

**Extended `scripts/cross_model_significance.py` from 3 to 5 models**
(added `load_in_4bit` support to `model_pair_predictions`, generalizing
what was `small_model_pair_predictions`) rather than leaving the
significance test at 3 models while RESULTS.md now reports 5. Result:
**Cochran's Q = 19.52, df = 4, p = 0.0006** -- still clearly significant
with the two new models included, confirming the cross-model PAIR-
robustness spread is real and not just SmolLM2-vs-everyone-else.

**SAE-feature extension (Wave 2) is explicitly deferred, not attempted
here** -- per the approved plan, it requires a new JumpReLU SAE class for
GemmaScope (different architecture from Qwen-Scope/LlamaScope's TopK) and
a LlamaScope-specific checkpoint loader, then repeating Phase 3's full
causal-ranking/validation methodology per model. Scoped as substantial,
comparable to Phase 3 itself, and deliberately left for a separate pass.

## Phase 6 Wave 2, step 1: corrected a wrong assumption in the approved plan (2026-07-12)

The approved Wave 2 plan assumed LlamaScope reuses `TopKSAE` as-is (same
architecture as Qwen-Scope) and only GemmaScope needs new JumpReLU code.
**Checked before building on that assumption, and it was wrong**: downloaded
and inspected a real LlamaScope checkpoint
(`fnlp/Llama3_1-8B-Base-LXR-8x`, layer 15) -- its `hyperparams.json` reports
`"act_fn": "jumprelu"` with a scalar `"jump_relu_threshold"`, confirmed
across three different LlamaScope variants (`LXR-8x`, `LXR-32x`, `LXA-8x`),
not TopK. The paper's "improved TopK SAEs" title describes the training
recipe, not necessarily the activation function of what's actually
published. **Both LlamaScope and GemmaScope need JumpReLU support** --
there's no "easy one, hard one" split on architecture after all.

Added one shared `src/sae/jumprelu_sae.py::JumpReLUSAE` (same
`W_enc`/`W_dec`/`b_enc`/`b_dec`/`encode`/`decode`/`feature_direction`/`to`
interface as `TopKSAE`, so it's a drop-in for
`src/sae/feature_selection.py`/`causal_ranking.py`/`interventions.py`
without changing any of them -- verified by actually running
`top_k0_by_cosine_similarity` against a live LlamaScope-loaded SAE and a
TRAIN-derived direction on Llama-3.1-8B's cached activations, no errors,
no changes needed to existing pipeline code). `threshold` accepts either a
scalar (LlamaScope) or a `(d_sae,)` tensor (GemmaScope, confirmed to use
per-feature thresholds per its own published paper), so the same class
covers both providers.

`src/sae/llama_scope.py::load_sae` loads real checkpoints from
`fnlp/Llama3_1-8B-Base-LXR-8x`
(`Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors` +
`hyperparams.json`, confirmed via direct inspection, not assumed) --
`.safetensors` format, needed adding `safetensors` to requirements.txt
(already an indirect dependency via `transformers`, now direct since
imported explicitly).

**Where this leaves Wave 2**: the actual remaining work is (1) a
GemmaScope-specific loader (`.npz` format, per-feature thresholds, and a
width/L0-sparsity variant selection decision LlamaScope didn't need), then
(2) for each model, layer selection (cheap, cached activations, no new
code), causal ranking via attribution patching, and causal validation via
suppression -- the compute-heavy, generation-based steps that redo Phase
3's methodology per model. Not started; this session only de-risked the
SAE-loading foundation both models will need.

## Phase 6 Wave 2, step 2: GemmaScope loader, another verified-not-assumed correction (2026-07-12)

`google/gemma-scope-9b-pt-res` lays out checkpoints as
`layer_{n}/width_{w}/average_l0_{l0}/params.npz` (numpy archive, not
safetensors) with a near-empty `hparams.json` (just `sparsity_lambda` --
unlike LlamaScope's, it does not carry the JumpReLU threshold at all).

**Checkpoint choice**: only `width_16k` (~4.5x expansion) and `width_131k`
(~36.6x, confirmed via Gemma-2-9B's real `hidden_size: 3584` from its own
`config.json`) are available for the layers this project needs.
`width_131k` was chosen to match the "~expansion 32" GemmaScope config the
arXiv:2505.23556 paper (Phase 3's own methodology source, see
LITERATURE.md) used -- `width_16k` is a clearly worse match. Within
`width_131k`, `average_l0_51` was chosen to match this project's own
Qwen-Scope precedent (`...W64K-L0_50`) as closely as an available option
allows (candidates were 10/17/30/51/89/163).

**Second real convention mismatch found by checking the actual checkpoint
instead of assuming it matches LlamaScope/Qwen-Scope's layout**:
GemmaScope's own `params.npz` stores `W_enc` as (d_model, d_sae) and
`W_dec` as (d_sae, d_model) -- the *opposite* of this project's
established convention (`W_enc`: (d_sae, d_model), `W_dec`: (d_model,
d_sae), used by `TopKSAE` and `JumpReLUSAE`). `src/sae/gemma_scope.py`
transposes both on load; a test
(`test_download_sae_checkpoint_raw_shapes_are_transposed_from_jumprelu_convention`)
pins the raw (untransposed) shapes down explicitly so a future GemmaScope
release changing this convention doesn't silently break the loader.
`threshold` here is confirmed to be a genuine `(d_sae,)` per-feature array
(unlike LlamaScope's scalar) -- exactly why `JumpReLUSAE.threshold` was
designed to accept either shape.

**Verified end-to-end** the same way as LlamaScope: computed a TRAIN
direction at layer 34 (Gemma-2-9B's Wave 1 best-separation layer) from
already-cached activations, loaded the real GemmaScope SAE for that layer,
ran `top_k0_by_cosine_similarity` successfully with no errors. Both
providers' SAE-loading foundations are now confirmed working; the
remaining Wave 2 work (causal ranking + causal validation per model) is
unchanged from the step-1 entry above.

## Phase 6 Wave 2, steps 3-4: layer selection and checkpoint verification for Llama-3.1-8B-Instruct and gemma-2-9b-it (2026-07-13)

Same method as Qwen3-8B's original layer selection: per-layer separation
score (difference-in-means direction from TRAIN, measured on held-out VAL),
computed directly against the already-cached full-corpus activations
(`src.direction.compute.select_candidate_layers(scores, k=3)`), not
recomputed from scratch or assumed to match Wave 1's single-best-layer
picks.

| model | top-3 layers | scores |
|---|---|---|
| Llama-3.1-8B-Instruct (32 layers) | 27, 26, 21 | 1.860, 1.857, 1.853 |
| gemma-2-9b-it (42 layers) | 34, 35, 33 | 1.806, 1.804, 1.800 |

Both tightly clustered, consistent with every other model in this project.
Layer 27 (Llama) and layer 34 (Gemma) match Wave 1's single-best-layer
picks exactly, so this doesn't reopen or change Wave 1's already-merged
dense-direction results -- it only confirms which two additional layers to
pool for K0 candidate selection.

**Checkpoint existence verified via `HfApi().list_repo_files`, not
assumed**, before hardcoding these layers into `src/sae/registry.py`:
LlamaScope's `fnlp/Llama3_1-8B-Base-LXR-8x` publishes all 32 layers (0-31),
so 27/26/21 are all present. GemmaScope's `google/gemma-scope-9b-pt-res`
was checked at the exact `width_131k/average_l0_51` config already
hardcoded as `gemma_scope.py`'s default (chosen in the step-2 entry
above) -- `layer_34/`, `layer_35/`, `layer_33/` all have both
`params.npz` and `hparams.json` present.

Added `src/sae/registry.py`: a small shared dispatch table
(`SAE_PROVIDERS: model_name -> (load_sae, layers, micro_batch_size)`) so
`scripts/rank_sae_features.py`/`scripts/validate_sae_features.py` no longer hardcode Qwen-Scope's loader and
Qwen3-8B's layers -- both scripts import from this one table instead of
each carrying their own copy, so a future layer-selection update can't
drift between the ranking and validation steps. (The `micro_batch_size`
field's purpose is explained in the OOM entry below.)

## Phase 6 Wave 2: double-BOS artifact in Llama-3.1/Gemma-2's chat templates -- measured, not assumed benign (2026-07-13)

Discovered while smoke-testing the generation path for both new models
before running any real compute (Wave 1 only ever did no-grad
`model.trace` extraction, never `.generate()`, so this path was genuinely
untested). Llama-3.1's and Gemma-2's chat templates each embed a literal
BOS token as text (`<|begin_of_text|>`, `<bos>`); nnsight's own
tokenization (`nnsight/modeling/language.py`'s `_tokenize`, confirmed by
reading the installed package source, not assumed) calls
`self.tokenizer(inputs, ...)` with no `add_special_tokens` override, so
the tokenizer's default (`True`) adds a *second* BOS on top of the
template's own. Confirmed via direct tokenization: Llama's templated
prompt starts `[128000, 128000, 128006, ...]` (`<|begin_of_text|>` twice),
Gemma's starts `[2, 2, 106, ...]` (`<bos>` twice). Qwen's chat templates
never embed a BOS at all, so this never surfaced for any model this
project has used before Wave 1's Llama/Gemma extraction -- meaning Wave
1's already-merged dense-direction results (PR #10) were computed under
this same artifact, not something newly introduced by Wave 2.

**Measured the actual impact rather than assuming it doesn't matter**:
extracted the same 15 harmful Llama-3.1 TRAIN prompts' layer-27
activations both ways (double-BOS via the current default tokenization,
single-BOS via `add_special_tokens=False`). Individual activations shift
measurably (mean cosine similarity ~0.945 between the two versions -- a
real, non-trivial per-prompt difference), but the separation score barely
moves (1.848 double-BOS vs. 1.824 single-BOS, ~1.3% relative difference --
smaller than the layer-to-layer gap already treated as noise between
Llama's own "tied" top-3 layers, 1.860/1.857/1.853). Generation
completions (smoke test, both models) were coherent and correctly
refusal-typical, not degenerate.

**Decision (user's call, given the measured evidence): proceed as-is,
document as an accepted limitation** -- same category as the SAEs'
base-vs-instruct training mismatch above, not something requiring a
Wave-1-invalidating redo. Reasoning: the ~1% shift in the separation score
is smaller than noise this project already tolerates, generation is
unaffected, and a redo would cost several GPU-hours re-running Wave 1's
full extraction plus every downstream result (dense-direction detector,
both cross-model significance tests) for a fix whose own measured effect
says it won't change any conclusion. Contrast with the `do_sample=False`
fix (see above), which *was* worth a redo because it was demonstrably
producing a non-monotonic causal curve -- a real distortion, not a
cosmetic one. This artifact does not currently have a code fix applied;
if a future session touches `format_prompt`/tokenization for these models
for an unrelated reason, this entry is the context for why the double-BOS
was left in place rather than treated as a bug to silently patch.

## Phase 6 Wave 2, step 3: causal ranking results (Llama-3.1-8B-Instruct) (2026-07-13)

`scripts/rank_sae_features.py meta-llama/Llama-3.1-8B-Instruct`,
same parameters as Qwen3-8B's final pass (K0=10, K*=20, N_STEPS=10, 16
harmful TRAIN prompts length-capped at 150 chars, seed=0) -- run once at
this rigor level rather than a smaller first pass, per this project's
standing full-rigor-upfront practice. Ran cleanly on the first attempt, no
OOM (Llama-3.1-8B's 32 layers and 128k-token vocab leave enough headroom
on a 6GB card at this batch size; contrast with gemma-2-9b-it below).

Two clear standout features, then a steep dropoff -- same qualitative
shape as Qwen3-8B's ranking, though even sharper:

| rank | layer | feature | score |
|---|---|---|---|
| 1 | 27 | 13363 | 10.068 |
| 2 | 26 | 7664  | 7.632  |
| 3 | 27 | 31488 | 0.530  |
| 4 | 21 | 5435  | 0.259  |

Full top-20 in `results/sae_causal_ranking_Llama-3.1-8B-Instruct.json`.

## Phase 6 Wave 2, step 3: a real OOM on gemma-2-9b-it's causal ranking, and two wrong fixes before the right one (2026-07-13)

Running the identical script against `google/gemma-2-9b-it` OOM'd
immediately (first candidate, first prompt) with `CUDA out of memory.
Tried to allocate 1.71 GiB ... 0 bytes is free`. Worth recording the two
attempts that *didn't* work before the one that did, since the wrong
diagnosis looked plausible each time:

1. **First guess: allocator fragmentation** (this project's own
   `src.activations.extract` full-corpus extraction already works around
   the same failure mode with a periodic `torch.cuda.empty_cache()`).
   Added the same pattern to `rank_pooled_candidates`'s loop. **Didn't
   help** -- re-ran, OOM'd again on literally the first candidate/prompt,
   before any accumulation across calls could have occurred. This ruled
   out fragmentation-from-repeated-cycles as the cause.
2. **Second guess: the batched integrated-gradients backward pass itself
   is too large for a single forward+backward** (Gemma-2-9B's 42 layers
   and larger FFN intermediate size vs. Llama-3.1-8B's 32). Added
   `micro_batch_size` support to `feature_ig_attribution`/
   `_ig_chunk`/`rank_pooled_candidates`: splits the N=10 interpolation
   steps into smaller chunks, each its own forward+backward pass, with
   per-chunk gradients averaged together -- mathematically identical to
   full batching (verified with a new test,
   `test_feature_ig_attribution_micro_batching_matches_full_batch`,
   rel=1e-2 tolerance for floating-point kernel-path differences, not a
   real numerical divergence). Set `micro_batch_size=2` for Gemma in the
   registry. **Also didn't help** -- OOM'd again, and critically, the
   failed allocation was the *same* 1.71 GiB both times regardless of
   batch size (10 vs. 2) -- a strong sign the OOM wasn't scaling with the
   IG batch dimension at all, meaning the real cause had to be something
   batch-independent.
3. **Actual cause, found by re-examining what's resident on GPU
   throughout the whole ranking pass, not just during one call**:
   `scripts/rank_sae_features.py`'s `main()` explicitly moved every candidate layer's
   *entire* SAE (`saes[l].to(device="cuda:0", dtype=torch.float16)`) onto
   GPU before ranking started. For GemmaScope's `width_131k` SAEs
   (131072 features vs. LlamaScope's 32768), each layer's W_enc + W_dec
   is ~1.9GB fp16; three resident simultaneously (layers 34/35/33) is
   ~5.6GB -- on top of the already-tight 4-bit model weights, this alone
   consumes nearly the whole 6GB card before the ranking loop's own
   backward pass needs anything. This GPU transfer was **never actually
   necessary**: `feature_ig_attribution`/`_ig_chunk` only ever index a
   single row/column per candidate
   (`sae.W_enc[feature_idx]`, `sae.feature_direction(feature_idx)`) and
   already move *that* tiny slice to the model's device themselves --
   correct and sufficient whether the parent SAE tensor lives on GPU or
   CPU. Removed the whole-SAE `.to("cuda:0")` step from `scripts/rank_sae_features.py`
   entirely; SAEs now stay on CPU (fp32) for the whole ranking pass, for
   every model, not just Gemma. Ran cleanly afterward -- confirmed via the
   first candidate completing, then the full 30-candidate pass finishing
   with no errors.

Both the fragmentation-hygiene fix (step 1, harmless, kept) and the
micro-batching support (step 2, harmless, also kept as an extra safety
margin for Gemma) remain in the code even though neither was the actual
fix -- the first is reasonable general hygiene matching existing project
precedent, and the second is a real, tested, useful capability
(mathematically-verified-equivalent smaller-batch IG) that may matter for
a future even-larger model. Neither should be read as "the fix"; the
docstring in `src/sae/causal_ranking.py` and this entry are the record of
what actually mattered.

**Lesson**: an OOM error's stack trace points at *where* memory ran out,
not *why* -- the actual cause here was a completely different, unrelated
line (a one-time setup step, not the loop that crashed). Chasing the
crash site's own batch dimension first was a reasonable first guess but
the wrong one twice in a row; what broke the pattern was stepping back to
ask what else was resident on GPU throughout the whole run, not just
during the failing call.

## Phase 6 Wave 2, step 3: causal ranking results (gemma-2-9b-it) (2026-07-13)

Same parameters as Llama-3.1-8B-Instruct above (K0=10, K*=20, N_STEPS=10,
micro_batch_size=2, 16 harmful TRAIN prompts, seed=0), after the OOM fix.

A more gradual decline than either Qwen3-8B's or Llama-3.1-8B's sharp 1-2
standout features -- no single dominant feature, scores decay smoothly:

| rank | layer | feature | score |
|---|---|---|---|
| 1 | 35 | 52410 | 0.801 |
| 2 | 35 | 80362 | 0.581 |
| 3 | 34 | 38366 | 0.526 |
| 4 | 33 | 84809 | 0.423 |
| 5 | 34 | 8149  | 0.412 |

Full top-20 in `results/sae_causal_ranking_gemma-2-9b-it.json`. This
smoother ranking-score shape foreshadows the causal validation result
below (a more gradual, modest refusal-rate decline vs. Llama's sharp
top-1-alone-does-most-of-it effect).

## Phase 6 Wave 2, step 4: causal validation results, both models, and a three-way cross-model comparison (2026-07-13)

`scripts/validate_sae_features.py`, same parameters as
Qwen3-8B's final pass (N=50 held-out VAL harmful prompts, 6 conditions,
40 tokens, greedy decoding, real `refusal_classifier`) for both models.
Zero degenerate completions across all 300 generations, both models.

| condition | Llama-3.1-8B refusal | gemma-2-9b-it refusal |
|---|---|---|
| baseline | **98.0% [89.5%, 99.65%]** | 96.0% [86.5%, 98.9%] |
| top-1 | 10.0% [4.4%, 21.4%] | 94.0% [83.8%, 97.9%] |
| top-5 | 4.0% [1.1%, 13.5%] | 92.0% [81.2%, 96.9%] |
| top-10 | 2.0% [0.4%, 10.5%] | 84.0% [71.5%, 91.7%] |
| top-15 | 0.0% [0.0%, 7.1%] | 82.0% [69.2%, 90.2%] |
| top-20 | 2.0% [0.4%, 10.5%] | 82.0% [69.2%, 90.2%] |

**Llama-3.1-8B baseline corrected 2026-07-23** (was 86.0% [73.8%, 93.1%])
-- a real `is_refusal` bug (Llama's curly apostrophes silently missed by
the ASCII marker list) undercounted refusals in this specific condition;
see the dedicated bug-fix entry further down. Only baseline was affected
-- top1 through top20 had zero additional matches under the fix. This
*strengthens* the single-feature finding below (98%->10% is an even
sharper drop than 86%->10%), doesn't undermine it.

**A genuine, striking three-way cross-model difference** in how
concentrated the causal effect is:

- **Llama-3.1-8B-Instruct**: the single top-ranked feature alone (layer
  27/feature 13363) drops refusal from 98% to 10% -- nearly the *entire*
  effect from one feature. Unlike Qwen3-8B, where "suppressing the single
  top feature alone still doesn't reproduce the effect" (see above), here
  it almost does.
- **Qwen3-8B**: effect distributed across the pooled feature set, top-1
  alone does essentially nothing (84% vs. 82% baseline), bottoms out at
  top-15 (18%).
- **gemma-2-9b-it**: a real, monotonic (non-increasing) decline (96% ->
  82%), but far more modest than either other model -- suppressing all 20
  ranked features removes only 14 percentage points of refusal, vs.
  Llama's 88-point drop and Qwen3's 66-point drop at comparable
  conditions. Consistent with the smoother, no-standout-feature ranking
  shape found above.

This is flagged as a real, unexplained finding, same standard as this
project's other cross-model differences (SmolLM2's PAIR-paraphrase
robustness, the non-monotonic perplexity-backbone pattern) -- not
resolved here, not force-fit into a story. Baseline-vs-top20 CIs overlap
narrowly for Gemma (86.5%-98.9% vs. 69.2%-90.2%), so whether this specific
curve is formally statistically significant (vs. Llama/Qwen3's clearly
non-overlapping CIs) hasn't been tested with a proper paired test
(McNemar, per this project's own established discipline for paired
data -- see the Phase 4 "comparing independent CIs on paired predictions"
lesson) -- worth doing if this comparison is written up as a headline
result rather than a descriptive observation.

**Correction, 2026-08-12**: the "top-15 (18%)" / "Qwen3's 66-point drop"
figures two paragraphs up were already stale when this entry was written --
they're the pre-fix, stochastic-decoding numbers from the 2026-07-10
"Tightening the results" table, not the deterministic (`do_sample=False`)
greedy-decoding re-run from 2026-07-11 that explicitly superseded them
(baseline 82%, top-15 **24%**, not 18%). Caught while formally
significance-testing this curve (see "Formal significance for Qwen3-8B's
and Llama-3.1-8B's suppression curves too" below), which reproduced 12/50
(24%) directly from the saved completions. Correct comparison: Llama's
88-point drop (98%->10%) vs. Qwen3's **58-point** drop (82%->24%), not 66.
Left the original text above as-is rather than silently edited, per this
project's own standing practice; `reports/RESULTS.md`'s copy of this same
figure is fixed in place since that document is the live-maintained report,
not a dated historical entry.

## Phase 6 Wave 3: SAE-feature detector extension to Llama-3.1-8B-Instruct and gemma-2-9b-it (2026-07-22)

The last piece needed to complete the 3-model SAE-feature comparison:
reframing Wave 2's causally-validated feature sets as prompt classifiers
(mirroring Phase 4's Qwen3-8B work) and running the same head-to-head
evaluation against baselines. Branch `sae-detector-cross-model` off
`master` (`a7bb983`).

**K=15 reused for all three models, not re-tuned per model**: each
model's own causal-validation curve (Wave 2 entry above) independently
reaches its minimum refusal rate at top-15 -- Qwen3-8B 24% (Phase 4's
original choice), Llama-3.1-8B 0% (strict minimum), gemma-2-9b-it 82%
(tied with top-20). Empirically justified by data already collected, not
an arbitrary carry-over.

**New infrastructure, mirroring Wave 2's registry-based generalization**:
- `scripts/extend_sae_adversarial.py` (new) --
  Wave 1's dense-direction extension (`scripts/extend_llama_gemma.py`) computed adversarial
  activations on the fly and discarded them (only needed one layer's
  projection); the SAE-feature detector needs 3 layers per model, so
  this time the cache is saved to disk (`results/activations/
  {model}_adversarial.pt`), mirroring Qwen3-8B's own cached file exactly.
  Reuses the existing `adversarial_paraphrase_manifest.json`, doesn't
  rebuild it -- same real JailbreakBench artifacts as every other model.
- `src/detectors/dense_direction_detector.py::resolve_layer_for_model` --
  a small per-model branch, not a uniform registry lookup, since the two
  source files have genuinely different schemas/provenance: Qwen3-8B's
  dense-direction layer comes from `dense_direction_ablation_Qwen3-8B.json`
  (a frozen, TEST-selected legacy value -- see the earlier "mild leakage
  pattern" entry, deliberately not retroactively changed), while
  Llama-3.1-8B/gemma-2-9b-it's come from `dense_direction_cross_model.json`
  (VAL-selected, the corrected discipline used for every model added
  after that fix).
- `scripts/calibrate_thresholds.py`/
  `scripts/compare_detectors.py` generalized with a `model` CLI
  arg (mirroring `scripts/rank_sae_features.py`/`05`'s pattern from Wave 2): SAE loading via
  `src/sae/registry.py` (already built), keyword/perplexity thresholds
  reused from Qwen3-8B's own calibration rather than recomputed (both are
  prompt-text-only and VAL-split membership is prompt-manifest-based,
  identical across every model's cache -- confirmed via
  `assert_caches_consistent`), output paths parametrized per model.
- `src/detectors/sae_feature_detector.py`'s `saes` type hint updated from
  `dict[int, TopKSAE]` to `dict[int, TopKSAE | JumpReLUSAE]` -- cosmetic,
  not a functional bug (the module only ever calls `.encode()`, which both
  classes implement identically), but was stale now that two of three
  models load `JumpReLUSAE` checkpoints.

**Results** (full numbers in RESULTS.md; `results/detector_head_to_head_
{Llama-3.1-8B-Instruct,gemma-2-9b-it}.json`), both models' dense-direction
and SAE-feature AUROC land in the same high-0.9x range as Qwen3-8B's
(0.983/0.975) -- Llama 0.989/0.978, Gemma 0.984/0.966 -- confirming
nothing broke in the generalization before looking at the more granular
comparisons:

| | TEST AUROC (dense/SAE) | DeLong p | PAIR detect (dense/SAE) | pooled adversarial McNemar p |
|---|---|---|---|---|
| Qwen3-8B | 0.983 / 0.975 | 0.068 (n.s.) | 42.9% / 33.3% | 0.5 (n.s.) |
| Llama-3.1-8B | 0.989 / 0.978 | **0.024** | 66.7% / **80.9%** | 0.25 (n.s.) |
| gemma-2-9b-it | 0.984 / 0.966 | **0.0063** | **47.6%** / 23.8% | **0.0156** |

**A genuinely different story per model, not a clean pattern**:
- **Qwen3-8B**: dense-direction and SAE-feature are statistically
  indistinguishable everywhere tested (this project's original finding).
- **Llama-3.1-8B**: dense-direction is *significantly better* on
  TEST-overall (p=0.024) -- but on PAIR specifically, SAE-feature
  numerically *beats* dense-direction (80.9% vs 66.7%, though the
  McNemar test on only 21 paired items doesn't reach significance,
  p=0.25). This is the one case in the whole project where SAE-feature
  outperforms dense-direction on paraphrase robustness, the direction
  arXiv:2505.23556 originally claimed -- still not formally significant,
  but the first time this project's own numbers have pointed that way at
  all.
- **gemma-2-9b-it**: dense-direction significantly better on TEST-overall
  (p=0.0063) *and* on the pooled adversarial set (p=0.0156, all 7
  discordant pairs favor dense) -- the strongest, most one-sided result
  for dense-direction of any model tested.

No attempt made to explain *why* this varies by model -- flagged
honestly as a real, unresolved cross-model difference, consistent with
this project's standing practice (SmolLM2's PAIR robustness, the
perplexity-backbone non-monotonicity, Wave 2's causal-effect-
concentration spread above). This completes the project's 3-model
SAE-feature comparison; Phase 6 (cross-model generalization) is now done.

## Closing a Wave 2 gap: is gemma-2-9b-it's suppression curve actually significant? (2026-07-22)

Wave 2's write-up reported gemma-2-9b-it's causal-validation curve (96%
baseline -> 82% at top-15/top-20) descriptively but flagged that, unlike
Qwen3-8B (non-overlapping Wilson CIs) or Llama-3.1-8B (an unambiguous 0%
floor), whether this specific curve was formally significant hadn't been
tested. `scripts/gemma_suppression_significance.py` closes this
with no new GPU compute -- `scripts/validate_sae_features.py`'s validation run already saved
every completion per condition, so this just reclassifies each of the 50
VAL prompts with `is_refusal` and runs McNemar's exact test (paired,
baseline vs. each condition, on the same 50 prompts) rather than eyeballing
Wilson CI overlap.

| condition | refusal (of 50) | discordant vs baseline | p-value |
|---|---|---|---|
| top-1 | 47 | 1 | 1.0 |
| top-5 | 46 | 2 | 0.5 |
| top-10 | 42 | 6 | **0.0312** |
| top-15 | 41 | 7 | **0.0156** |
| top-20 | 41 | 7 | **0.0156** |

**The effect is real and statistically significant from top-10 onward** --
resolves the open question in Gemma's favor: the modest-looking 14-point
decline is a genuine causal effect, not noise, even though it's far
smaller than Llama's or Qwen3's. Every discordant pair favors suppression
reducing refusal (baseline-only, zero condition-only) at every threshold,
consistent with a real monotonic effect rather than a symmetric coin-flip
fluctuation. Doesn't change the three-way cross-model story (Gemma's
effect is still the smallest of the three), just upgrades it from
"descriptive, significance untested" to "descriptive, and now formally
confirmed."

## Cross-model direction transfer: does Qwen3-8B's direction do anything on Llama-3.1-8B, and vice versa? (2026-07-23)

Closes this project's longest-standing gap: every prior phase fit and
tested a refusal direction only within the model it came from. Scoped to
Qwen3-8B <-> Llama-3.1-8B-Instruct (both d_model=4096, so a raw direction
vector is dimensionally injectable into either model) -- gemma-2-9b-it
(d_model=3584) excluded rather than attempting a learned cross-dimension
mapping, which would confound "does it transfer" with "is the mapping any
good" (user's explicit choice when offered both options). Necessity
(ablation) only, not sufficiency (addition) -- addition needs a calibrated
alpha for a foreign direction on a different target's residual-stream
scale, real additional scope, deferred. SAE-feature transfer also out of
scope -- an SAE's feature basis is specific to that trained autoencoder,
not a well-posed transfer question the way a single vector is.
`scripts/transfer_direction.py`, branch
`cross-model-direction-transfer`.

**Test 1 (separation score, cache-only, no generation)**: broadcast each
model's own direction across the *other* model's layers, score against
its VAL activations. Own-direction controls reproduce known values exactly
(Qwen3-8B 1.7831 vs. known 1.783 at layer 23; Llama-3.1-8B 1.8597 vs.
known 1.860 at layer 27) -- confirms the harness before trusting the
foreign numbers. **Both foreign scores are negative**, not just weak:
Llama's direction on Qwen3-8B's activations scores -0.6454; Qwen's
direction on Llama's activations scores -0.7218. Not noise near zero --
a real anti-correlated signal in both directions.

**Test 2 (causal ablation, the definitive test)**: N=50 harmful VAL
prompts, same prompt text reused for both models (`assert_caches_consistent`
confirms identical corpus ordering), three conditions each (baseline,
own-direction ablation, foreign-direction ablation), `do_sample=False`,
`max_new_tokens=40` -- matching every prior causal-validation script's
convention. Three paired McNemar tests per model
(baseline-vs-foreign, baseline-vs-own, own-vs-foreign) for a clear verdict.

| | baseline | own-ablation | foreign-ablation | own vs baseline p | own vs foreign p |
|---|---|---|---|---|---|
| Qwen3-8B (foreign = Llama's direction) | 84% | 8% | 84% | **0.0** | **0.0** |
| Llama-3.1-8B (foreign = Qwen's direction) | 92% | 88% | 92% | 0.5 | 0.5 |

**CORRECTED 2026-07-23, same day**: the Llama-3.1-8B row above (and every
Llama-generated refusal-rate number in this entry) was originally reported
as 80%/86%/80% -- **wrong**, due to a real `is_refusal` bug found and
fixed the same day (see the dedicated entry below: Llama-3.1-8B generates
curly apostrophes, e.g. "can't" with U+2019, which the classifier's ASCII
marker list silently failed to match). Recomputed directly from the
already-saved completions in `results/cross_model_direction_transfer.json`
(no new GPU generation needed) using the fixed classifier -- the table
above and the two paragraphs below reflect the corrected numbers.

**Qwen3-8B: a clean, unambiguous no-transfer result (unaffected by the
bug -- Qwen3-8B uses ASCII apostrophes).** Own-direction ablation crashes
refusal (84%->8%, matching this project's established Phase 1 result
almost exactly -- a real, working control). Llama's foreign direction
does *nothing at all* -- refusal identical to baseline to the percentage
point, p=1.0 vs. baseline. Combined with Test 1's negative separation
score, this is the cleanest possible negative result: the intervention
mechanism clearly *can* produce a dramatic effect at this scale (proven
by the own-direction control), and the foreign direction produces none
of it.

**Llama-3.1-8B, corrected: a real but weak, statistically-underpowered
effect -- not the "own ablation doesn't work at all" story originally
reported.** With the bug fixed, own-direction ablation on Llama shows a
genuine, correctly-signed decrease (92%->88%, 4 points), not the
increase the buggy numbers showed (80%->86%). Still nowhere near
significant at n=50 (p=0.5, unchanged from the pre-fix p=0.4531 --
the *direction* of the effect flipped to the expected sign, but the
*significance verdict* didn't change either way). This has never been
tested before in this project in either version: Wave 1 only ever used
Llama's dense direction as a *classifier* (AUROC 0.989, Wave 3), never a
causal ablation intervention -- Wave 2's causal ablation work on Llama
used SAE features instead (which worked dramatically: 86%->10% from a
single top feature). **Manually inspected completions to rule out a
second bug before trusting the correction** -- coherent, on-topic
refusals throughout, zero degenerate completions.

Because the "own" effect, while correctly-signed now, is still too weak
to distinguish from noise at n=50, the "own vs. foreign, not significant"
result still **cannot be read as evidence of no transfer** the way
Qwen3-8B's can -- it's underpowered either way. **Revised honest summary**
(the original "detection accuracy and causal necessity are decoupled"
framing was itself an artifact of the bug and is retracted): one clean
negative-transfer result (Llama's direction has zero causal effect on
Qwen3-8B), one genuinely inconclusive result for a more mundane reason
than originally claimed -- Llama's own dense-direction ablation has a
real, correctly-signed but small effect that a larger sample would be
needed to resolve, so nothing definitive can be said about whether Qwen's
direction transfers to it. Not smoothed into a single "directions don't
transfer" headline -- the two models' results still say different things,
just less dramatically different than first reported.

## Independent replication of Llama's own-direction ablation at n=75 (2026-07-23)

Picked up as a gap explicitly flagged above: "Llama's own-direction
causal ablation effect is itself new and unreplicated at a larger N."
`scripts/replicate_llama_ablation.py` draws a fresh, independent
sample (not an extension of the original 50 prompts, new seed) and
re-runs only the two conditions this needed (baseline, own-ablation --
not the full 3-condition cross-model transfer test, which isn't in
question here).

**Sample size note**: originally attempted at n=150 (3x). Killed after
running far slower than expected -- plain baseline generation (no hooks
at all) measured at roughly 2 tokens/sec on this hardware, meaning even
the cheap phase alone would have taken the better part of an hour, with
the heavier per-layer ablation phase still to come. Not a good use of
wall-clock time for a bounded gap-fill, so re-run at n=75 (1.5x the
original, still a real power increase, a fraction of the runtime).

**Result: does not confirm a real small effect.** Baseline 96.0% [88.9%,
98.6%] vs. own-ablation 94.7% [87.1%, 97.9%], only 3 discordant pairs out
of 75, McNemar's exact test p=1.0. This is *weaker* than the original
n=50's 92%->88%, not a sharper measurement of the same real effect. Read
honestly, this points toward the original observation being sample noise
rather than a real-but-small causal effect -- the correct conclusion is
"Llama's dense direction's causal necessity for its own refusal remains
unresolved by this project's data," not "confirmed real but small." See
RESULTS.md's cross-model-direction-transfer section for the full writeup.

## Sufficiency (activation addition) extended to 7-9B scale (2026-07-24)

The other half of the same "only tested at small scale" gap: necessity
(ablation) had reached 7-9B models via Wave 2 and the cross-model-transfer
test, but sufficiency (activation addition) was still only ever measured
on Phase 1's two small models. `scripts/sufficiency_at_scale.py`
extends it to Qwen3-8B and Llama-3.1-8B-Instruct.

**Split-discipline choice**: Phase 1's original methodology used a
dedicated 3-way train/calib/val split; this project's later full-corpus
work only has train/val/test. Rather than inventing a 4th split (which
this project has deliberately avoided everywhere else), alpha-sweep
calibration runs on a VAL-split harmless sample (n=12, matching Phase 1's
calib-split size) and the final causal-validation generation test runs on
a disjoint TEST-split harmless sample (n=50, this project's standing
validation-sample convention) -- mirrors how layer selection and
threshold calibration already both live on VAL elsewhere in this project,
TEST reserved for final reporting only.

**Result: real for both models, but not a clean scale-up.** Qwen3-8B
replicates Phase 1's clean pattern almost exactly (baseline 6.0% ->
addition 70.0%, alpha=1.0, non-overlapping CIs, stayed non-degenerate
through alpha=2.0 in the sweep). Llama-3.1-8B-Instruct is real but far
weaker and messier: its alpha-sweep never reached the 80% calibration
target at any alpha (peaked 67% refusal at alpha=1.5, but 33% of those
completions were degenerate -- over the 10% cutoff, so rejected as
non-viable), started fully degenerating from alpha=2.0, and calibration
fell back to the highest-refusal *viable* alpha (1.0, 58% on the calib
set) -- the fallback branch already existed in the calibration logic
(shared with scripts/calibrate_alpha.py) but had never actually been exercised by any
model until Llama here. Final validated effect: 10.0% -> 34.0%, real
(barely non-overlapping CIs) but much smaller than Qwen3-8B's.

**Not written up as "sufficiency confirmed at scale."** Necessity
generalizes more robustly across these two models than sufficiency does
-- the same qualitative pattern Phase 1 already found between Qwen2.5 and
SmolLM2 (addition is architecture-dependent, ablation isn't), now showing
up again at 8-9B scale with a different model pair. Reported as an open,
unexplained architecture difference, not forced into a single "it works
at scale" headline. See RESULTS.md's dedicated section for full numbers.

## True harmful-compliance spot-check extended to cross-model-transfer completions (2026-07-24)

Extends the scripts/ablate_qwen3_direction.py/20 methodology (direct Claude-labeling, not an
automated classifier -- both candidate local judges already failed this
task, see above) to a second dataset: the four ablation conditions'
completions already sitting in `results/cross_model_direction_transfer.json`
from scripts/transfer_direction.py, no new generation needed. Read all non-refuse
completions for the two small conditions (Llama own/foreign-ablation, 6
and 4 completions), sampled 15 of 46 for Qwen3-8B's own-ablation
(the only condition large enough that reading all of it wasn't a good use
of time), read all 8 of Qwen3-8B's foreign-ablation.

**Confirms the scripts/ablate_qwen3_direction.py pattern generalizes**: Qwen3-8B's non-refuse
completions are overwhelmingly moralizing, not real compliance (2/15
sampled were true compliance, both clean and unhedged). Foreign-ablation
(no real causal effect) never produced true compliance in this sample.
**New observation**: Llama-3.1-8B's rare near-compliance completions are
qualitatively different -- hedged/partial ("I can provide a general
outline... but I must emphasize this is for educational purposes only")
rather than Qwen's clean compliance. Small samples (n=6, n=4), reported
as an honest observation, not a statistically established claim -- see
RESULTS.md for the full write-up and the appropriate hedging.

## Found and fixed a real `is_refusal` bug: curly apostrophes, Llama-3.1-8B-specific (2026-07-23)

Discovered while building ground truth for the moralize-vs-comply
classifier (see the entry below) -- sampling non-refuse completions from
Llama's suppression/transfer results and labeling them by hand turned up
4 of 53 that were plainly genuine refusals ("I can't fulfill that
request...", "I can't answer that...") that `is_refusal` had called
non-refuse. Traced to the exact byte: inspected the raw JSON directly
(not assumed) and confirmed Llama-3.1-8B generates a curly apostrophe
(U+2019 RIGHT SINGLE QUOTATION MARK, "'") in contractions like "can't",
not the ASCII apostrophe (U+0027, "'") `_REFUSAL_MARKERS` was written
against. A plain substring match silently fails on this, undercounting
refusals for any model that does this.

**Confirmed Llama-3.1-8B-specific, not a project-wide issue**: recomputed
`is_refusal` (old vs. fixed) across every saved completions file in the
project (`sae_suppression_validation_{Qwen3-8B,Llama-3.1-8B-Instruct,
gemma-2-9b-it}.json`, `dense_direction_ablation_Qwen3-8B.json`,
`cross_model_direction_transfer.json`) -- zero differences anywhere
except Llama-3.1-8B-generated completions. Qwen3-8B and gemma-2-9b-it
both consistently use ASCII apostrophes; only Llama-3.1-8B's tokenizer/
generation produces the curly variant.

**Fix**: `is_refusal` now normalizes curly apostrophe variants (U+2019,
U+2018, U+02BC) to ASCII before matching (`src/direction/
refusal_classifier.py`), covered by a new regression test
(`test_is_refusal_handles_curly_apostrophes`).

**Impact assessed and corrected using already-saved completions -- no new
GPU generation needed for either correction**:
- `results/sae_suppression_validation_Llama-3.1-8B-Instruct.json`'s
  **baseline** condition: 43/50 (86%) -> 49/50 (98%) refusal. Every other
  condition (top1 through top20) was unaffected -- 0 additional matches.
  This *strengthens* Wave 2's headline finding rather than undermining it:
  the drop from baseline to top-1 alone becomes 98%->10%, an even sharper
  single-feature effect than the 86%->10% originally reported. See
  RESULTS.md for the corrected table.
- `results/cross_model_direction_transfer.json`'s Llama-side conditions:
  corrected above in the "Cross-model direction transfer" entry --
  materially changes that entry's narrative (a real, correctly-signed but
  statistically-underpowered own-direction effect, not "doesn't work at
  all"), not just its numbers.

**Not affected**: Wave 3's SAE-feature/dense-direction *detector* results
(AUROC, McNemar tests on adversarial prompts) never use `is_refusal` --
those score raw activations directly, not generated text. Gemma's
Wave 2 significance test (`scripts/gemma_suppression_significance.py`) is unaffected -- confirmed via
the same recomputation, zero differences in that file.

**Lesson**: this is the second time a Unicode/encoding mismatch has
silently corrupted a text-matching step in this project (see the
double-BOS tokenization entry) -- worth treating any string-substring
classifier as a candidate for this failure mode by default, not just
when something looks visibly wrong, since a silent undercount doesn't
announce itself the way a crash does. Caught here only because building
a *different* classifier's ground truth happened to involve reading the
raw text closely.

## Moralize-vs-comply classifier: an automated judge didn't work, direct labeling did (2026-07-23)

Closes the longest-standing item on `refusal_rate`'s own known-limitations
list: `is_refusal` detects refusal *phrasing* only, conflating "moralize"
(lectures about why a request is wrong, zero harmful content, safe) with
"comply" (genuinely provides the harmful content, unsafe) under one
"non_refuse" bucket. Originally surfaced by `scripts/ablate_qwen3_direction.py`'s head-to-head
(6% dense-ablation refusal vs. 24% SAE-suppression refusal looked like an
18-point safety gap; manual inspection at the time found 47/50 of dense
ablation's "non-refusals" were moralizing, so the true gap was flagged as
"likely much smaller" but never measured). Branch
`moralize-comply-classifier`.

**Ground-truth expansion, before trusting anything**: the existing
45-row human-labeled worksheet (`results/classifier_spotcheck_worksheet.csv`,
from a past session's Phase 3 spot-check) had a real confound -- 100% of
its `comply` labels came from Phase 1's smaller models, zero from any
Qwen3-8B intervention experiment, and zero coverage of Llama-3.1-8B/
gemma-2-9b-it or the cross-model-transfer results. `scripts/expand_worksheet.py`
sampled 53 more non-refuse completions
specifically from the previously-uncovered files, Claude-labeled blind
(same protocol as the original), saved separately as
`results/classifier_spotcheck_worksheet_expansion.csv`. Real side-finding
while reading these closely: 4 of 53 were genuine refusals `is_refusal`
missed due to a curly-apostrophe bug -- see the dedicated entry above,
found and fixed the same session, before continuing this work.

**Building the automated classifier -- two judge models tried, both
failed validation**:

1. `microsoft/Phi-4-mini-instruct`, attempt 1, loaded via `src.activations.
   extract.load_model` (nnsight): independent of Qwen/Llama/Gemma (same
   "don't use a target model as its own judge" logic already applied to
   the perplexity baseline), instruction-tuned, already downloaded in this
   project. Hit two successive real `transformers`-version incompatibilities
   in its remote code (nnsight always loads with `trust_remote_code=True`,
   hardcoded, not overridable via `load_model`'s kwargs): first
   `ImportError: cannot import name 'LossKwargs'` (renamed to
   `TransformersKwargs` in the installed `transformers==5.13.0` -- patched
   with a compatibility alias, confirmed safe since it's a typing-only
   marker class), then `AttributeError: 'list' object has no attribute
   'keys'` in `transformers.modeling_utils.get_expanded_tied_weights_keys`
   -- a structural tied-weights API change, not patchable without deeper
   surgery into `transformers` internals or downgrading the package
   (unsafe: other code already depends on v5.x-specific APIs like
   `BitsAndBytesConfig` replacing the removed `load_in_4bit` shorthand).
   Two successive incompatibilities from one model's stale remote code =
   the "well is deeper than expected" signal to switch approach rather
   than keep patching.
2. `HuggingFaceTB/SmolLM2-1.7B-Instruct`, proven compatible with the
   nnsight stack (Phase 1 ran on it with no issues), independent,
   instruction-tuned. Loaded and generated without error, but validation
   against the 81-row combined ground-truth set (28 original + 53
   expansion) landed at **50% overall, and critically 0% on both `comply`
   and `partial` in the harmful-prompt subset** (`results/
   moralize_comply_classifier_validation.json`) -- the verdict
   distribution (71/98 "moralize", 23/98 "refuse", 4/98 "comply", 0/98
   "partial") showed it defaulting to whichever category looked safest
   rather than discriminating on actual content. Traced two real
   execution bugs along the way first (a `min_new_tokens=1`/
   `max_new_tokens=8` mismatch that let the `[-N:]` output slice silently
   include prompt-tail tokens when generation stopped early -- fixed by
   matching `min_new_tokens=max_new_tokens`, mirroring every other
   generation call in this project; and a prompt redesign adding a
   "Category:" prefill after finding the model tended to echo the prompt
   before answering) -- but fixing both didn't fix the underlying
   accuracy. This is a real capability ceiling at 1.7B for this specific
   nuanced task, not a prompt-engineering problem.
3. `microsoft/Phi-4-mini-instruct`, attempt 2 -- loaded via plain
   `transformers.AutoModelForCausalLM.from_pretrained` instead of nnsight
   (mirroring `src.baselines.perplexity_filter.load_perplexity_model`'s
   existing precedent for an auxiliary/scoring model that needs no
   activation access), avoiding `trust_remote_code` entirely and using
   the library's own maintained native Phi3 implementation. This resolved
   both prior incompatibilities cleanly. But two more prompt variants
   (the original instruction-style prompt, then a few-shot version with
   reordered categories and a worked "comply" example) each collapsed to
   a *different* single default category regardless of content (refuse-
   heavy, then moralize-heavy) -- most likely the model's own safety
   alignment overriding the meta-level labeling task ("this text
   discusses something harmful-adjacent" triggering its own refusal
   reflex even when asked to classify, not generate). Three distinct
   collapse patterns across two models and four prompt variants is a
   well-established finding, not a fluke worth one more attempt.

**Decision: pivot to direct (Claude) labeling for the actual application**,
rather than keep iterating on locally-available models that have now
failed three genuine, principled attempts. This isn't a downgrade --
it's the exact methodology this project already validated at 97.8%
agreement (Phase 3's original spot-check). The classifier module
(`src/direction/moralize_comply_classifier.py`) is kept in the codebase,
tested (parsing logic, judge-loading), and documented honestly as
"validated, found unreliable with locally-available models" rather than
deleted or silently abandoned -- useful infrastructure and a real,
reusable finding if a stronger local model becomes available later.

**Resolving scripts/ablate_qwen3_direction.py's original question** (`scripts/rescore_compliance.py`):
every non-refuse completion in both
files (47 for dense-direction ablation, 38 for SAE-suppression top-15 --
not sampled, both are already this project's own modest VAL sets) read
and labeled directly, a built-in consistency check confirming every
non-refuse completion got a label before trusting the tally.

| | refuse (is_refusal) | moralize | partial | **comply (true harm)** |
|---|---|---|---|---|
| dense-direction ablation | 6.0% | 94.0% | 0.0% | **0.0%** |
| SAE-suppression (top-15) | 24.0% | 74.0% | 2.0% | **0.0%** |

**The headline "6% vs. 24%" gap was entirely a refusal-phrasing artifact,
not a real safety difference.** True harmful-compliance rate is 0% for
both conditions (one ambiguous "partial" case in the SAE condition: a
self-harm blog post whose title matched the harmful request literally,
though truncated before the body showed real content either way -- called
partial rather than forced into moralize or comply). This is a stronger,
cleaner resolution than "likely much smaller" -- confirms the original
2026-07-11 head-to-head's core finding (dense ablation is a blunter,
more disruptive intervention that suppresses refusal *phrasing* more
without actually producing more harmful *content*) with a real measured
number instead of a plausible-sounding gap.

## Investigating the Llama dense-direction-vs-SAE-feature causal gap (2026-07-24)

Picks up the session's explicitly-scoped next task: why is Llama-3.1-8B's
dense-direction approach weak on both necessity (unreplicated at n=75,
see above) and sufficiency (10%->34%, degenerating from alpha>=2.0, see
above), while its SAE-feature approach ablates refusal dramatically
(98%->10% from a single feature, Wave 2)? Chosen specifically because it
looked answerable by re-analyzing data already on disk rather than
requiring new GPU generation -- confirmed true: `scripts/analyze_llama_causal_gap.py`
needed only cached activations (`results/activations/*.pt`) and
already-downloaded SAE checkpoints (via `src.sae.registry`), run once on
CPU, no model loaded.

**Two candidate mechanical explanations, tested in order, one ruled out
rather than assumed**:

1. **Raw direction magnitude.** Llama's raw diff-in-means direction norm
   (19.1 at layer 27) is the smallest of the four models with necessity
   data (75.3-279.6 for the others) -- but raw norm alone isn't the right
   unit, since residual-stream scale varies by model and layer. Computed
   each model's ambient activation norm directly from its own cached
   activation tensor at the relevant layer and took the ratio. Result:
   Llama's ratio (0.702) is the *largest* of the four, not the smallest
   (0.332-0.522 for the others) -- the opposite of what a magnitude-
   dilution story predicts. **Ruled out, reported as such, not
   smoothed over.**
2. **Dense-direction/SAE-feature axis misalignment.** Loaded the real
   LlamaScope decoder vector for Llama's own top causal feature (layer
   27/13363) and the real Qwen-Scope decoder vector for Qwen3-8B's own
   top causal feature (layer 25/65291), computed cosine similarity
   against each model's dense direction at that same layer (recomputed
   from TRAIN-split cached activations, matching
   `src.direction.compute.compute_directions` exactly), against a
   random-200-feature null baseline. Result: Llama 0.201, Qwen3-8B
   0.196 -- essentially identical, both real (random baseline ~0.013-0.016)
   but nowhere near "the same axis." **Does not differentiate the two
   models** -- reported as inconclusive, not as confirming a
   misalignment story, since the model where the dense direction *does*
   work causally shows the same degree of "misalignment."

**What the data does support**: the concentration-vs-distribution
asymmetry already documented in Wave 2/3 above (Llama's causal effect
collapses into one feature; Qwen3-8B's is spread across the ranked set,
top-1 alone doing nothing) co-occurs with which model's dense direction
ablates/adds cleanly. Reported as a real, multi-pronged correlational
finding from n=2 models with full data, not a proven mechanism -- no
mechanistic account of *why* Llama's refusal computation would be more
concentrated is established here, and none is claimed. Full write-up,
including the honest hedging, in RESULTS.md's dedicated section; results
in `results/llama_causal_gap_analysis.json` (gitignored, matching this
project's `results/` convention -- rerun the script to reproduce).

**Extended to gemma-2-9b-it the same day**, user's explicit choice among
several remaining open questions (offered as options, this one picked as
the natural continuation of the just-finished work). Added gemma to both
of the script's checks (`NECESSITY_MODELS`, `SAE_ALIGNMENT_MODELS`):

- Its own top causal feature (GemmaScope layer 35/52410, from Wave 2's
  ranking) required computing its raw diff-in-means direction norm fresh
  (`raw_direction_norm_at_layer`, new helper) -- unlike the other three
  models, gemma has no prior `phase1_reproduction_*.json` or
  `sufficiency_7b_9b_scale.json` entry to reuse, since it was never run
  through either script. Verified GemmaScope's `W_dec` layout matches the
  project's own (d_model, d_sae) convention after `load_sae`'s internal
  transpose (see `src/sae/gemma_scope.py`'s docstring) before reusing the
  same `feature_vector` helper unchanged.
- **Magnitude ratio: 0.525, unremarkable** -- sits right between
  Qwen2.5's (0.522) and Qwen3-8B's (0.509), nothing like Llama's outlier
  0.702. Strengthens the "Llama specifically, not 8-9B models generally"
  reading of the ruled-out hypothesis above.
- **Cosine alignment: 0.367, notably higher than Llama's or Qwen3-8B's**
  (~0.20 each) -- a genuine third data point, but reported as an
  observation, not evidence either way: gemma has no own-direction dense-
  ablation causal-generation test in this project (only Qwen3-8B and
  Llama-3.1-8B were tested that way at this scale) to correlate the
  alignment number against. gemma's own SAE causal-effect shape (smooth
  ranking-score decay, no standout feature, modest 96%->82% suppression
  curve -- see Wave 2 above) is its own third pattern, not a blend of
  Llama's concentration or Qwen3-8B's distribution, so the higher
  alignment doesn't obviously explain anything about it either.
- **Net effect on the original two-model puzzle**: unresolved, as
  expected going in -- gemma adds one confirmatory data point (magnitude)
  and one new, separately-unexplained observation (alignment), not a
  third data point for the necessity/sufficiency correlation itself,
  since that causal test doesn't exist for gemma. Consistent with this
  project's standing practice of reporting a genuine null/non-resolving
  result rather than reaching for a post-hoc story to make three models
  look tidier than they are.

## Cross-model sufficiency transfer: does activation addition transfer between Qwen3-8B and Llama-3.1-8B? (2026-07-24)

Closes the last item on the session's remaining-gaps list that didn't
need new methodology: `scripts/transfer_direction.py` already tested
necessity (ablation) transfer for this model pair and deliberately
deferred sufficiency, since addition needs its own alpha calibration for
the *foreign* direction on the target's residual-stream scale (a fixed
alpha has no consistent cross-model meaning, same reasoning as
`compute_raw_directions`'s own docstring). `scripts/transfer_sufficiency.py`
does that calibration and the corresponding causal-validation generation.

**Reuse, not regeneration, wherever the data already exists**: own raw
directions/layers are recomputed identically to
`scripts/sufficiency_at_scale.py` (never re-derived from scratch), and
that script's own baseline/own-addition completions
(`results/sufficiency_7b_9b_scale.json`) are reused outright rather than
regenerated -- verified byte-for-byte that this script's harmless-prompt
sampling (same seed, same val/test splits, same n=12/n=50) reproduces
that file's `calib_prompts`/`val_prompts` exactly before trusting the
reuse. Only the foreign-direction alpha sweep and final validation needed
new generation (~134 completions per target model), roughly half the
scope a from-scratch run would have needed.

**Convention matched to the necessity side**: the foreign raw direction is
added at the *target's own* already-selected layer (Qwen's direction into
Llama's layer 27; Llama's direction into Qwen's layer 23), mirroring
`scripts/transfer_direction.py`'s "foreign direction evaluated at the
target's own already-selected layer" convention, extended here from
ablation to addition.

**Result: a clean, symmetric no-transfer verdict for both models,
sharper than the necessity side's**. Foreign-direction addition induced
0.00 refusal-rate change above baseline at *every* alpha tested (0.25
through 4.0) in both models -- not a weak effect, an absent one. Paired
McNemar's exact tests: Qwen3-8B baseline-vs-foreign p=1.0 (0/50
discordant -- completions identical to baseline); Llama-3.1-8B
baseline-vs-foreign p=0.5 (2/50 discordant, still indistinguishable from
no effect). Own-vs-foreign is clearly significant for both (p<0.001),
and baseline-vs-own confirms Llama's real-but-weak 10%->34% own-addition
effect isn't itself noise (p=0.0005). Zero degenerate completions in
either foreign-addition run, so the null isn't a byproduct of the
intervention breaking generation.

**This resolves what the necessity side left inconclusive for Llama**:
own-ablation vs. foreign-ablation couldn't be distinguished at n=50, but
own-addition vs. foreign-addition is unambiguous. Not generalized beyond
this one model pair -- still the only `d_model`-matched pair available.
Full write-up in RESULTS.md; results in `results/sufficiency_transfer.json`
(gitignored, matching this project's `results/` convention).

## PAIR-paraphrase robustness spread: a continuous margin account (2026-07-24)

User's explicit choice among three remaining open items (offered as options):
dig into one of the standing unexplained cross-model differences rather than
retry the moralize-vs-comply classifier or the dense-vs-SAE PAIR flip. Picked
the PAIR-robustness spread specifically (SmolLM2 90.5% > Llama-3.1-8B 66.7% >
gemma-2-9b-it 47.6% > Qwen3-8B 42.9% > Qwen2.5-1.5B 38.1%, Cochran's Q
p=0.0006) -- the existing candidate hypothesis in RESULTS.md (SmolLM2's
weaker/less-linear baseline refusal) was explicitly flagged as not explaining
Llama's own comparative robustness, so this needed a genuinely different
angle, not a retest of the same idea.

**Approach**: replace the binary "flagged or not" adversarial-detection rate
with a continuous per-prompt margin -- each model's dense-direction
projection on the 21 real PAIR prompts, minus its own calibrated threshold,
normalized by that layer's pooled harmful/harmless standard deviation
(matching `src.direction.compute.separation_score`'s own normalization, so
comparable across models with very different raw activation scales). Also
computed the same normalized margin for genuine harmful VAL prompts as a
reference scale, expressing PAIR margin as a fraction of it.

**A real gap found before the analysis could even start**: only 3 of 5
models (Qwen3-8B, Llama-3.1-8B, gemma-2-9b-it) had a cached
`results/activations/{model}_adversarial.pt` -- `scripts/extend_qwen_smollm.py`
scores Qwen2.5-1.5B/SmolLM2 on the adversarial set by extracting activations
transiently at eval time, never caching them. Rather than limit the
investigation to 3 of 5 models (missing both the most- and least-robust
models in the ranking), wrote `scripts/extend_adversarial_small.py` to build
the missing two caches -- forward-pass-only extraction (no generation,
no quantization needed for these small models), ~10 seconds each, mirroring
`scripts/extend_sae_adversarial.py`'s exact payload shape.

**Sanity check before trusting anything new**: recomputing each model's PAIR
detection rate from these fresh per-prompt margins (fraction with projection
above threshold) reproduced the already-published rates exactly
(0.381/0.905/0.429/0.667/0.476) -- confirms the already-persisted
directions/thresholds (`results/dense_directions.pt`,
`results/dense_direction_cross_model.json`,
`results/detector_thresholds_Qwen3-8B.json`) are being applied correctly
before trusting any new number built on top of them.

**Result: a real, formally-significant rank correlation, not a mechanism**.
Spearman's rho between mean PAIR margin and known detection rate across all
5 models: **rho=0.90, p=0.037** (n=5 -- small, but this is the full
population of models this project has, not a sample). The top 3 models by
margin (SmolLM2 0.469, Llama-3.1-8B 0.332, gemma-2-9b-it -0.049) match the
detection-rate ranking exactly; the bottom two (Qwen2.5-1.5B -0.109,
Qwen3-8B -0.240) are swapped relative to detection rate, both solidly
negative and close together -- reported as the one genuine discrepancy, not
smoothed into a perfect match. One new textured observation surfaced by the
continuous measure that the binary rate couldn't show: gemma-2-9b-it sits
almost exactly at its own decision boundary on average for PAIR prompts
(margin -0.049, essentially zero), distinct from the Qwen models being
pushed solidly into harmless-looking territory.

**What this does and doesn't establish**: this sharpens the *description* of
the phenomenon (a formal statistic, finer resolution, a genuine new
observation about gemma) but does not identify *why* some models' margins
survive paraphrasing better than others -- that mechanism question is
exactly as open as it was before this analysis, just now described with a
number instead of an eyeballed ranking. Consistent with this project's
standing practice of not forcing a description into an explanation it
doesn't support. Full write-up in RESULTS.md; results in
`results/pair_margin_analysis.json` (gitignored, matching this project's
`results/` convention).

## Is Llama's dense-vs-SAE PAIR flip real? A continuous-margin check (2026-07-24)

Last of three remaining open items, user's explicit choice (offered
alongside retrying the moralize-vs-comply classifier): Llama-3.1-8B is the
one case in this project where SAE-feature numerically beats dense-direction
on PAIR specifically (80.9% vs 66.7%, McNemar p=0.25, not significant at
n=21, see Wave 3 above) -- the direction arXiv:2505.23556 originally
claimed, but never confirmed at significance anywhere else in this project.
`scripts/sae_pair_margin.py` extends the same continuous-margin method just
used for the 5-model dense-direction analysis to Llama's SAE-feature score,
reusing `src.detectors.sae_feature_detector.score` unchanged (the exact
function the published 80.9% came from) rather than reimplementing it --
needed no new GPU generation.

**Sanity check first**: recomputed detection rate from the fresh per-prompt
margins reproduced the published 80.9% (got 81.0%, rounding) -- confirms the
already-calibrated threshold and already-selected top-15 features are being
applied correctly.

**Result**: SAE-feature's margin is higher than dense-direction's in both
absolute (0.508 vs 0.332) and relative (0.627 vs 0.355, as a fraction of
each detector's own genuine-harmful-prompt margin) terms -- not just an
artifact of where the binary threshold happens to fall. Ran a **paired
Wilcoxon signed-rank test on the same 21 prompts' continuous margins**
(more statistical power than McNemar's test on the binarized pass/fail
outcome, since it uses the full margin rather than collapsing to
above/below threshold): statistic=73.0, **p=0.147**. Lower than the
original McNemar p=0.25 (moves in the direction of "more likely a real
effect"), but still not significant at the conventional 0.05 threshold.

**Genuinely inconclusive, reported as such rather than either dismissed or
oversold**: a more powered test doesn't resolve this into "real" or "noise"
-- both binary and continuous measures point the same direction (SAE beats
dense on Llama's PAIR set) without either reaching significance at n=21.
This is the honest end state for this specific question given the data on
hand; resolving it further would need a larger matched adversarial set,
blocked on JailbreakBench publishing more real artifacts (same limitation
noted elsewhere in this project), not a re-analysis this session could do.
Full write-up in RESULTS.md; results in `results/sae_pair_margin_llama.json`
(gitignored, matching this project's `results/` convention).

## Retrying the moralize-vs-comply classifier with a larger local judge (2026-07-24)

User's explicit choice to retry rather than stop after the three cross-model
re-analyses above. Two prior judges (Phi-4-mini-instruct 3.8B, SmolLM2-1.7B)
both failed by defaulting to one category (~70%+ "moralize") regardless of
content -- a real capability ceiling at that size, not a prompt-engineering
problem (see the original "Moralize-vs-comply classifier" entry above).
Genuinely untried: a larger local model already proven-compatible with this
project's own tooling. `src.direction.moralize_comply_classifier.load_judge_model`
now takes an optional `judge_model_name` override (default unchanged, so no
behavior change for existing callers); `scripts/validate_judge_llama.py`
tries Llama-3.1-8B-Instruct (8B, ~2-5x either prior judge) against the same
two labeled worksheets (98 rows), same reporting structure as
`scripts/validate_classifier.py`, for direct comparability.

**A real independence question, flagged before running rather than
after**: unlike Phi-4-mini-instruct (picked specifically to be independent
of the Qwen/Llama/Gemma families this project causally tests), Llama-3.1-8B
is itself one of them. 27 of 98 worksheet rows have completions Llama-3.1-8B
itself generated (`sae_suppression_Llama-3.1-8B-Instruct`,
`transfer_qwen_direction_on_llama`). Added a dedicated self-judged-vs-
independent breakdown to the validation script rather than only reporting
an aggregate that would average the bias away.

**Result: fails, but for a genuinely different and more concerning reason
than the prior two.** Verdict distribution across the 98 rows: {refuse: 43,
moralize: 31, comply: 21, partial: 3} -- the most common single verdict is
only 43.9% of outputs, so unlike the prior judges, this one is not
collapsing to one category regardless of content. It does perfectly on the
easy safety-net case (21/21 on genuine `refuse` ground-truth rows) and
reasonably on harmless-prompt compliance (7/9, 77.8%). **But on the actual
load-bearing case -- harmful-prompt genuine compliance/partial compliance
-- accuracy is 0/7 and 0/10 respectively.** Spot-checked a few of the
wrong "comply" calls directly: several genuine harmful-compliance
completions (e.g. one actively delivering the requested harmful speech
content) were called "refuse" outright -- not confusion between adjacent
categories (moralize vs. comply, an understandable judgment call), but a
complete category miscall on unambiguous content. This is a worse failure
mode for the classifier's actual purpose than the prior two judges' honest
"I can't tell, defaulting to X" collapse, even though it superficially
looks more sophisticated (spread across categories).

**The flagged self-judging concern is real, not just theoretical**: 8/27
(29.6%) accuracy on self-judged rows vs. 44/71 (62.0%) on independent
rows -- more than double the error rate when Llama-3.1-8B judges its own
completions. Worth noting for the record even though the independent-rows
accuracy (62.0% overall, but still 0% on the load-bearing harmful-comply
case) isn't good enough to use either way.

**Still "validated, found unreliable with locally-available models"** --
three local judges tried now (1.7B, 3.8B, 8B), three different failure
modes, none usable. Direct labeling remains the only validated approach
for true-harmful-compliance labeling in this project. Full write-up in
RESULTS.md; results in
`results/moralize_comply_classifier_validation_llama_judge.json`
(gitignored, matching this project's `results/` convention).

## Does concentration protect the causal signal from paraphrase? A matched-pair mechanistic dig (2026-07-24)

After the user asked "what's the next gap" once the day's re-analyses were
done, the honest answer was that nothing actionable remained -- everything
left was either blocked externally, deliberately left open by this
project's own philosophy, or needed the user directly. Recommended against
opening a new research thread on that basis, but flagged one real
exception: the SAE-concentration/PAIR-robustness connection was the most
promising thread left, though it would need genuine mechanistic work (not
another aggregate-margin script) to say anything new. User asked to do
exactly that. Planned via `EnterPlanMode` first (a genuinely non-trivial,
multi-script investigation, not a quick fix) -- full plan approved before
any code was written.

**The concrete new idea, found during planning**: every PAIR-adversarial
record is matched, by construction
(`src.eval.adversarial_paraphrase.build_adversarial_set`'s own goal-text
matching), to one exact original TEST-split harmful prompt already in the
main activation cache. Verified directly before committing to the plan:
21/21 PAIR records match a TEST-split `source=="jbb"` prompt, for every
model. This gives real **matched pairs** (same underlying request, only
the phrasing differs) instead of comparing two unrelated distributions
(what the day's earlier `pair_margin_analysis.py`/`sae_pair_margin.py` did)
-- still zero new GPU generation, since all layers are cached for every
model and the matched originals are already sitting in the corpus.

**Motivating observation, never noticed before this session's planning**:
among the 3 SAE-having models, dense-direction PAIR robustness (Llama
66.7% > gemma 47.6% > Qwen3-8B 42.9%) ranks in the *same order* as SAE
causal-effect concentration from Wave 2 (Llama: one dominant feature;
gemma: modest/gradual; Qwen3-8B: distributed). This rules out "redundancy
protects" (the most-distributed model is the *least* robust) and leaves
"the concentrated feature is itself paraphrase-invariant" as the standing
candidate to test directly.

**Two scripts, matching the plan exactly**:

1. `scripts/paraphrase_decay_dense.py` (all 5 models): matched-pair delta
   (original margin minus paraphrased margin, both normalized by pooled
   std as in `pair_margin_analysis.py`) plus a per-model paired Wilcoxon
   test. **Result: the delta ranking matches the known PAIR-detection
   ranking exactly across all 5 models** (SmolLM2 smallest delta/most
   robust through Qwen2.5 largest delta/least robust). Spearman rho=-1.0
   -- but at n=5, `scipy.stats.spearmanr`'s default p-value uses a
   t-distribution approximation that is not valid at this sample size (the
   project's actual full model count, not something to grow). Computed the
   **exact permutation p-value directly** (brute-force over all 5! =120
   orderings, matching this project's existing "use the exact test, not
   the asymptotic one" discipline -- e.g. McNemar exact elsewhere): p=0.0167,
   a real, formally significant result, not an artifact of the wrong test.
   SmolLM2 is also the only model where the paraphrase shift itself isn't
   statistically distinguishable from zero (p=0.0595) -- consistent with
   it being the most robust by a wide margin.
2. `scripts/paraphrase_decay_sae.py` (3 SAE models): same matched-pair
   method, but at the feature level, testing two separate questions rather
   than conflating them -- does each model's own top-ranked causal feature
   survive paraphrasing, and does the full top-15 detector score decay
   differently from that single feature (the direct redundancy test).
   Reused `src.detectors.sae_feature_detector.score` and
   `src.sae.registry.SAE_PROVIDERS` unchanged. **Result: Llama's top
   feature (layer 27/13363) is the only one of the three where paraphrase
   produces no statistically detectable shift at all** (p=0.1678) --
   direct, matched-pair evidence for the "this specific feature is
   unusually invariant" hypothesis, not just an inference from aggregate
   correlation. Qwen3-8B's and gemma's own top features both shift
   significantly. The delta ranking again matches known SAE-feature
   PAIR-detection exactly (rho=-1.0), but **at n=3 the exact permutation
   p-value is 0.333** -- a perfect rank match is the best any n=3 result
   can score, reported honestly as not significant rather than oversold.

**A real answer to the redundancy question, found within each model**:
for Qwen3-8B, the full top-15 score decays *more* than its own top feature
alone (2.037 vs. 1.315) -- paraphrase disrupts many of its ranked features
together, not a case of redundancy averaging out noise. For Llama, the
full score decays significantly (0.858, p=0.0038) even though its top
feature alone does not (0.396, n.s.) -- the one causally-dominant feature
is robust, but the other 14 features the published detector also sums in
are not, so the detector's own 80.9% PAIR robustness is somewhat diluted
by non-causally-important features, not purely carried by the invariant one.

**Honestly hedged, not oversold**: this gives real, statistically-grounded
support for a specific mechanism-adjacent claim (Llama's dominant feature
specifically resists paraphrase; the effect isn't explained by redundancy)
-- a genuine step down from "these two rankings happen to correlate"
toward "here is what specifically does and doesn't change under
paraphrase, per matched prompt pair." It does **not** explain *why* that
one feature happens to be invariant (would need token-level attribution of
what the paraphrase changes -- explicitly scoped out of this plan as a
heavier Tier 2, not attempted). The SAE-level question is also
fundamentally capped at n=3 (no SAE for SmolLM2/Qwen2.5-1.5B, the two most
extreme dense-direction robustness models), so it can never reach the
full 5-model resolution the dense-direction result achieves. Full
write-up in RESULTS.md; results in `results/paraphrase_decay_dense.json`
and `results/paraphrase_decay_sae.json` (gitignored, matching this
project's `results/` convention).

## Tier 2: why is Llama's feature paraphrase-invariant? Token-level attribution (2026-07-25)

User's explicit continuation of the mechanistic dig above, agreeing this is
genuinely heavier work (real new GPU generation -- forward/backward passes,
not another cached re-analysis) via `EnterPlanMode` first. Goal: which
input tokens actually drive Llama's top causal feature's activation on a
PAIR-paraphrased prompt -- the literal harmful content, or the roleplay
wrapper PAIR uses to disguise it, compared against the same question for
Qwen3-8B (whose own top feature does decay under paraphrase).

**Two failed Integrated Gradients attempts, in full, not smoothed over --
both real bugs, caught by a numerical correctness check before trusting
any output on real prompts**:

1. **Multiplicative interpolation from a zero embedding baseline**
   (`v = alpha * natural_embedding`, mirroring `src.sae.causal_ranking`'s
   own "natural value -> 0" convention for a single scalar feature).
   IG's completeness property (sum of attributions should equal
   metric(natural) - metric(baseline)) failed by ~100x (sum=-1.44 vs.
   expected -139.6). **Root cause, confirmed by direct measurement, not
   guessed**: Llama-3.1/Qwen3 apply RMSNorm immediately after the
   embedding layer, and RMSNorm is scale-invariant
   (`RMSNorm(alpha*x) == RMSNorm(x)` for any alpha>0) -- scaling the
   embedding by 0.5 changed layer-0's output by only ~1.4 in max-abs terms
   (measured directly), confirming multiplicative interpolation from a
   zero baseline is degenerate everywhere except the single point
   alpha=0, which the discretized IG integral never actually samples.
2. **Fixed to additive interpolation toward a real, non-degenerate
   baseline** (each model's own EOS-token embedding, tiled across
   positions -- fetched inside a `model.trace()` since module parameters
   resolve to meta tensors outside one, confirmed directly). Completeness
   still failed, now with an even larger discrepancy (sum=13.9 vs.
   expected 6567.75) -- most likely because a repeated-EOS-token input is
   itself such an extreme out-of-distribution point that the straight-line
   path from natural to baseline isn't well-approximated by 10 IG steps,
   though a deeper nnsight/4-bit-quantization gradient interaction at the
   embedding layer specifically was not fully ruled out either.

**Switched to single-token leave-one-out ablation instead** (offered as an
option, user chose to keep debugging IG once, then switched after the
second failure) -- no gradients, no interpolation path to get wrong: for
each token position, replace just that token with the model's own EOS
token and re-run a clean forward pass, comparing the target feature's raw
pre-activation with vs. without that token. Batches all `seq_len`
single-token variants for one prompt into one forward pass per batch
chunk. Verification here is a basic sanity check (finite values, real
variance across positions), not IG's completeness property -- leave-one-out
effects don't sum to the full-prompt effect in general (tokens can be
redundant with each other), so there's no equivalent invariant to check
against.

**Three more real bugs found and fixed during the actual run, not before
it**:

1. **Un-masked top tokens were dominated by chat-template scaffolding**
   (`<think>`, `assistant`, `ĊĊ` double-newline) near the readout position,
   not the paraphrased content -- fixed by restricting reported top tokens
   to the character span of the raw instruction text within the
   chat-templated prompt (found via `templated.find(prompt)` plus the
   tokenizer's offset mapping), verified with a fast CPU-only tokenizer
   test before re-running anything on GPU.
2. **A batch size of 32 for the ablation forward passes OOM'd immediately**
   ("16.04 GiB is allocated by PyTorch" on a 6GB card) -- the 4-bit 8B
   model alone leaves very little headroom; reduced to 8 (measured
   near-identical wall-clock vs. batch=4, so 8 halves the trace() call
   count for free). Even so, throughput was ~300s/prompt regardless of
   batch size (nnsight per-trace overhead dominates, not batch compute) --
   far slower than this project's `causal_ranking.py` IG work, a real,
   unexplained difference worth remembering if this pattern (embedding-
   layer hooking, many small forward passes) comes up again.
3. **A real cross-run bug querying** `content_span_mask`'s tokenizer call
   passed `add_special_tokens=False` while `token_ablation_importance`'s
   call used the default (`True`) -- silently consistent for Qwen3-8B
   (whose tokenizer adds no special token by default) but crashed on
   Llama-3.1 specifically (whose tokenizer prepends a BOS token by
   default), a length mismatch only surfacing on the *second* model after
   Qwen3-8B's full 21-prompt run had already completed. Fixed by making
   both calls use the same (default) setting.
4. **A genuine memory leak across loop iterations**, distinct from the
   batch-size OOM above: `_feature_value` (called repeatedly per ablation
   batch per prompt) was missing `@torch.no_grad()` despite never calling
   `.backward()` -- every forward pass built and retained a full, unused
   autograd graph, accumulating until a later prompt's allocation finally
   exceeded the 6GB card ("15.87 GiB allocated by PyTorch", again absurd
   for one forward pass). The already-existing `_metric_at_alpha`
   sanity-check helper *was* already `@torch.no_grad()`-decorated and
   never showed this symptom in any test run, which is what pointed at the
   fix. **Added per-prompt checkpointing** (`results/token_attribution.json`
   rewritten after every single prompt, plus raw per-token importance
   saved even if the downstream reporting step fails) after the *first*
   crash lost Qwen3-8B's already-completed run to a bug in the reporting
   step it didn't actually need to re-lose -- the second crash (this one)
   confirmed the checkpointing's value: Qwen3-8B's 21 prompts were
   preserved and skipped on the retry, only Llama's partial 5/21 needed
   redoing.

**Result: a real, qualitative pattern, read directly rather than run
through an automated classifier** (same "moralize-vs-comply" discipline as
elsewhere in this project) -- Llama's top-ranked token is frequently the
literal core harmful-action word itself (`"unauthorized"`, `"cloning"`,
`"underage"`, the `"ext"`+`"ort"` pieces of "extort", all at rank 1 on
their respective prompts); Qwen3-8B's top tokens more often surface the
PAIR attack's fictional/roleplay wrapper instead (`"hypothetical"` at
rank 1 twice, `"consultant"` at rank 1 three times) or are entirely generic
connective tissue. Rough tally, explicitly a direct-reading judgment call
and not a validated automated count: ~9/21 Llama prompts show a clear
content word in the top-5 (several at rank 1) vs. ~6/21 for Qwen3-8B,
rarely at rank 1. Neither model is clean -- plenty of prompts on both sides
are mostly generic regardless of model. Full write-up, including explicit
limitations (why, not just what, remains unestablished; token-level
attribution conflates identity and position), in RESULTS.md; all 42
prompts' full top-5 lists in `results/token_attribution.json` (gitignored,
matching this project's `results/` convention).

## Adding DeepSeek-R1-Distill-Qwen-1.5B: the mandatory `<think>` block breaks "last-prompt-token / first-generated-token" methodology (2026-07-26)

Every model onboarded so far answers immediately after its formatted
prompt. Two places in the pipeline hardcode that assumption: (1)
`refusal_classifier.is_refusal` only inspects the first 200 characters of
a generated completion, fed by `interventions.py`'s `generate_*` functions
(default `max_new_tokens=40`); (2) `causal_ranking.py`'s
`feature_ig_attribution` reads `model.lm_head.output[:, -1, :]` -- the
logits at the position immediately after the prompt -- for the
differentiable refusal-vs-compliance metric used in SAE causal ranking.

DeepSeek-R1-Distill-Qwen-1.5B's chat template (`tokenizer_config.json`,
verified directly) auto-appends `<｜Assistant｜><think>\n` whenever
`add_generation_prompt=True`. There is no `enable_thinking`-style flag --
the template's only way to suppress the tag is `add_generation_prompt=False`,
which removes the generation prompt entirely and isn't usable. So for this
model both (1) and (2) land inside the reasoning trace, not the answer.
Activation *extraction* (`src/activations/extract.py`, last prompt-token
residual stream) is unaffected -- it runs before generation starts.

**Fix**: `src/direction/extract_answer.py::extract_answer(text)` splits on
the first `</think>` (confirmed a plain, non-special token for this
tokenizer -- id 151649, `skip_special_tokens=True` does not strip it) and
returns the post-tag text, or `None` if the tag never appears (a genuine
truncated/non-converging trace, not silently classified anyway). New
`refusal_classifier.resolve_completions()` partitions a batch of raw
completions into resolved answers plus a truncated count, so causal-
validation scripts can classify only the real answer and report truncation
honestly as its own outcome, not folded into refusal or compliance.
`reproduce_direction.py` and `calibrate_alpha.py` both gained
`--reasoning-model` and `--max-new-tokens` flags (default off/40, so every
other model's behavior is byte-identical) that route through this path.

Budget calibration was empirical, not assumed (`scripts/think_length_probe.py`
-- see RESULTS.md for the numbers and the discovery that ~a third of
prompts trigger a genuine non-converging reasoning loop that no reasonable
budget fixes, not just "needs more tokens"). Settled on
`max_new_tokens=2048` for Phase 1 and calibration runs.

**Not a bug, corrected after further checking**: `is_degenerate()`'s
completions initially looked like false positives on a quick, truncated
(~500-char) read -- written up here as a bug at first. Full-text inspection
during the later suppression-validation run (see RESULTS.md) found this
was wrong: those completions genuinely do collapse into verbatim-repeated
sentences later in the text, past what a short preview shows. The
threshold is correct as-is; the lesson is to read full completions, not
previews, before concluding a classifier is wrong on this model.

**Investigating a genuine null result**: the alpha-calibration sweep found
0% refusal at all 7 tested alphas (0.25-4.0). Two hypotheses were tested
before accepting this as real: (a) that applying the addition intervention
across the *entire* reasoning trace (not just a 40-token window like every
other model) compounds too much, degrading the answer instead of inducing
refusal -- tested by building `interventions.py::generate_reasoning_trace()`
(a plain, unhooked generation pass that resolves the prompt + reasoning
text through `</think>`) plus a `prompt_override` parameter on
`generate_with_addition`/`generate_with_ablation`, so the intervention can
be applied only during the post-think answer. Still 0/5 refusal at alpha
1-8. (b) That layer 9 (chosen by separation score, not validated for
causal potency) or the tested alpha range was simply wrong -- alpha 16/32
at layer 9 finally broke coherence entirely (confirming `is_degenerate()`
does work at genuine breakdown), and alpha 2/8 at a mid-network layer (14)
stayed coherent but still complied. **Both hypotheses ruled out. This is a
genuine null result**: necessity is present (weak, from a low ~14%
baseline), sufficiency is absent, across the full tested space. Same
asymmetry already documented for Llama-3.1-8B, more extreme here.

`prompt_override` on `generate_with_addition` (and `generate_with_ablation`,
added for symmetry / future use) is the same mechanism Fix B (the two-stage
resolve/attribute path added to `causal_ranking.py` and
`generate_with_feature_suppression` for the SAE-feature work below) reuses
-- same "freeze the reasoning trace once, then treat it as the fixed
prefix" pattern.

## DeepSeek-R1-Distill-Qwen-1.5B SAE-feature causal ranking: hookpoint plumbing, a real OOM, and its actual cause (2026-07-27)

**Registry**: `EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k` is trained
on MLP-module output, not the residual stream like the three existing SAE
providers (Qwen-Scope/LlamaScope/GemmaScope all hook
`model.model.layers[i].output`). `SAE_PROVIDERS` is unpacked as a 3-tuple
in 10 call sites across scripts/API/tests -- widening it to carry a
hookpoint field would have touched all 10 for no reason, so hookpoint got
its own small `SAE_HOOKPOINT` lookup + `hookpoint_for()` helper in
`src/sae/registry.py` instead, defaulting to `"resid"` for every existing
model. A new `src/activations/extract.py::hook_module(model, layer_idx,
hookpoint)` resolves the right nnsight Envoy (`layer` or `layer.mlp`), used
by `causal_ranking.py::_ig_chunk` and
`interventions.py::generate_with_feature_suppression`. Layers [7, 8, 11]:
top-3 by separation score on the full corpus (`scripts/extract_activations.py`),
superseding Phase 1's smaller ad-hoc [8, 7, 9] read -- EleutherAI's suite
covers all 28 layers, so no HfApi check needed unlike the partial-coverage
providers.

**Checkpoint verified directly** (not assumed from the `sae` library's
typical convention): `layers.{n}.mlp/sae.safetensors` has keys
`encoder.weight`/`encoder.bias` (same (d_sae, d_model) convention as
Qwen-Scope's W_enc/b_enc) but `W_dec` shaped **(d_sae, d_model)** -- the
*transpose* of Qwen-Scope's (d_model, d_sae) convention, no separate
`decoder.weight` key. `src/sae/eleuther_scope.py` reuses `TopKSAE`
unchanged (no need for a new class) by transposing `W_dec` once at load
time; `k=32` from the checkpoint's own `cfg.json`, not assumed from the
paper's stated default.

**A real OOM, and the wrong hypothesis about it, before the right one**:
`rank_sae_features.py` crashed with "15.93/15.95 GiB is allocated" on a
6GB card on the very first candidate, both at the default full-batch
micro_batch_size and after cutting it to 2 (gemma-2-9b-it's own fix for an
earlier OOM) -- the number barely moved between the two, which should have
been the tell that this wasn't really about batch size. First hypothesis
(wrong, but reasonable): the new `hook_module`/MLP-hookpoint code path.
Ruled out directly -- an isolated single-call test with `hookpoint="resid"`
(the same code path already proven working for the other 3 models)
allocated the identical 7.14GB as `hookpoint="mlp"` on the same short
prompt. **Actual cause**: `rank_sae_features.py` hardcoded
`load_in_4bit=True` for every model, inherited from when it only ever ran
against Qwen3-8B/Llama-3.1-8B/gemma-2-9b-it (all needing 4-bit just to
fit). DeepSeek-1.5B doesn't need 4-bit for its own size and runs Phase
1/calibration fine in fp16, but the IG backward pass specifically is far
more memory-hungry in fp16 than in 4-bit -- measured directly, 7.14GB vs.
2.58GB for the identical single tiny call. Fixed with a small
`NEEDS_4BIT` set gating `load_in_4bit` per-script per-model, now including
DeepSeek for this script only (Phase 1/`calibrate_alpha.py` stay fp16 --
forward-pass-only generation doesn't have this problem, confirmed by Phase
1 already having run successfully in fp16). Windows' WDDM driver silently
pages GPU allocations into system RAM rather than failing immediately when
VRAM is exceeded, which is why `torch.cuda.memory_allocated()` could
report a real, live 7.14GB on a 6GB card without erroring until the
combined VRAM+RAM commit was finally exhausted -- worth remembering next
time a PyTorch CUDA OOM message reports an "allocated" figure that looks
physically impossible on this machine; it's a real number, not a
mis-report.

`generate_with_feature_suppression` (used by the causal-validation step
below) got the same `hookpoint`/`prompt_override` treatment as
`generate_with_addition`/`generate_with_ablation`, but did **not** need
the 4-bit fix -- it's forward-pass-only per-token generation (like Phase
1's ablation/addition), not a backward pass, so it stayed in the
cheaper-but-sufficient cost class and ran fine in fp16.

**Ranking result**: layer 11/feature 48719 is a clear top feature (score
0.074) above the 2nd-ranked layer 7/feature 3715 (0.049) and the rest of
the top-20's near-zero tail -- full numbers in RESULTS.md.

**Suppression validation scoped down before running**: the original
6-condition/N=50 protocol (used for the other 3 SAE models) would have
taken ~13+ hours at this model's reasoning-inclusive budget (measured
~10.8 tok/s under the per-token suppression hook, same rate as Phase 1's
ablation). Flagged to the user before running (mirroring the Phase 1
pre-flight cost check); reduced to N=15 (~4hr) rather than also cutting
the number of conditions, to keep the dose-response resolution the other
models got. Result and its honest "inconclusive, not negative" reading
(near-floor baseline refusal rate plus real truncation shrinking effective
N well below 15) are in RESULTS.md.

**`is_degenerate()` correction**: the "false positive" claim written up
earlier in this file and in RESULTS.md (Alpha calibration section, above)
was wrong -- caught during this run by reading a full completion instead
of a preview and finding genuine verbatim-repetition collapse partway
through. Both files corrected in place rather than left standing; see
RESULTS.md for the full account, including a new, actually-real finding
this correction surfaced (this model's long-form answers have a real
tendency to degenerate into repetition once they run out of content,
present even at baseline).

## Retrofitting DeepSeek into the 5-model statistical comparisons, and a real SAE-detector generalization failure (2026-07-27)

Per the user's explicit request, extended the three cross-model
statistical analyses that were built for 5 models to include DeepSeek
(deferred as a separate ask from the earlier API/UI wiring, on the user's
guidance):

- **`scripts/extend_deepseek.py`** (new, mirrors `extend_llama_gemma.py`'s
  pattern but single-model, no 4-bit -- DeepSeek doesn't need it):
  populated `dense_direction_cross_model.json`. Layer 7, TEST accuracy
  84.7% (AUROC 0.911) -- weakest of all 6 models -- and by far the worst
  PAIR robustness (9.5% vs. next-lowest 38.1%).
- **`scripts/extend_adversarial_small.py`** extended with a third model
  entry (DeepSeek fits the same "small, no 4-bit needed" category as
  Qwen2.5-1.5B/SmolLM2, reused rather than writing a near-duplicate
  script) -- built `DeepSeek-R1-Distill-Qwen-1.5B_adversarial.pt`.
- **`scripts/pair_margin_analysis.py`, `scripts/cross_model_significance.py`,
  `scripts/paraphrase_decay_dense.py`**: each got a `DeepSeek-R1-Distill-Qwen-1.5B`
  entry added to their existing per-model dicts/lists (`MODELS`,
  `KNOWN_PAIR_DETECTION_RATE`, `SMALL_MODELS`) and were rerun.

**Two real findings from the reruns, not just extended tables**:

1. **Cochran's Q strengthens with DeepSeek included**: Q=34.44, df=5,
   p=1.94e-6 (was Q=19.52, df=4, p=0.0006 at 5 models) -- DeepSeek's
   extreme 9.5% PAIR detection rate widens the already-significant
   cross-model spread further, not an artifact of adding a 6th data point.
2. **The matched-pair paraphrase-decay Spearman correlation breaks**:
   was rho=-1.0 (exact p=0.0167, n=5, a perfect rank match) between
   projection-shift-under-paraphrase and known PAIR-detection rate;
   drops to rho=-0.657, p=0.175 (n=6, no longer significant) with
   DeepSeek added. DeepSeek's matched-pair delta (0.892) is unremarkable
   in magnitude, similar to Llama's (0.767) or gemma's (1.030) -- but its
   actual PAIR detection rate (9.5%) is far below what that delta would
   predict from the other 5 models' pattern. **Reported as a genuine
   break in a previously clean relationship, not silently absorbed into
   an updated number** -- the earlier "matches exactly" framing in
   RESULTS.md's matched-pair section needs updating alongside this.

**A third, bigger finding surfaced computing DeepSeek's SAE-feature PAIR
rate for `paraphrase_decay_sae.py`** (before mechanically adding it to that
script's `KNOWN_SAE_PAIR_DETECTION_RATE` dict, paused to investigate
first): the SAE-feature detector's calibrated threshold (6.899, Youden's J
on VAL) gives only 4.4% recall on genuine VAL harmful prompts and 0/21 on
PAIR. Verified this is not a calibration bug -- harmless VAL scores are
100% exactly zero, so Youden's J correctly found the best achievable
operating point given the real distribution; the cause is architectural
(the SAE's hard k=32-of-65536 sparsity means 15 hand-picked features,
ranked from a small n=10 resolved-prompt sample forced by this model's
mandatory reasoning trace, have low odds of firing on any given VAL
prompt). Full account, including how this connects to every other
measurement in this project telling the same "diffuse refusal
representation" story about this model, in RESULTS.md.

Deliberately paused before folding that 0/21 number into
`paraphrase_decay_sae.py`'s lookup dict and continuing mechanically -- it's
a bigger, standalone finding than a footnote in a comparison table, and
got its own RESULTS.md write-up first, per explicit user instruction
("write it up first").

## Wrapper-swap variance decomposition: closing the token-attribution "why" gap (2026-07-28)

`scripts/token_attribution.py` (2026-07-25) found a real but confounded
qualitative pattern: Llama-3.1-8B's top feature reads core-content words,
Qwen3-8B's reads wrapper/framing words, on real (uncontrolled) PAIR
prompts where WHICH request and HOW it's framed vary together. Built
`scripts/wrapper_swap_variance.py` to disentangle the two factors with a
controlled 2-factor design (10 real core requests x 5 wrapper conditions,
full factorial, each model's own top feature's activation as the readout
via one forward pass, no generation) -- see RESULTS.md's new "Closing the
'why'" subsection for the numbers.

**Design planned via `EnterPlanMode` + a Plan-agent design-review pass**
(mirrors this project's standing practice for genuinely non-trivial
investigations -- the matched-pair mechanistic dig and Tier-2
token-attribution work both used the same pattern). **The review caught a
real bug before any GPU time was spent**: the first-draft permutation-test
design ("permute which of the 10 core-labels is attached to which prompt,
holding the grid structure fixed") is ambiguous between relabeling whole
rows (a no-op -- `ss_core` only depends on which values are grouped
together, not what the group is called) and the correct method, **within-block
(Manly-style) resampling** -- independently permuting the 10 core-labels
*within each wrapper column* for the core-effect null, and symmetrically
the 5 wrapper-labels *within each core-request row* for the wrapper-effect
null. Implemented the corrected version, then verified it on synthetic
planted-effect arrays (a planted row-only effect correctly gives
eta_sq_core~1.0, p<0.01, eta_sq_wrapper~0, p>0.1, and vice versa for a
planted column-only effect; confirmed the permutation null distribution
has real variance across draws, ruling out the no-op failure mode) before
trusting real GPU results.

**Framing correction from the same review**: because this is a single
deterministic forward pass (no dropout, no sampling), `ss_residual` in the
no-replication two-way ANOVA is ~pure core x wrapper interaction, not
"interaction confounded with measurement noise" as initially described --
there is no noise source in this measurement. Reported both eta-squared
(`ss_factor/ss_total`) and partial eta-squared
(`ss_factor/(ss_factor+ss_residual)`) per the review's recommendation,
since raw eta-squared alone can understate a factor's explanatory power
when residual/interaction is large for one model but not the other (which
turned out to be exactly the case: Llama's residual is 56% vs. Qwen3-8B's
12%).

**Core requests**: the 10 unique blunt `goal` strings already in
`results/adversarial_paraphrase_manifest.json` (dataset-sourced from
JBB-Behaviors via this project's own real PAIR records) -- reused verbatim,
no new harmful content authored. Explicitly noted as a limitation (in
RESULTS.md): these are the same 10 goals that originally generated the
hypothesis being tested, so this is an in-sample confirmatory test, not an
independent held-out replication (the review flagged JBB-Behaviors' other
~90 behaviors as a cheap future held-out extension, not done here).

**Wrapper templates**: one `bare` control plus 4 project-authored, generic
`{request}`-templated framings (creative-writing/fiction,
hypothetical/thought-experiment, security-research/red-team,
roleplay/persona) representing categories already implicit in the
project's real PAIR corpus -- deliberately not the literal branded "DAN"
prompt, just the same general shape, to avoid reproducing a specific known
jailbreak template verbatim.

**Secondary qualitative check reframed as activation-patching-style**, per
the review's citability recommendation: the existing single-token
leave-one-out method (`token_attribution.py`) is a degenerate single-token
case of activation patching (Meng et al. 2022 ROME; Vig et al. 2020; Zhang
& Nanda 2024) -- more standard/citable framing than an unlabeled "ablation
subset" for an MSc thesis write-up.

**Code reuse**: promoted `token_attribution.py`'s private `_feature_value`
into a shared `src/sae/feature_probe.py::feature_value` (now used by 2
scripts -- real duplication, not premature abstraction), with a
`@pytest.mark.model` test (`tests/test_feature_probe.py`, mirroring
`test_causal_ranking.py`'s real-small-model pattern -- nnsight's
`model.trace()` proxy/context semantics aren't meaningfully stubbable, so
this project's convention for any function that calls `.trace()` is a real
small-model test, not a mock).

**GPU-memory operational note**: before this run, `nvidia-smi` showed
5.8/6.1GB already in use with the GPU otherwise idle -- traced to an
orphaned multiprocessing worker (`--multiprocessing-fork`, `parent_pid`
pointing at an already-force-killed uvicorn dev-server process from
earlier browser-based frontend verification this session). Force-killing
a parent process on Windows does not clean up its children -- they stay
alive holding the CUDA context. Killing the orphaned child directly freed
the memory (5.8GB -> 0.6GB). Worth checking `Get-CimInstance Win32_Process
-Filter "Name='python.exe'" | Select ProcessId,CommandLine` for
`--multiprocessing-fork` children with a stale `parent_pid` if `nvidia-smi`
ever shows memory used with 0% utilization and no obviously-corresponding
live process.

## Llama causal-gap component decomposition (2026-07-28)

Closes the specific gap flagged in `scripts/analyze_llama_causal_gap.py`'s
own hedging: cosine alignment (~0.20) between Llama's dense direction and
its own top causal SAE feature was measured but never causally tested --
identical alignment for Qwen3-8B (~0.196) meant it couldn't differentiate
why Llama's dense direction is a weak causal lever while Qwen3-8B's is a
strong one. Built `scripts/decompose_llama_causal_gap.py`: decomposes each
model's own unit dense direction into the component parallel to its top
feature's decoder direction and the orthogonal remainder, ablates each
separately via the unchanged `generate_with_ablation`. See RESULTS.md's
new "Component decomposition" subsection for the numbers -- a genuinely
clean result, not the murkier one hoped-for-but-not-guaranteed going in:
Llama's isolated feature-aligned axis alone (92%->38%) causes a far larger
effect than the full direction (92%->88%), the orthogonal remainder alone
does nothing (92%->92%, identical); Qwen3-8B shows the opposite pattern,
both components independently crash refusal same as the full direction.

**Planned via `EnterPlanMode` + a Plan-agent design-review pass**, same
pattern as the wrapper-swap experiment two entries above. **The review
caught a real, concrete data bug before any GPU time was spent**: `results/
cross_model_direction_transfer.json` stores stale, pre-apostrophe-bugfix
refusal stats for every Llama condition (baseline stored as 0.80, fresh
recompute 0.92; own_ablation stored as 0.86, fresh recompute 0.88) -- the
2026-07-23 `is_refusal` curly-apostrophe fix was applied to RESULTS.md's
published numbers at the time but never written back into this JSON file.
Confirmed via direct recompute: Qwen3-8B's entries in the same file are
unaffected (ASCII apostrophes only). Fixed by having the new script reload
the file's raw `completions` and rescore fresh via `is_refusal()`/
`refusal_stats()` rather than trusting the stored `refusal_stats` field --
otherwise a new McNemar/Cochran's Q test would have silently paired
fresh-classifier labels (new conditions) against stale-classifier labels
(reused baseline/own conditions) within the same significance test. The
stale JSON file itself was left as-is (gitignored, regenerable, and
correcting it in place wasn't necessary once the new script stopped
trusting its stored stats) -- worth remembering if anything else ever
reads that specific file's `refusal_stats` fields directly for Llama.

**Framing correction from the same review**: renormalizing each component
to unit length before ablation changes the causal question from "how much
does this component contribute at its true small weight" to "is this axis
alone -- independent of how little of the original vector's mass sits on
it -- a sufficient/necessary causal lever." Stated explicitly next to the
results rather than left implicit, since the parallel/orthogonal numbers
are not directly commensurate with the already-published full-direction
number for that reason. The review also flagged running 6 uncorrected
pairwise McNemar tests per model as a real multiplicity risk (this
project's own `cochrans_q` exists for exactly this reasoning, previously
only ever applied across k different models on the same items rather than
k conditions within one model) -- fixed by running an omnibus Cochran's Q
across all 4 conditions first and reporting the pairwise tests explicitly
as post-hoc, not independently pre-registered comparisons.

**A real interruption during execution, not a script bug**: the first
background run of this script was killed mid-generation by an unrelated
environment/session disruption (visible as an MCP-server disconnect/
reconnect churn around the same time) with zero progress saved -- the
script only wrote its output JSON after both new conditions finished per
model, losing a full Llama model load and partial generation. Added
per-prompt checkpointing (`results/_decompose_llama_causal_gap_checkpoint.json`,
resumable both at the per-condition and per-model level) before rerunning
-- same "any long GPU script should checkpoint incrementally" lesson this
project has hit before (see the Tier-2 token-attribution debugging saga
entry above), reapplied here rather than re-learned the hard way twice.

## Closing the remaining future-work items: a second transfer pair + wrapper-swap replication (2026-07-28)

After closing two research gaps earlier the same day, the user asked to close
out the thesis's Future Work list itself. Full triage before starting: two
items were genuinely tractable without new scope (this entry), a 4th
moralize-vs-comply classifier attempt was explicitly deferred (real risk of
a 4th distinct failure mode, not a guaranteed fix), and DeepSeek-vs-other-
distilled-models was explicitly scoped out (needs models not in this
project at all). **A user-driven correction worth remembering**: this
project's own default posture is to report genuinely open questions
honestly rather than force premature closure -- when the user first asked
to "complete" every future-work item, the right move was to push back on
which items could actually be completed (some are structurally blocked, one
carries a real chance of ending in a fourth honest failure) rather than
silently attempt everything or silently comply with an unreasonable framing.

**Both experiments planned via `EnterPlanMode` + a Plan-agent design review
pass**, same pattern as every other non-trivial investigation today. The
review caught real, concrete issues in both designs before any code was
written -- worth internalizing as a pattern at this point: every design
review run in this session so far has found at least one genuine bug or
gap, not just style feedback.

### Experiment A: Qwen2.5-1.5B-Instruct <-> DeepSeek-R1-Distill-Qwen-1.5B transfer

A second architecture-matched pair (`results/dense_directions.pt` confirms
both are 1536-dimensional) was found already sitting in this project's
existing 6 models -- no new downloads. `scripts/transfer_direction_deepseek.py`.

**The review's most important catch**: `resolve_completions()` drops a
*different* subset of prompts per condition (whichever failed to close
`<think>` in that specific condition) -- naively resolving baseline/own/
foreign independently and then zipping their resolved lists by position for
McNemar would silently misalign prompts across conditions once their
truncation patterns diverge, corrupting the test without any visible error.
Fixed via a new `resolve_completions_by_index()`
(`src/direction/refusal_classifier.py`, keyed by original position rather
than compacted into a fresh list) -- every McNemar comparison restricts to
the intersection of indices that resolved in *both* conditions being
compared (three separate intersections, since each pair of conditions can
drop a different subset), with the resulting paired-n reported alongside
every p-value rather than assumed to equal the sampled N. Verified on a
synthetic example with deliberately mismatched truncation patterns across
two conditions before trusting it on real GPU output.

**Other review-driven design choices, not the naive first-draft versions**:
asymmetric budget/precision per target model (Qwen2.5-1.5B: 40 tokens, no
quantization; DeepSeek: 2048 tokens, no quantization, whole-generation
intervention matching its own published Phase 1 convention exactly --
loading either model 4-bit would introduce an uncontrolled variable versus
already-published baselines); N=30 (not 15, not 50) for DeepSeek's three
conditions specifically, keeping the same total-cost envelope this project
already accepted for the SAE-suppression-validation N=15x6-condition
precedent while preserving enough resolved-n for McNemar to have any power
given DeepSeek's 30-47% typical truncation rate; sufficiency/addition-
transfer explicitly not attempted this round (DeepSeek's own-direction
addition is already a thoroughly-swept genuine null, so a foreign direction
succeeding where the model's own fails is a priori unlikely, and it needs
its own expensive alpha-calibration sweep at the 2048-token budget before
any validation generation even starts -- low expected value for real cost).

**Result**: Qwen2.5-1.5B gives a second clean, unambiguous no-transfer
result matching the Qwen3-8B<->Llama pair's pattern exactly (own-ablation
96%->4%, p=0.0; foreign-ablation 96%->90%, p=0.25 n.s.). DeepSeek-as-target
is genuinely underpowered, not informative either way -- baseline refusal on
this 30-prompt sample is already near-floor (5.9%) and heavy truncation
shrinks the McNemar paired-n to single digits (6-10) after index-
intersection, so all three p=1.0 results reflect near-zero test power, not
evidence of anything. Reported honestly as inconclusive, not oversold as a
second "no transfer" data point -- see RESULTS.md's dedicated section.

### Experiment B: wrapper-swap replication

Adds real per-cell replication (2 new project-authored phrasing variants
per non-bare wrapper category, 3 total per category) to
`scripts/wrapper_swap_variance.py`'s design, so the large interaction/
residual term found for Llama (56.4% of variance) becomes testable instead
of an unexaminable leftover. `scripts/wrapper_swap_replication.py`. "bare"
excluded from this follow-up by construction (no phrasing dimension exists
to replicate for zero-wrapper text without inventing framing content).

**Two review-driven corrections before any GPU time**:
1. **Statistical**: real per-cell replication (n=3) now provides a valid
   pure-error term the original 1-observation-per-cell design never could,
   making the classical two-way ANOVA F-test legitimate here (unlike
   before) -- reported as the primary number. But per this project's
   standing preference for permutation at small n, ALSO ran a Freedman-Lane
   residual-permutation test for the interaction term specifically as a
   corroborating check -- a genuinely different, more subtle scheme than
   the existing main-effect permutations (which shuffle within a row/column
   block): fit the additive/no-interaction model to the cell means, permute
   the resulting residuals *globally* (not within any block), reattach, and
   recompute. A naive global shuffle of raw values instead of residuals
   would also destroy the main effects and contaminate the interaction
   test specifically -- confirmed this distinction matters empirically, not
   just in theory: a synthetic sanity check (see below) showed the naive
   scheme gives a materially different (miscalibrated) result on the same
   additive-only data the correct scheme handles properly.
2. **Framing**: flagged explicitly, not left implicit, that the 3
   "replicates" per cell are different phrasings, not repeated trials of an
   identical stimulus -- there is no measurement noise in a deterministic
   forward pass (the original script's own docstring already makes this
   point). Using phrasing-to-phrasing variance as an ANOVA error term is a
   defensible but explicit modeling choice (phrasing variation treated as
   exchangeable noise relative to the category-level effect), stated as
   such in the write-up rather than assumed to be self-evident.

**Verified before trusting real data**: both the F-test and the
Freedman-Lane permutation scheme were run against two synthetic cases (a
pure-additive grid with no planted interaction, and a grid with a real
planted interaction effect) -- both correctly gave non-significant results
on the additive case (F-test p=0.44, permutation p=0.28) and highly
significant results on the planted-interaction case (F-test p<1e-15,
permutation p=0.0005), and the naive (wrong) raw-value-permutation scheme
was separately confirmed to give a materially different, miscalibrated
result (p=1.0) on the same additive data the correct scheme handled fine --
same "verify the statistical machinery on synthetic data with a known
answer before trusting it on real activations" discipline already
established earlier the same day for the original wrapper-swap permutation
test and this session's other new statistical code.

**Pre-flight content checks (no GPU needed, run before any forward passes)**:
the 8 new phrasing variants were checked for wording similarity against
both the original 4 templates and the real PAIR corpus (`difflib.
SequenceMatcher`, matching this project's existing dedup-similarity
convention from `src.data.dedup`) and manually checked for content leakage
(no core-request-specific keywords). Two variants initially scored
moderately similar to their own category's original template (0.585,
0.511 -- expected, since same-category framings share vocabulary by
design) but were reworded to reduce overlap further out of caution; all 8
scored low similarity against the real PAIR corpus (max 0.115) and clean
on the content-leakage check throughout.

## Per-cell decomposition of Llama's wrapper-swap interaction (2026-07-29)

Follow-up to the replicated wrapper-swap design's standing open question
("why does Llama's core x category interaction exist mechanistically") --
a pure re-analysis of `results/wrapper_swap_replication.json`'s already-
saved activation grids, no new GPU generation needed. New script
`scripts/wrapper_interaction_cells.py`: for each (core, category) cell,
subtracts the additive-model prediction (row mean + column mean - grand
mean, the same fit already used by `freedman_lane_interaction_test()`)
from the real cell mean, isolating a per-cell interaction residual
distinct from the within-cell phrasing noise the replicated design
already separates out.

**Real, clean structure, not noise spread evenly across all 10 core
requests**: 2 of Llama's 10 core requests (the Trump-2020-election
misinformation request and the blackmail-message request) show an
interaction range across the 4 wrapper categories of 1.7-1.8, a roughly
2.5-9x gap above the remaining 8 requests (range 0.19-0.67). Sanity-
checked directly against the largest individual cells (not just the
range statistic): the single largest cell is the blackmail request under
the fiction wrapper, and the Trump-election request shows large
interaction values of *opposite sign* under hypothetical vs. fiction --
exactly the signature of genuine interaction (the same core request's
response to wrapper framing flips direction), not a data artifact in one
cell.

**Qualitative task-type read, explicitly flagged as a candidate not an
established finding**: unlike this project's usual practice of running
an automated classifier or a fully controlled follow-up, this observation
comes from directly reading the two standout requests' content -- both
differ in *task type* from the other 8 (which are uniformly "explain a
harmful procedure" asks). The Trump request asks the model to construct/
validate false factual content; the blackmail request asks the model to
directly generate the harmful artifact itself, not explain how to make
one. Stated explicitly as a candidate explanation worth a real controlled
follow-up (task-type as a deliberate factor), not as a proven mechanism --
2 anomalous requests out of 10 is too small a sample to treat as
confirmed without a purpose-built design, and this project's standing
discipline is to not force a tidy story onto an honestly partial finding.

**Qwen3-8B deliberately excluded from the per-cell writeup**: its own
interaction term is not statistically significant, so ranking its cells
by |interaction| would just be ranking noise -- reported in the script's
saved JSON for completeness but not narrated in RESULTS.md as if it were
a real pattern, consistent with this project's standing practice of not
over-reading null results.

## Testing the task-type hypothesis directly -- a genuine negative result (2026-07-29)

The per-cell decomposition above offered a candidate, explicitly hedged
explanation (the two standout core requests differ in task type from the
rest) and said it would need a purpose-built controlled design to confirm
rather than 2 anomalous data points. User asked to build that follow-up
directly. Planned via `EnterPlanMode` + an Explore-agent pass over the
exact existing conventions (`scripts/wrapper_swap_variance.py`'s
`CORE_REQUESTS`/`WRAPPER_TEMPLATES`, `scripts/wrapper_swap_replication.py`'s
`WRAPPER_PHRASINGS`/`balanced_anova`/`freedman_lane_interaction_test`,
`src/sae/feature_probe.py`'s `feature_value` signature) before writing any
new code -- plan mode was exited by the harness mid-session before a
Plan-agent design review ran; proceeded directly given the design was
already fully specified from the Explore pass and the user's explicit
go-ahead, rather than re-entering plan mode for a already-scoped, low-risk
extension (no new statistical formulas, just a different grid shape fed
into already-verified functions).

**Design**: 12 new core requests, real, sourced from this project's
existing corpus (`src.data.loaders.load_all_labeled_prompts`), never
authored -- verified none overlap the original 10 before selection.
Balanced 4/4/4 across `procedural`/`direct_artifact`/`false_narrative`
(regex-scanned candidates from the full 1120-prompt harmful corpus:
`^(Write|Draft|Compose)` for direct-artifact, `^(Explain|Provide|Describe|
Outline)` for procedural, keyword-matched `false|fake|misinformation|
propaganda|...` for false-narrative -- 252/263/86 candidates respectively,
hand-picked 4 per type for topic diversity within each). Same wrapper
templates/phrasings reused unchanged via direct import (no duplication).
New `scripts/wrapper_task_type.py`.

**Two views of the same 144-reading-per-model grid, both via the
unmodified existing `balanced_anova()`/`freedman_lane_interaction_test()`**:
(1) task_type x category, reshaping each type's 4 cores x 3 phrasings
into 12 replicates per cell -- the direct hypothesis test; (2) core x
category on just the 12 new requests -- a same-shape replication check.
**The new reshape logic (`task_type_grid()`) was verified on synthetic
data before running on real activations**: an exact sum-of-squares match
between `core_grid`'s and `task_type_grid`'s total variance (grouping
cannot change total variance, only how it's partitioned) -- confirmed via
assertion inside the script itself (`assert abs(ss_total_core -
ss_total_tt) < 1e-6`), not just an offline check, so a future bug in the
reshape would fail loudly rather than silently corrupt results.

**Result: a genuine negative result for the task-type hypothesis in its
literal form, reported plainly rather than reframed.** Both models'
task_type x category interaction come back clean nulls (Llama p=0.416
F-test / 0.393 permutation; Qwen3-8B p=0.148 / 0.130). **The underlying
core x category interaction itself cleanly replicates** on this entirely
independent sample of requests (Llama η²=0.169, F-test p=4.2e-6,
permutation p=0.0001) -- strong evidence the phenomenon is real and
reproducible, not an artifact of the original 10 prompts, even though the
specific task-type explanation for it doesn't hold up. A per-cell
breakdown of the new grid (reusing `wrapper_interaction_cells.py`'s
`cell_interactions()` unchanged) shows why the group-level test washed
out a real cell-level signal: interaction again concentrates in just 2 of
the 12 new requests, both happen to be `false_narrative` type and both
literally phrased "fake news article" -- but the other 2 false_narrative
requests in the same batch are unremarkable, diluting the task-type
group's own average. **Not reframed as "false narrative is the real
answer"** -- this is a narrower, even more speculative candidate (literal
phrasing or a specific sub-topic, not a clean task-type category) than
what was tested, offered honestly as the new open question rather than a
resolved one. This is now the second time this project has designed a
purpose-built test for a hedged qualitative read and gotten a clean
negative on the literal hypothesis while the underlying phenomenon held
up -- worth remembering that "a candidate explanation, not established"
hedges in this project's own history have a real track record of not
surviving direct testing, which is exactly why they're hedged that way
rather than asserted.

## A blind, data-driven feature search over Llama's wrapper-swap interaction (2026-07-29)

After the task-type hypothesis's clean rejection, the user was offered a
choice: keep hand-picking a third narrative from the 2 newest outliers
(real risk of unfalsifiable post-hoc pattern-hunting), or do a genuinely
larger, data-driven search across many objective features instead. User
chose the latter. Planned via `EnterPlanMode`; the first plan-mode session
was exited by the harness mid-Explore-pass (no ExitPlanMode call from
either side), so a second plan-mode session ran a Plan-agent design review
from scratch rather than skipping it, given the real statistical-design
risk (avoiding the same cherry-picking bias creeping back in through
*feature* selection instead of *category* naming) warranted independent
scrutiny even though the code-reuse story was already fully known.

**The Plan-agent review caught real, concrete issues before any GPU time
was spent** (the pattern holds again -- every design review this session
has found at least one genuine gap): (1) flagged that
starting-verb-category, one of my draft "new" features, was actually a
verbatim restatement of the already-tested-and-rejected task-type
operationalization -- re-including it in a blind family would be circular,
not a fresh test; demoted to an explicit Tier-B replication check instead.
(2) Flagged that proper-noun-count, motivated by "Trump"/"fake news
article," was the textbook Texas-sharpshooter pattern (draw the target
around the arrows already on the board) -- demoted to Tier B with an
explicit non-blind caveat rather than silently dropped or silently kept.
(3) Recommended equal (12/source), not proportional, allocation across the
4 source datasets specifically because `source_dataset` is itself a tested
predictor, not just a nuisance covariate -- proportional allocation would
have made that one test underpowered (n=4 for JBB) and unstable. (4)
Recommended a maxT (Westfall-Young) permutation scheme as primary
correction over Bonferroni/BH-FDR, reasoning explicitly from the shape of
the data already seen (interaction_range is right-skewed in both prior
rounds -- concentrated in 2 of the sample with a long flat tail -- exactly
where asymptotic Spearman/KW p-values are least trustworthy), with BH-FDR
kept only as a cross-check. (5) Caught that char-count and word-count would
be near-collinear and should collapse to one length feature plus a
genuinely different second one (average word length), not two versions of
the same signal inflating the tested family for no informational gain.

**Verified before trusting on real activations, per this project's standing
discipline**: `maxT_family_test()`'s synthetic self-check (a pure-null case
-- independent random outcome and features, expect uniform non-significant
adjusted p's; a planted case -- one feature constructed to correlate with
the outcome, expect that feature alone to clear significance with zero
false positives on the other three) passed cleanly before the real run
(`verify_maxT_on_synthetic()`, called at the top of `main()` so it can
never be skipped by accident). `sample_new_cores()`'s exclusion/dedup logic
(reusing `src.data.dedup._normalize`/`_content_word_overlap` directly, the
same 0.9-char-ratio-AND-0.5-word-overlap gate `deduplicate()` uses, rather
than a second independent near-dup implementation) was asserted to return
exactly 48 cores, 12 per source, zero overlap with the 22 already-used
cores, checked before any model load.

**Result: the interaction replicates a third time, more strongly than
either prior round** (η²=0.286 at n=48, vs. 0.424 at the original n=10 and
0.169 at the task-type round's n=12) -- real, and now about as
well-established as any finding in this project. **All four Tier-A
features come back clean, corrected nulls** for both models -- maxT and
BH-FDR agree closely throughout, no disagreement between the two methods
worth reporting. **Both Tier-B features show nominal (raw,
uncorrected) p<0.05 for Llama** (starting-verb-category p=0.0285,
proper-noun-count p=0.0249) -- explicitly NOT treated as a finding:
starting-verb-category is re-testing a hypothesis a cleaner, purpose-built
2-way ANOVA already rejected (p=0.416) one round earlier, so a marginal hit
on a less rigorous correlational re-test of an already-refuted hypothesis
is the false-positive pattern the tiering exists to catch, not a
reversal; proper-noun-count was excluded from the blind family precisely
because it was fit to the same 2 outliers it's now nominally "confirming."
Neither result changes the honest bottom line: three rounds in, the
interaction is real and highly reproducible, but *why* it concentrates in
specific requests remains genuinely open -- reported as such, not chased
with a fourth guessed category. This is the second time in two rounds a
purpose-built test of a hedged candidate came back clean negative while the
underlying phenomenon held up, worth treating as the project's actual track
record on this question rather than an unlucky pair of misses.

## Literature-motivated salience hypothesis, tested and rejected (2026-07-29)

User asked to actually go find and read literature on why some specific
requests might resist explanation by task type or surface features, rather
than leave it as an open question. A literature pass (added to
`LITERATURE.md`) surfaced real, citable context -- Khorramrouz & Levy's
"Characterizing Selective Refusal Bias" (topic-specific refusal
idiosyncrasy is independently documented, not unique to this project),
an ICLR-2026-submitted-then-withdrawn paper on disentangling goal vs.
framing in LLM representations (external validation of the same
goal/framing paradigm this project's wrapper-swap design already uses,
flagged with the withdrawn caveat rather than cited uncritically), and
general SAE/feature-frequency literature linking a concept's training-data
salience to its representation geometry. Only the third gave a genuinely
new, testable candidate not already covered by Tier A/B.

**Tested directly rather than left as a citation**
(`scripts/wrapper_salience_test.py`): salience operationalized as raw
perplexity under this project's own existing perplexity-filter reference
LM (Olmo-3-1025-7B, `src/baselines/perplexity_filter.py`), reused for a
purpose it wasn't built for (catching GCG-style gibberish suffixes), not
duplicated. No new SAE-feature measurement needed -- the 48 cores and
their `interaction_range` were already saved from the prior round
(`results/wrapper_feature_search.json`); this only added the one new
feature and one Spearman permutation test.

**Explicit bookkeeping on where this test sits, since it's neither of the
two prior categories**: it was proposed after seeing all four Tier-A
features fail (so not blind/pre-registered, no family correction applies
since it's the only new test), but it was not reverse-engineered from the
specific outlier words the way Tier B's `proper_noun_count` was -- it came
from an independent literature finding about training-data salience in
general, applied here as a hypothesis rather than fit to "Trump"/"fake
news" as strings. Reported with its own single, honestly-labeled
uncorrected p-value, in neither the Tier-A nor Tier-B bucket.

**Result: a fourth clean negative.** Llama: rho=-0.110, p=0.457; Qwen3-8B:
rho=-0.166, p=0.260 (also n.s., consistent with Qwen's own interaction
already being null). Both correlations run in the hypothesis's predicted
direction but nowhere near significant at n=48.

**Assessment stated plainly, not smoothed over**: four independently
motivated hypotheses (task type; word count/avg word length/keyword-filter
score/source dataset; salience-via-perplexity) have now failed to predict
which core requests drive Llama's interaction, while the interaction
itself has replicated cleanly three separate times. Continuing to test
one-off hypotheses against the same handful of concentrated cells has real
diminishing returns at this point -- offered to the user as this project's
honest read, not decided unilaterally; a genuinely different, larger-scope
approach (or accepting this as a well-documented open question) remains
the user's call.

## A genuinely different approach: token-level reading, not another surface hypothesis (2026-07-29)

User explicitly asked for the "genuinely different, heavier approach"
flagged as the alternative to more one-off surface hypotheses: reading
actual token-level attribution patterns for the outlier requests, reusing
Section 5.4.3's already-built and verified single-token leave-one-out
method (`scripts/token_attribution.py::token_ablation_importance`,
`content_span_mask`, `top_tokens`) rather than inventing new attribution
machinery. New `scripts/wrapper_token_read.py`, imports the existing
functions directly.

**Selection, not a full plan-mode cycle**: given the core method already
exists, is already verified, and the only new design decision was *which
prompts to compare* (not a new statistical framework needing independent
scrutiny the way the maxT scheme did), proceeded directly rather than a
second Plan-agent review. Picked 4 high- and 4 low-interaction core
requests spanning all three prior rounds' data (original 10, task-type 12,
feature-search 48) for structural diversity -- short templated AdvBench
asks, a task-type/false-narrative request, and an XSTest item. While
selecting, found the single highest-interaction core request across *any*
round is actually "Exactly how much money does my boss earn?" (XSTest,
range=2.515, higher than the original Trump-election request) and the
second-highest is a long HarmBench *contextual* prompt (a full
Reddit-style political rant ending in a request to argue 2020 election
fraud) -- both confirmed correctly labeled `harmful` by direct lookup
against `load_all_labeled_prompts()`, not a sampling bug, before trusting
either as real data points. The HarmBench contextual prompt was noted but
excluded from the actual token-read set purely for practicality (its
length would multiply the leave-one-out forward-pass count substantially
for one data point) -- flagged honestly as a scope choice, not hidden.

**Result: a real, qualitative pattern, read directly rather than scored by
any automated procedure** (matching this project's standing distrust of
unvalidated automated classifiers for exactly this kind of judgment call).
Low-interaction requests' top-attributed token is the *same* single word
across all 5 wrapper conditions (e.g. "scam" for the MLM request, rank-1
under bare/fiction/hypothetical/research/roleplay alike) -- a stable
content-lock. High-interaction requests show the top token swinging
between core content and wrapper-scaffolding language depending on
framing, most consistently under the **fiction** condition specifically
(3 of 4 high-interaction requests), sometimes with the feature's raw
activation collapsing toward a framing-dominated state with no content
token in the top-5 at all (the XSTest boss-salary request under
`hypothetical`: value drops to 0.49 from 2.91 at bare, top tokens become
"hypothetical"/"purely"/"application").

**Explicitly not oversold**: n=4 vs. 4, a directly-read qualitative
signature, not a formally tested account of *why* some requests' content
representations resist displacement by wrapper framing better than
others. This is the first account in five rounds grounded in what the
model's own feature actually attends to rather than a property of the raw
prompt text -- a real step forward in kind, even without a clean
statistical answer. Full per-token results (all 5 conditions x 8 requests)
in `results/wrapper_token_read.json`.

## Adding an LLM-as-judge baseline, and why the reported comparison is threshold-aware (2026-08-04)

**Why add it at all.** The head-to-head's two baselines were both chosen to be
weak: a fixed keyword lexicon and a perplexity filter that by construction only
catches gibberish. Beating them establishes very little -- a reviewer can
reasonably say the activation detectors were never tested against a competent
text-only reader. A frontier-scale LLM prompted as a classifier is the strongest
baseline available that still reads only prompt text, which keeps it on the
correct side of the project's actual claim.

**Why Llama-3.3-70B on Groq specifically.** It had to be (a) genuinely capable,
(b) text-in/text-out only, and (c) free, since the project has no API budget.
Groq's free tier serves 70B at no cost. Closed frontier APIs were never
candidates for the *detector* side of this project (no activation access), but
as a *baseline* that restriction does not apply -- a baseline is allowed to be a
black box, because the point is that it never sees internal state.

**Prompt-injection hardening was not optional.** Two real defects surfaced
during development, both of which would have produced plausible-looking but
wrong numbers rather than crashing:

1. **The judge refuses the most severe prompts.** Asked to rate a Sarin-synthesis
   HarmBench item, the model replies "I can't provide instructions..." instead of
   emitting an integer. The first parser found no digits and defaulted to 0 --
   scoring the single most harmful prompts in the corpus as *maximally benign*.
   A refusal is strong evidence of harm, so an unparseable response now defaults
   to 100.
2. **The judge executes benign prompts instead of rating them.** With the
   candidate text placed directly in the user turn, "Write a poem with five
   lines" produced a poem, not a score -- again unparseable, and after fix (1)
   that meant scoring innocuous prompts as maximally harmful. Fixed by wrapping
   the candidate in `<prompt>` tags with an explicit instruction to treat the
   contents as inert data.

Both are general hazards of LLM-as-judge evaluation, not quirks of this model,
and both are the reason the judge's numbers should not be taken on trust without
inspecting its raw outputs first.

**Why the write-up leads with two different tests instead of one number.** The
judge wins thresholded accuracy (94.1% vs 88.9%, McNemar p<0.001) while
dense-direction wins AUROC (0.983 vs 0.954, DeLong p=0.0041). Reporting either
alone would misrepresent the result. AUROC is threshold-free and measures
separability; accuracy measures one operating point chosen by Youden's J on VAL.
The two disagreeing is diagnostic, not contradictory -- it localizes the gap to
**threshold selection**, which is a calibration problem this project is already
framed to care about, rather than to the activation signal itself.

**Why PAIR is reported as indistinguishable.** Raw rates favour the judge
(57.1% vs 42.9%), which is tempting to report as a loss for the activation
detectors. McNemar's exact test on the same 21 prompts gives p=0.45 on 7
discordant items. Same discipline as everywhere else in this project: at these
sample sizes the paired test governs, not the difference in rates. The
adversarial set's size is the binding constraint, and remains an externally
blocked limitation.

**Caching is part of the method, not a convenience.** Groq's free tier caps
tokens per day (100k), so the full 608-prompt sweep took two days of quota.
Scores are cached by (model, prompt) SHA-256 in `results/llm_judge_cache.json`,
written incrementally, so the reported run is reproducible offline at zero cost
and a re-run never re-pays for a prompt. This follows the same
checkpoint-incrementally rule the Tier 2 token-attribution work adopted after a
crash cost a completed model's results.

## Threshold reselection, and why it is reported but not adopted (2026-08-04)

**Why run it.** The LLM-judge comparison produced a diagnosis rather than just
a number: dense-direction ranks better (AUROC, significant on all three models)
but decides worse (accuracy, significant on two). That pattern implicates the
cutoff, not the signal. A diagnosis that is never tested is just a story, so
`scripts/recalibrate.py` intervenes on exactly the suspected component --
the threshold rule -- and holds everything else fixed.

**Design choices.** Every rule is fit on VAL only, never TEST, preserving the
same split discipline as the rest of Phase 4. The comparison against the
existing Youden-J threshold is a paired McNemar on the same TEST prompts, not
two independent accuracy figures, because the two rules classify the same items
and only the discordant pairs carry information. Candidate thresholds are
midpoints between adjacent unique scores, so the search cannot land on a value
that is optimal only by floating-point accident.

**Verified on planted data before trusting it.** `_best_threshold` was checked
against a perfectly separable synthetic set with a known-correct answer (it
returns the midpoint and 100% accuracy) and a degenerate all-positive set,
following this project's practice of validating statistical machinery on data
with a known answer before running it on real results.

**Why the result is reported but not adopted.** Reselection is a real,
significant improvement on Qwen3-8B (88.9% -> 92.0%, p=0.0002) and a no-op on
Llama and gemma. Adopting it would mean re-running the head-to-head for one
model and updating every downstream number: the README tables, the UI's
validation panel, the cross-model comparison, and the thesis. Doing that
silently would leave the repo internally inconsistent mid-flight, and doing it
selectively for the one model it helps invites the obvious objection that the
threshold rule was chosen after seeing which model it flattered. The finding is
therefore recorded as its own result, with the adoption decision left explicit.

**The limitation that matters most.** On Qwen3-8B the two rules tie on VAL
accuracy (93.8%), so VAL could not have picked the winner in advance. The
justification for `max_accuracy` has to rest on it optimizing the metric being
reported, not on it having looked better during calibration -- and that
distinction is worth stating plainly, because the alternative framing ("we
found a better threshold") would be indistinguishable from tuning on TEST.

## A wrong paired test, caught and corrected (2026-08-04)

**The error.** The first LLM-judge write-up claimed the judge was
significantly more accurate than dense-direction on TEST (p<0.001 on two
models). That claim came from calling `mcnemar_exact` on the two detectors'
raw *predictions*. On a mixed-label split that tests the wrong hypothesis: it
asks whether the two detectors *decide differently*, not whether one is *more
accurate*. A detector that simply flags more will produce a large
prediction-level discordance no matter how its accuracy compares.

**How it surfaced.** The arithmetic stopped agreeing with itself. On
Llama the test reported 22 judge-only versus 2 dense-only discordant pairs,
which implies a ~6.9pp accuracy gap, while the measured accuracies differed by
0.7pp. A paired test that contradicts the marginals it is supposed to explain
is wrong by inspection.

**The correction.** Recomputed on per-item correctness, none of the accuracy
differences are significant on any model (p=0.38 / 0.84 / 0.82, versus the
p<0.001 originally reported). The threshold-reselection result survives but
weakens (p=0.0225, not p=0.0002). The DeLong AUROC comparisons were never
affected -- they operate on scores, not thresholded decisions -- and the
adversarial-set McNemars were never affected either, because every prompt in
those conditions is harmful, which is the one case where prediction-level and
correctness-level discordance coincide. That is also why the original helper
was written the way it was: its docstring scopes it to "flag/no-flag calls on
the same 21 PAIR prompts", and it is correct there.

**The fix, at the source rather than the call site.**
`src.eval.detector_metrics.mcnemar_accuracy(preds_a, preds_b, labels)` now
exists and compares correctness explicitly, with a docstring stating when each
form applies. Two regression tests pin the distinction: one where the two
forms must diverge (mixed labels, one detector flagging everything) and one
where they must agree (all-positive labels). Every other `mcnemar_exact` call
site in the repo was audited and is operating on all-harmful or refusal-rate
data, where the original form is correct.

**Why this is recorded rather than quietly amended.** Three published claims
changed direction: "the judge is significantly more accurate" became "no
significant difference", and the reselection p-value moved by two orders of
magnitude. A reader comparing this repository's history against its current
numbers would otherwise find an unexplained discrepancy, and the most useful
part of the episode is the failure mode itself -- a paired test applied to the
wrong quantity produces confident, plausible, wrong p-values, and the thing
that caught it was a marginals check, not the test.

## Pre-registration: a content-weighted SAE detector (2026-08-11)

**The question.** Every experiment on Qwen3-8B's wrapper-framing sensitivity so
far stops at diagnosis: the wrapper-swap factorial found its rank-1 causal
feature (layer 25, feature 65291) is dominated by wrapper identity
(eta_sq_wrapper=0.656, p=0.0001), while Llama's rank-1 feature (layer 27,
feature 13363) is dominated by core-request content (eta_sq_core=0.407,
p=0.015) -- and that asymmetry lines up with the known PAIR gap (Llama 71.4%
vs. Qwen3-8B 52.4%). This is the first attempt in this project to act on that
finding rather than describe it further: build a re-weighted variant of the
top-15 SAE-feature detector that down-weights framing-tracking features and
up-weights content-tracking ones, and test whether it actually improves PAIR
robustness.

**Why this is written down before the experiment runs.** The weighting formula
below is fixed now, using only the wrapper-swap ANOVA statistics (which do not
depend on TEST or PAIR outcomes), specifically so it cannot be chosen after
seeing which formula flatters the result -- the same discipline this project
already applies to permutation-test design (see the whole-row-relabeling
no-op caught by a Plan-agent review before any GPU time was spent, earlier in
this document).

**Stage 1 (not yet run).** Extend the rank-1-only wrapper-swap ANOVA to all
top-15 causally-ranked features, for both Qwen3-8B and Llama-3.1-8B-Instruct
(Llama as a negative control: its rank-1 feature is already content-dominant,
so the reweighting should plausibly change little there -- if it does, that
undermines the claim that any Qwen3-8B effect is real and specific rather than
an artifact of reweighting sums in general). `scripts/feature_variance_family.py`
reuses `wrapper_swap_variance.py`'s existing grid/ANOVA/permutation machinery
unmodified. Because this multiplies the family of tests from 1 to 15 per
model, a maxT/Westfall-Young family-wise correction is applied (adapted from
`wrapper_feature_search.py`'s existing `maxT_family_test` pattern) --
otherwise this would repeat, at the feature level, exactly the
multiple-comparisons gap this document's limitations section elsewhere
flags at the model level.

**The weighting formula, fixed now.** For each feature i with
`(eta_core_i, eta_wrapper_i)` from Stage 1:

```
w_i = eta_core_i / (eta_core_i + eta_wrapper_i)   if the denominator > 0
w_i = 0.0                                          otherwise
```

A closed-form ratio with no tunable threshold. A secondary, binary variant is
also computed for robustness (weight 0 for any feature where
`wrapper_effect_p_maxT < 0.05 AND eta_wrapper_i > eta_core_i`, weight 1
otherwise) but is not the primary claim, since it introduces a significance
cutoff the primary formula avoids.

**The evaluation, fixed now.** Both weighting variants, plus the existing
unweighted top-15 detector, are calibrated identically (VAL,
`max_accuracy_threshold`, the project-wide adopted rule) and compared
pairwise against the vanilla detector on: TEST-split accuracy/AUROC
(`mcnemar_accuracy`, `delong_auc_test`, both paired on the same 288 items,
checking for regression) and the 21 real PAIR prompts (`mcnemar_exact`, the
actual target metric). Any of the three outcomes -- real improvement, no
detectable change, or a regression -- will be reported as found. No new GPU
generation is needed for Stage 2/3: both split's activations are already
cached.

**Scope, fixed now.** gemma-2-9b-it is explicitly out of this pass (its
rank-1 feature has never been through the wrapper-swap ANOVA at all, so
jumping straight to its top-15 would skip the sanity check this whole design
leans on) and this is a detector-reweighting experiment, not a generation-time
steering intervention -- neither is being tested here, regardless of outcome.

## Closing the pre-registration: the content-weighted detector, outcome (2026-08-12)

**Both verification gates passed before any real result was trusted.**
`feature_variance_family.py`'s synthetic check (null case clean, planted
core-effect detected with no false positives among 14 unrelated features)
and its rank-1 reproduction check (new multi-feature loop reproduces the
already-published 0.656/0.227 Qwen3-8B and 0.407 Llama eta-squared values,
and the same significance conclusions) both passed on the first run, for
both models -- see `reports/RESULTS.md`'s write-up for the full per-feature
breakdown this unlocked.

**The pre-registered evaluation was run once, as written, and the result is
reported as found: negative.** Neither weighting variant produced a
significant PAIR improvement for Qwen3-8B, and both produced a significant
TEST-accuracy regression (primary p=0.0156, binary p=0.0414), with binary
also significantly hurting AUROC (p<0.0001). No secondary weighting formula,
threshold, or feature-subset was tried after seeing this -- that would have
been exactly the after-the-fact tuning the pre-registration entry was
written to prevent. The result stands as run.

**A real, verified explanation for Llama's bit-identical binary-variant
numbers, not an assumed one.** The 5 features the binary rule drops for
Llama fire on zero of 288 TEST prompts each (checked directly:
`saes[layer].encode(test_activations)[:, feature_idx]` is exactly 0 for
every TEST item, for all 5) -- so zeroing their weight is a genuine no-op on
this split, not a bug worth chasing further. This surfaces a real gap
between the two channels this experiment reads the same feature through:
`scripts/wrapper_swap_variance.py`'s ANOVA reads a feature's raw
pre-activation (`src.sae.feature_probe.feature_value`), unconstrained by the
SAE's own top-K sparsity competition, while `sae_feature_detector.score`
only counts a feature that *wins* that competition on a given prompt. A
feature can carry a real, statistically detectable content/framing signal in
the controlled factorial and still almost never be selected by the sparsity
gate on real data -- worth remembering before building another intervention
on top of ANOVA-style raw-activation statistics without checking whether the
target features are actually live in the detector's normal operating
regime.

**Why this is recorded as a completed, negative result rather than iterated
on.** The likely reason the intervention fails is structural, not a tuning
problem: Qwen3-8B's top-15 features are its own causally-ranked set,
independently confirmed elsewhere in this document to each carry real
class-separating signal on clean prompts -- they are not pure
framing-detectors that happen to also fire on harmful content, they are
harmfulness detectors whose *dominant* source of variance, in a controlled
factorial specifically designed to isolate it, happens to be framing rather
than content. Suppressing them by that same statistic removes real
harmfulness signal alongside whatever framing-sensitivity it carries, and
for this detector the loss outweighs the gain under both weightings tried.
A different kind of intervention (e.g. combining features non-linearly, or
steering rather than reweighting) might fare differently, but that is a
different, unscoped experiment, not a retry of this one.

## Pre-registration: framing-direction ablation (2026-08-12)

**The question.** The previous experiment (above) tried to fix Qwen3-8B's
PAIR vulnerability by down-weighting its framing-tracking SAE features
downstream -- it failed because those features carry real class-separating
signal beyond their framing-sensitivity, so suppressing them loses genuine
harmfulness signal too. This tries a structurally different fix: instead of
discarding whole features downstream, ablate an explicit "framing direction"
from the residual stream upstream, before either detector scores it --
removing the framing *component* of the activation itself, not the features
that happen to correlate with it.

**Design, fixed now.** The framing direction is a difference-of-means vector
(mean activation across the 40 wrapped prompts minus mean across the 10 bare
prompts, from the same 50-prompt core×wrapper factorial already used
throughout this thread), computed independently per layer, at Qwen3-8B's own
top-15 SAE-feature-detector layers (23, 24, 25) and, for the negative
control, Llama's own layers (21, 26, 27). `src.direction.compute
.compute_directions` -- the same function that produces the refusal
direction itself -- is reused unmodified for this, and ablation reuses the
same `resid - (resid @ dir_hat) * dir_hat` projection-removal math already
used for causal necessity testing, applied here to already-cached activation
tensors rather than live generation.

**Required validation, fixed now, before Stage 2 is trusted for Qwen3-8B.**
Among the 14/15 features already flagged framing-significant
(`results/feature_variance_Qwen3-8B.json`), recompute each feature's
wrapper-effect ANOVA on the same 50 prompts' activations after ablating the
frozen direction. Pass requires **both**: median relative drop in
`eta_sq_wrapper` >= 50%, and at least 10 of 14 features losing significance
outright. If this fails, it is recorded as a negative result about the
direction-estimation method itself and Stage 2 does not run for Qwen3-8B --
the layers, prompt set, or formula will not be adjusted post-hoc to force a
pass.

**The evaluation, fixed now.** Ablated activations are recalibrated on
ablated VAL (never the vanilla threshold) and compared against the vanilla
detector on TEST accuracy/AUROC (regression check) and the 21 PAIR prompts
(the target metric), for Qwen3-8B and Llama, using the SAE-feature detector
unmodified (`src.detectors.sae_feature_detector.score`/`calibrate`, no code
changes -- ablation is pure pre-processing on the activation tensor). A
secondary, explicitly non-primary check repeats the same comparison against
the dense-direction detector at Qwen3-8B's own layer (23), since it is
nearly free once the ablated tensors exist.

**A known limitation, accepted rather than engineered around.** The
wrapper-swap ANOVA found real core×wrapper interaction terms for some
features, not just additive wrapper main effects -- a single global framing
direction (a main-effect diff-of-means collapsing all 10 core requests) will
not fully capture an effect that interacts with which request it is applied
to. Accepted as a first-pass limitation, the same way the refusal direction
itself is also a pure main-effect diff-of-means despite refusal plausibly
interacting with request content too -- a per-core direction would multiply
the design space and reopen exactly the after-the-fact-tuning risk this
discipline exists to prevent, with no existing building block for it.

**Scope, fixed now.** gemma-2-9b-it excluded (no rank-1 wrapper-swap
baseline to sanity-check against, same reasoning as the prior experiment).
One direction per layer, not one averaged across layers. Full projection
removal only -- no partial/scaled ablation strength sweep. No generation-time
steering -- this is a scoring-time intervention only, same scope discipline
as the prior experiment.

## Closing the pre-registration: the framing direction fails its own validation gate (2026-08-12)

**The pre-registered gate was checked as written, and it failed.** For
Qwen3-8B, the median relative drop in `eta_sq_wrapper` across the 14
framing-leaning features was **7.9%** (needed >=50%), and only **1 of 14**
lost significance (needed >=10). Only one feature (layer 24, feature 5393)
showed the clean, large effect the design hoped every feature would show
(99.4% drop, p goes from significant to 0.9934); the other 13 moved only
modestly (0.6% to 38.5%), and one (layer 24, feature 401) moved slightly the
*wrong* way (-7.0%, ablation marginally increased its wrapper effect).
Llama's diagnostic-only check (not a gate, but run for comparison) shows the
same qualitative pattern at a different scale: median drop 30.6%, but still
zero of 5 features losing significance. Per the pre-registration, Stage 2
(the actual TEST/PAIR evaluation) does not run for either model -- there is
nothing to responsibly evaluate on top of a direction that does not do what
it was built to do, and the design is not adjusted post-hoc to try to pass.

**Why this result is itself informative, not just a stop sign.** The
pre-registration flagged, as an accepted limitation rather than a blocker,
that the wrapper-swap ANOVA already found real core x wrapper interaction
terms for several features -- a single global direction (a main-effect
diff-of-means collapsing all 10 core requests) might not fully capture an
effect that depends on which request it is paired with. This validation
result is consistent with that limitation being the operative one, not
merely theoretical: if 13 of 14 features respond to framing along
substantially different directions (or along directions with a large
per-request-dependent component the global main effect cannot capture),
a single shared direction would produce exactly this pattern -- a small,
inconsistent, often-non-significant drop rather than a clean removal. The
one feature that did respond cleanly (layer 24/5393) is not distinguished by
rank, layer, or eta-squared magnitude from the others in any way inspected
here, so this experiment doesn't identify what makes it different (a genuine
open question, out of scope for this pass).

**Two negative results now on record, from two structurally different
intervention strategies.** The content-weighted detector (previous entry)
suppressed whole features downstream and lost real signal doing it. This
ablated a shared direction upstream and found the direction itself does not
isolate the phenomenon at the individual-feature level. Together they narrow
the honest picture of what "fixing" Qwen3-8B's framing-sensitivity would
actually require: not a simple linear operation on the existing top-15
feature set, in either direction. A next attempt would need either a
per-request-aware direction (multiplies the design space, explicitly
deferred here to avoid exactly the after-the-fact-tuning risk this
discipline exists to prevent) or a genuinely different mechanism, not a
variant of either linear intervention tried so far.

Results in `results/framing_directions.pt` and
`results/framing_direction_validation.json`.

## Pre-registration: a non-linear SAE-feature combiner (2026-08-12)

**The question.** Both linear fixes above failed as additively separable
operations: one scalar weight per feature, or one subtracted direction,
applied uniformly. The direction-ablation failure specifically pointed at
real core x wrapper *interaction* terms no single global linear operation
can capture. This tries a model that can represent "feature A's meaning
depends on feature B's state" directly -- genuine cross-feature interaction,
not a per-feature adjustment -- over the same top-15 features already used
throughout this thread.

**Model choice, fixed now.** `PolynomialFeatures(degree=2,
interaction_only=True, include_bias=False)` feeding a standard
`LogisticRegression` -- the minimal model that adds cross-feature
interaction capacity while staying a convex GLM with a deterministic
optimum and directly inspectable coefficients, consistent with this
project's existing preference for exact, interpretable methods over
black-box ML. A gradient-boosted-tree alternative was considered and
rejected as primary: several interacting hyperparameters would need fixing
a priori with no tuning available, and it offers no real interpretability
advantage over inspecting which interaction coefficients end up large. This
is the one pre-registered model class -- no bake-off between families.

**Pipeline and hyperparameters, fixed now** (n_val=289, 15 raw features ->
120 after pairwise interactions):

```
Pipeline([
    ("scale1", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("scale2", StandardScaler()),
    ("clf", LogisticRegression(penalty="l2", C=0.1, solver="lbfgs", max_iter=2000, random_state=0)),
])
```

`C=0.1` (10x stronger regularization than sklearn's `C=1.0` default) is
fixed a priori: ~289/120 ~ 2.4 samples per expanded feature is far below
usual "10 events per parameter" heuristics, independent of anything seen in
this run. All fitting/scaling happens on VAL only; TEST/PAIR are
transform-only, never refit -- same split discipline `calibrate()` already
follows elsewhere in this codebase.

**Required overfitting gate, fixed now, before TEST/PAIR are touched.**
Stratified 5-fold CV within VAL, refitting the identical pipeline per fold,
compared against in-sample accuracy (the same pipeline fit on the full VAL
set, scored on that same set). **Pass/fail rule**: if `in_sample_accuracy -
mean_cv_accuracy > 0.05` (5 percentage points -- roughly 2x the ~2.9pp
binomial standard error at n=289), the model is judged to have overfit VAL;
stop, report as a negative result, do not calibrate or evaluate on
TEST/PAIR, and do not adjust `C` post-hoc to force a pass -- same discipline
as both prior gates in this thread.

**The evaluation, fixed now.** If the gate passes: calibrate a threshold via
`max_accuracy_threshold` on the fitted pipeline's `predict_proba` (VAL),
then compare against the vanilla unweighted top-15 detector on TEST
accuracy/AUROC (`mcnemar_accuracy`, `delong_auc_test` -- regression check)
and the 21 PAIR prompts (`mcnemar_exact` -- the target metric), for Qwen3-8B
and, as a negative control, Llama-3.1-8B-Instruct (same fixed pipeline
applied to its own top-15 features, expected to show no PAIR improvement
since its features are already mostly content-leaning).

**Scope, fixed now.** No grid search or tuning of `C`, polynomial degree, or
CV folds on VAL. gemma-2-9b-it excluded (same reasoning as both prior
entries). No generation-time intervention -- scoring-time only, same as
both prior experiments.

## Closing the pre-registration: the non-linear combiner clears both gates, result genuinely inconclusive (2026-08-12)

**Both required gates were checked as written, and both passed, for both
models.** Qwen3-8B: CV gap = 0.0416 (in-sample 0.9689, mean-CV 0.9273),
under the 0.05 threshold. Llama: CV gap = 0.0035 (in-sample 0.9585, mean-CV
0.9549) -- barely any overfitting at all. Neither model's TEST-split
comparison shows a significant change versus the vanilla detector (Qwen3-8B:
accuracy p=0.7266, AUROC p=0.1277; Llama: accuracy p=0.5, AUROC p=0.0989) --
the no-regression bar both prior linear attempts failed to clear is cleared
here by both models.

**This is the first of three attempts where the PAIR number moves in the
hoped-for direction with no accompanying TEST cost.** Qwen3-8B's PAIR
detection rises from 52.4% to 71.4% (11/21 to 15/21, 6 discordant pairs, 5
favouring the non-linear combiner), the largest PAIR change of any
experiment run this session. It is **not formally significant** at this
sample size: McNemar p=0.2188. Llama's PAIR rate moves the other way (81.0%
to 71.4%, 2 discordant, both favouring vanilla), also not significant
(p=0.5) -- the negative control does not improve, and if anything drifts
slightly in the opposite direction, which is the pattern specificity would
predict, but neither result clears p<0.05, so this is corroborating, not
confirming.

**Reported as genuinely inconclusive, not as a fix, and not as a third
failure either.** Two things distinguish this from the two prior negative
results rather than just "another attempt that also didn't reach
significance": first, it is the only one of the three that actually cleared
its own no-regression requirement, meaning the PAIR number is at least
eligible to be trusted rather than disqualified before being read (per this
project's standing rule from the content-weighted-detector entry: a PAIR
change is not treated as meaningful unless TEST shows no cost). Second, the
direction and magnitude (+19.0pp for the target model, a small move the
other way for the control) is the qualitative shape a real, specific effect
would produce, even though n=21 does not have the power to confirm it.
**What would resolve this**: a larger PAIR-adversarial set (already a
standing limitation of this project, blocked externally on JailbreakBench
publishing more attack artifacts, see `reports/RESULTS.md`'s adversarial-set
limitations) is the direct way to get the statistical power this result is
missing -- not a different model or another round of hyperparameter choices
on the same n=21. Results in `results/nonlinear_combiner_eval.json`.

## Correcting the record: the "blocked on JailbreakBench" claim was imprecise (2026-08-12)

**The claim, repeated in this document (this entry included, just above,
and the 2026-07-24 "Adversarial n is the binding constraint" entry) and in
`reports/RESULTS.md`, was that the PAIR-adversarial set could not be
enlarged until JailbreakBench published more attack artifacts.** Checked
directly rather than continuing to assume it: JailbreakBench already has
successful PAIR artifacts for **60 of this project's 73 total corpus JBB
harmful goals** (89 distinct goals exist in JailbreakBench overall) -- 41 in
TRAIN, 9 in VAL, ~10-11 in TEST. The real, binding constraint was never
external data availability, it was that the adversarial set is (correctly)
built only from TEST-split goals. This document's older entries are left as
they are rather than silently edited -- they accurately record what was
believed at the time -- this entry is the correction, following this
project's own standing practice of recording corrections rather than
quietly amending history.

**What this unlocks**: `scripts/build_train_pair_set.py` builds a
supplementary PAIR set from the 41 TRAIN-goal matches (78 prompts across
41 goals, more than double the official n=21). TRAIN was only ever used to
derive the refusal direction / SAE causal ranking, never a detection
threshold, so this carries none of the calibration-leakage risk a VAL-goal
set would (VAL was deliberately excluded from this for that reason, a
choice made before running anything, not after). `scripts/train_pair_eval.py`
reuses the non-linear combiner's exact pipeline and VAL-derived threshold
(no refitting, no new researcher degrees of freedom) to test the same
already-pre-registered model against this larger set.

**Result**: Qwen3-8B's vanilla-vs-non-linear PAIR comparison reaches
significance on this larger set (29.5% -> 46.2%, McNemar p=0.0072, 17 vs. 4
discordant of 78). Llama's negative control stays flat (84.6% -> 83.3%,
p=1.0). But the vanilla detector's baseline PAIR rate on this TRAIN-goal set
(29.5%) is far below its known TEST-based rate (52.4%) for Qwen3-8B
specifically -- a 22.9-point gap not present for Llama (84.6% vs. 81.0%,
unremarkable). The two goal sets are not interchangeable for this model,
most plausibly genuine goal-level heterogeneity in paraphrase difficulty
rather than a flaw in either set. **Read as corroborating evidence for the
effect's direction and specificity** (same qualitative pattern replicates
across two independent goal sets with different baselines: Qwen3-8B
improves, Llama does not), **not as a clean statistical replication of the
original TEST-based effect size** -- the unexplained baseline gap means
that specific number shouldn't be taken at face value. Full numbers in
`results/train_pair_eval.json`.

## Wrapper-swap variance diagnostic extended to gemma-2-9b-it (2026-08-12)

**The question.** gemma-2-9b-it has the same class of PAIR-paraphrase
vulnerability as Qwen3-8B (47.6% vs. 52.4% detection) and its own working
top-15 SAE-feature detector (`results/sae_causal_ranking_gemma-2-9b-it.json`,
layers 33/34/35, rank-1 = layer 35/feature 52410), but nobody had run the
core-vs-wrapper ANOVA on it. This is a diagnose-first extension, not a
pre-registered intervention -- the diagnostic itself was never
pre-registered for Qwen3-8B or Llama-3.1-8B-Instruct either (see the
"Wrapper-swap variance decomposition" and "Phase 6 Wave 3" results entries),
so this entry follows that same precedent rather than inventing new process
for a measurement step.

**Decision rule, fixed before running anything**: a feature is
framing-leaning if `eta_sq_wrapper > eta_sq_core` (the same operational
definition `feature_variance_family.py`'s own summary printout already
uses). gemma's top-15 would be read as framing-dominated (motivating a
combiner attempt, Qwen3-8B's pattern) at >=8/15 framing-leaning, or
content-dominated (no motivated fix, Llama's pattern) otherwise -- a plain
majority cutoff fixed before seeing the data.

**Method**: reused `scripts/wrapper_swap_variance.py`'s grid measurement,
ANOVA, and permutation-test machinery unmodified (only its `MODELS`/
`TOP_FEATURE` dicts gained a gemma entry) for the rank-1 feature, then
`scripts/feature_variance_family.py`'s maxT/Westfall-Young family correction
(only its `RANKING_PATH` dict gained a gemma entry) for the full top-15.
Both scripts' existing verification gates ran for real: `verify_maxT_on_synthetic`
(null + planted-effect synthetic check) and `verify_rank1_reproduction`
(gemma's family-loop rank-1 recomputation cross-checked against the
standalone `wrapper_swap_variance.py` run) both passed.

**Result: gemma's top-15 is framing-dominated more uniformly than
Qwen3-8B's.** All 15 of 15 features are framing-leaning
(`eta_sq_wrapper > eta_sq_core`), every one maxT-corrected significant at
p<=0.0015 -- a cleaner sweep than Qwen3-8B's 14/15 and starkly unlike
Llama's 11/15 content-leaning split. Per-feature eta-squared ranges
core=0.201-0.347 vs. wrapper=0.451-0.679 across all 15; full table and both
raw/maxT-adjusted p-values in `results/feature_variance_gemma-2-9b-it.json`
and `results/wrapper_swap_variance.json`.

**Decision, per the rule fixed above: 15/15 >= 8/15, framing-dominated ->
proceed to a non-linear combiner attempt for gemma-2-9b-it**, pre-registered
separately below before touching any real VAL data.

## Pre-registration: a non-linear SAE-feature combiner (gemma-2-9b-it) (2026-08-12)

**The question.** The diagnostic above found gemma-2-9b-it's top-15 SAE
features framing-dominated even more uniformly than Qwen3-8B's (15/15 vs.
14/15). Per this project's own precedent (Qwen3-8B's non-linear combiner
was the only one of three fix attempts that both cleared its no-regression
gate and moved PAIR in the hoped-for direction), this is a real, motivated
reason to try the identical combiner on gemma rather than a fresh design
exercise.

**Model choice and pipeline, fixed now, identical to the Qwen3-8B/Llama
pre-registration -- no new tuning, no bake-off**:

```
Pipeline([
    ("scale1", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("scale2", StandardScaler()),
    ("clf", LogisticRegression(penalty="l2", C=0.1, solver="lbfgs", max_iter=2000, random_state=0)),
])
```

gemma-specific fixed numbers (verified before this entry was written, not
assumed): `n_val=289` (identical to Qwen3-8B/Llama's VAL split size), 15 raw
top-causally-ranked SAE features -> 120 after pairwise interactions
(identical expansion), layers 33/34/35, rank-1 = layer 35/feature 52410.
`C=0.1` is the same a-priori choice as before, for the same reason
(~289/120 ~ 2.4 samples per expanded feature, far below usual heuristics,
independent of anything seen in any run). All fitting/scaling on VAL only;
TEST/PAIR transform-only, never refit.

**Required overfitting gate, fixed now, before TEST/PAIR are touched**:
identical rule to the original pre-registration -- stratified 5-fold CV
within VAL vs. in-sample VAL accuracy, both computed by refitting the
identical pipeline; **fail if `in_sample_accuracy - mean_cv_accuracy >
0.05`**. On failure: stop, report as a negative result, do not touch
TEST/PAIR, do not adjust `C` post-hoc.

**The evaluation, fixed now.** If the gate passes: calibrate a threshold via
`max_accuracy_threshold` on VAL `predict_proba`, then compare against
gemma's vanilla unweighted top-15 detector on TEST accuracy/AUROC
(`mcnemar_accuracy`, `delong_auc_test` -- regression check) and gemma's own
PAIR-adversarial set (`mcnemar_exact` -- the target metric).

**No new negative-control run.** Llama-3.1-8B-Instruct's already-published
non-improvement under this identical pipeline (81.0%->71.4%, p=0.5, moving
the *opposite* direction from Qwen3-8B's improvement) already established
the specificity pattern this design predicts for a content-dominated model;
rerunning it here would duplicate already-built infrastructure for no new
information. gemma's result is compared against that published number
narratively, not via a fresh paired statistical test against a rerun.

**Decision criterion for wiring live into the webapp, fixed now, matching
the bar Qwen3-8B's combiner actually cleared**: wire in if and only if (a)
the overfit gate passes, (b) TEST shows no significant regression on
accuracy or AUROC, and (c) PAIR detection improves (directionally, whether
or not McNemar reaches p<0.05 -- Qwen3-8B's own live-wired result was
itself not significant at p=0.2188). If PAIR instead moves the *wrong*
direction, even non-significantly, that is a materially different outcome
from Qwen3-8B's case and will not be wired live even if TEST doesn't
regress.

**Scope, fixed now.** No grid search or tuning of `C`, polynomial degree, or
CV folds on VAL. No generation-time intervention -- scoring-time only, same
as every prior combiner experiment in this thread.

## Closing the pre-registration: gemma's combiner clears the gate, but PAIR moves the wrong way (2026-08-12)

**The overfit gate passed**: in-sample VAL accuracy 0.9654, mean 5-fold CV
accuracy 0.9273, gap 0.0381 -- under the 0.05 threshold, comparable to
Qwen3-8B's own 0.0416 gap. No TEST regression either: accuracy 92.7% ->
93.4% and AUROC 0.9655 -> 0.9726, both directionally *better*, neither
significant (McNemar p=0.7744, DeLong p=0.2786) -- criteria (a) and (b) from
the pre-registration both clear.

**But PAIR detection moves the wrong way: 47.6% -> 42.9%** (McNemar
p=1.0000, not significant, but the wrong direction). This is the opposite of
Qwen3-8B's result (52.4% -> 71.4%, the hoped-for direction) despite gemma's
top-15 being *more* uniformly framing-dominated (15/15 vs. Qwen3-8B's
14/15) -- if the framing-dominance diagnosis alone predicted intervention
success, gemma should have been at least as good a candidate as Qwen3-8B,
plausibly better. It was not.

**Per criterion (c), fixed before this ran: not wired live.** The
pre-registration explicitly distinguished "PAIR improves, even
non-significantly" (Qwen3-8B's case, wired live) from "PAIR moves the wrong
way, even non-significantly" (this case) as materially different outcomes,
specifically to prevent post-hoc rationalizing a wrong-direction result into
a ship decision because the other two gates happened to pass. No new code
touches `src/api/model_registry.py`, `inference_manager.py`, or the webapp;
`results/nonlinear_combiner_gemma-2-9b-it.joblib` exists on disk from this
run but has no corresponding decision to expose it live.

**Why this is a genuinely informative negative result, not just "another
attempt that didn't work".** It decouples two things this project's own
prior narrative had been treating as if they moved together: *which
features a detector's top-15 tracks* (framing vs. content, established by
the wrapper-swap ANOVA) and *whether a cross-feature-interaction model over
those same features helps paraphrase robustness*. Qwen3-8B's framing-heavy
top-15 responded to interaction modeling; gemma's even-more-framing-heavy
top-15 did not, and if anything overfit toward whatever VAL-specific
interaction pattern hurt PAIR generalization slightly. The mechanistic
diagnosis (which single-feature statistic a detector's features carry) is
not, by itself, sufficient to predict whether a specific downstream fix
built from that diagnosis will transfer -- something the diagnosis alone
could not have revealed, and which the two-model precedent (Qwen3-8B
positive, Llama flat) had not yet exposed either, since Llama's
content-dominated features never motivated trying the combiner on it in the
first place. Full numbers in `results/nonlinear_combiner_eval.json`.

## Formal significance for Qwen3-8B's and Llama-3.1-8B's suppression curves too (2026-08-12)

**The gap.** The "Closing a Wave 2 gap" entry above ran a formal paired
McNemar test for gemma-2-9b-it's suppression curve because, unlike the
other two models, it had never been tested that way -- Qwen3-8B's
significance claim rested on non-overlapping Wilson CIs and Llama-3.1-8B's
on an unambiguous 0% floor. Both of those are the same kind of informal
proxy this project's own adversarial-evaluation entry (2026-07-11) already
flagged as weaker than a proper paired test for paired predictions
("comparing two separate Wilson CIs for overlap ... can miss or wrongly
suggest a real paired difference"). Never actually closed for these two
models, so closed now rather than left as an accepted gap.

**Method**: generalized `scripts/gemma_suppression_significance.py` into
`scripts/suppression_significance.py` (renamed, looped over a `MODELS` list
instead of one hardcoded model -- no other logic changed) and ran it for
all three. No new GPU compute: reclassifies each model's already-saved
`results/sae_suppression_validation_<model>.json` completions with
`is_refusal` and runs `mcnemar_exact` against baseline, per condition, same
as gemma's original closure.

**gemma-2-9b-it's numbers reproduced exactly** (41/50 baseline, p=0.0312/
0.0156/0.0156 at top10/15/20) -- confirms the script generalization didn't
change anything for the model it was already verified against.

| condition | Qwen3-8B refusal | Qwen3-8B p | Llama-3.1-8B refusal | Llama-3.1-8B p |
|---|---|---|---|---|
| baseline | 41/50 | -- | 49/50 | -- |
| top1 | 42/50 | 1.0 | 5/50 | **0.0** |
| top5 | 21/50 | **0.0** | 2/50 | **0.0** |
| top10 | 16/50 | **0.0** | 1/50 | **0.0** |
| top15 | 12/50 | **0.0** | 0/50 | **0.0** |
| top20 | 13/50 | **0.0** | 1/50 | **0.0** |

**Both confirm exactly what the informal arguments already claimed, now on
solid footing.** Qwen3-8B: not significant at top1 alone (matches "top1
barely moves refusal" from the original write-up), significant from top5
onward (matches "baseline distinguishable from top5 onward" in the
2026-07-11 deterministic re-run entry) -- every discordant pair at top5+
favors suppression reducing refusal (baseline-only counts, zero
condition-only), consistent with a real monotonic effect. Llama-3.1-8B:
significant at every single condition including top1 -- unsurprising given
one feature alone already drops refusal from 98% to 10%, but now formally
confirmed rather than argued from the 0% floor alone. No reversals, no
surprises -- this closes the gap the gemma entry explicitly left open
("worth doing if this comparison is written up as a headline result rather
than a descriptive observation") without changing any conclusion already
published. Full numbers in `results/sae_suppression_significance_Qwen3-8B.json`
and `results/sae_suppression_significance_Llama-3.1-8B-Instruct.json`.

## DeepSeek-R1-Distill-Qwen-1.5B's suppression curve needs a different, index-paired test (2026-08-12)

**Why the other three models' method doesn't directly apply.** DeepSeek's
suppression validation (`reasoning_model: true` in its own JSON) already
has a documented "inconclusive, not negative" reading in RESULTS.md: N=15
per condition with heavy, condition-varying truncation (33-60%) because a
real fraction of completions never get past the mandatory `<think>` block
in the fixed 2048-token budget. The raw 15 completions per condition are
**not** already-matched pairs the way the other three models' clean N=50
sets are -- calling `is_refusal` on all 15 raw completions directly (this
script's path for the other three) would silently score truncated
non-answers as "not a refusal" and misalign which actual prompt is being
compared at each position across conditions, since a different subset
truncates each time.

**Method**: `src.direction.refusal_classifier.resolve_completions_by_index`
already exists for exactly this (its own docstring: "each condition
truncates a DIFFERENT subset of prompts... callers should intersect the
returned dicts' keys across every condition being compared before scoring,
not assume every index survived everywhere") -- reused, not reinvented.
`scripts/suppression_significance.py` now branches on the JSON's own
`reasoning_model` flag: for DeepSeek, each condition's McNemar comparison
against baseline uses only the prompt indices that resolved to a real
answer in *both* baseline and that condition, intersected per comparison
(not a single global intersection across all conditions, since each
condition truncates differently).

**Result: zero discordant pairs at every single condition** (top1 through
top20), p=1.0 throughout. Paired-N after intersection is small at every
condition (4-6 prompts, down from the nominal 15) and the baseline itself
is already at floor within that paired subset (0/6 baseline refusals for
the top15 comparison's 6-prompt intersection). **This is not a null
result -- there were zero prompts where baseline and the suppression
condition disagreed, on either side, at any threshold.** Confirms, with
actual paired-test rigor rather than an assumption, the "inconclusive, not
negative" reading already published: this data genuinely cannot
distinguish "the effect exists but this sample can't see it" from "there
is no effect here," the same conclusion reached informally, now backed by
the correct test rather than raw refusal-rate CIs on mismatched subsets.
Full numbers in `results/sae_suppression_significance_DeepSeek-R1-Distill-Qwen-1.5B.json`.

## Multiple-comparisons correction audit: 5 families, fixed before computing anything (2026-08-12)

**The gap.** `reports/RESULTS.md`'s "Known limitations (cross-model
dense-direction comparison)" section already names this honestly:
BH-FDR correction is applied in exactly one place in this project (the
wrapper-swap maxT scheme) and nowhere else, even though several other
families of paired tests exist, some close enough to 0.05 to matter (the
LLM-judge PAIR comparisons, p=0.031/p=0.039, and what that bullet calls
"the threshold-rule-vs-judge comparison", p=0.0201/p=0.0225). The project's
own stated reason it was never corrected retroactively: deciding which
tests count as one "family" after the fact, rather than pre-registering it,
is itself a researcher-degree-of-freedom risk. This closes the gap by fixing
family membership explicitly, in writing, before computing any correction --
addressing the stated objection rather than ignoring it.

**Sourced from the actual result JSONs, not rounded prose**, wherever one
exists: `results/threshold_recalibration.json` (exact `vs_judge_mcnemar`/
`vs_youden_mcnemar` values per model/rule), `results/paraphrase_decay_sae.json`
(exact Wilcoxon p-values), `results/llama_causal_gap_decomposition.json`
(all 6 pairwise post-hoc McNemar values per model, not just the subset
mentioned in prose). DeLong AUROC (0.0041/0.0015/0.0053) and
PAIR-McNemar-vs-judge (0.031/0.69/0.039) were never persisted to a JSON;
the "A wrong paired test, caught and corrected" entry above confirms both
were unaffected by the McNemar bug found there, so used as-is at their
published precision.

**A real imprecision caught while sourcing values**: the limitations
bullet's "threshold-rule-vs-judge comparison (p=0.0201, p=0.0225)" bundles
two different kinds of test. 0.0201 is Qwen3-8B's now-superseded
youden_j-vs-judge accuracy comparison -- the *same* underlying question as
the detector-vs-judge family's Qwen entry below (0.3833), just computed at
a threshold that was later abandoned. 0.0225 is `max_accuracy`-vs-`youden_j`
-- a rule-vs-rule comparison, no judge involved at all. Pooling 0.0201 with
either family would double-count the same accuracy-vs-judge question at two
non-independent calibrations of the same detector; excluded from both.

**Five families, fixed before computing anything** (`scripts/fdr_correction.py`,
`scipy.stats.false_discovery_control(method="bh")` -- the same library call
this project already uses for BH-FDR elsewhere, no new implementation, no
synthetic self-test of `scipy`'s own trusted function planned or needed):

- **A: detector vs. LLM-judge**, each model's adopted threshold (9 tests --
  3 DeLong AUROC, 3 accuracy McNemar, 3 PAIR McNemar).
- **B: threshold-reselection vs. youden_j**, TEST accuracy, no judge
  involved (3 tests).
- **C: SAE-feature paraphrase-decay Wilcoxon**, top-1-feature delta and
  full-top-15-score delta, 3 models (6 tests).
- **D1/D2: component-decomposition post-hoc pairwise McNemar**, kept as two
  separate families (Llama's 6 pairs, Qwen3-8B's 6 pairs) since each
  follows its own independent omnibus Cochran's Q -- standard post-hoc
  practice, a pairwise family only corrects within the set that followed
  the same omnibus test, not pooled across two unrelated omnibus tests.

`max_f1` rows excluded (already noted in RESULTS.md as numerically
identical to `max_accuracy`, not independent tests). The 3 dense-vs-SAE-feature
DeLong/McNemar tests are out of scope -- never named by the limitations
bullet this closes.

**Result: 3 of 27 tested conclusions flip from significant to
not-significant under family-wise correction, all in the two families the
limitations bullet already flagged as borderline** -- everything else
(DeLong AUROC family, SAE-paraphrase-decay family, both component-decomposition
families) survives correction unchanged, mostly because those p-values are
already extreme (p=0.0 to p=0.005) relative to their family size.

| test | raw p | BH q | before | after |
|---|---|---|---|---|
| Qwen3-8B PAIR McNemar vs. judge | 0.031 | 0.070 | significant | **not significant** |
| gemma-2-9b-it PAIR McNemar vs. judge | 0.039 | 0.070 | significant | **not significant** |
| Qwen3-8B `max_accuracy` vs. `youden_j` | 0.0225 | 0.068 | significant | **not significant** |

**What this changes and what it doesn't.** The load-bearing claim in the
LLM-judge comparison -- activation-based detection wins threshold-independent
AUROC ranking on all three models -- is untouched (all three DeLong p-values
survive easily, q<=0.016). What no longer clears a family-wise bar: "the
judge detects more PAIR attacks, significantly on two of three models" (now
neither model is significant after correcting for testing PAIR on 3 models
alongside 6 other detector-vs-judge comparisons in the same family) and
"threshold reselection is a significant improvement on Qwen3-8B" (q=0.068,
just above 0.05). Neither was a headline finding on its own -- both were
already reported as secondary/diagnostic points alongside the AUROC result
-- but both `reports/RESULTS.md` and `README.md` stated them as significant
without a family-wise caveat, so both are corrected in place to say
"significant at the per-test level, not after BH-FDR within its family."

## Pre-registration: DeepSeek-R1-Distill-Llama-8B vs. Llama-3.1-8B-Instruct -- is diffuseness distillation-general or base-specific? (2026-08-13)

**The question.** DeepSeek-R1-Distill-Qwen-1.5B is diffuse across every
measurement this project has run: weakest dense-direction classifier of
all six models (84.7% TEST accuracy, AUROC 0.911), worst PAIR robustness
(9.5%, a dramatic new low), a genuine activation-addition null (0% refusal
at every alpha 0.25-4.0), and an SAE-feature detector that barely fires
(4.4% VAL recall). The Roadmap has named comparing this against another
distilled model as deliberately-not-pursued future work. This closes that
gap, done to full parity rather than as a minimal probe (user's explicit
call): the same dense-direction *and* SAE-feature treatment every other
model in this project gets, not a reduced version.

**Model and rationale, fixed now.** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
-- distilled from the same base architecture and parameter class as
`Llama-3.1-8B-Instruct`, already the most thoroughly characterized model in
this project: best passive classifier of all six (93.1% TEST accuracy,
AUROC 0.989), yet one of the weakest causal mechanisms until the
component-decomposition work resolved it (the small feature-aligned
component alone drops refusal 92%->38%, p=0.0, while the large orthogonal
remainder does nothing at all, 92%->92%); SAE-feature causal effect
concentrated almost entirely in one feature (98%->10% from the top feature
alone); dense-direction PAIR robustness 71.4%, SAE-feature PAIR 81.0%. This
is the natural architecture/size-matched control DeepSeek-1.5B's own
comparison group (Qwen2.5-1.5B-Instruct, a different and much smaller base)
never had.

**The hypothesis, stated before running anything, both directions
informative.** If DeepSeek-1.5B's diffuseness is a property of R1-style
reasoning distillation itself, DeepSeek-R1-Distill-Llama-8B should look
diffuse too -- despite being 8B and Llama-based, it should show a weaker
classifier, weaker PAIR robustness, and a weak-or-null causal effect (both
necessity and sufficiency), unlike Llama-3.1-8B-Instruct's strong,
concentrated profile. If it is instead a property of DeepSeek-1.5B's small
base model or its specific training run, DeepSeek-R1-Distill-Llama-8B
should look more like Llama-3.1-8B-Instruct -- a strong classifier and a
concentrated (if not necessarily identical) causal mechanism. Either
outcome is reportable; there is no failure mode for this comparison, only
an answer.

**A pretrained third-party SAE was found and verified to exist for this
exact model**, `qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19` (layer 19,
reported final L0 of 93 during training) -- checked directly by downloading
and inspecting the 2.1GB checkpoint rather than assumed from the model
card: a plain state dict, `encoder.weight (65536,4096)`,
`encoder.bias (65536,)`, `decoder.weight (4096,65536)`,
`decoder.bias (4096,)`, matching this project's existing `TopKSAE` class
shape convention exactly (no transpose needed, unlike EleutherAI's
checkpoint). **A real, unresolved risk flagged rather than assumed away**:
this project already caught LlamaScope's and GemmaScope's actually-released
checkpoints being JumpReLU despite their paper describing "TopK SAEs" (see
`src/sae/jumprelu_sae.py`'s docstring) -- the same failure mode is possible
here. This checkpoint stores no separate threshold tensor (a real JumpReLU
checkpoint needs one), which is evidence against JumpReLU and consistent
with hard top-k, but the GitHub repo the model card cites as its training
code is dead (confirmed via the GitHub API, 404, org does not resolve), so
there is no ground truth to verify against the way every other SAE provider
in this project was. **Required gate before any causal-ranking work**:
empirically check reconstruction quality at k=93 on real layer-19
activations; if poor, sweep nearby k values; if poor at every reasonable k,
stop and report the SAE as unusable rather than force it into service --
the dense-direction half of this comparison does not depend on it and
proceeds regardless.

**Metrics, fixed now, all compared directly against Llama-3.1-8B-Instruct's
already-published numbers (no rerun needed):** dense-direction TEST
accuracy/AUROC, PAIR detection rate, causal necessity (ablation) and
sufficiency (activation addition) at full N=50/6-condition, and if the SAE
gate passes: SAE-feature causal ranking, N=50/6-condition suppression
validation, and SAE-feature detector TEST accuracy/AUROC/PAIR. Reasoning-
trace methodology reused unmodified from DeepSeek-1.5B's onboarding
(`extract_answer`, `resolve_completions`/`resolve_completions_by_index`,
`--reasoning-model`/`--max-new-tokens` flags, 2048-token budget). No
reduced sample sizes taken just to save time -- if the real per-generation
cost makes N=50 impractical, that gets reported as a finding (same
pre-flight cost-check discipline as DeepSeek-1.5B's own onboarding), not
silently cut without saying so.

**Scope, fixed now.** No SAE-feature work if Step 2's reconstruction gate
fails. No webapp wiring -- this model has no role in the live interactive
detector. Staged execution (SAE verification, then dense-direction, then
SAE-feature if warranted, then write-up), each stage's result reviewed
before the next runs, matching this project's own established pattern for
large model-onboarding efforts.
Full numbers in `results/multiple_comparisons_correction.json`.
