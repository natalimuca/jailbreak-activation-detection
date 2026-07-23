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
  [LITERATURE.md](LITERATURE.md)) does the same thing with GemmaScope/LlamaScope
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

`scripts/05_causal_validate_sae_features.py`: baseline vs. suppressing the
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

- **Ranking pass**: `N_EVAL_PROMPTS` 8 -> 16 (`scripts/04`). Doubles cost
  (~3.8hrs estimated vs the original ~2hrs) but tightens which exact
  features land in the top-20, not just the validation's confidence in
  them.
- **Validation pass**: `N_VAL_PROMPTS` 25 -> 50, plus two intermediate
  conditions (`top10`, `top15`) added alongside the existing
  baseline/top1/top5/top20, for a smoother dose-response curve rather than
  3 sparse points (`scripts/05`). ~1.75hrs estimated for 6 conditions x 50
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
to just do). Re-ran `scripts/01_reproduce_refusal_direction.py` for both
models and `scripts/02_calibrate_addition_alpha.py` for both, no code
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

`scripts/06_dense_direction_ablation_qwen3.py`: Phase 1's ablation method
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
`scripts/07_sample_completions_for_labeling.py` draws a stratified sample
(3 per source/condition group, 45 total across all 15 groups from Phase 1,
Phase 3, and the head-to-head) and writes a CSV worksheet with the
classifier's own verdict deliberately hidden (saved separately to
`results/classifier_spotcheck_reference.json`) to avoid anchoring the
labeler's judgment. `scripts/08_score_classifier_agreement.py` joins the
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
not source) -- awaiting the user's labels before `scripts/08` can report
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
detectors (`scripts/10_calibrate_detector_thresholds.py`, via Youden's J --
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
choice: closest in spirit to how `scripts/02_calibrate_addition_alpha.py`
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
just-merged Qwen3-8B pipeline (`scripts/06` -> `scripts/10` -> `scripts/11`)
selects the dense-direction detector's layer via **TEST**-split separation
score (`scripts/06`'s docstring explains this was to avoid VAL, which was
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

**Empirical outcome after rerunning `scripts/10`/`scripts/11`** (full
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
   the formula subtly wrong). `scripts/13_cross_model_significance.py`
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

**Wave 1 execution**: reused `scripts/03_extract_all_activations.py`
unchanged (fully generic, no new extraction code needed) for both models,
`--4bit` (8-9B params on a 6GB GPU). Full-corpus extraction took ~1h45m
(Llama-3.1-8B-Instruct) and ~2h08m (Gemma-2-9B-it). New
`scripts/14_extend_dense_direction_llama_gemma.py` mirrors
`scripts/12`'s pattern exactly (same `select_layer_and_calibrate`,
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

**Extended `scripts/13_cross_model_significance.py` from 3 to 5 models**
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
`scripts/04`/`scripts/05` no longer hardcode Qwen-Scope's loader and
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

`scripts/04_causal_rank_sae_features.py meta-llama/Llama-3.1-8B-Instruct`,
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
   `scripts/04`'s `main()` explicitly moved every candidate layer's
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
   CPU. Removed the whole-SAE `.to("cuda:0")` step from `scripts/04`
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

`scripts/05_causal_validate_sae_features.py`, same parameters as
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
- `scripts/15_extend_sae_adversarial_cache_llama_gemma.py` (new) --
  Wave 1's dense-direction extension (`scripts/14`) computed adversarial
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
- `scripts/10_calibrate_detector_thresholds.py`/
  `scripts/11_head_to_head_detectors.py` generalized with a `model` CLI
  arg (mirroring `scripts/04`/`05`'s pattern from Wave 2): SAE loading via
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
tested. `scripts/16_test_gemma_suppression_significance.py` closes this
with no new GPU compute -- `scripts/05`'s validation run already saved
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
`scripts/17_cross_model_direction_transfer.py`, branch
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
`scripts/22_llama_own_ablation_larger_n.py` draws a fresh, independent
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
on Phase 1's two small models. `scripts/23_sufficiency_7b_9b_scale.py`
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
(shared with scripts/02) but had never actually been exercised by any
model until Llama here. Final validated effect: 10.0% -> 34.0%, real
(barely non-overlapping CIs) but much smaller than Qwen3-8B's.

**Not written up as "sufficiency confirmed at scale."** Necessity
generalizes more robustly across these two models than sufficiency does
-- the same qualitative pattern Phase 1 already found between Qwen2.5 and
SmolLM2 (addition is architecture-dependent, ablation isn't), now showing
up again at 8-9B scale with a different model pair. Reported as an open,
unexplained architecture difference, not forced into a single "it works
at scale" headline. See RESULTS.md's dedicated section for full numbers.

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
Wave 2 significance test (`scripts/16`) is unaffected -- confirmed via
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
"non_refuse" bucket. Originally surfaced by `scripts/06`'s head-to-head
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
gemma-2-9b-it or the cross-model-transfer results. `scripts/18_expand_
moralize_comply_worksheet.py` sampled 53 more non-refuse completions
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

**Resolving scripts/06's original question** (`scripts/20_rescore_
scripts06_harmful_compliance.py`): every non-refuse completion in both
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
