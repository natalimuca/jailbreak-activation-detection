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
