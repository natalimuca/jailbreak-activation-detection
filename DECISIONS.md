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

**Results (2026-07-09, `results/sae_causal_ranking_Qwen3-8B.json`):** ran
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
