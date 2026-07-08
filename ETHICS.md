# Ethics review: draft summary

**Status: DRAFT.** This is prepared material for you to review, adapt, and
submit to your advisor/department -- it is not a substitute for actual
sign-off, which only your institution can give. Nothing in this repo should
be treated as cleared for use beyond your own research until that happens.

## What this research does

Detects jailbreak / harmful-intent prompts using an LLM's internal
activations, as a more robust alternative to surface-level (keyword or
perplexity) filtering. This is **defensive** research: the goal is a better
detector, not new attack techniques. No novel jailbreak methods are being
developed or published as part of this work.

## Data sources

All harmful-instruction data comes from four established, publicly published
safety-research benchmarks, used exactly as their authors intend:

- AdvBench (Zou et al. 2023, [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks))
- HarmBench (Mazeika et al. 2024, [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench))
- JailbreakBench / JBB-Behaviors (Chao et al. 2024, NeurIPS Datasets and Benchmarks track)
- XSTest (Röttger et al. 2023)

No harmful content was authored, sourced, or scraped from elsewhere for this
project -- every harmful prompt already exists in a public, citable,
peer-reviewed-adjacent safety benchmark, used by dozens of other published
papers surveyed in [LITERATURE.md](LITERATURE.md).

## What gets generated locally

Testing whether an activation direction *causally* controls refusal
(Phase 1) requires actually running harmful prompts through open-weight
models with the refusal mechanism disabled, to observe whether the model
complies. This does produce real harmful completions on the local machine
during experiments.

**Handling**: raw model completions are never committed to this git
repository (`results/` is gitignored -- see `.gitignore`) and are not
shared, published, or used for any purpose beyond confirming the causal
effect exists. Only aggregate statistics (refusal rates, confidence
intervals -- no actual harmful text) are recorded in version control, in
[RESULTS.md](RESULTS.md).

## Models used

Open-weight instruction-tuned models already publicly released and
downloadable by anyone (Qwen, SmolLM2, and later Llama/Gemma/Qwen variants
per [DECISIONS.md](DECISIONS.md)). This research doesn't create new
capability -- these models' ability to produce harmful content when
safety training is bypassed is already a known, publicly documented
property (that's precisely what the underlying published research,
Arditi et al. 2024, demonstrates).

## Risk assessment (self-assessed, not a substitute for institutional review)

Low risk: uses only established public benchmarks and already-public
open-weight models, generates no content intended for external use, and the
end goal (a better jailbreak detector) is protective rather than harmful.
The main things worth flagging to a reviewer: (1) local storage of
model-generated harmful text during experiments, handled via gitignore as
above, and (2) eventual publication of a working detector's methodology,
which is standard practice in this research area (see the papers surveyed
in LITERATURE.md, all of which publish similar methodology).

## What to actually do with this

Bring this document (or your own version of it) to your advisor/department
as a starting point for whatever your institution's actual process requires
-- this project cannot determine what that process is or complete it.
