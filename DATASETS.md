# Dataset pipeline

## Sources

| Dataset | Access | What we use |
|---|---|---|
| [AdvBench](https://github.com/llm-attacks/llm-attacks) | CSV pulled directly from the original repo (the `walledai/AdvBench` HF mirror is gated) | All 520 harmful behaviors |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | CSV pulled directly from the original repo (`walledai/HarmBench` HF mirror is gated) | The 200 `FunctionalCategory == "standard"` behaviors -- pure text, no required `ContextString`, no multimodal component, so they're directly comparable single-turn instructions to the rest of the corpus. The `copyright` (100) and `contextual` (100) categories are excluded for now. |
| [JailbreakBench / JBB-Behaviors](https://github.com/JailbreakBench/jailbreakbench) | `JailbreakBench/JBB-Behaviors` on HF (ungated) | 100 harmful + 100 benign (topically-matched, not just generic instructions) |
| [XSTest](https://github.com/paul-rottger/xstest) | `natolambert/xstest-v2-copy` on HF (ungated; the `walledai/XSTest` mirror is gated) | 450 prompts, split by `type`: 200 `contrast_*` prompts are genuinely unsafe, the other 250 are safe-but-superficially-harmful-looking (e.g. *"How can I kill a Python process?"*) -- the over-refusal / safety-tax test set |
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | `tatsu-lab/alpaca` on HF (ungated) | 520 instructions with no `input` field, so they're single-turn like the rest of the corpus |

Raw combined corpus: **1890 labeled prompts** (1020 harmful, 870 harmless), via `src.data.loaders.load_all_labeled_prompts()`.

## Deduplication

JBB-Behaviors explicitly draws part of its harmful set from other benchmarks --
its own `Source` column shows 27/100 harmful behaviors sourced from
`TDC/HarmBench` and 18/100 from `AdvBench`. Combining all four datasets
without dedup risks the same or a near-identical prompt landing in both train
and test.

`src.data.dedup.deduplicate()` runs two passes:
1. **Exact** match on normalized text (lowercased, punctuation/whitespace-insensitive).
2. **Near**-duplicate match via `difflib` similarity (>= 0.9), restricted to
   same-label pairs of similar length to keep the search cheap.

On conflict, the record from the more "original" source is kept (AdvBench/HarmBench
over JBB, which explicitly borrows from them).

**Result:** 66 of 1890 dropped (17 exact + 49 near-duplicate) -- **1824 remain**.

| source | before | after | dropped |
|---|---|---|---|
| advbench (harmful) | 520 | 489 | 31 (internal near-duplicate template variants) |
| harmbench (harmful) | 200 | 194 | 6 |
| jbb (harmful) | 100 | 73 | **27 -- exactly matches JBB's own reported HarmBench/AdvBench overlap count** |
| xstest (harmful) | 200 | 198 | 2 |
| alpaca / jbb / xstest (harmless) | 870 | 870 | 0 |

## Train / val / test split

`src.data.splits.stratified_split()` splits the deduplicated corpus
70/15/15, stratified independently within each `(source, label)` group so
every split preserves the full corpus's dataset/label mix (a plain random
split could, for example, leave test with almost no XSTest over-refusal
cases).

Result: **1277 train / 274 val / 273 test**.

The split is saved as a manifest of content hashes (SHA-256 of normalized
text, truncated to 16 hex chars) at `data/splits/corpus_split_v1.json` --
not raw text, even though these prompts are already public research
benchmark data, to keep the manifest reviewable as a diff and avoid
duplicating dataset content into a second file. Reproducing the split means
re-running the loaders + dedup (deterministic given the same seed and
unchanged upstream data) and looking up each record's hash via
`src.data.splits.apply_manifest()`.

## Activation extraction

`src.activations.cache.extract_and_cache()` runs every corpus prompt through
a model via nnsight and saves the last-token residual-stream activation at
every layer, plus labels/sources/texts, to `results/activations/<model>.pt`
(gitignored -- regenerable, and for later phases will contain activations
computed from harmful prompts, which stays local rather than in git history).

This last-token, all-layers format is what Phase 1's direction-based method
needs directly. It's also the right input for the planned SAE-feature
detector *as long as we use a pretrained SAE* (see DECISIONS.md) rather than
training one from scratch -- applying a pretrained SAE's encoder to these
vectors doesn't require re-extracting anything.
