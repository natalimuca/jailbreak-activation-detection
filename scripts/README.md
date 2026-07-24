# Scripts

Each script is a standalone experiment or pipeline step, run manually and
in order (no orchestrator). Filenames used to carry a numeric prefix
encoding that order; this table replaces it.

| # | Script | What it does |
|---|---|---|
| 1 | `reproduce_direction.py` | Reproduce Arditi et al.'s refusal-direction finding (Phase 1) |
| 2 | `calibrate_alpha.py` | Calibrate the activation-addition coefficient per model |
| 3 | `extract_activations.py` | Extract and cache full-corpus residual-stream activations |
| 4 | `rank_sae_features.py` | Causally rank candidate SAE features via attribution patching |
| 5 | `validate_sae_features.py` | Causal-validate the ranked SAE features via suppression |
| 6 | `ablate_qwen3_direction.py` | Dense-direction ablation on Qwen3-8B |
| 7 | `sample_for_labeling.py` | Sample completions for the refusal-classifier spot-check |
| 8 | `score_agreement.py` | Score classifier-vs-human agreement on the sample |
| 9 | `build_adversarial_set.py` | Build the real JailbreakBench adversarial paraphrase set |
| 10 | `calibrate_thresholds.py` | Calibrate all four detectors' decision thresholds on VAL |
| 11 | `compare_detectors.py` | Head-to-head detector comparison on TEST + adversarial set |
| 12 | `extend_qwen_smollm.py` | Extend dense-direction detector to Qwen2.5-1.5B/SmolLM2 |
| 13 | `cross_model_significance.py` | Formal significance tests (DeLong, Cochran's Q) across models |
| 14 | `extend_llama_gemma.py` | Extend dense-direction detector to Llama-3.1-8B/Gemma-2-9B |
| 15 | `extend_sae_adversarial.py` | Extend SAE-feature adversarial-set cache to Llama/Gemma |
| 16 | `gemma_suppression_significance.py` | Significance test for Gemma's SAE-suppression curve |
| 17 | `transfer_direction.py` | Cross-model refusal-direction transfer (Qwen3-8B <-> Llama-3.1-8B) |
| 18 | `expand_worksheet.py` | Expand the moralize-vs-comply labeling worksheet |
| 19 | `validate_classifier.py` | Validate the moralize-vs-comply classifier against local judges |
| 20 | `rescore_compliance.py` | Rescore scripts/ablate_qwen3_direction.py's result for true harmful compliance |
| 21 | `export_directions.py` | Export precomputed dense-direction vectors for the live-inference API |
| 22 | `replicate_llama_ablation.py` | Independent replication of Llama's own-ablation effect at larger N |
| 23 | `sufficiency_at_scale.py` | Sufficiency (activation addition) at 7-9B scale |
| 24 | `analyze_llama_causal_gap.py` | Investigate why Llama's dense-direction necessity/sufficiency are weak vs. its SAE feature |
| 25 | `transfer_sufficiency.py` | Cross-model sufficiency transfer (Qwen3-8B <-> Llama-3.1-8B) |

Full rationale for every methodology choice is in
[DECISIONS.md](../DECISIONS.md); results are in
[RESULTS.md](../RESULTS.md); how each technique works is in
[METHODOLOGY.md](../METHODOLOGY.md).
