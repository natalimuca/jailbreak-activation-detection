"""Live single-prompt scoring for all four Phase 4 detectors, for the
interactive UI backend (Phase 7).

The core constraint driving this module's design: the local GPU (6GB) can
hold at most one heavy model at a time. A target model (up to 8B, 4-bit) and
the perplexity filter's backbone (Olmo-3-1025-7B, 4-bit) together would
need ~8-9GB, so they can never be resident simultaneously -- `DetectorInferenceManager`
keeps at most one of {target model, perplexity model} loaded, evicting
the other before loading whichever is needed next. SAE weights are the
exception: `src.sae.*`'s `load_sae` defaults to `device="cpu"` (this is
already how scripts/rank_sae_features.py/05's causal ranking/validation avoid competing with
the target model for VRAM), so SAE weights are cached on CPU per
currently-resident target model and never touch this GPU-residency budget.

Every score is computed by calling the actual Phase 4 detector code
(`src.detectors.*`, `src.baselines.*`) against already-calibrated
thresholds from `src.api.model_registry` -- this module only handles
model residency and live activation extraction, never scoring logic
itself, so a live prompt's classification can never silently drift from
what reports/RESULTS.md reports.
"""

from __future__ import annotations

import os
import threading

import torch

from src.activations.extract import format_prompt, load_model
from src.api.model_registry import ModelConfig, load_model_config
from src.baselines.keyword_filter import is_flagged as keyword_is_flagged
from src.baselines.keyword_filter import matched_keywords
from src.baselines.llm_judge import is_flagged as judge_is_flagged
from src.baselines.llm_judge import load_cache as load_judge_cache
from src.baselines.llm_judge import save_cache as save_judge_cache
from src.baselines.llm_judge import score as judge_score
from src.baselines.perplexity_filter import compute_perplexity, load_perplexity_model
from src.baselines.perplexity_filter import is_flagged as perplexity_is_flagged
from src.detectors.dense_direction_detector import is_flagged as dense_is_flagged
from src.detectors.dense_direction_detector import project as dense_project
from src.detectors.nonlinear_combiner_detector import is_flagged as nonlinear_is_flagged
from src.detectors.nonlinear_combiner_detector import load_pipeline as load_nonlinear_pipeline
from src.detectors.nonlinear_combiner_detector import score as nonlinear_score
from src.detectors.sae_feature_detector import is_flagged as sae_is_flagged
from src.detectors.sae_feature_detector import score as sae_score
from src.sae.registry import SAE_PROVIDERS

ALL_DETECTORS = ("keyword", "perplexity", "llm_judge", "dense_direction", "sae_feature", "nonlinear_combiner")


class DetectorInferenceManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target_model = None
        self._target_hf_name: str | None = None
        self._ppl_model = None
        self._ppl_tokenizer = None
        self._sae_cache: dict[tuple[str, int], object] = {}
        self._nonlinear_cache: dict[str, object] = {}

    # -- GPU residency -----------------------------------------------------

    def _evict_target(self) -> None:
        if self._target_model is not None:
            del self._target_model
            self._target_model = None
            self._target_hf_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _evict_perplexity(self) -> None:
        if self._ppl_model is not None:
            del self._ppl_model, self._ppl_tokenizer
            self._ppl_model = None
            self._ppl_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_target_resident(self, config: ModelConfig) -> None:
        if self._target_hf_name == config.hf_name:
            return
        self._evict_perplexity()
        self._evict_target()
        self._target_model = load_model(config.hf_name, load_in_4bit=config.load_in_4bit)
        self._target_hf_name = config.hf_name
        # Bound CPU-side SAE memory: only the currently-resident model's SAE
        # weights are worth keeping around (Qwen-Scope layers alone are
        # ~2GB each -- see src/sae/qwen_scope.py's docstring).
        self._sae_cache = {k: v for k, v in self._sae_cache.items() if k[0] == config.hf_name}

    def _ensure_perplexity_resident(self) -> None:
        if self._ppl_model is not None:
            return
        self._evict_target()
        self._ppl_model, self._ppl_tokenizer = load_perplexity_model()

    def _get_sae(self, hf_name: str, layer: int):
        key = (hf_name, layer)
        if key not in self._sae_cache:
            load_sae_fn = SAE_PROVIDERS[hf_name][0]
            self._sae_cache[key] = load_sae_fn(layer)
        return self._sae_cache[key]

    def _get_nonlinear_pipeline(self, cache_label: str, pipeline_path):
        # Unlike _sae_cache (multi-GB weights, pruned on model swap), fitted
        # sklearn pipelines are tiny -- kept for the manager's whole lifetime.
        if cache_label not in self._nonlinear_cache:
            self._nonlinear_cache[cache_label] = load_nonlinear_pipeline(pipeline_path)
        return self._nonlinear_cache[cache_label]

    # -- Single-prompt activation extraction --------------------------------

    @torch.no_grad()
    def _extract_layers(self, prompt_text: str, layers: list[int]) -> dict[int, torch.Tensor]:
        """Residual-stream activations at the last prompt token, for exactly
        the layers this request needs (not every layer, unlike
        `src.activations.extract.get_last_token_resid_acts`'s full-corpus
        extraction). Layers must be visited in ascending order inside the
        trace -- nnsight raises `MissedProviderError` on out-of-order Envoy
        access (see reports/DECISIONS.md), so callers must pass `layers` sorted."""
        prompt = format_prompt(self._target_model, prompt_text)
        saved = {}
        with self._target_model.trace(prompt, use_cache=False):
            for layer_idx in layers:
                saved[layer_idx] = self._target_model.model.layers[layer_idx].output[:, -1, :].save()
        return {layer_idx: t.detach().cpu().float() for layer_idx, t in saved.items()}

    # -- Per-detector scoring ------------------------------------------------

    def _score_keyword(self, prompt: str, config: ModelConfig) -> dict:
        matched = matched_keywords(prompt)
        threshold = config.keyword_threshold
        return {
            "flagged": keyword_is_flagged(prompt, threshold=int(threshold)),
            "score": len(matched),
            "threshold": threshold,
            "matched_terms": matched,
        }

    def _score_perplexity(self, prompt: str, config: ModelConfig) -> dict:
        ppl = compute_perplexity(prompt, self._ppl_model, self._ppl_tokenizer)
        return {
            "flagged": perplexity_is_flagged(ppl, config.perplexity_threshold),
            "score": ppl,
            "threshold": config.perplexity_threshold,
        }

    def _score_llm_judge(self, prompt: str, config: ModelConfig) -> dict:
        """Text-only, so the score never varies by target model -- but it is
        the one detector that leaves this machine, and Groq's free tier can
        be exhausted or the key absent. Both report as unavailable rather
        than failing the whole request, matching how the SAE panel reports
        models without a pretrained suite."""
        if config.llm_judge_threshold is None:
            return {"available": False, "reason": "no calibrated llm_judge threshold"}
        if not os.environ.get("GROQ_API_KEY"):
            return {"available": False, "reason": "GROQ_API_KEY not set"}
        cache = load_judge_cache()
        try:
            score = judge_score(prompt, cache)
        except Exception as exc:
            return {"available": False, "reason": f"judge API unavailable ({type(exc).__name__})"}
        save_judge_cache(cache)
        return {
            "available": True,
            "flagged": judge_is_flagged(score, config.llm_judge_threshold),
            "score": score,
            "threshold": config.llm_judge_threshold,
        }

    def _score_dense_direction(self, layer_acts: dict[int, torch.Tensor], config: ModelConfig) -> dict:
        acts = layer_acts[config.dense_direction_layer]
        s = dense_project(acts, config.dense_direction).item()
        return {
            "flagged": dense_is_flagged(s, config.dense_direction_threshold),
            "score": s,
            "threshold": config.dense_direction_threshold,
            "layer": config.dense_direction_layer,
        }

    def _score_sae_feature(self, hf_name: str, layer_acts: dict[int, torch.Tensor], config: ModelConfig) -> dict:
        sae_cfg = config.sae_feature
        if sae_cfg is None:
            return {"available": False, "reason": f"no pretrained SAE suite for {config.cache_label}"}

        saes = {layer: self._get_sae(hf_name, layer) for layer in sae_cfg.layers}
        acts_by_layer = {layer: layer_acts[layer] for layer in sae_cfg.layers}
        total = sae_score(acts_by_layer, saes, sae_cfg.features).item()

        # Recomputed per-feature (not just derived from the total) so the
        # "why" breakdown always matches what sae_score actually summed.
        contributions = [
            {"layer": layer, "feature": feature_idx, "contribution": saes[layer].encode(acts_by_layer[layer])[0, feature_idx].item()}
            for layer, feature_idx in sae_cfg.features
        ]
        contributions.sort(key=lambda c: c["contribution"], reverse=True)

        return {
            "available": True,
            "flagged": sae_is_flagged(total, sae_cfg.threshold),
            "score": total,
            "threshold": sae_cfg.threshold,
            "top_features": contributions[:5],
        }

    def _score_nonlinear_combiner(self, hf_name: str, layer_acts: dict[int, torch.Tensor], config: ModelConfig) -> dict:
        nc_cfg = config.nonlinear_combiner
        if nc_cfg is None:
            return {"available": False, "reason": f"no pre-registered non-linear combiner for {config.cache_label}"}

        saes = {layer: self._get_sae(hf_name, layer) for layer in nc_cfg.layers}
        acts_by_layer = {layer: layer_acts[layer] for layer in nc_cfg.layers}
        pipeline = self._get_nonlinear_pipeline(config.cache_label, nc_cfg.pipeline_path)
        total = nonlinear_score(acts_by_layer, saes, nc_cfg.features, pipeline).item()

        return {
            "available": True,
            "flagged": nonlinear_is_flagged(total, nc_cfg.threshold),
            "score": total,
            "threshold": nc_cfg.threshold,
        }

    # -- Public entry point --------------------------------------------------

    def analyze(self, prompt: str, hf_name: str, detectors: list[str] | None = None) -> dict:
        detectors = list(detectors) if detectors is not None else list(ALL_DETECTORS)
        unknown = set(detectors) - set(ALL_DETECTORS)
        if unknown:
            raise ValueError(f"Unknown detector(s): {sorted(unknown)}. Valid: {ALL_DETECTORS}")

        config = load_model_config(hf_name)
        result: dict = {}

        with self._lock:
            if "keyword" in detectors:
                result["keyword"] = self._score_keyword(prompt, config)

            if "llm_judge" in detectors:
                result["llm_judge"] = self._score_llm_judge(prompt, config)

            needs_target = "dense_direction" in detectors or (
                "sae_feature" in detectors and config.sae_feature is not None
            ) or (
                "nonlinear_combiner" in detectors and config.nonlinear_combiner is not None
            )
            layer_acts: dict[int, torch.Tensor] = {}
            if needs_target:
                self._ensure_target_resident(config)
                layers_needed = set()
                if "dense_direction" in detectors:
                    layers_needed.add(config.dense_direction_layer)
                if "sae_feature" in detectors and config.sae_feature is not None:
                    layers_needed.update(config.sae_feature.layers)
                if "nonlinear_combiner" in detectors and config.nonlinear_combiner is not None:
                    layers_needed.update(config.nonlinear_combiner.layers)
                layer_acts = self._extract_layers(prompt, sorted(layers_needed))

            if "dense_direction" in detectors:
                result["dense_direction"] = self._score_dense_direction(layer_acts, config)

            if "sae_feature" in detectors:
                result["sae_feature"] = self._score_sae_feature(hf_name, layer_acts, config)

            if "nonlinear_combiner" in detectors:
                result["nonlinear_combiner"] = self._score_nonlinear_combiner(hf_name, layer_acts, config)

            if "perplexity" in detectors:
                self._ensure_perplexity_resident()
                result["perplexity"] = self._score_perplexity(prompt, config)

        return result
