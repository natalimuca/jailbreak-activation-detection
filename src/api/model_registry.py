"""Per-model configuration for the interactive detector UI (Phase 7):
which layer/direction/threshold each detector uses for a given target
model, assembled from artifacts every earlier phase already produced
(`results/detector_thresholds_*.json`, `results/dense_direction_cross_model.json`,
`results/dense_directions.pt` from `scripts/export_directions.py`,
`results/sae_causal_ranking_*.json`) -- this module never recalibrates or
re-derives anything, it only looks up already-computed values, so a live
prompt's score is always directly comparable to the numbers already
reported in RESULTS.md.

Only 3 of the 5 models have a pretrained SAE suite (Qwen3-8B,
Llama-3.1-8B-Instruct, gemma-2-9b-it -- see `src.sae.registry.SAE_PROVIDERS`);
the other two (Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct) get
`sae_feature=None` rather than a fabricated fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.detectors.sae_feature_detector import load_top_features
from src.sae.registry import SAE_PROVIDERS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
DENSE_DIRECTIONS_PATH = RESULTS_DIR / "dense_directions.pt"
CROSS_MODEL_RESULTS_PATH = RESULTS_DIR / "dense_direction_cross_model.json"

# hf_name -> cache_label ("Qwen/Qwen3-8B" -> "Qwen3-8B"), same mapping used
# throughout scripts/extend_qwen_smollm.py, 14, 21 -- kept as one table here since the API
# needs it in both directions (list models for the frontend, resolve a
# model_name from a request body).
MODELS: dict[str, str] = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B-Instruct",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
}

SAE_K = 15


@dataclass
class SAEFeatureConfig:
    layers: list[int]
    features: list[tuple[int, int]]
    threshold: float


@dataclass
class ModelConfig:
    hf_name: str
    cache_label: str
    load_in_4bit: bool
    dense_direction_layer: int
    dense_direction: torch.Tensor
    dense_direction_threshold: float
    keyword_threshold: float
    perplexity_threshold: float
    sae_feature: SAEFeatureConfig | None


def _load_thresholds(cache_label: str, results_dir: Path) -> dict:
    """Every model's four thresholds, plus dense-direction layer and (for the
    3 SAE-covered models) SAE layers/k, live in
    `detector_thresholds_<cache_label>.json` -- but that file was only ever
    generated for the 3 models scripts/calibrate_thresholds.py was run against directly
    (Qwen3-8B, Llama-3.1-8B-Instruct, gemma-2-9b-it). For the other two,
    scripts/calibrate_thresholds.py's own documented convention is reused here: keyword/perplexity
    thresholds are model-agnostic (prompt-text-only) so Qwen3-8B's values
    apply unchanged, and the dense-direction layer/threshold come from
    `dense_direction_cross_model.json` instead."""
    per_model_path = results_dir / f"detector_thresholds_{cache_label}.json"
    if per_model_path.exists():
        with open(per_model_path) as fh:
            return json.load(fh)

    with open(results_dir / "detector_thresholds_Qwen3-8B.json") as fh:
        qwen3 = json.load(fh)
    with open(results_dir / "dense_direction_cross_model.json") as fh:
        cross_model = json.load(fh)[cache_label]

    return {
        "dense_direction_layer": cross_model["layer"],
        "thresholds": {
            "keyword": qwen3["thresholds"]["keyword"],
            "perplexity": qwen3["thresholds"]["perplexity"],
            "dense_direction": cross_model["threshold"],
        },
    }


def _load_sae_feature_config(hf_name: str, cache_label: str, thresholds_payload: dict, results_dir: Path) -> SAEFeatureConfig | None:
    if hf_name not in SAE_PROVIDERS or "sae_feature" not in thresholds_payload["thresholds"]:
        return None
    ranking_path = results_dir / f"sae_causal_ranking_{cache_label}.json"
    k = thresholds_payload.get("sae_feature_k", SAE_K)
    features = load_top_features(k=k, path=ranking_path)
    layers = thresholds_payload.get("sae_feature_layers") or sorted({layer for layer, _ in features})
    return SAEFeatureConfig(
        layers=layers,
        features=features,
        threshold=thresholds_payload["thresholds"]["sae_feature"],
    )


def load_model_config(hf_name: str, results_dir: Path = RESULTS_DIR) -> ModelConfig:
    if hf_name not in MODELS:
        raise KeyError(f"Unknown model {hf_name!r}. Known models: {list(MODELS)}")
    cache_label = MODELS[hf_name]

    thresholds_payload = _load_thresholds(cache_label, results_dir)

    directions = torch.load(results_dir / "dense_directions.pt", weights_only=False)
    if cache_label not in directions:
        raise KeyError(
            f"No exported dense direction for {cache_label!r} in {results_dir / 'dense_directions.pt'} "
            "-- run scripts/export_directions.py."
        )
    layer = thresholds_payload["dense_direction_layer"]
    exported = directions[cache_label]
    if exported["layer"] != layer:
        raise ValueError(
            f"{cache_label}: exported dense-direction layer ({exported['layer']}) doesn't match "
            f"the calibrated-threshold layer ({layer}) -- results/dense_directions.pt is stale, "
            "rerun scripts/export_directions.py."
        )

    return ModelConfig(
        hf_name=hf_name,
        cache_label=cache_label,
        load_in_4bit=cache_label in {"Qwen3-8B", "Llama-3.1-8B-Instruct", "gemma-2-9b-it"},
        dense_direction_layer=layer,
        dense_direction=exported["direction"],
        dense_direction_threshold=thresholds_payload["thresholds"]["dense_direction"],
        keyword_threshold=thresholds_payload["thresholds"]["keyword"],
        perplexity_threshold=thresholds_payload["thresholds"]["perplexity"],
        sae_feature=_load_sae_feature_config(hf_name, cache_label, thresholds_payload, results_dir),
    )


def list_models(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Model list for the frontend's model picker: display name plus which
    detectors are actually available, so the UI can grey out SAE-feature for
    the 2 models without a pretrained SAE suite instead of erroring on
    selection."""
    out = []
    for hf_name, cache_label in MODELS.items():
        config = load_model_config(hf_name, results_dir)
        out.append(
            {
                "hf_name": hf_name,
                "cache_label": cache_label,
                "sae_feature_available": config.sae_feature is not None,
            }
        )
    return out
