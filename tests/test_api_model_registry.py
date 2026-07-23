"""Tests for src.api.model_registry against synthetic results/ artifacts
(tmp_path), not this machine's real (gitignored, multi-GB) results/
directory -- mirrors test_sae_feature_detector.py's `load_top_features`
test, which does the same with a fabricated ranking JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.api.model_registry import MODELS, list_models, load_model_config


def _write_model_artifacts(results_dir: Path, cache_label: str, with_sae: bool) -> None:
    thresholds = {
        "model": cache_label,
        "n_val": 10,
        "dense_direction_layer": 5,
        "sae_feature_k": 2,
        "sae_feature_layers": [5, 6],
        "thresholds": {"keyword": 1.0, "perplexity": 15.0, "dense_direction": 10.0},
    }
    if with_sae:
        thresholds["thresholds"]["sae_feature"] = 3.0
    (results_dir / f"detector_thresholds_{cache_label}.json").write_text(json.dumps(thresholds))

    if with_sae:
        ranking = {"ranked_features": [{"layer": 5, "feature": 0, "score": 2.0}, {"layer": 6, "feature": 1, "score": 1.5}]}
        (results_dir / f"sae_causal_ranking_{cache_label}.json").write_text(json.dumps(ranking))

    directions_path = results_dir / "dense_directions.pt"
    directions = torch.load(directions_path, weights_only=False) if directions_path.exists() else {}
    directions[cache_label] = {"layer": 5, "direction": torch.tensor([1.0, 0.0])}
    torch.save(directions, directions_path)


def test_load_model_config_reads_dedicated_thresholds_file(tmp_path):
    _write_model_artifacts(tmp_path, "Qwen3-8B", with_sae=True)

    config = load_model_config("Qwen/Qwen3-8B", results_dir=tmp_path)

    assert config.cache_label == "Qwen3-8B"
    assert config.dense_direction_layer == 5
    assert config.dense_direction_threshold == 10.0
    assert config.keyword_threshold == 1.0
    assert config.perplexity_threshold == 15.0
    assert torch.equal(config.dense_direction, torch.tensor([1.0, 0.0]))
    assert config.sae_feature is not None
    assert config.sae_feature.layers == [5, 6]
    assert config.sae_feature.features == [(5, 0), (6, 1)]
    assert config.sae_feature.threshold == 3.0


def test_load_model_config_falls_back_to_cross_model_file_when_no_dedicated_thresholds(tmp_path):
    # Qwen3-8B stands in for the "shared" keyword/perplexity threshold source
    # scripts/10 documents reusing for models it wasn't run against directly.
    _write_model_artifacts(tmp_path, "Qwen3-8B", with_sae=True)
    _write_model_artifacts(tmp_path, "Qwen2.5-1.5B-Instruct", with_sae=False)
    (tmp_path / "detector_thresholds_Qwen2.5-1.5B-Instruct.json").unlink()  # force the fallback path
    (tmp_path / "dense_direction_cross_model.json").write_text(
        json.dumps({"Qwen2.5-1.5B-Instruct": {"layer": 5, "threshold": 12.0}})
    )

    config = load_model_config("Qwen/Qwen2.5-1.5B-Instruct", results_dir=tmp_path)

    assert config.dense_direction_layer == 5
    assert config.dense_direction_threshold == 12.0
    assert config.keyword_threshold == 1.0  # reused from Qwen3-8B, not recomputed
    assert config.perplexity_threshold == 15.0  # reused from Qwen3-8B, not recomputed
    assert config.sae_feature is None  # not in SAE_PROVIDERS, regardless of thresholds payload


def test_load_model_config_unknown_model_raises_key_error(tmp_path):
    with pytest.raises(KeyError):
        load_model_config("not/a-real-model", results_dir=tmp_path)


def test_load_model_config_missing_exported_direction_raises_key_error(tmp_path):
    _write_model_artifacts(tmp_path, "Qwen3-8B", with_sae=True)
    directions = torch.load(tmp_path / "dense_directions.pt", weights_only=False)
    del directions["Qwen3-8B"]
    torch.save(directions, tmp_path / "dense_directions.pt")

    with pytest.raises(KeyError):
        load_model_config("Qwen/Qwen3-8B", results_dir=tmp_path)


def test_load_model_config_stale_exported_layer_raises_value_error(tmp_path):
    _write_model_artifacts(tmp_path, "Qwen3-8B", with_sae=True)
    directions = torch.load(tmp_path / "dense_directions.pt", weights_only=False)
    directions["Qwen3-8B"]["layer"] = 999  # no longer matches the threshold file's layer (5)
    torch.save(directions, tmp_path / "dense_directions.pt")

    with pytest.raises(ValueError):
        load_model_config("Qwen/Qwen3-8B", results_dir=tmp_path)


def test_list_models_reports_sae_availability_per_model(tmp_path):
    sae_covered = {"Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"}
    for hf_name, cache_label in MODELS.items():
        _write_model_artifacts(tmp_path, cache_label, with_sae=hf_name in sae_covered)

    models = list_models(results_dir=tmp_path)

    assert len(models) == len(MODELS)
    sae_flags = {m["hf_name"]: m["sae_feature_available"] for m in models}
    assert sae_flags["Qwen/Qwen3-8B"] is True
    assert sae_flags["meta-llama/Llama-3.1-8B-Instruct"] is True
    assert sae_flags["google/gemma-2-9b-it"] is True
    assert sae_flags["Qwen/Qwen2.5-1.5B-Instruct"] is False
    assert sae_flags["HuggingFaceTB/SmolLM2-1.7B-Instruct"] is False
