"""Tests for the FastAPI request/response layer (src.api.main) -- request
validation and error translation only. `get_manager` and `list_models` are
overridden with fakes so these never touch a GPU or this machine's real
(gitignored) results/ directory, matching test_api_inference_manager.py's
approach to the same GPU-code-testing problem."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_manager


class _FakeManager:
    def analyze(self, prompt, model_name, detectors):
        if model_name == "bad/model":
            raise KeyError("Unknown model 'bad/model'")
        if detectors == ["not_a_real_detector"]:
            raise ValueError("Unknown detector(s): ['not_a_real_detector']")
        return {"keyword": {"flagged": False, "score": 0, "threshold": 1.0, "matched_terms": []}}


@pytest.fixture
def client():
    app.dependency_overrides[get_manager] = lambda: _FakeManager()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_analyze_returns_detector_results(client):
    response = client.post("/api/analyze", json={"prompt": "hi", "model_name": "any/model"})
    assert response.status_code == 200
    assert response.json()["keyword"]["flagged"] is False


def test_analyze_unknown_model_returns_400(client):
    response = client.post("/api/analyze", json={"prompt": "hi", "model_name": "bad/model"})
    assert response.status_code == 400


def test_analyze_unknown_detector_returns_400(client):
    response = client.post(
        "/api/analyze", json={"prompt": "hi", "model_name": "any/model", "detectors": ["not_a_real_detector"]}
    )
    assert response.status_code == 400


def test_analyze_rejects_empty_prompt(client):
    response = client.post("/api/analyze", json={"prompt": "", "model_name": "any/model"})
    assert response.status_code == 422


def test_get_models_returns_registry_output(client, monkeypatch):
    fake_models = [{"hf_name": "x/y", "cache_label": "y", "sae_feature_available": True}]
    monkeypatch.setattr("src.api.main.list_models", lambda: fake_models)

    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == fake_models
