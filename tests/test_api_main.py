"""Tests for the FastAPI request/response layer (src.api.main) -- request
validation and error translation only. `get_manager` and `list_models` are
overridden with fakes so these never touch a GPU or this machine's real
(gitignored) results/ directory, matching test_api_inference_manager.py's
approach to the same GPU-code-testing problem."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import src.api.main as main
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


def test_landing_page_serves_landing_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Jailbreak Detection via Internal Activations" in response.text


def test_get_models_returns_registry_output(client, monkeypatch):
    fake_models = [{"hf_name": "x/y", "cache_label": "y", "sae_feature_available": True, "nonlinear_combiner_available": False}]
    monkeypatch.setattr("src.api.main.list_models", lambda: fake_models)

    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == fake_models


def test_get_examples_reports_unavailable_when_manifest_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "EXAMPLES_PATH", tmp_path / "absent.json")
    body = client.get("/api/examples").json()
    assert body["available"] is False
    assert body["examples"] == []


def test_get_examples_caps_records_per_method(client, monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"text": f"t{i}", "goal": f"g{i}", "behavior": f"b{i}", "method": method}
                for method in ("PAIR", "GCG")
                for i in range(10)
            ]
        )
    )
    monkeypatch.setattr(main, "EXAMPLES_PATH", manifest)
    monkeypatch.setattr(main, "EXAMPLES_PER_METHOD", 2)

    body = client.get("/api/examples").json()
    assert body["available"] is True
    assert [e["method"] for e in body["examples"]] == ["GCG", "GCG", "PAIR", "PAIR"]
    assert [e["id"] for e in body["examples"]] == [0, 1, 2, 3]


def test_get_attribution_reports_unavailable_when_results_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ATTRIBUTION_PATH", tmp_path / "absent.json")
    body = client.get("/api/attribution").json()
    assert body["available"] is False
    assert body["models"] == {}


def test_get_attribution_returns_parallel_token_and_importance_lists(client, monkeypatch, tmp_path):
    path = tmp_path / "token_attribution.json"
    path.write_text(
        json.dumps(
            {
                "ModelA": [
                    {
                        "goal": "g",
                        "text": "t",
                        "tokens": ["\u0120alpha", "beta"],
                        "importance": [0.5, -0.25],
                        "top_tokens": [["\u0120alpha", 0.5]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "ATTRIBUTION_PATH", path)

    body = client.get("/api/attribution").json()
    record = body["models"]["ModelA"][0]
    assert body["available"] is True
    assert len(record["tokens"]) == len(record["importance"])
    assert record["importance"] == [0.5, -0.25]
