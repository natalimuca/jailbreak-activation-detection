"""FastAPI app for the interactive detector UI (Phase 7): a thin HTTP layer
over `src.api.inference_manager.DetectorInferenceManager` and
`src.api.model_registry`. All scoring/GPU-residency logic lives in those
modules -- this file only does request validation and error translation, so
it stays testable without a GPU (see `tests/test_api_main.py`, which
overrides `get_manager` with a fake).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.api.inference_manager import DetectorInferenceManager
from src.api.model_registry import list_models
from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AttributionResponse,
    ExamplesResponse,
    ModelInfo,
)

WEBAPP_DIR = Path(__file__).resolve().parents[2] / "webapp"
# results/ is gitignored, so a fresh clone has no adversarial manifest -- the
# endpoint reports that as available=false rather than 404ing, matching how
# the SAE panel reports models without a pretrained SAE suite.
EXAMPLES_PATH = Path(__file__).resolve().parents[2] / "results" / "adversarial_paraphrase_manifest.json"
EXAMPLES_PER_METHOD = 4
ATTRIBUTION_PATH = Path(__file__).resolve().parents[2] / "results" / "token_attribution.json"

app = FastAPI(title="Jailbreak Activation Detector")

_manager = DetectorInferenceManager()


def get_manager() -> DetectorInferenceManager:
    return _manager


@app.get("/api/models", response_model=list[ModelInfo])
def get_models() -> list[dict]:
    return list_models()


@app.get("/api/examples", response_model=ExamplesResponse)
def get_examples() -> dict:
    if not EXAMPLES_PATH.exists():
        return {
            "available": False,
            "reason": "adversarial manifest not found -- run scripts/build_adversarial_set.py",
            "examples": [],
        }
    with open(EXAMPLES_PATH) as fh:
        records = json.load(fh)
    examples: list[dict] = []
    for method in sorted({r["method"] for r in records}):
        for i, record in enumerate(r for r in records if r["method"] == method):
            if i >= EXAMPLES_PER_METHOD:
                break
            examples.append(
                {
                    "id": len(examples),
                    "method": method,
                    "behavior": record["behavior"],
                    "goal": record["goal"],
                    "text": record["text"],
                }
            )
    return {"available": True, "examples": examples}


@app.get("/api/attribution", response_model=AttributionResponse)
def get_attribution() -> dict:
    if not ATTRIBUTION_PATH.exists():
        return {
            "available": False,
            "reason": "no attribution run found -- run scripts/token_attribution.py",
            "models": {},
        }
    with open(ATTRIBUTION_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {
        "available": True,
        "models": {
            model: [
                {
                    "goal": record["goal"],
                    "text": record["text"],
                    "tokens": record["tokens"],
                    "importance": record["importance"],
                    "top_tokens": record["top_tokens"],
                }
                for record in records
            ]
            for model, records in raw.items()
        },
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, manager: DetectorInferenceManager = Depends(get_manager)) -> dict:
    try:
        return manager.analyze(request.prompt, request.model_name, request.detectors)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# Mounted last, after every /api/* route above -- Starlette matches routes in
# registration order, so this catch-all only ever serves the frontend's
# static files and never shadows the API.
app.mount("/", StaticFiles(directory=WEBAPP_DIR, html=True), name="webapp")
