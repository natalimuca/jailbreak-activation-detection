"""FastAPI app for the interactive detector UI (Phase 7): a thin HTTP layer
over `src.api.inference_manager.DetectorInferenceManager` and
`src.api.model_registry`. All scoring/GPU-residency logic lives in those
modules -- this file only does request validation and error translation, so
it stays testable without a GPU (see `tests/test_api_main.py`, which
overrides `get_manager` with a fake).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from src.api.inference_manager import DetectorInferenceManager
from src.api.model_registry import list_models
from src.api.schemas import AnalyzeRequest, AnalyzeResponse, ModelInfo

app = FastAPI(title="Jailbreak Activation Detector")

_manager = DetectorInferenceManager()


def get_manager() -> DetectorInferenceManager:
    return _manager


@app.get("/api/models", response_model=list[ModelInfo])
def get_models() -> list[dict]:
    return list_models()


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, manager: DetectorInferenceManager = Depends(get_manager)) -> dict:
    try:
        return manager.analyze(request.prompt, request.model_name, request.detectors)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
