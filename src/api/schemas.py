"""Pydantic request/response contracts for the interactive detector UI's
API (Phase 7). Kept separate from `inference_manager.py` so the contract
can be unit-tested (serialization, validation errors) without touching
any GPU code."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.api.inference_manager import ALL_DETECTORS


class ModelInfo(BaseModel):
    hf_name: str
    cache_label: str
    sae_feature_available: bool


class AnalyzeRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model_name: str
    detectors: list[str] | None = Field(
        default=None,
        description=f"Subset of {list(ALL_DETECTORS)} to run; omit for all four.",
    )


class KeywordResult(BaseModel):
    flagged: bool
    score: int
    threshold: float
    matched_terms: list[str]


class PerplexityResult(BaseModel):
    flagged: bool
    score: float
    threshold: float


class DenseDirectionResult(BaseModel):
    flagged: bool
    score: float
    threshold: float
    layer: int


class SAEFeatureContribution(BaseModel):
    layer: int
    feature: int
    contribution: float


class SAEFeatureResult(BaseModel):
    available: bool
    reason: str | None = None
    flagged: bool | None = None
    score: float | None = None
    threshold: float | None = None
    top_features: list[SAEFeatureContribution] | None = None


class ExamplePrompt(BaseModel):
    id: int
    method: str
    behavior: str
    goal: str
    text: str


class ExamplesResponse(BaseModel):
    available: bool
    reason: str | None = None
    examples: list[ExamplePrompt] = []


class AttributionPrompt(BaseModel):
    goal: str
    text: str
    tokens: list[str]
    importance: list[float]
    top_tokens: list[tuple[str, float]]


class AttributionResponse(BaseModel):
    available: bool
    reason: str | None = None
    models: dict[str, list[AttributionPrompt]] = {}


class AnalyzeResponse(BaseModel):
    keyword: KeywordResult | None = None
    perplexity: PerplexityResult | None = None
    dense_direction: DenseDirectionResult | None = None
    sae_feature: SAEFeatureResult | None = None
