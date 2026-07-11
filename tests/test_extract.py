"""Regression test for 4-bit model loading -- a transformers>=5 upgrade removed
the legacy `load_in_4bit=True` shorthand kwarg (it now must go through
BitsAndBytesConfig), which silently broke `load_model(..., load_in_4bit=True)`
until a real Qwen3-8B extraction run hit the TypeError. Uses the small,
already-cached SmolLM2 model so this catches the regression without needing
a large download."""

import pytest

pytestmark = pytest.mark.model

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def test_load_model_with_4bit_quantization_does_not_raise():
    from src.activations.extract import load_model

    model = load_model(MODEL_NAME, load_in_4bit=True)
    assert model is not None
