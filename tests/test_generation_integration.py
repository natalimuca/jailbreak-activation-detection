"""End-to-end sanity check for the generation-based intervention functions.
Loads a real (small) model and runs actual generation -- slow and needs a
GPU, so marked separately from the fast default suite (see pytest.ini)."""

import pytest
import torch

pytestmark = pytest.mark.model

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


@pytest.fixture(scope="module")
def model():
    from src.activations.extract import load_model

    return load_model(MODEL_NAME)


def test_generate_baseline_produces_nonempty_text(model):
    from src.direction.interventions import generate_baseline

    completion = generate_baseline(model, "Explain how photosynthesis works", max_new_tokens=10)
    assert isinstance(completion, str)
    assert len(completion.strip()) > 0


def test_generate_with_ablation_runs_without_error(model):
    from src.activations.extract import n_layers
    from src.direction.interventions import generate_with_ablation

    d_model = model.config.hidden_size
    direction = torch.zeros(d_model)
    direction[0] = 1.0  # arbitrary unit direction -- just checking the intervention wiring works
    completion = generate_with_ablation(model, "Write a short poem about the sea", direction, max_new_tokens=10)
    assert isinstance(completion, str)
    assert len(completion.strip()) > 0


def test_generate_with_addition_runs_without_error(model):
    from src.direction.interventions import generate_with_addition

    d_model = model.config.hidden_size
    raw_direction = torch.zeros(d_model)  # zero vector -- addition should be a no-op, just checking it runs
    completion = generate_with_addition(
        model, "Give me a tip for staying focused while studying", raw_direction, layer_idx=5, alpha=1.0, max_new_tokens=10
    )
    assert isinstance(completion, str)
    assert len(completion.strip()) > 0
