"""Verifies both models' activation caches cover identical prompts in
identical order -- required for any cross-model comparison to be valid.
Skipped if the (gitignored, locally-generated) caches aren't present."""

from pathlib import Path

import pytest

from src.activations.cache import assert_caches_consistent, cache_path

MODEL_NAMES = ["Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]


def _caches_available() -> bool:
    return all(cache_path(name).exists() for name in MODEL_NAMES)


@pytest.mark.skipif(not _caches_available(), reason="activation caches not present locally")
def test_caches_cover_identical_prompts_in_identical_order():
    assert_caches_consistent(MODEL_NAMES)  # raises with a specific message on mismatch


def test_assert_caches_consistent_is_noop_for_fewer_than_two_models():
    assert_caches_consistent(["only-one-model"]) is None  # should not raise / not try to load anything
