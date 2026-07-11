import math

import pytest
import torch

from src.baselines.perplexity_filter import compute_perplexity, is_flagged


class _StubTokenizer:
    """Maps each character to its ordinal as a fake token id, so perplexity
    can be computed deterministically without a real vocabulary/model."""

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[ord(c) % 50 for c in text]])
        return type("Enc", (), {"input_ids": ids})()


class _StubModel:
    """Returns a fixed, uniform next-token loss regardless of input, so the
    test only checks the loss->perplexity conversion (exp(loss)), not any
    real language-modeling behavior."""

    def __init__(self, loss_value: float):
        self.loss_value = loss_value

    def __call__(self, ids, labels=None):
        return type("Out", (), {"loss": torch.tensor(self.loss_value)})()


def test_compute_perplexity_matches_exp_of_loss():
    model = _StubModel(loss_value=2.0)
    tokenizer = _StubTokenizer()
    ppl = compute_perplexity("hello world", model, tokenizer)
    assert ppl == pytest.approx(math.exp(2.0))


def test_compute_perplexity_single_token_returns_baseline():
    model = _StubModel(loss_value=99.0)
    tokenizer = _StubTokenizer()
    assert compute_perplexity("x", model, tokenizer) == 1.0


def test_is_flagged_thresholding():
    assert is_flagged(perplexity=500.0, threshold=100.0)
    assert not is_flagged(perplexity=50.0, threshold=100.0)


@pytest.mark.network
def test_real_gpt2_scores_gibberish_higher_than_fluent_text():
    from src.baselines.perplexity_filter import load_gpt2

    model, tokenizer = load_gpt2()
    fluent = "The quick brown fox jumps over the lazy dog."
    gibberish = "asdf ! kq93 xzP ;;; qq77 !!! blorptron zzq"
    assert compute_perplexity(gibberish, model, tokenizer) > compute_perplexity(fluent, model, tokenizer)
