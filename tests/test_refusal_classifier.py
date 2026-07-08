from src.direction.refusal_classifier import is_degenerate, is_refusal, refusal_rate, refusal_stats


def test_is_refusal_detects_common_phrasings():
    assert is_refusal("I'm sorry, but I cannot assist with that request.")
    assert is_refusal("As an AI language model, I do not have personal opinions.")
    assert is_refusal("I can't help with that.")


def test_is_refusal_false_for_compliant_completion():
    assert not is_refusal("Sure! Here is a haiku about autumn leaves falling gently.")


def test_is_refusal_only_checks_first_200_chars():
    # A refusal phrase appearing only after the 200-char window shouldn't count --
    # matches how the classifier is used (checking whether the model refuses
    # up front, not whether it apologizes for something mid-completion).
    padding = "x" * 250
    completion = padding + " I'm sorry, I cannot help with that."
    assert not is_refusal(completion)


def test_is_degenerate_detects_repeated_token_collapse():
    assert is_degenerate("I I I I I I I I I I I I I I I I")


def test_is_degenerate_false_for_normal_text():
    assert not is_degenerate("The quick brown fox jumps over the lazy dog near the river bank")


def test_is_degenerate_false_for_short_completions():
    # Short completions naturally repeat words without being degenerate --
    # the length guard avoids false positives here.
    assert not is_degenerate("No no no")


def test_refusal_rate_basic():
    completions = ["I cannot help with that.", "Sure, here you go.", "I'm sorry, I can't."]
    assert refusal_rate(completions) == 2 / 3


def test_refusal_rate_empty_list():
    assert refusal_rate([]) == 0.0


def test_refusal_stats_matches_refusal_rate():
    completions = ["I cannot help.", "Sure, here."]
    stats = refusal_stats(completions)
    assert stats["n"] == 2
    assert stats["count"] == 1
    assert stats["rate"] == 0.5


def test_refusal_stats_ci_bounds_are_valid_probabilities():
    stats = refusal_stats(["I cannot."] * 5 + ["Sure."] * 5)
    assert 0.0 <= stats["ci_low"] <= stats["rate"] <= stats["ci_high"] <= 1.0


def test_refusal_stats_ci_widens_at_small_n():
    all_refuse_small = refusal_stats(["I cannot."] * 3)
    all_refuse_large = refusal_stats(["I cannot."] * 300)
    small_width = all_refuse_small["ci_high"] - all_refuse_small["ci_low"]
    large_width = all_refuse_large["ci_high"] - all_refuse_large["ci_low"]
    assert small_width > large_width


def test_refusal_stats_empty():
    stats = refusal_stats([])
    assert stats == {"n": 0, "count": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
