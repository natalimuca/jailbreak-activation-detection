from src.direction.moralize_comply_classifier import _parse_verdict


def test_parse_verdict_exact_word():
    assert _parse_verdict("moralize") == "moralize"
    assert _parse_verdict("comply") == "comply"
    assert _parse_verdict("partial") == "partial"
    assert _parse_verdict("refuse") == "refuse"


def test_parse_verdict_strips_punctuation_and_case():
    assert _parse_verdict("Moralize.") == "moralize"
    assert _parse_verdict("COMPLY,") == "comply"


def test_parse_verdict_takes_first_word_only():
    # Not a substring search over the whole output -- only the first word
    # counts, so a negated mention later in the text doesn't get matched.
    assert _parse_verdict("moralize (does not comply)") == "moralize"


def test_parse_verdict_unparseable_for_unrecognized_output():
    assert _parse_verdict("I'm not sure how to classify this.") == "unparseable"


def test_parse_verdict_unparseable_for_empty_output():
    assert _parse_verdict("") == "unparseable"
    assert _parse_verdict("   ") == "unparseable"
