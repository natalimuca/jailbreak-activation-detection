"""Integration tests for the dataset loaders -- these hit the network (HF
Hub / GitHub raw) since the loaders fetch real benchmark data on every call
by design (no bundled copies of the datasets in this repo). Skipped
automatically if there's no network access, rather than failing the suite."""

import pytest

from src.data.loaders import (
    load_all_labeled_prompts,
    load_harmbench,
    load_harmful,
    load_harmless,
    load_jbb_benign,
    load_jbb_harmful,
    load_xstest_safe,
    load_xstest_unsafe,
)

pytestmark = pytest.mark.network


@pytest.mark.parametrize(
    "loader,expected_min",
    [
        (load_harmful, 20),
        (load_harmless, 20),
        (load_harmbench, 20),
        (load_jbb_harmful, 20),
        (load_jbb_benign, 20),
        (load_xstest_safe, 20),
        (load_xstest_unsafe, 20),
    ],
)
def test_loader_returns_nonempty_string_list(loader, expected_min):
    prompts = loader(n=expected_min)
    assert len(prompts) == expected_min
    assert all(isinstance(p, str) and p.strip() for p in prompts)


def test_load_all_labeled_prompts_schema_and_labels():
    records = load_all_labeled_prompts()
    assert len(records) > 1000
    for r in records[:50]:
        assert set(r.keys()) == {"text", "label", "source"}
        assert r["label"] in {"harmful", "harmless"}
        assert r["source"] in {"advbench", "harmbench", "jbb", "alpaca", "xstest"}


def test_load_all_labeled_prompts_has_both_labels_from_multiple_sources():
    records = load_all_labeled_prompts()
    sources_by_label = {}
    for r in records:
        sources_by_label.setdefault(r["label"], set()).add(r["source"])
    assert len(sources_by_label["harmful"]) >= 3
    assert len(sources_by_label["harmless"]) >= 2
