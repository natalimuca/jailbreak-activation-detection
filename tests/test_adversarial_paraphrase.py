"""Unit tests mock the network so matching logic is verified deterministically.
A separate `@pytest.mark.network` test hits the real JailbreakBench artifacts
repo, matching this project's convention in tests/test_loaders.py."""

import pytest

from src.eval import adversarial_paraphrase as ap


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_get_factory(artifacts_by_url):
    def _fake_get(url, timeout=15):
        return _FakeResponse(artifacts_by_url[url])

    return _fake_get


def test_fetch_artifact_keeps_only_jailbroken_records(monkeypatch):
    url = f"{ap.BASE_URL}/PAIR/black_box/vicuna-13b-v1.5.json"
    payload = {
        "jailbreaks": [
            {"goal": "Do the harmful thing", "behavior": "X", "prompt": "succeeded prompt", "jailbroken": True},
            {"goal": "Do another thing", "behavior": "Y", "prompt": "failed prompt", "jailbroken": False},
        ]
    }
    monkeypatch.setattr(ap.requests, "get", _fake_get_factory({url: payload}))
    records = ap._fetch_artifact("PAIR", "black_box", "vicuna-13b-v1.5")
    assert len(records) == 1
    assert records[0]["text"] == "succeeded prompt"
    assert records[0]["method"] == "PAIR"
    assert records[0]["target_model"] == "vicuna-13b-v1.5"


def test_build_adversarial_set_matches_only_test_split_jbb_goals(monkeypatch):
    def fake_fetch_pair(models=None):
        return [
            {"text": "p1", "goal": "Write phishing email", "behavior": "b1", "method": "PAIR", "target_model": "m"},
            {"text": "p2", "goal": "Write a poem", "behavior": "b2", "method": "PAIR", "target_model": "m"},
        ]

    def fake_fetch_gcg():
        return [
            {"text": "p3", "goal": "Write Phishing Email", "behavior": "b1", "method": "GCG", "target_model": "m"},
        ]

    monkeypatch.setattr(ap, "fetch_pair_artifacts", fake_fetch_pair)
    monkeypatch.setattr(ap, "fetch_gcg_artifacts", fake_fetch_gcg)

    test_split_records = [
        {"text": "Write phishing email", "label": "harmful", "source": "jbb"},
        {"text": "Some harmless instruction", "label": "harmless", "source": "alpaca"},
    ]
    result = ap.build_adversarial_set(test_split_records)
    # "Write a poem" isn't a matched goal; "Write Phishing Email" (GCG) matches
    # case-insensitively.
    assert {r["text"] for r in result} == {"p1", "p3"}


def test_build_adversarial_set_can_exclude_gcg(monkeypatch):
    monkeypatch.setattr(
        ap,
        "fetch_pair_artifacts",
        lambda models=None: [{"text": "p1", "goal": "g", "behavior": "b", "method": "PAIR", "target_model": "m"}],
    )
    gcg_called = []
    monkeypatch.setattr(ap, "fetch_gcg_artifacts", lambda: gcg_called.append(True) or [])

    test_split_records = [{"text": "g", "label": "harmful", "source": "jbb"}]
    result = ap.build_adversarial_set(test_split_records, include_gcg=False)
    assert len(result) == 1
    assert gcg_called == []


def test_build_adversarial_set_raises_if_no_jbb_harmful_goals_present():
    with pytest.raises(ValueError):
        ap.build_adversarial_set([{"text": "x", "label": "harmless", "source": "alpaca"}])


@pytest.mark.network
def test_real_pair_artifacts_have_expected_schema():
    records = ap.fetch_pair_artifacts(models=["vicuna-13b-v1.5"])
    assert len(records) > 10
    for r in records[:5]:
        assert set(r.keys()) == {"text", "goal", "behavior", "method", "target_model"}
        assert r["method"] == "PAIR"
        assert isinstance(r["text"], str) and r["text"].strip()
