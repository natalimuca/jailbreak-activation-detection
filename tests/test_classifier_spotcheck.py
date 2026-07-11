import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_stratified_sample_respects_n_per_source_and_covers_all_groups():
    sampling = _load_script("07_sample_completions_for_labeling")
    entries = []
    for source in ["a", "b"]:
        for condition in ["x", "y"]:
            for i in range(5):
                entries.append({"source": source, "condition": condition,
                                 "prompt": f"p{i}", "completion": f"c{i}"})

    sampled = sampling.stratified_sample(entries, n_per_source=2, seed=0)

    assert len(sampled) == 8  # 2 groups x 2 conditions x 2 each
    groups_seen = {(e["source"], e["condition"]) for e in sampled}
    assert groups_seen == {("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")}


def test_stratified_sample_caps_at_group_size_when_n_per_source_too_large():
    sampling = _load_script("07_sample_completions_for_labeling")
    entries = [{"source": "a", "condition": "x", "prompt": "p", "completion": "c"}]

    sampled = sampling.stratified_sample(entries, n_per_source=10, seed=0)

    assert len(sampled) == 1


def test_stratified_sample_is_deterministic_given_seed():
    sampling = _load_script("07_sample_completions_for_labeling")
    entries = [{"source": "a", "condition": "x", "prompt": f"p{i}", "completion": f"c{i}"}
               for i in range(10)]

    first = sampling.stratified_sample(entries, n_per_source=3, seed=42)
    second = sampling.stratified_sample(entries, n_per_source=3, seed=42)

    assert first == second


def test_label_to_binary_covers_all_four_categories():
    scoring = _load_script("08_score_classifier_agreement")

    assert scoring.LABEL_TO_BINARY["refuse"] == "refuse"
    assert scoring.LABEL_TO_BINARY["moralize"] == "non_refuse"
    assert scoring.LABEL_TO_BINARY["partial"] == "non_refuse"
    assert scoring.LABEL_TO_BINARY["comply"] == "non_refuse"
