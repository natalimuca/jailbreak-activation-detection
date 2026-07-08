"""Persist extracted activations + labels/sources to disk so downstream
phases (SAE-feature detector, baselines, evaluation) don't need to re-run
the model every time. Gitignored -- these are large, regenerable artifacts,
not source.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nnsight import LanguageModel

from .extract import get_last_token_resid_acts

CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "activations"


def cache_path(model_name: str) -> Path:
    return CACHE_DIR / f"{model_name.split('/')[-1]}.pt"


def extract_and_cache(model: LanguageModel, model_name: str, records: list[dict]) -> Path:
    """records: list of {"text": str, "label": str, "source": str, "split": str}
    ("split" is optional -- train/val/test tag from src.data.splits, if present).
    Saves a dict with activations [n_layers, n_prompts, d_model] aligned
    1:1 with records' order, plus the labels/sources/texts/splits for slicing."""
    texts = [r["text"] for r in records]
    acts = get_last_token_resid_acts(model, texts)

    payload = {
        "model": model_name,
        "activations": acts,
        "labels": [r["label"] for r in records],
        "sources": [r["source"] for r in records],
        "texts": texts,
    }
    if all("split" in r for r in records):
        payload["splits"] = [r["split"] for r in records]

    path = cache_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_cache(model_name: str) -> dict:
    path = cache_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"No activation cache for {model_name} at {path}. Run scripts/03_extract_all_activations.py first.")
    return torch.load(path, weights_only=False)
