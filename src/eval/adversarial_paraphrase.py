"""Fetches real, published jailbreak attack artifacts from JailbreakBench
(github.com/JailbreakBench/artifacts) and matches them against this
project's own JBB-sourced harmful behaviors already in the TEST split --
Phase 4's adversarial paraphrase/robustness test set.

Two attack families are used, for two different failure modes:

- **PAIR** (black_box): fluent, roleplay-style paraphrases of the harmful
  goal -- the "adversarial paraphrase" case this phase is named for,
  expected to slip past a keyword filter (no lexicon vocabulary) but NOT a
  perplexity filter (still fluent, low-perplexity text).
- **GCG** (white_box + transfer): the harmful goal plus an optimized
  gibberish suffix -- the classic case perplexity filters ARE built to
  catch (Alon & Kamfonas 2023). Kept as a contrasting condition, not the
  headline "paraphrase" case.

Only `jailbroken: true` records are kept -- attacks that actually succeeded
against their original target model, i.e. real disguised-harmful prompts,
not failed attempts. This is explicitly NOT a claim that these prompts
would jailbreak this project's own models (Qwen3-8B, Qwen2.5, SmolLM2) --
that was never tested. Only the *prompt surface form* is reused here, to
test whether it fools a *prompt classifier*, independent of whether it
would successfully jailbreak our target model.
"""

from __future__ import annotations

import requests

BASE_URL = "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts"

PAIR_TARGET_MODELS = ["vicuna-13b-v1.5", "llama-2-7b-chat-hf", "gpt-3.5-turbo-1106", "gpt-4-0125-preview"]
GCG_WHITE_BOX_MODELS = ["vicuna-13b-v1.5", "llama-2-7b-chat-hf"]
GCG_TRANSFER_MODELS = ["gpt-3.5-turbo-1106", "gpt-4-0125-preview"]


def _fetch_artifact(method: str, subdir: str, model: str) -> list[dict]:
    url = f"{BASE_URL}/{method}/{subdir}/{model}.json"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    records = []
    for r in data["jailbreaks"]:
        if not r.get("jailbroken"):
            continue
        records.append(
            {
                "text": r["prompt"],
                "goal": r["goal"],
                "behavior": r["behavior"],
                "method": method,
                "target_model": model,
            }
        )
    return records


def fetch_pair_artifacts(models: list[str] = PAIR_TARGET_MODELS) -> list[dict]:
    records = []
    for model in models:
        records.extend(_fetch_artifact("PAIR", "black_box", model))
    return records


def fetch_gcg_artifacts() -> list[dict]:
    records = []
    for model in GCG_WHITE_BOX_MODELS:
        records.extend(_fetch_artifact("GCG", "white_box", model))
    for model in GCG_TRANSFER_MODELS:
        records.extend(_fetch_artifact("GCG", "transfer", model))
    return records


def build_adversarial_set(test_split_records: list[dict], include_gcg: bool = True) -> list[dict]:
    """test_split_records: this project's own TEST-split records (e.g.
    `src.data.splits.apply_manifest(...)["test"]`) -- used only to restrict
    matching to goals already held out from every detector's calibration,
    so the adversarial set doesn't leak against anything tuned on
    TRAIN/VAL. Matches on exact (case/whitespace-normalized) goal text
    against records with source == "jbb" and label == "harmful"."""
    allowed_goals = {
        r["text"].strip().lower()
        for r in test_split_records
        if r.get("source") == "jbb" and r.get("label") == "harmful"
    }
    if not allowed_goals:
        raise ValueError(
            "No JBB harmful goals found in the given TEST-split records -- "
            "check that test_split_records includes the full corpus's test split."
        )

    artifacts = fetch_pair_artifacts()
    if include_gcg:
        artifacts += fetch_gcg_artifacts()

    return [r for r in artifacts if r["goal"].strip().lower() in allowed_goals]
