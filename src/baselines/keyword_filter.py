"""Curated keyword/phrase lexicon for harmful-intent prompt detection --
the classic decades-old content-moderation baseline (match a fixed lexicon
against the input text), included here specifically as a *weak* baseline
for Phase 4's head-to-head: it can only catch requests that use recognizable
surface vocabulary, so it's expected to degrade sharply on paraphrased or
fictionally-framed harmful requests (the adversarial paraphrase set this
project builds from JailbreakBench artifacts).

Terms are grouped by the harm categories AdvBench/HarmBench/JBB-Behaviors
already use (weapons/explosives, malware/hacking, drugs, violence, fraud,
illegal activity) -- generic category vocabulary, not copied from any
specific benchmark behavior string, so this doesn't just memorize the
training corpus's exact phrasing.
"""

from __future__ import annotations

_KEYWORDS = [
    # weapons / explosives
    "bomb", "explosive", "detonator", "grenade", "assassinate", "assassination",
    "firearm", "gun schematic", "improvised weapon", "chemical weapon",
    "biological weapon", "nerve agent", "poison", "nuclear device",
    # malware / hacking
    "malware", "ransomware", "virus", "trojan", "keylogger", "spyware",
    "ddos", "botnet", "sql injection", "exploit", "zero-day", "hack into",
    "bypass security", "disable antivirus", "phishing", "backdoor",
    "crack the password", "brute force password",
    # drugs
    "synthesize methamphetamine", "synthesize meth", "cook meth",
    "manufacture drugs", "illegal drug", "narcotic synthesis",
    # violence / self-harm
    "kill someone", "murder someone", "torture", "how to commit suicide",
    "mass shooting", "terrorist attack", "ethnic cleansing", "genocide",
    # fraud / financial crime
    "launder money", "money laundering", "counterfeit currency",
    "identity theft", "credit card fraud", "steal credentials", "tax evasion scheme",
    "insider trading",
    # illegal activity / trafficking
    "human trafficking", "child exploitation", "smuggle contraband",
    "sell stolen goods", "forge documents", "forge a passport",
]


def matched_keywords(prompt: str) -> list[str]:
    text = prompt.lower()
    return [kw for kw in _KEYWORDS if kw in text]


def score(prompt: str) -> int:
    """Number of distinct lexicon terms found in the prompt."""
    return len(matched_keywords(prompt))


def is_flagged(prompt: str, threshold: int = 1) -> bool:
    return score(prompt) >= threshold
