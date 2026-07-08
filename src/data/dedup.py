"""Deduplication across the four source datasets.

JBB-Behaviors explicitly draws part of its harmful set from HarmBench/TDC and
AdvBench (verified: 27% and 18% of its 100 harmful behaviors respectively, via
its own `Source` column), so combining all four datasets without dedup would
put the same or near-identical prompt in both train and test in some splits --
a real leakage risk, not a hypothetical one.

Two passes:
  1. Exact dedup on normalized text (case/whitespace/punctuation-insensitive).
  2. Near-duplicate dedup via difflib similarity, restricted to same-label
     pairs with similar length (keeps this roughly linear instead of a full
     O(n^2) scan over the whole corpus).

When a duplicate is found, the record from the earlier dataset in
`_SOURCE_PRIORITY` is kept (i.e. prefer the original source over a benchmark
that borrowed the behavior), and the drop is logged for the provenance report.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict

_SOURCE_PRIORITY = {"advbench": 0, "harmbench": 1, "jbb": 2, "alpaca": 3, "xstest": 4}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate(records: list[dict], near_dup_threshold: float = 0.9) -> tuple[list[dict], list[dict]]:
    """records: list of {"text", "label", "source"}.
    Returns (kept_records, dropped_records) -- dropped_records additionally
    carry "duplicate_of" (the text it matched) and "match_type" (exact/near)."""
    records = sorted(records, key=lambda r: _SOURCE_PRIORITY.get(r["source"], 99))

    kept: list[dict] = []
    dropped: list[dict] = []
    seen_exact: dict[str, dict] = {}
    # Bucket kept records by (label, length-bucket) to keep near-dup search cheap.
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for r in records:
        norm = _normalize(r["text"])

        if norm in seen_exact:
            dropped.append({**r, "duplicate_of": seen_exact[norm]["text"], "match_type": "exact"})
            continue

        length_bucket = len(norm) // 10
        candidates = (
            buckets[(r["label"], length_bucket - 1)]
            + buckets[(r["label"], length_bucket)]
            + buckets[(r["label"], length_bucket + 1)]
        )
        near_match = None
        for c in candidates:
            if difflib.SequenceMatcher(None, norm, _normalize(c["text"])).ratio() >= near_dup_threshold:
                near_match = c
                break

        if near_match is not None:
            dropped.append({**r, "duplicate_of": near_match["text"], "match_type": "near"})
            continue

        seen_exact[norm] = r
        buckets[(r["label"], length_bucket)].append(r)
        kept.append(r)

    return kept, dropped
