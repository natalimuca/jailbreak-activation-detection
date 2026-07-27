from __future__ import annotations

THINK_CLOSE = "</think>"
THINK_CLOSE_ID = 151649


def extract_answer(completion: str) -> str | None:
    if THINK_CLOSE not in completion:
        return None
    return completion.split(THINK_CLOSE, 1)[1]
