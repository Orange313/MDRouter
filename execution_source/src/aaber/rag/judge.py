from __future__ import annotations

from aaber.common.text import keyword_overlap


def evidence_support_score(answer: str, evidence_text: str) -> float:
    return keyword_overlap(answer, evidence_text)

