from __future__ import annotations

from aaber.data.schema import EvidenceUnit, Query
from aaber.rag.prompt import build_answer_prompt


class ExtractiveAnswerGenerator:
    def generate(self, query: Query, evidence: list[EvidenceUnit]) -> str:
        prompt = build_answer_prompt(query, evidence)
        return prompt

