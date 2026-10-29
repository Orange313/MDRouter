from __future__ import annotations

from aaber.actions.base import RetrievalAction
from aaber.common.text import keyword_overlap
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query


class VLMVerifyAction(RetrievalAction):
    """Cheap stand-in for selective VLM verification."""

    name = "vlm-verify"
    default_cost = 20.0

    def execute(self, query: Query, seed_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        scored = []
        for unit in seed_units:
            score = keyword_overlap(query.question, unit.searchable_text)
            if unit.image_path and any(term in query.question.lower() for term in ["image", "figure", "chart", "table"]):
                score += 0.1
            scored.append((unit.unit_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            EvidenceResult(query.query_id, unit_id, score, self.name, rank=idx)
            for idx, (unit_id, score) in enumerate(scored[:top_k], start=1)
        ]

