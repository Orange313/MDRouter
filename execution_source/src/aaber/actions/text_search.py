from __future__ import annotations

from aaber.actions.base import RetrievalAction
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query
from aaber.index.base import RetrieverIndex


class TextSearchAction(RetrievalAction):
    name = "text-search"
    default_cost = 1.0

    def __init__(self, index: RetrieverIndex) -> None:
        self.index = index

    def execute(self, query: Query, seed_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        results = self.index.search(query, top_k=top_k)
        for result in results:
            result.source_action = self.name
        return results

