from __future__ import annotations

from aaber.actions.base import RetrievalAction
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query


class MultiVectorRerankAction(RetrievalAction):
    name = "multivector-rerank"
    default_cost = 8.0

    def __init__(self, index) -> None:
        self.index = index

    def execute(self, query: Query, seed_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        results = self.index.rerank(query, seed_units or self.index.units, top_k=top_k)
        for result in results:
            result.source_action = self.name
        return results


class MultiVectorSearchAction(RetrievalAction):
    name = "multivector-search"
    default_cost = 8.0

    def __init__(self, index) -> None:
        self.index = index

    def execute(self, query: Query, seed_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        results = self.index.search(query, top_k=top_k)
        for result in results:
            result.source_action = self.name
        return results
