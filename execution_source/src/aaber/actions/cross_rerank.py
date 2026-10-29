from __future__ import annotations

from aaber.actions.base import RetrievalAction
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query


class CrossEncoderRerankAction(RetrievalAction):
    name = "cross-rerank"
    # Frozen from the five-action server benchmark: approximately 5.6x BM25.
    default_cost = 5.6

    def __init__(self, index) -> None:
        self.index = index

    def execute(self, query: Query, seed_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        results = self.index.rerank(query, seed_units or self.index.units, top_k=top_k)
        for result in results:
            result.source_action = self.name
        return results

    def execute_batch_candidates(
        self,
        queries: list[Query],
        seed_units_by_query: dict[str, list[EvidenceUnit]],
        top_k: int = 10,
    ) -> dict[str, list[EvidenceResult]]:
        results = self.index.rerank_many(queries, seed_units_by_query, top_k=top_k)
        for rows in results.values():
            for result in rows:
                result.source_action = self.name
        return results
