from __future__ import annotations

from aaber.actions.base import RetrievalAction
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query
from aaber.index.base import RetrieverIndex


class ImageSearchAction(RetrievalAction):
    name = "image-search"
    default_cost = 3.0

    def __init__(self, index: RetrieverIndex) -> None:
        self.index = index

    def execute(self, query: Query, seed_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        results = self.index.search(query, top_k=top_k)
        for result in results:
            result.source_action = self.name
        return results

    def execute_batch(self, queries: list[Query], top_k: int = 10) -> dict[str, list[EvidenceResult]]:
        if hasattr(self.index, "search_batch"):
            by_query = self.index.search_batch(queries, top_k=top_k)  # type: ignore[attr-defined]
        else:
            by_query = {query.query_id: self.index.search(query, top_k=top_k) for query in queries}
        for results in by_query.values():
            for result in results:
                result.source_action = self.name
        return by_query
