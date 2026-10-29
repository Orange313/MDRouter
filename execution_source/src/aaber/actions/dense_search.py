from __future__ import annotations

from aaber.actions.base import RetrievalAction
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query


class DenseSearchAction(RetrievalAction):
    """Independent semantic text retrieval using a sentence-transformer index."""

    name = "dense-search"
    # Frozen from the five-action server benchmark: 9.071 / 14.363 ~= 0.632,
    # rounded to two decimals for the budgeted routing experiments.
    default_cost = 0.63

    def __init__(self, index) -> None:
        self.index = index

    def execute(
        self,
        query: Query,
        seed_units: list[EvidenceUnit],
        top_k: int = 10,
    ) -> list[EvidenceResult]:
        results = self.index.search(query, top_k=top_k)
        for result in results:
            result.source_action = self.name
        return results
