from __future__ import annotations

from abc import ABC, abstractmethod

from aaber.data.schema import EvidenceResult, EvidenceUnit, Query


class RetrievalAction(ABC):
    name: str
    default_cost: float = 1.0

    def estimate_cost(self, query: Query, unit: EvidenceUnit | None = None) -> float:
        return self.default_cost

    @abstractmethod
    def execute(
        self,
        query: Query,
        seed_units: list[EvidenceUnit],
        top_k: int = 10,
    ) -> list[EvidenceResult]:
        raise NotImplementedError

