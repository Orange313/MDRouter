from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from aaber.data.schema import EvidenceResult, EvidenceUnit, Query


class RetrieverIndex(ABC):
    name: str

    @abstractmethod
    def build(self, units: list[EvidenceUnit]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "RetrieverIndex":
        raise NotImplementedError

