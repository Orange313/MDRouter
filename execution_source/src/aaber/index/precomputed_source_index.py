from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from aaber.common.io import read_jsonl
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query
from aaber.index.base import RetrieverIndex


class PrecomputedSourceIndex(RetrieverIndex):
    """Retriever backed by cached candidate rows for one or more source actions."""

    name = "precomputed_source"

    def __init__(self, results_by_query: dict[str, list[EvidenceResult]] | None = None, name: str | None = None) -> None:
        self.results_by_query = results_by_query or {}
        if name:
            self.name = name

    @classmethod
    def from_candidate_files(
        cls,
        paths: list[Path],
        source_actions: set[str],
        name: str = "precomputed_source",
    ) -> "PrecomputedSourceIndex":
        grouped: dict[str, list[EvidenceResult]] = defaultdict(list)
        for path in paths:
            if not path.exists():
                continue
            for row in read_jsonl(path):
                result = EvidenceResult.from_dict(row)
                if result.source_action not in source_actions:
                    continue
                grouped[result.query_id].append(result)

        ordered = {
            query_id: sorted(rows, key=lambda row: (row.rank or 999999, -row.score))
            for query_id, rows in grouped.items()
        }
        return cls(results_by_query=ordered, name=name)

    def build(self, units: list[EvidenceUnit]) -> None:
        return None

    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        rows = self.results_by_query.get(query.query_id, [])[:top_k]
        return [
            EvidenceResult(
                query_id=row.query_id,
                unit_id=row.unit_id,
                score=row.score,
                source_action=row.source_action,
                rank=idx,
                metadata=dict(row.metadata),
            )
            for idx, row in enumerate(rows, start=1)
        ]

    def search_batch(self, queries: list[Query], top_k: int = 10) -> dict[str, list[EvidenceResult]]:
        return {query.query_id: self.search(query, top_k=top_k) for query in queries}

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("PrecomputedSourceIndex is loaded from candidate files and is not saved.")

    @classmethod
    def load(cls, path: str | Path) -> "PrecomputedSourceIndex":
        raise NotImplementedError("Use PrecomputedSourceIndex.from_candidate_files instead.")

    def has_results(self) -> bool:
        return any(self.results_by_query.values())

