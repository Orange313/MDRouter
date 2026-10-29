from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from aaber.common.io import read_jsonl, write_jsonl
from aaber.data.schema import EvidenceUnit, Query, Relation


class DatasetAdapter(ABC):
    dataset: str

    @abstractmethod
    def load_queries(self, split: str) -> list[Query]:
        raise NotImplementedError

    @abstractmethod
    def load_qrels(self, split: str) -> dict[str, set[str]]:
        raise NotImplementedError

    @abstractmethod
    def build_units(self, split: str) -> list[EvidenceUnit]:
        raise NotImplementedError

    def build_relations(self, split: str) -> list[Relation]:
        return []


class FileDatasetAdapter(DatasetAdapter):
    """Adapter for pre-normalized local JSONL files."""

    def __init__(
        self,
        dataset: str,
        queries_path: str | Path,
        qrels_path: str | Path,
        units_path: str | Path,
        relations_path: str | Path | None = None,
    ) -> None:
        self.dataset = dataset
        self.queries_path = Path(queries_path)
        self.qrels_path = Path(qrels_path)
        self.units_path = Path(units_path)
        self.relations_path = Path(relations_path) if relations_path else None

    def load_queries(self, split: str) -> list[Query]:
        queries = [Query.from_dict(row) for row in read_jsonl(self.queries_path)]
        for query in queries:
            if not query.dataset:
                query.dataset = self.dataset
        return _filter_split(queries, split)

    def load_qrels(self, split: str) -> dict[str, set[str]]:
        qrels: dict[str, set[str]] = {}
        for row in read_jsonl(self.qrels_path):
            row_split = row.get("split")
            if row_split is not None and split != "all" and row_split != split:
                continue
            query_id = str(row["query_id"])
            positives = row.get("positive_unit_ids", row.get("positive_page_ids", []))
            qrels[query_id] = {str(item) for item in positives}
        return qrels

    def build_units(self, split: str) -> list[EvidenceUnit]:
        units = [EvidenceUnit.from_dict(row) for row in read_jsonl(self.units_path)]
        return _filter_split(units, split)

    def build_relations(self, split: str) -> list[Relation]:
        if not self.relations_path:
            return []
        relations = [Relation.from_dict(row) for row in read_jsonl(self.relations_path)]
        return relations


def _filter_split(items: list, split: str) -> list:
    if split == "all":
        return items
    filtered = []
    has_split = False
    for item in items:
        metadata = getattr(item, "metadata", {}) or {}
        item_split = metadata.get("split")
        if item_split is not None:
            has_split = True
        if item_split is None or item_split == split:
            filtered.append(item)
    return filtered if has_split else items


def write_prepared_dataset(
    out_dir: str | Path,
    queries: list[Query],
    qrels: dict[str, set[str]],
    units: list[EvidenceUnit],
    relations: list[Relation],
) -> None:
    target = Path(out_dir)
    write_jsonl(target / "queries.jsonl", [q.to_dict() for q in queries])
    write_jsonl(
        target / "qrels.jsonl",
        [
            {"query_id": query_id, "positive_unit_ids": sorted(positive_ids)}
            for query_id, positive_ids in sorted(qrels.items())
        ],
    )
    write_jsonl(target / "units" / "evidence_units.jsonl", [u.to_dict() for u in units])
    write_jsonl(target / "units" / "relations.jsonl", [r.to_dict() for r in relations])

