from __future__ import annotations

import hashlib
import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from aaber.data.dataset_base import DatasetAdapter
from aaber.data.schema import EvidenceUnit, Query, Relation


PAGE_ID_RE = re.compile(r"^(?P<doc>.+?)_p(?P<page>\d+)\.[A-Za-z0-9]+$")


class VisRAGMPDocVQAAdapter(DatasetAdapter):
    """Read the VisRAG MP-DocVQA retrieval parquet export."""

    def __init__(
        self,
        dataset: str,
        corpus_parquet: str | Path,
        queries_parquet: str | Path,
        qrels_parquet: str | Path,
        image_dir: str | Path,
        source_name: str = "openbmb/VisRAG-Ret-Test-MP-DocVQA",
        split_config: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.corpus_parquet = Path(corpus_parquet)
        self.queries_parquet = Path(queries_parquet)
        self.qrels_parquet = Path(qrels_parquet)
        self.image_dir = Path(image_dir)
        self.source_name = source_name
        self.split_config = split_config or {}
        self.split_ratios = _split_ratios(self.split_config)
        self.split_seed = int(self.split_config.get("seed", 13)) if isinstance(self.split_config, dict) else 13
        self._corpus_rows: list[dict[str, Any]] | None = None
        self._query_rows: list[dict[str, Any]] | None = None
        self._qrel_rows: list[dict[str, Any]] | None = None
        self._query_split_by_id: dict[str, str] | None = None

    def load_queries(self, split: str) -> list[Query]:
        split_by_id = self._build_query_split_assignments()
        queries: list[Query] = []
        for row in self._load_query_rows():
            query_id = str(row.get("query-id") or "")
            if not query_id:
                continue
            row_split = split_by_id[query_id]
            if split != "all" and row_split != split:
                continue
            answer = row.get("answer")
            queries.append(
                Query(
                    query_id=query_id,
                    question=str(row.get("query") or ""),
                    dataset=self.dataset,
                    query_type=_infer_query_type(str(row.get("query") or "")),
                    answer="; ".join(str(item) for item in answer) if isinstance(answer, list) else None,
                    metadata={
                        "split": row_split,
                        "source": self.source_name,
                        "is_numerical": int(row.get("is_numerical") or 0),
                    },
                )
            )
        return queries

    def load_qrels(self, split: str) -> dict[str, set[str]]:
        split_by_id = self._build_query_split_assignments()
        qrels: dict[str, set[str]] = {}
        for row in self._load_qrel_rows():
            query_id = str(row.get("query-id") or "")
            corpus_id = str(row.get("corpus-id") or "")
            if not query_id or not corpus_id:
                continue
            if split != "all" and split_by_id.get(query_id) != split:
                continue
            if int(row.get("score") or 0) <= 0:
                continue
            qrels.setdefault(query_id, set()).add(_unit_id(corpus_id))
        return qrels

    def build_units(self, split: str) -> list[EvidenceUnit]:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        units: list[EvidenceUnit] = []
        for row in self._load_corpus_rows():
            corpus_id = str(row.get("corpus-id") or "")
            if not corpus_id:
                continue
            doc_id, page_id = _parse_doc_page(corpus_id)
            image_path = self._save_image(row.get("image"), corpus_id)
            units.append(
                EvidenceUnit(
                    unit_id=_unit_id(corpus_id),
                    doc_id=doc_id,
                    page_id=page_id,
                    unit_type="page",
                    text="",
                    caption="",
                    context_text=f"MP-DocVQA document {doc_id} page {page_id}",
                    image_path=str(image_path).replace("\\", "/") if image_path else None,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    layout_type="page",
                    relations=[],
                    quality={
                        "ocr_confidence": 0.0,
                        "image_resolution": 0.0,
                        "blur_score": 0.0,
                        "caption_missing": True,
                    },
                    metadata={
                        "source": self.source_name,
                        "corpus_id": corpus_id,
                        "image_filename": corpus_id,
                        "doc_id": doc_id,
                        "page_id": page_id,
                    },
                )
            )
        return units

    def build_relations(self, split: str) -> list[Relation]:
        pages_by_doc: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for row in self._load_corpus_rows():
            corpus_id = str(row.get("corpus-id") or "")
            if not corpus_id:
                continue
            doc_id, page_id = _parse_doc_page(corpus_id)
            pages_by_doc[doc_id].append((_unit_id(corpus_id), page_id))

        relations: list[Relation] = []
        for doc_id, pages in sorted(pages_by_doc.items()):
            unique_pages = sorted(set(pages), key=lambda item: (item[1], item[0]))
            if len(unique_pages) < 2:
                continue
            for (left_id, left_page), (right_id, right_page) in itertools.combinations(unique_pages, 2):
                distance = abs(right_page - left_page)
                score = max(0.2, 0.55 - 0.03 * min(distance, 10))
                relations.append(
                    Relation(
                        relation_id=f"same_doc_{_short_relation_id(left_id)}_{_short_relation_id(right_id)}",
                        src_unit_id=left_id,
                        dst_unit_id=right_id,
                        relation_type="same_doc",
                        score=score,
                        evidence=f"{doc_id} pages {left_page} and {right_page} are in the same document",
                        metadata={"doc_id": doc_id, "page_distance": distance},
                    )
                )
        return relations

    def _load_corpus_rows(self) -> list[dict[str, Any]]:
        if self._corpus_rows is None:
            self._corpus_rows = _read_parquet_rows(self.corpus_parquet)
        return self._corpus_rows

    def _load_query_rows(self) -> list[dict[str, Any]]:
        if self._query_rows is None:
            self._query_rows = _read_parquet_rows(self.queries_parquet)
        return self._query_rows

    def _load_qrel_rows(self) -> list[dict[str, Any]]:
        if self._qrel_rows is None:
            self._qrel_rows = _read_parquet_rows(self.qrels_parquet)
        return self._qrel_rows

    def _build_query_split_assignments(self) -> dict[str, str]:
        if self._query_split_by_id is not None:
            return self._query_split_by_id
        rows = self._load_query_rows()
        if not self.split_ratios:
            self._query_split_by_id = {str(row["query-id"]): "test" for row in rows if row.get("query-id")}
            return self._query_split_by_id

        names = list(self.split_ratios)
        total_ratio = sum(self.split_ratios.values()) or 1.0
        counts: list[int] = []
        remaining = len(rows)
        for name in names[:-1]:
            count = int(len(rows) * self.split_ratios[name] / total_ratio)
            counts.append(count)
            remaining -= count
        counts.append(remaining)

        def sort_key(row: dict[str, Any]) -> str:
            stable = "|".join(
                [
                    str(self.split_seed),
                    str(row.get("query-id") or ""),
                    str(row.get("query") or ""),
                ]
            )
            return hashlib.md5(stable.encode("utf-8")).hexdigest()

        ordered = sorted(rows, key=sort_key)
        split_by_id: dict[str, str] = {}
        cursor = 0
        for name, count in zip(names, counts):
            for row in ordered[cursor : cursor + count]:
                query_id = str(row.get("query-id") or "")
                if query_id:
                    split_by_id[query_id] = name
            cursor += count
        self._query_split_by_id = split_by_id
        return split_by_id

    def _save_image(self, image_obj: Any, corpus_id: str) -> Path | None:
        if not isinstance(image_obj, dict):
            return None
        data = image_obj.get("bytes")
        if not data:
            return None
        suffix = Path(str(image_obj.get("path") or corpus_id)).suffix or ".jpg"
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,6}", suffix):
            suffix = ".jpg"
        out_path = self.image_dir / f"{_sanitize_id(corpus_id)}{suffix}"
        if not out_path.exists():
            out_path.write_bytes(data)
        return out_path


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading parquet requires pyarrow. Install with: python -m pip install pyarrow") from exc

    pf = pq.ParquetFile(path)
    rows: list[dict[str, Any]] = []
    for row_group_idx in range(pf.num_row_groups):
        rows.extend(pf.read_row_group(row_group_idx).to_pylist())
    return rows


def _parse_doc_page(corpus_id: str) -> tuple[str, int]:
    name = Path(corpus_id).name
    match = PAGE_ID_RE.match(name)
    if not match:
        return Path(name).stem, 1
    return match.group("doc"), int(match.group("page"))


def _unit_id(corpus_id: str) -> str:
    return f"visrag_mpdocvqa_{_sanitize_id(corpus_id)}"


def _sanitize_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_")
    if not cleaned:
        cleaned = "item"
    if len(cleaned) > 96:
        digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
        cleaned = f"{cleaned[:80].rstrip('_')}_{digest}"
    return cleaned


def _short_relation_id(value: str) -> str:
    if len(value) <= 48:
        return value
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:10]
    return f"{value[:36]}_{digest}"


def _split_ratios(config: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(config, dict):
        return {}
    ratios: dict[str, float] = {}
    for name in ["train", "dev", "test"]:
        if name in config:
            ratios[name] = float(config[name])
    return ratios


def _infer_query_type(question: str) -> str:
    lower = question.lower()
    if any(term in lower for term in ["table", "row", "column", "cell", "committee", "meeting"]):
        return "table-centric"
    if any(term in lower for term in ["page", "previous", "next", "document", "section"]):
        return "multi-page"
    if any(term in lower for term in ["chart", "plot", "axis", "graph", "figure"]):
        return "chart-centric"
    if any(term in lower for term in ["picture", "image", "shown"]):
        return "image-centric"
    return "text-centric"
