from __future__ import annotations

"""Adapter for the public `weaviate/IRPAPERS` CSV release.

IRPAPERS supplies a shared page corpus and one question per target page, but
does not publish train/dev/test query partitions.  This adapter therefore
creates a deterministic *target-paper-group* query split.  The full corpus is
available to every split, as is standard for retrieval evaluation; only router
supervision and configuration selection are isolated by the target paper of a
question.
"""

import base64
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from aaber.data.dataset_base import DatasetAdapter
from aaber.data.schema import EvidenceUnit, Query, Relation


class IRPapersCSVAdapter(DatasetAdapter):
    """Load page images/OCR and questions from the official IRPAPERS CSVs."""

    def __init__(
        self,
        *,
        dataset: str,
        docs_csv: str | Path,
        queries_csv: str | Path,
        image_dir: str | Path,
        source_name: str = "weaviate/IRPAPERS",
        split_config: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.docs_csv = Path(docs_csv)
        self.queries_csv = Path(queries_csv)
        self.image_dir = Path(image_dir)
        self.source_name = source_name
        self.split_config = split_config or {}
        self.split_ratios = _split_ratios(self.split_config)
        self.split_seed = int(self.split_config.get("seed", 20260718))
        self._query_rows: list[dict[str, str]] | None = None
        self._split_by_query_id: dict[str, str] | None = None

    def load_queries(self, split: str) -> list[Query]:
        split_by_id = self._build_split_assignments()
        queries: list[Query] = []
        for row in self._load_query_rows():
            query_id = _query_id(row)
            row_split = split_by_id[query_id]
            if split != "all" and row_split != split:
                continue
            question = str(row.get("question") or "").strip()
            queries.append(
                Query(
                    query_id=query_id,
                    question=question,
                    dataset=self.dataset,
                    query_type=_infer_query_type(question),
                    answer=str(row.get("answer") or "").strip() or None,
                    metadata={
                        "split": row_split,
                        "source": self.source_name,
                        "target_pdf_id": _pdf_id(row, "pdf_id"),
                        "target_page_number": _page_number(row),
                        "pdf_title": str(row.get("pdf_title") or "").strip(),
                        "split_group": f"pdf:{_pdf_id(row, 'pdf_id')}",
                    },
                )
            )
        return queries

    def load_qrels(self, split: str) -> dict[str, set[str]]:
        split_by_id = self._build_split_assignments()
        qrels: dict[str, set[str]] = {}
        for row in self._load_query_rows():
            query_id = _query_id(row)
            if split != "all" and split_by_id[query_id] != split:
                continue
            qrels[query_id] = {_unit_id(_pdf_id(row, "pdf_id"), _page_number(row))}
        return qrels

    def build_units(self, split: str) -> list[EvidenceUnit]:
        """Build the shared page corpus; ``split`` intentionally does not filter it."""
        if not self.docs_csv.exists():
            raise FileNotFoundError(f"Missing IRPAPERS documents CSV: {self.docs_csv}")
        _raise_csv_field_limit()
        self.image_dir.mkdir(parents=True, exist_ok=True)
        units: list[EvidenceUnit] = []
        with self.docs_csv.open("r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=2):
                pdf_id = _pdf_id(row, "pdf_id_x", "pdf_id")
                page_number = _page_number(row)
                if not pdf_id or page_number < 1:
                    raise ValueError(f"Invalid IRPAPERS document row {index}: pdf_id={pdf_id!r}, page={page_number!r}")
                image_path = self._save_image(row, pdf_id, page_number, index)
                units.append(
                    EvidenceUnit(
                        unit_id=_unit_id(pdf_id, page_number),
                        doc_id=_doc_id(pdf_id),
                        page_id=page_number,
                        unit_type="page",
                        text=str(row.get("transcription") or "").strip(),
                        caption="",
                        context_text=str(row.get("pdf_title") or "").strip(),
                        image_path=str(image_path).replace("\\", "/"),
                        bbox=(0.0, 0.0, 0.0, 0.0),
                        layout_type="page",
                        relations=[],
                        quality={"ocr_source": "IRPAPERS transcription", "image_available": True},
                        metadata={
                            "source": self.source_name,
                            "pdf_id": pdf_id,
                            "pdf_name": str(row.get("pdf_name") or "").strip(),
                            "pdf_title": str(row.get("pdf_title") or "").strip(),
                            "page_number": page_number,
                        },
                    )
                )
        if not units:
            raise ValueError(f"No document pages found in {self.docs_csv}")
        return units

    def build_relations(self, split: str) -> list[Relation]:
        # The binary Text--Visual protocol does not use relations.  The config
        # also disables automatic relation construction to avoid an unrelated
        # corpus-wide preprocessing action.
        return []

    def _load_query_rows(self) -> list[dict[str, str]]:
        if self._query_rows is not None:
            return self._query_rows
        if not self.queries_csv.exists():
            raise FileNotFoundError(f"Missing IRPAPERS queries CSV: {self.queries_csv}")
        _raise_csv_field_limit()
        with self.queries_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No queries found in {self.queries_csv}")
        missing = [row for row in rows if not _query_id(row) or not _pdf_id(row, "pdf_id") or _page_number(row) < 1]
        if missing:
            raise ValueError(f"IRPAPERS queries contain {len(missing)} rows without a valid ID, target paper, or page")
        if len({_query_id(row) for row in rows}) != len(rows):
            raise ValueError("IRPAPERS query IDs must be unique after canonicalization")
        self._query_rows = rows
        return rows

    def _build_split_assignments(self) -> dict[str, str]:
        if self._split_by_query_id is not None:
            return self._split_by_query_id
        rows = self._load_query_rows()
        if not self.split_ratios:
            self._split_by_query_id = {_query_id(row): "test" for row in rows}
            return self._split_by_query_id

        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[_pdf_id(row, "pdf_id")].append(row)
        names = list(self.split_ratios)
        total_ratio = sum(self.split_ratios.values()) or 1.0
        targets = {name: len(rows) * self.split_ratios[name] / total_ratio for name in names}
        assigned_counts = {name: 0 for name in names}
        assignments: dict[str, str] = {}
        for group_id, group_rows in sorted(groups.items(), key=lambda item: _stable_number(item[0], self.split_seed)):
            chosen = max(
                names,
                key=lambda name: (
                    (targets[name] - assigned_counts[name]) / max(targets[name], 1.0),
                    -assigned_counts[name],
                    -names.index(name),
                ),
            )
            for row in group_rows:
                assignments[_query_id(row)] = chosen
            assigned_counts[chosen] += len(group_rows)
        self._split_by_query_id = assignments
        return assignments

    def _save_image(self, row: dict[str, str], pdf_id: str, page_number: int, row_number: int) -> Path:
        encoded = str(row.get("base64_str") or "").strip()
        if encoded.startswith("data:image") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        if not encoded:
            raise ValueError(f"IRPAPERS document row {row_number} has no page image")
        path = self.image_dir / f"{_doc_id(pdf_id)}_p{page_number:04d}.png"
        if path.exists() and path.stat().st_size > 0:
            return path
        try:
            payload = base64.b64decode(encoded, validate=True)
        except Exception as exc:  # pragma: no cover - input-data error path
            raise ValueError(f"Could not decode IRPAPERS page image at document row {row_number}") from exc
        if not payload:
            raise ValueError(f"IRPAPERS document row {row_number} decoded to an empty page image")
        path.write_bytes(payload)
        return path


def _split_ratios(config: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(config, dict):
        return {}
    return {name: float(config[name]) for name in ("train", "dev", "test") if name in config}


def _raise_csv_field_limit() -> None:
    """Permit IRPAPERS' base64-encoded page-image cells in the stdlib CSV reader."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _stable_number(value: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:irpapers-target-paper:{value}".encode("utf-8")).hexdigest()[:16], 16)


def _pdf_id(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _page_number(row: dict[str, str]) -> int:
    value = str(row.get("page_number") or "").strip()
    try:
        return int(float(value))
    except ValueError:
        return 0


def _query_id(row: dict[str, str]) -> str:
    dataset_id = str(row.get("dataset_id") or "").strip()
    pdf_id = _pdf_id(row, "pdf_id")
    return f"irpapers_q_{_safe_id(pdf_id)}_{_safe_id(dataset_id)}" if dataset_id and pdf_id else ""


def _doc_id(pdf_id: str) -> str:
    return f"irpapers_pdf_{_safe_id(pdf_id)}"


def _unit_id(pdf_id: str, page_number: int) -> str:
    return f"{_doc_id(pdf_id)}_p{page_number:04d}"


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "unknown"


def _infer_query_type(question: str) -> str:
    lower = question.lower()
    if any(term in lower for term in ("table", "row", "column", "cell")):
        return "table-centric"
    if any(term in lower for term in ("chart", "plot", "axis", "graph", "figure", "diagram")):
        return "chart-centric"
    if any(term in lower for term in ("layout", "left", "right", "above", "below", "color", "image", "picture")):
        return "layout-centric"
    return "text-centric"
