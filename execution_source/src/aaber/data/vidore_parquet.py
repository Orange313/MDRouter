from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from aaber.data.dataset_base import DatasetAdapter
from aaber.data.schema import EvidenceUnit, Query, Relation


class VidoreDocVQATesseractParquetAdapter(DatasetAdapter):
    """Read vidore/docvqa_test_subsampled_tesseract directly from parquet."""

    def __init__(
        self,
        dataset: str,
        parquet_path: str | Path,
        image_dir: str | Path,
        source_name: str = "vidore/docvqa_test_subsampled_tesseract",
        split_config: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.parquet_path = Path(parquet_path)
        self.image_dir = Path(image_dir)
        self.source_name = source_name
        self.split_config = split_config or {}
        self.split_ratios = _split_ratios(self.split_config)
        self.split_seed = int(self.split_config.get("seed", 13)) if isinstance(self.split_config, dict) else 13
        self.split_group_by = str(self.split_config.get("group_by", "")).strip() if isinstance(self.split_config, dict) else ""
        self._rows: list[dict[str, Any]] | None = None
        self._split_by_idx: dict[int, str] | None = None

    def load_queries(self, split: str) -> list[Query]:
        queries: list[Query] = []
        for idx, row, row_split in self._iter_rows(split):
            image_filename = str(row.get("image_filename") or f"page_{idx:06d}")
            queries.append(
                Query(
                    query_id=self._query_id(idx),
                    question=str(row.get("query") or ""),
                    dataset=self.dataset,
                    query_type=_infer_query_type(str(row.get("query") or "")),
                    answer=None,
                    metadata={
                        "split": row_split,
                        "source": self.source_name,
                        "image_filename": image_filename,
                    },
                )
            )
        return queries

    def load_qrels(self, split: str) -> dict[str, set[str]]:
        qrels: dict[str, set[str]] = {}
        for idx, row, _row_split in self._iter_rows(split):
            image_filename = str(row.get("image_filename") or f"page_{idx:06d}")
            qrels[self._query_id(idx)] = {self._unit_id(image_filename)}
        return qrels

    def build_units(self, split: str) -> list[EvidenceUnit]:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        units_by_id: dict[str, EvidenceUnit] = {}
        for idx, row, row_split in self._iter_rows(split):
            image_filename = str(row.get("image_filename") or f"page_{idx:06d}")
            unit_id = self._unit_id(image_filename)
            if unit_id in units_by_id:
                continue
            image_path = self._save_image(row.get("image"), image_filename)
            units_by_id[unit_id] = EvidenceUnit(
                unit_id=unit_id,
                doc_id=unit_id,
                page_id=1,
                unit_type="page",
                text=str(row.get("text_description") or ""),
                caption="",
                context_text=f"DocVQA page {image_filename}",
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
                    "split": row_split,
                    "source": self.source_name,
                    "image_filename": image_filename,
                },
            )
        return list(units_by_id.values())

    def build_relations(self, split: str) -> list[Relation]:
        return []

    def _iter_rows(self, split: str):
        rows = self._load_rows()
        split_by_idx = self._build_split_assignments()
        for idx, row in enumerate(rows):
            row_split = split_by_idx[idx]
            if split == "all" or row_split == split:
                yield idx, row, row_split

    def _load_rows(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading ViDoRe parquet requires pyarrow. Install with: python -m pip install pyarrow") from exc

        pf = pq.ParquetFile(self.parquet_path)
        rows: list[dict[str, Any]] = []
        for row_group_idx in range(pf.num_row_groups):
            rows.extend(pf.read_row_group(row_group_idx).to_pylist())
        self._rows = rows
        return rows

    def _build_split_assignments(self) -> dict[int, str]:
        if self._split_by_idx is not None:
            return self._split_by_idx
        rows = self._load_rows()
        if not self.split_ratios:
            self._split_by_idx = {idx: "test" for idx in range(len(rows))}
            return self._split_by_idx

        if self.split_group_by:
            self._split_by_idx = self._build_grouped_split_assignments(rows)
            return self._split_by_idx

        names = list(self.split_ratios)
        total_ratio = sum(self.split_ratios.values()) or 1.0
        counts: list[int] = []
        remaining = len(rows)
        for name in names[:-1]:
            count = int(len(rows) * self.split_ratios[name] / total_ratio)
            counts.append(count)
            remaining -= count
        counts.append(remaining)

        def sort_key(idx: int) -> str:
            row = rows[idx]
            stable = "|".join(
                [
                    str(self.split_seed),
                    str(idx),
                    str(row.get("image_filename") or ""),
                    str(row.get("query") or ""),
                ]
            )
            return hashlib.md5(stable.encode("utf-8")).hexdigest()

        ordered_indices = sorted(range(len(rows)), key=sort_key)
        split_by_idx: dict[int, str] = {}
        cursor = 0
        for name, count in zip(names, counts):
            for idx in ordered_indices[cursor : cursor + count]:
                split_by_idx[idx] = name
            cursor += count
        self._split_by_idx = split_by_idx
        return split_by_idx

    def _build_grouped_split_assignments(self, rows: list[dict[str, Any]]) -> dict[int, str]:
        """Assign complete row groups to one split while balancing query counts."""
        names = list(self.split_ratios)
        total_ratio = sum(self.split_ratios.values()) or 1.0
        targets = {
            name: len(rows) * self.split_ratios[name] / total_ratio
            for name in names
        }
        grouped: dict[str, list[int]] = {}
        for idx, row in enumerate(rows):
            value = row.get(self.split_group_by)
            if value in {None, ""}:
                raise ValueError(
                    f"Cannot group split by {self.split_group_by!r}: row {idx} has no value"
                )
            grouped.setdefault(str(value), []).append(idx)

        def group_sort_key(item: tuple[str, list[int]]) -> str:
            group_id, _indices = item
            stable = f"{self.split_seed}|{self.split_group_by}|{group_id}"
            return hashlib.md5(stable.encode("utf-8")).hexdigest()

        counts = {name: 0 for name in names}
        split_by_idx: dict[int, str] = {}
        for _group_id, indices in sorted(grouped.items(), key=group_sort_key):
            chosen = max(
                names,
                key=lambda name: (
                    (targets[name] - counts[name]) / max(targets[name], 1.0),
                    -counts[name],
                    -names.index(name),
                ),
            )
            for idx in indices:
                split_by_idx[idx] = chosen
            counts[chosen] += len(indices)
        return split_by_idx

    def _save_image(self, image_obj: Any, image_filename: str) -> Path | None:
        if not isinstance(image_obj, dict):
            return None
        data = image_obj.get("bytes")
        if not data:
            return None
        suffix = Path(str(image_obj.get("path") or "")).suffix or ".png"
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,6}", suffix):
            suffix = ".png"
        out_path = self.image_dir / f"{_sanitize_id(image_filename)}{suffix}"
        if not out_path.exists():
            out_path.write_bytes(data)
        return out_path

    @staticmethod
    def _query_id(idx: int) -> str:
        return f"docvqa_q_{idx:06d}"

    @staticmethod
    def _unit_id(image_filename: str) -> str:
        return f"docvqa_{_sanitize_id(image_filename)}"


def _sanitize_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_")
    if not cleaned:
        cleaned = "item"
    if len(cleaned) > 80:
        digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
        cleaned = f"{cleaned[:64].rstrip('_')}_{digest}"
    return cleaned


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
    if any(term in lower for term in ["table", "row", "column", "cell"]):
        return "table-centric"
    if any(term in lower for term in ["chart", "plot", "axis", "graph", "figure"]):
        return "chart-centric"
    if any(term in lower for term in ["picture", "image", "shown"]):
        return "image-centric"
    return "text-centric"
