from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from aaber.common.io import read_json, write_json
from aaber.common.text import keyword_overlap, tokenize
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query
from aaber.index.base import RetrieverIndex


class CrossEncoderRerankIndex(RetrieverIndex):
    """Query-document reranker with an optional sentence-transformers CrossEncoder backend."""

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 128,
        device: str | None = None,
        fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.fallback = fallback
        self.units: list[EvidenceUnit] = []
        self._model: Any | None = None

    def build(self, units: list[EvidenceUnit]) -> None:
        self.units = units

    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        return self.rerank(query, self.units, top_k=top_k)

    def rerank(self, query: Query, candidate_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        candidates = [unit for unit in candidate_units if unit.searchable_text.strip()]
        if not candidates:
            return []
        scores = self._score(query, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            EvidenceResult(
                query_id=query.query_id,
                unit_id=unit.unit_id,
                score=float(score),
                source_action=self.name,
                rank=rank,
                metadata={"model_name": self.model_name, "fallback": self.fallback or self._model is None},
            )
            for rank, (unit, score) in enumerate(ranked, start=1)
        ]

    def rerank_many(
        self,
        queries: list[Query],
        candidate_units_by_query: dict[str, list[EvidenceUnit]],
        top_k: int = 10,
    ) -> dict[str, list[EvidenceResult]]:
        flat_queries: list[Query] = []
        flat_units: list[EvidenceUnit] = []
        for query in queries:
            for unit in candidate_units_by_query.get(query.query_id, []):
                if unit.searchable_text.strip():
                    flat_queries.append(query)
                    flat_units.append(unit)
        if not flat_units:
            return {query.query_id: [] for query in queries}
        scores = self._score_many(flat_queries, flat_units)
        grouped: dict[str, list[tuple[EvidenceUnit, float]]] = {query.query_id: [] for query in queries}
        for query, unit, score in zip(flat_queries, flat_units, scores):
            grouped.setdefault(query.query_id, []).append((unit, float(score)))

        output: dict[str, list[EvidenceResult]] = {}
        for query in queries:
            ranked = sorted(grouped.get(query.query_id, []), key=lambda item: item[1], reverse=True)[:top_k]
            output[query.query_id] = [
                EvidenceResult(
                    query_id=query.query_id,
                    unit_id=unit.unit_id,
                    score=float(score),
                    source_action=self.name,
                    rank=rank,
                    metadata={"model_name": self.model_name, "fallback": self.fallback or self._model is None},
                )
                for rank, (unit, score) in enumerate(ranked, start=1)
            ]
        return output

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        write_json(
            target / "index.json",
            {
                "name": self.name,
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "device": self.device,
                "fallback": self.fallback,
                "units": [unit.to_dict() for unit in self.units],
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "CrossEncoderRerankIndex":
        data = read_json(Path(path) / "index.json")
        index = cls(
            model_name=str(data.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
            batch_size=int(data.get("batch_size", 32)),
            device=data.get("device"),
            fallback=bool(data.get("fallback", False)),
        )
        index.units = [EvidenceUnit.from_dict(row) for row in data["units"]]
        return index

    def _score(self, query: Query, units: list[EvidenceUnit]) -> list[float]:
        return self._score_many([query] * len(units), units)

    def _score_many(self, queries: list[Query], units: list[EvidenceUnit]) -> list[float]:
        model = self._load_model_or_none()
        if model is None:
            return [_lexical_rerank_score(query, unit) for query, unit in zip(queries, units)]
        pairs = [[query.question, _unit_text(unit)] for query, unit in zip(queries, units)]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(score) for score in scores]

    def _load_model_or_none(self) -> Any | None:
        if self.fallback:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return None
        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model


def _unit_text(unit: EvidenceUnit) -> str:
    text = " ".join(part for part in [unit.text, unit.caption, unit.context_text] if part)
    return text[:768]


def _lexical_rerank_score(query: Query, unit: EvidenceUnit) -> float:
    text = _unit_text(unit)
    query_tokens = tokenize(query.question)
    unit_tokens = tokenize(text)
    if not query_tokens or not unit_tokens:
        return 0.0
    query_set = set(query_tokens)
    unit_set = set(unit_tokens)
    overlap = len(query_set & unit_set) / max(1, len(query_set))
    coverage = len(query_set & unit_set) / max(1, len(unit_set))
    phrase_bonus = 1.0 if query.question.lower() in text.lower() else 0.0
    field_bonus = 0.0
    for term in ["title", "date", "name", "number", "axis", "photo", "figure", "chart"]:
        if term in query_set and term in unit_set:
            field_bonus += 0.05
    length_penalty = 1.0 / math.sqrt(max(1.0, len(unit_tokens) / 80.0))
    return 2.0 * overlap + 0.5 * coverage + 0.2 * phrase_bonus + field_bonus + 0.1 * length_penalty
