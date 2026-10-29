from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from aaber.common.io import read_json, write_json
from aaber.common.text import tokenize
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query
from aaber.index.base import RetrieverIndex


class BM25Index(RetrieverIndex):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.units: list[EvidenceUnit] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_doc_len = 0.0

    def build(self, units: list[EvidenceUnit]) -> None:
        self.units = units
        self.doc_tokens = [tokenize(unit.searchable_text) for unit in units]
        self.doc_freq = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.avg_doc_len = sum(len(tokens) for tokens in self.doc_tokens) / max(1, len(self.doc_tokens))

    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        query_terms = tokenize(query.question)
        scores: list[tuple[str, float]] = []
        total_docs = max(1, len(self.units))
        for unit, tokens in zip(self.units, self.doc_tokens):
            tf = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in query_terms:
                if tf[term] == 0:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf[term] + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-9))
                score += idf * tf[term] * (self.k1 + 1) / max(denom, 1e-9)
            if score > 0:
                scores.append((unit.unit_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [
            EvidenceResult(query.query_id, unit_id, score, self.name, rank=idx)
            for idx, (unit_id, score) in enumerate(scores[:top_k], start=1)
        ]

    def save(self, path: str | Path) -> None:
        write_json(Path(path) / "index.json", self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        data = read_json(Path(path) / "index.json")
        index = cls(k1=float(data.get("k1", 1.5)), b=float(data.get("b", 0.75)))
        index.units = [EvidenceUnit.from_dict(row) for row in data["units"]]
        index.doc_tokens = [list(row) for row in data["doc_tokens"]]
        index.doc_freq = Counter(data["doc_freq"])
        index.avg_doc_len = float(data.get("avg_doc_len", 0.0))
        return index

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "k1": self.k1,
            "b": self.b,
            "units": [u.to_dict() for u in self.units],
            "doc_tokens": self.doc_tokens,
            "doc_freq": dict(self.doc_freq),
            "avg_doc_len": self.avg_doc_len,
        }

