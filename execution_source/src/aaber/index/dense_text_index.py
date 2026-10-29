from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from aaber.common.io import read_json, write_json
from aaber.common.text import cosine_counter, token_counter
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query
from aaber.index.base import RetrieverIndex


class DenseTextIndex(RetrieverIndex):
    """Dependency-free lexical dense placeholder.

    This keeps the retriever API runnable. Replace this class with a
    sentence-transformer + FAISS implementation for real experiments.
    """

    name = "dense_text"

    def __init__(self) -> None:
        self.units: list[EvidenceUnit] = []
        self.vectors: list[Counter[str]] = []

    def build(self, units: list[EvidenceUnit]) -> None:
        self.units = units
        self.vectors = [token_counter(unit.searchable_text) for unit in units]

    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        query_vec = token_counter(query.question)
        scored = [
            (unit.unit_id, cosine_counter(query_vec, vec))
            for unit, vec in zip(self.units, self.vectors)
        ]
        scored = [(unit_id, score) for unit_id, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            EvidenceResult(query.query_id, unit_id, score, self.name, rank=idx)
            for idx, (unit_id, score) in enumerate(scored[:top_k], start=1)
        ]

    def save(self, path: str | Path) -> None:
        write_json(Path(path) / "index.json", self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "DenseTextIndex":
        data = read_json(Path(path) / "index.json")
        index = cls()
        index.units = [EvidenceUnit.from_dict(row) for row in data["units"]]
        index.vectors = [Counter(row) for row in data["vectors"]]
        return index

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "units": [u.to_dict() for u in self.units],
            "vectors": [dict(v) for v in self.vectors],
        }


class SentenceTransformerTextIndex(RetrieverIndex):
    """Real dense text retriever backed by sentence-transformers when installed."""

    name = "sentence_transformer"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.units: list[EvidenceUnit] = []
        self.embeddings: np.ndarray | None = None
        self._model: Any | None = None

    def build(self, units: list[EvidenceUnit]) -> None:
        model = self._load_model()
        self.units = units
        texts = [unit.searchable_text for unit in units]
        self.embeddings = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        if self.embeddings is None:
            return []
        model = self._load_model()
        query_embedding = model.encode(
            [query.question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ query_embedding
        top_idx = np.argsort(-scores)[:top_k]
        return [
            EvidenceResult(
                query_id=query.query_id,
                unit_id=self.units[int(idx)].unit_id,
                score=float(scores[int(idx)]),
                source_action=self.name,
                rank=rank,
            )
            for rank, idx in enumerate(top_idx, start=1)
            if float(scores[int(idx)]) > 0
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        write_json(
            target / "index.json",
            {
                "name": self.name,
                "model_name": self.model_name,
                "units": [unit.to_dict() for unit in self.units],
            },
        )
        if self.embeddings is not None:
            np.savez_compressed(target / "embeddings.npz", embeddings=self.embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "SentenceTransformerTextIndex":
        target = Path(path)
        data = read_json(target / "index.json")
        index = cls(model_name=str(data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")))
        index.units = [EvidenceUnit.from_dict(row) for row in data["units"]]
        emb_path = target / "embeddings.npz"
        if emb_path.exists():
            index.embeddings = np.load(emb_path)["embeddings"]
        return index

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence_transformer index requires sentence-transformers. "
                "Install with: python -m pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model
