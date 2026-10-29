from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from aaber.common.io import read_json, write_json
from aaber.common.text import cosine_counter, has_any, token_counter
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query
from aaber.index.base import RetrieverIndex

VISUAL_TERMS = ["image", "figure", "fig", "chart", "plot", "diagram", "table", "图", "表"]


class ImageTextIndex(RetrieverIndex):
    """Text-to-visual placeholder for CLIP/SigLIP style retrieval."""

    name = "clip_image"

    def __init__(self) -> None:
        self.units: list[EvidenceUnit] = []
        self.vectors: list[Counter[str]] = []

    def build(self, units: list[EvidenceUnit]) -> None:
        self.units = units
        self.vectors = [
            token_counter(" ".join([unit.caption, unit.context_text, unit.text if unit.image_path else ""]))
            for unit in units
        ]

    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        query_vec = token_counter(query.question)
        visual_bonus = has_any(query.question, VISUAL_TERMS)
        scored: list[tuple[str, float]] = []
        for unit, vec in zip(self.units, self.vectors):
            score = cosine_counter(query_vec, vec)
            if unit.image_path:
                score += 0.05 * visual_bonus
            if score > 0:
                scored.append((unit.unit_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            EvidenceResult(query.query_id, unit_id, score, self.name, rank=idx)
            for idx, (unit_id, score) in enumerate(scored[:top_k], start=1)
        ]

    def search_batch(self, queries: list[Query], top_k: int = 10) -> dict[str, list[EvidenceResult]]:
        return {query.query_id: self.search(query, top_k=top_k) for query in queries}

    def save(self, path: str | Path) -> None:
        write_json(Path(path) / "index.json", self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ImageTextIndex":
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


class TransformersImageTextIndex(RetrieverIndex):
    """Real image-text retriever backed by Transformers CLIP/SigLIP models."""

    name = "clip_transformers"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 8) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.units: list[EvidenceUnit] = []
        self.embeddings: np.ndarray | None = None
        self.valid_unit_indices: list[int] = []
        self._model: Any | None = None
        self._processor: Any | None = None

    def build(self, units: list[EvidenceUnit]) -> None:
        self.units = units
        model, processor, torch = self._load_model()
        valid_items = [(idx, unit) for idx, unit in enumerate(units) if unit.image_path and Path(unit.image_path).exists()]
        self.valid_unit_indices = [idx for idx, _ in valid_items]
        if not valid_items:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
            return

        from PIL import Image

        vectors: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(valid_items), self.batch_size):
                batch = valid_items[start : start + self.batch_size]
                images = [Image.open(unit.image_path).convert("RGB") for _, unit in batch]
                inputs = processor(images=images, return_tensors="pt")
                features = _get_image_features(model, inputs)
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
                vectors.append(features.cpu().numpy().astype(np.float32))
        self.embeddings = np.vstack(vectors)

    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []
        model, processor, torch = self._load_model()
        model.eval()
        with torch.no_grad():
            inputs = processor(text=[query.question], return_tensors="pt", padding=True, truncation=True)
            text_features = _get_text_features(model, inputs)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            query_vec = text_features.cpu().numpy().astype(np.float32)[0]

        scores = self.embeddings @ query_vec
        top_idx = np.argsort(-scores)[:top_k]
        results: list[EvidenceResult] = []
        for rank, emb_idx in enumerate(top_idx, start=1):
            unit = self.units[self.valid_unit_indices[int(emb_idx)]]
            results.append(
                EvidenceResult(
                    query_id=query.query_id,
                    unit_id=unit.unit_id,
                    score=float(scores[int(emb_idx)]),
                    source_action=self.name,
                    rank=rank,
                )
            )
        return results

    def search_batch(self, queries: list[Query], top_k: int = 10) -> dict[str, list[EvidenceResult]]:
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return {query.query_id: [] for query in queries}
        model, processor, torch = self._load_model()
        model.eval()
        output: dict[str, list[EvidenceResult]] = {}
        with torch.no_grad():
            for start in range(0, len(queries), self.batch_size):
                batch = queries[start : start + self.batch_size]
                inputs = processor(
                    text=[query.question for query in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                text_features = _get_text_features(model, inputs)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
                query_vectors = text_features.cpu().numpy().astype(np.float32)
                score_matrix = query_vectors @ self.embeddings.T
                for row_idx, query in enumerate(batch):
                    scores = score_matrix[row_idx]
                    top_idx = np.argsort(-scores)[:top_k]
                    results: list[EvidenceResult] = []
                    for rank, emb_idx in enumerate(top_idx, start=1):
                        unit = self.units[self.valid_unit_indices[int(emb_idx)]]
                        results.append(
                            EvidenceResult(
                                query_id=query.query_id,
                                unit_id=unit.unit_id,
                                score=float(scores[int(emb_idx)]),
                                source_action=self.name,
                                rank=rank,
                            )
                        )
                    output[query.query_id] = results
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
                "units": [unit.to_dict() for unit in self.units],
                "valid_unit_indices": self.valid_unit_indices,
            },
        )
        if self.embeddings is not None:
            np.savez_compressed(target / "embeddings.npz", embeddings=self.embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "TransformersImageTextIndex":
        target = Path(path)
        data = read_json(target / "index.json")
        index = cls(
            model_name=str(data.get("model_name", "openai/clip-vit-base-patch32")),
            batch_size=int(data.get("batch_size", 8)),
        )
        index.units = [EvidenceUnit.from_dict(row) for row in data["units"]]
        index.valid_unit_indices = [int(idx) for idx in data.get("valid_unit_indices", [])]
        emb_path = target / "embeddings.npz"
        if emb_path.exists():
            index.embeddings = np.load(emb_path)["embeddings"]
        return index

    def _load_model(self):
        if self._model is not None and self._processor is not None:
            import torch

            return self._model, self._processor, torch
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "clip_transformers index requires torch and transformers. "
                "Install with: python -m pip install torch transformers"
            ) from exc
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        return self._model, self._processor, torch


def _get_image_features(model: Any, inputs: dict[str, Any]):
    if hasattr(model, "get_image_features"):
        return _as_feature_tensor(model.get_image_features(**inputs))
    outputs = model(**inputs)
    return _as_feature_tensor(outputs, preferred_keys=("image_embeds", "pooler_output", "last_hidden_state"))


def _get_text_features(model: Any, inputs: dict[str, Any]):
    if hasattr(model, "get_text_features"):
        return _as_feature_tensor(model.get_text_features(**inputs))
    outputs = model(**inputs)
    return _as_feature_tensor(outputs, preferred_keys=("text_embeds", "pooler_output", "last_hidden_state"))


def _as_feature_tensor(value: Any, preferred_keys: tuple[str, ...] = ("pooler_output", "image_embeds", "text_embeds")):
    if hasattr(value, "norm"):
        return value
    for key in preferred_keys:
        if hasattr(value, key):
            tensor = getattr(value, key)
            if hasattr(tensor, "norm"):
                if key == "last_hidden_state" and len(tensor.shape) == 3:
                    return tensor[:, 0]
                return tensor
        if isinstance(value, dict) and key in value:
            tensor = value[key]
            if hasattr(tensor, "norm"):
                if key == "last_hidden_state" and len(tensor.shape) == 3:
                    return tensor[:, 0]
                return tensor
    raise RuntimeError("Model output does not expose text features.")
