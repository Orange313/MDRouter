from __future__ import annotations

from pathlib import Path
from typing import Any

from aaber.common.io import read_json, write_json
from aaber.common.text import keyword_overlap
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query
from aaber.index.base import RetrieverIndex


class MultiVectorRerankIndex(RetrieverIndex):
    """ColPali/ColQwen-compatible reranker placeholder."""

    name = "colpali"

    def __init__(self) -> None:
        self.units: list[EvidenceUnit] = []

    def build(self, units: list[EvidenceUnit]) -> None:
        self.units = units

    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        scored = []
        for unit in self.units:
            score = 0.5 * keyword_overlap(query.question, unit.text)
            score += 0.3 * keyword_overlap(query.question, unit.caption)
            score += 0.2 * keyword_overlap(query.question, unit.context_text)
            if unit.image_path and any(term in query.question.lower() for term in ["chart", "figure", "image", "table"]):
                score += 0.1
            if score > 0:
                scored.append((unit.unit_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            EvidenceResult(query.query_id, unit_id, score, self.name, rank=idx)
            for idx, (unit_id, score) in enumerate(scored[:top_k], start=1)
        ]

    def rerank(self, query: Query, candidate_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        original_units = self.units
        self.units = candidate_units
        try:
            return self.search(query, top_k=top_k)
        finally:
            self.units = original_units

    def save(self, path: str | Path) -> None:
        write_json(Path(path) / "index.json", {"name": self.name, "units": [u.to_dict() for u in self.units]})

    @classmethod
    def load(cls, path: str | Path) -> "MultiVectorRerankIndex":
        data = read_json(Path(path) / "index.json")
        index = cls()
        index.units = [EvidenceUnit.from_dict(row) for row in data["units"]]
        return index


class ColPaliEngineIndex(RetrieverIndex):
    """Optional real ColPali/ColQwen multivector index via colpali-engine."""

    name = "colpali_engine"

    def __init__(self, model_name: str = "vidore/colpali-v1.3", batch_size: int = 2, device: str | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.units: list[EvidenceUnit] = []
        self.valid_unit_indices: list[int] = []
        self.embeddings: list[Any] = []
        self._model: Any | None = None
        self._processor: Any | None = None
        self._passage_embeddings_device_key: str | None = None
        self._passage_embeddings_device: list[Any] | None = None

    def build(self, units: list[EvidenceUnit]) -> None:
        model, processor, torch, device = self._load_model()
        self.units = units
        valid_items = [(idx, unit) for idx, unit in enumerate(units) if unit.image_path and Path(unit.image_path).exists()]
        self.valid_unit_indices = [idx for idx, _ in valid_items]
        self.embeddings = []
        self._passage_embeddings_device = None
        self._passage_embeddings_device_key = None
        if not valid_items:
            return

        from PIL import Image

        model.eval()
        with torch.no_grad():
            for start in range(0, len(valid_items), self.batch_size):
                batch = valid_items[start : start + self.batch_size]
                images = [Image.open(unit.image_path).convert("RGB") for _, unit in batch]
                inputs = processor.process_images(images).to(device)
                batch_embeddings = model(**inputs)
                for emb in batch_embeddings:
                    self.embeddings.append(emb.detach().cpu())

    def search(self, query: Query, top_k: int = 10) -> list[EvidenceResult]:
        if not self.embeddings:
            return []
        model, processor, torch, device = self._load_model()
        model.eval()
        with torch.no_grad():
            query_inputs = processor.process_queries([query.question]).to(device)
            query_embeddings = model(**query_inputs)
            passage_embeddings = self._passage_embeddings_for_device(device)
            scores = processor.score_multi_vector(query_embeddings, passage_embeddings, batch_size=64, device=device)[0]
        top_idx = torch.argsort(scores, descending=True)[:top_k].tolist()
        results: list[EvidenceResult] = []
        for rank, emb_idx in enumerate(top_idx, start=1):
            unit = self.units[self.valid_unit_indices[int(emb_idx)]]
            results.append(
                EvidenceResult(
                    query_id=query.query_id,
                    unit_id=unit.unit_id,
                    score=float(scores[int(emb_idx)].detach().cpu()),
                    source_action=self.name,
                    rank=rank,
                )
            )
        return results

    def rerank(self, query: Query, candidate_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        candidate_ids = {unit.unit_id for unit in candidate_units}
        if not candidate_ids:
            return self.search(query, top_k=top_k)

        selected: list[tuple[int, EvidenceUnit]] = []
        for embedding_index, unit_index in enumerate(self.valid_unit_indices):
            unit = self.units[unit_index]
            if unit.unit_id in candidate_ids:
                selected.append((embedding_index, unit))
        if not selected:
            return []

        model, processor, torch, device = self._load_model()
        model.eval()
        with torch.no_grad():
            query_inputs = processor.process_queries([query.question]).to(device)
            query_embeddings = model(**query_inputs)
            all_passage_embeddings = self._passage_embeddings_for_device(device)
            candidate_embeddings = [all_passage_embeddings[index] for index, _ in selected]
            scores = processor.score_multi_vector(
                query_embeddings,
                candidate_embeddings,
                batch_size=64,
                device=device,
            )[0]

        limit = min(top_k, len(selected))
        top_idx = torch.argsort(scores, descending=True)[:limit].tolist()
        results: list[EvidenceResult] = []
        for rank, candidate_index in enumerate(top_idx, start=1):
            unit = selected[int(candidate_index)][1]
            results.append(
                EvidenceResult(
                    query_id=query.query_id,
                    unit_id=unit.unit_id,
                    score=float(scores[int(candidate_index)].detach().cpu()),
                    source_action=self.name,
                    rank=rank,
                )
            )
        return results

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
                "units": [unit.to_dict() for unit in self.units],
                "valid_unit_indices": self.valid_unit_indices,
            },
        )
        if self.embeddings:
            import torch

            torch.save(self.embeddings, target / "embeddings.pt")

    @classmethod
    def load(cls, path: str | Path) -> "ColPaliEngineIndex":
        target = Path(path)
        data = read_json(target / "index.json")
        index = cls(
            model_name=str(data.get("model_name", "vidore/colpali-v1.3")),
            batch_size=int(data.get("batch_size", 2)),
            device=data.get("device"),
        )
        index.units = [EvidenceUnit.from_dict(row) for row in data["units"]]
        index.valid_unit_indices = [int(idx) for idx in data.get("valid_unit_indices", [])]
        emb_path = target / "embeddings.pt"
        if emb_path.exists():
            import torch

            index.embeddings = torch.load(emb_path, map_location="cpu")
        return index

    def _passage_embeddings_for_device(self, device):
        device_key = str(device)
        if self._passage_embeddings_device is None or self._passage_embeddings_device_key != device_key:
            self._passage_embeddings_device = [emb.to(device) for emb in self.embeddings]
            self._passage_embeddings_device_key = device_key
        return self._passage_embeddings_device

    def _load_model(self):
        if self._model is not None and self._processor is not None:
            import torch

            return self._model, self._processor, torch, self._resolve_device(torch)
        try:
            import torch
            model_cls, processor_cls = self._model_classes()
        except ImportError as exc:
            raise RuntimeError(
                "colpali_engine index requires colpali-engine and torch. "
                "Install with: python -m pip install colpali-engine torch"
            ) from exc
        device = self._resolve_device(torch)
        self._processor = processor_cls.from_pretrained(self.model_name)
        kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
        if device.type == "cuda":
            kwargs["device_map"] = f"cuda:{device.index or 0}"
        key_mapping = self._checkpoint_key_mapping()
        if key_mapping:
            kwargs["key_mapping"] = key_mapping
        self._model = model_cls.from_pretrained(self.model_name, **kwargs)
        if device.type != "cuda":
            self._model = self._model.to(device)
        return self._model, self._processor, torch, device

    def _resolve_device(self, torch):
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _model_classes(self):
        from colpali_engine.models import (
            ColPali,
            ColPaliProcessor,
            ColQwen2,
            ColQwen2Processor,
            ColQwen2_5,
            ColQwen2_5_Processor,
        )

        name = self.model_name.lower()
        if "colqwen2.5" in name or "colqwen2_5" in name:
            return ColQwen2_5, ColQwen2_5_Processor
        if "colqwen2" in name or "colqwen" in name:
            return ColQwen2, ColQwen2Processor
        return ColPali, ColPaliProcessor

    def _checkpoint_key_mapping(self) -> dict[str, str] | None:
        name = self.model_name.lower()
        if "colqwen2" in name or "colqwen" in name:
            return {
                r"^base_model\.model\.custom_text_proj": "custom_text_proj",
                r"^model\.embed_tokens": "language_model.embed_tokens",
                r"^model\.layers": "language_model.layers",
                r"^model\.norm": "language_model.norm",
            }
        return None
