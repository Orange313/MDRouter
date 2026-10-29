from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Query:
    query_id: str
    question: str
    dataset: str
    query_type: str | None = None
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Query":
        return cls(
            query_id=str(data["query_id"]),
            question=str(data.get("question", data.get("query", ""))),
            dataset=str(data.get("dataset", "")),
            query_type=data.get("query_type"),
            answer=data.get("answer"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceUnit:
    unit_id: str
    doc_id: str
    page_id: int
    unit_type: str
    text: str = ""
    caption: str = ""
    context_text: str = ""
    image_path: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    layout_type: str = "unknown"
    relations: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceUnit":
        bbox = data.get("bbox")
        return cls(
            unit_id=str(data["unit_id"]),
            doc_id=str(data.get("doc_id", "")),
            page_id=int(data.get("page_id", 0)),
            unit_type=str(data.get("unit_type", "page")),
            text=str(data.get("text", "")),
            caption=str(data.get("caption", "")),
            context_text=str(data.get("context_text", "")),
            image_path=data.get("image_path"),
            bbox=tuple(bbox) if bbox is not None else None,  # type: ignore[arg-type]
            layout_type=str(data.get("layout_type", data.get("unit_type", "unknown"))),
            relations=list(data.get("relations", [])),
            quality=dict(data.get("quality", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = list(self.bbox)
        return data

    @property
    def searchable_text(self) -> str:
        return " ".join(part for part in [self.text, self.caption, self.context_text] if part)


@dataclass
class Relation:
    relation_id: str
    src_unit_id: str
    dst_unit_id: str
    relation_type: str
    score: float = 1.0
    evidence: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        return cls(
            relation_id=str(data["relation_id"]),
            src_unit_id=str(data["src_unit_id"]),
            dst_unit_id=str(data["dst_unit_id"]),
            relation_type=str(data.get("relation_type", "related")),
            score=float(data.get("score", 1.0)),
            evidence=str(data.get("evidence", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceResult:
    query_id: str
    unit_id: str
    score: float
    source_action: str
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceResult":
        return cls(
            query_id=str(data["query_id"]),
            unit_id=str(data["unit_id"]),
            score=float(data.get("score", 0.0)),
            source_action=str(data.get("source_action", data.get("retriever", ""))),
            rank=data.get("rank"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionCandidate:
    query_id: str
    unit_id: str
    action_name: str
    predicted_gain: float
    estimated_cost: float
    priority: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionCandidate":
        return cls(
            query_id=str(data["query_id"]),
            unit_id=str(data["unit_id"]),
            action_name=str(data["action_name"]),
            predicted_gain=float(data.get("predicted_gain", 0.0)),
            estimated_cost=float(data.get("estimated_cost", data.get("cost", 1.0))),
            priority=float(data.get("priority", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteTrace:
    query_id: str
    budget: float
    used_budget: float
    selected_actions: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

