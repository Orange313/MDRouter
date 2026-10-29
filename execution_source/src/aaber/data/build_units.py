from __future__ import annotations

from aaber.data.schema import EvidenceUnit


def build_page_units(raw_pages: list[dict]) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    for row in raw_pages:
        doc_id = str(row.get("doc_id", row.get("document_id", "")))
        page_id = int(row.get("page_id", row.get("page", 0)))
        unit_id = str(row.get("unit_id", f"{doc_id}_p{page_id}"))
        units.append(
            EvidenceUnit(
                unit_id=unit_id,
                doc_id=doc_id,
                page_id=page_id,
                unit_type="page",
                text=str(row.get("text", row.get("ocr_text", ""))),
                caption=str(row.get("caption", "")),
                context_text=str(row.get("context_text", row.get("section", ""))),
                image_path=row.get("image_path"),
                bbox=tuple(row["bbox"]) if row.get("bbox") else None,  # type: ignore[arg-type]
                layout_type=str(row.get("layout_type", "page")),
                quality=dict(row.get("quality", {})),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return units


def attach_relation_ids(units: list[EvidenceUnit], relation_ids_by_unit: dict[str, list[str]]) -> list[EvidenceUnit]:
    for unit in units:
        existing = set(unit.relations)
        existing.update(relation_ids_by_unit.get(unit.unit_id, []))
        unit.relations = sorted(existing)
    return units

