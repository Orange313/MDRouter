from __future__ import annotations

from collections import defaultdict

from aaber.data.schema import EvidenceUnit, Relation


def relation_maps(relations: list[Relation]) -> tuple[dict[str, list[Relation]], dict[str, list[Relation]]]:
    outgoing: dict[str, list[Relation]] = defaultdict(list)
    incoming: dict[str, list[Relation]] = defaultdict(list)
    for relation in relations:
        outgoing[relation.src_unit_id].append(relation)
        incoming[relation.dst_unit_id].append(relation)
    return outgoing, incoming


def build_relation_features(
    unit: EvidenceUnit,
    outgoing: dict[str, list[Relation]],
    incoming: dict[str, list[Relation]],
) -> dict[str, float]:
    rels = outgoing.get(unit.unit_id, []) + incoming.get(unit.unit_id, [])
    types = {r.relation_type for r in rels}
    max_score = max((r.score for r in rels), default=0.0)
    max_by_type = {relation_type: max((r.score for r in rels if r.relation_type == relation_type), default=0.0) for relation_type in types}
    return {
        "relation_degree": float(len(rels)),
        "relation_max_score": float(max_score),
        "has_caption_link": float("caption_link" in types),
        "has_reference_link": float("reference_link" in types),
        "has_adjacent_page": float("adjacent_page" in types),
        "has_cross_page_link": float("cross_page" in types),
        "has_same_doc": float("same_doc" in types),
        "has_same_source": float("same_source" in types),
        "has_same_topic": float("same_topic" in types),
        "same_section_score": float(max_by_type.get("same_section", 0.0)),
        "cross_page_link_score": float(max_by_type.get("cross_page", 0.0)),
        "same_doc_score": float(max_by_type.get("same_doc", 0.0)),
        "same_source_score": float(max_by_type.get("same_source", 0.0)),
        "same_topic_score": float(max_by_type.get("same_topic", 0.0)),
        "relation_path_length": 1.0 if rels else 0.0,
    }
