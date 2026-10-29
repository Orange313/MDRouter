from __future__ import annotations

from aaber.common.text import cosine_counter, keyword_overlap, token_counter
from aaber.data.schema import EvidenceUnit, Query


def build_alignment_features(query: Query, unit: EvidenceUnit) -> dict[str, float]:
    query_vec = token_counter(query.question)
    text_vec = token_counter(unit.text)
    caption_vec = token_counter(unit.caption)
    context_vec = token_counter(unit.context_text)
    searchable_vec = token_counter(unit.searchable_text)
    return {
        "query_text_sim": cosine_counter(query_vec, text_vec),
        "query_caption_sim": cosine_counter(query_vec, caption_vec),
        "query_context_sim": cosine_counter(query_vec, context_vec),
        "query_unit_sim": cosine_counter(query_vec, searchable_vec),
        "keyword_overlap": keyword_overlap(query.question, unit.searchable_text),
        "image_text_sim": cosine_counter(caption_vec, text_vec),
        "caption_image_consistency": _caption_image_consistency(unit),
        "caption_missing_flag": float(not bool(unit.caption)),
        "has_image_flag": float(bool(unit.image_path)),
    }


def _caption_image_consistency(unit: EvidenceUnit) -> float:
    if not unit.caption or not unit.image_path:
        return 0.0
    return max(
        keyword_overlap(unit.caption, unit.text),
        keyword_overlap(unit.caption, unit.context_text),
        0.1,
    )

