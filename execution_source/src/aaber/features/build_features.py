from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from aaber.common.text import cosine_counter, keyword_overlap, score_entropy, token_counter
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query, Relation
from aaber.features.alignment_features import build_alignment_features
from aaber.features.layout_features import build_layout_features
from aaber.features.quality_features import build_quality_features
from aaber.features.query_features import build_query_features
from aaber.features.relation_features import build_relation_features, relation_maps


def build_candidate_features(
    queries: list[Query],
    units: list[EvidenceUnit],
    relations: list[Relation],
    candidates: Iterable[EvidenceResult],
    auxiliary_candidates: Iterable[EvidenceResult] | None = None,
) -> list[dict]:
    unit_by_id = {unit.unit_id: unit for unit in units}
    query_by_id = {query.query_id: query for query in queries}
    outgoing, incoming = relation_maps(relations)
    score_index: dict[tuple[str, str], list[EvidenceResult]] = defaultdict(list)
    base_candidates = list(candidates)
    auxiliary_candidates = list(auxiliary_candidates or [])
    for result in base_candidates:
        score_index[(result.query_id, result.unit_id)].append(result)
    query_source_rows = _query_source_features(base_candidates, auxiliary_candidates)
    query_uncertainty_rows = _query_uncertainty_features(base_candidates)

    rows: list[dict] = []
    for (query_id, unit_id), result_group in sorted(score_index.items()):
        query = query_by_id.get(query_id)
        unit = unit_by_id.get(unit_id)
        if query is None or unit is None:
            continue
        row: dict = {
            "query_id": query_id,
            "unit_id": unit_id,
            "max_candidate_score": max(r.score for r in result_group),
            "mean_candidate_score": sum(r.score for r in result_group) / len(result_group),
            "candidate_sources": ",".join(sorted({r.source_action for r in result_group})),
        }
        row.update(query_source_rows.get(query_id, {}))
        for result in result_group:
            row[f"{result.source_action}_score"] = max(row.get(f"{result.source_action}_score", 0.0), result.score)
            row[f"{result.source_action}_rank"] = min(
                float(row.get(f"{result.source_action}_rank", result.rank or 9999)),
                float(result.rank or 9999),
            )
        row.update(build_query_features(query))
        row.update(build_alignment_features(query, unit))
        row.update(build_relation_features(unit, outgoing, incoming))
        row.update(build_layout_features(unit))
        row.update(build_quality_features(unit))
        row.update(query_uncertainty_rows.get(query_id, _empty_query_uncertainty_features()))
        row["retrieval_score_variance"] = _score_variance([r.score for r in result_group])
        rows.append(row)
    return rows


class CandidateFeatureCache:
    """Static page-side feature cache for semantically identical online feature construction."""

    def __init__(self, units: list[EvidenceUnit], relations: list[Relation]) -> None:
        self.unit_by_id = {unit.unit_id: unit for unit in units}
        outgoing, incoming = relation_maps(relations)
        self.counters: dict[str, dict[str, Any]] = {}
        self.searchable_token_sets: dict[str, set[str]] = {}
        self.static_features: dict[str, dict[str, float]] = {}
        for unit in units:
            text_vec = token_counter(unit.text)
            caption_vec = token_counter(unit.caption)
            context_vec = token_counter(unit.context_text)
            searchable_vec = token_counter(unit.searchable_text)
            self.counters[unit.unit_id] = {
                "text": text_vec,
                "caption": caption_vec,
                "context": context_vec,
                "searchable": searchable_vec,
            }
            self.searchable_token_sets[unit.unit_id] = set(searchable_vec)
            static = {
                "image_text_sim": cosine_counter(caption_vec, text_vec),
                "caption_image_consistency": _cached_caption_image_consistency(unit),
                "caption_missing_flag": float(not bool(unit.caption)),
                "has_image_flag": float(bool(unit.image_path)),
            }
            static.update(build_relation_features(unit, outgoing, incoming))
            static.update(build_layout_features(unit))
            static.update(build_quality_features(unit))
            self.static_features[unit.unit_id] = static

    def alignment_features(
        self,
        unit_id: str,
        query_vec: Any,
        query_tokens: set[str],
    ) -> dict[str, float]:
        counters = self.counters[unit_id]
        searchable_tokens = self.searchable_token_sets[unit_id]
        return {
            "query_text_sim": cosine_counter(query_vec, counters["text"]),
            "query_caption_sim": cosine_counter(query_vec, counters["caption"]),
            "query_context_sim": cosine_counter(query_vec, counters["context"]),
            "query_unit_sim": cosine_counter(query_vec, counters["searchable"]),
            "keyword_overlap": (
                len(query_tokens & searchable_tokens) / max(1, len(query_tokens))
                if query_tokens and searchable_tokens
                else 0.0
            ),
        }


def build_candidate_features_cached(
    cache: CandidateFeatureCache,
    queries: list[Query],
    candidates: Iterable[EvidenceResult],
    auxiliary_candidates: Iterable[EvidenceResult] | None = None,
) -> list[dict]:
    """Cached equivalent of :func:`build_candidate_features` for online routing."""

    query_by_id = {query.query_id: query for query in queries}
    score_index: dict[tuple[str, str], list[EvidenceResult]] = defaultdict(list)
    base_candidates = list(candidates)
    auxiliary_candidates = list(auxiliary_candidates or [])
    for result in base_candidates:
        score_index[(result.query_id, result.unit_id)].append(result)
    query_source_rows = _query_source_features(base_candidates, auxiliary_candidates)
    query_uncertainty_rows = _query_uncertainty_features(base_candidates)
    query_features = {query.query_id: build_query_features(query) for query in queries}
    query_counters = {query.query_id: token_counter(query.question) for query in queries}
    query_token_sets = {query_id: set(counter) for query_id, counter in query_counters.items()}

    rows: list[dict] = []
    for (query_id, unit_id), result_group in sorted(score_index.items()):
        query = query_by_id.get(query_id)
        unit = cache.unit_by_id.get(unit_id)
        if query is None or unit is None:
            continue
        row: dict = {
            "query_id": query_id,
            "unit_id": unit_id,
            "max_candidate_score": max(result.score for result in result_group),
            "mean_candidate_score": sum(result.score for result in result_group) / len(result_group),
            "candidate_sources": ",".join(sorted({result.source_action for result in result_group})),
        }
        row.update(query_source_rows.get(query_id, {}))
        for result in result_group:
            row[f"{result.source_action}_score"] = max(
                row.get(f"{result.source_action}_score", 0.0), result.score
            )
            row[f"{result.source_action}_rank"] = min(
                float(row.get(f"{result.source_action}_rank", result.rank or 9999)),
                float(result.rank or 9999),
            )
        row.update(query_features[query_id])
        row.update(
            cache.alignment_features(
                unit_id,
                query_counters[query_id],
                query_token_sets[query_id],
            )
        )
        row.update(cache.static_features[unit_id])
        row.update(query_uncertainty_rows.get(query_id, _empty_query_uncertainty_features()))
        row["retrieval_score_variance"] = _score_variance([result.score for result in result_group])
        rows.append(row)
    return rows


def _cached_caption_image_consistency(unit: EvidenceUnit) -> float:
    if not unit.caption or not unit.image_path:
        return 0.0
    return max(
        keyword_overlap(unit.caption, unit.text),
        keyword_overlap(unit.caption, unit.context_text),
        0.1,
    )


def _query_source_features(
    base_candidates: list[EvidenceResult],
    auxiliary_candidates: list[EvidenceResult],
) -> dict[str, dict[str, float]]:
    base_by_query: dict[str, list[EvidenceResult]] = defaultdict(list)
    image_by_query: dict[str, list[EvidenceResult]] = defaultdict(list)
    for result in base_candidates:
        base_by_query[result.query_id].append(result)
    for result in auxiliary_candidates:
        if result.source_action in {"clip_transformers", "siglip_image", "clip_image", "image", "image-search"}:
            image_by_query[result.query_id].append(result)

    output: dict[str, dict[str, float]] = {}
    for query_id in sorted(set(base_by_query) | set(image_by_query)):
        base_rows = sorted(base_by_query.get(query_id, []), key=lambda row: row.rank or 999999)
        image_rows = sorted(image_by_query.get(query_id, []), key=lambda row: row.rank or 999999)
        base_ids_10 = {row.unit_id for row in base_rows[:10]}
        base_ids_20 = {row.unit_id for row in base_rows[:20]}
        image_ids_10 = {row.unit_id for row in image_rows[:10]}
        image_ids_20 = {row.unit_id for row in image_rows[:20]}
        base_top_score = max((row.score for row in base_rows), default=0.0)
        image_top_score = max((row.score for row in image_rows), default=0.0)
        image_scores = [row.score for row in image_rows[:10]]
        output[query_id] = {
            "base_candidate_count": float(len(base_rows)),
            "image_candidate_count": float(len(image_rows)),
            "image_has_candidates": float(bool(image_rows)),
            "base_top_score": float(base_top_score),
            "image_top_score": float(image_top_score),
            "image_vs_base_top_score_gap": float(image_top_score - base_top_score),
            "image_top_score_ratio": float(image_top_score / max(base_top_score, 1e-9)) if base_top_score else 0.0,
            "image_top_in_base_top10": float(bool(image_rows and image_rows[0].unit_id in base_ids_10)),
            "image_base_overlap_at10": float(len(image_ids_10 & base_ids_10) / max(1, len(image_ids_10))),
            "image_base_overlap_at20": float(len(image_ids_20 & base_ids_20) / max(1, len(image_ids_20))),
            "image_score_margin": float(image_scores[0] - image_scores[1]) if len(image_scores) >= 2 else (float(image_scores[0]) if image_scores else 0.0),
            "image_score_entropy": float(score_entropy(image_scores)),
        }
    return output


def _query_uncertainty_features(base_candidates: list[EvidenceResult]) -> dict[str, dict[str, float]]:
    scores_by_query: dict[str, list[float]] = defaultdict(list)
    for result in base_candidates:
        scores_by_query[result.query_id].append(result.score)

    output: dict[str, dict[str, float]] = {}
    for query_id, scores in scores_by_query.items():
        sorted_scores = sorted(scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else (sorted_scores[0] if sorted_scores else 0.0)
        output[query_id] = {
            "top1_top2_margin": float(margin),
            "score_entropy": float(score_entropy(sorted_scores[:10])),
        }
    return output


def _empty_query_uncertainty_features() -> dict[str, float]:
    return {"top1_top2_margin": 0.0, "score_entropy": 0.0}


def _score_variance(scores: list[float]) -> float:
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    return float(sum((score - avg) ** 2 for score in scores) / len(scores))


def feature_columns(rows: list[dict]) -> list[str]:
    ignored = {"query_id", "unit_id", "candidate_sources", "query_type"}
    cols: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in ignored:
                continue
            if isinstance(value, int | float):
                cols.add(key)
    return sorted(cols)
