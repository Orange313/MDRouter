from __future__ import annotations

from collections import defaultdict

from aaber.data.schema import EvidenceResult

DEFAULT_SOURCE_WEIGHTS = {
    "bm25": 1.0,
    "dense_text": 1.0,
    "clip_image": 1.0,
    "colpali": 1.2,
    "text-search": 1.0,
    "dense-search": 1.0,
    "image-search": 1.0,
    "cross-rerank": 1.35,
    "relation-expand": 0.6,
    "multivector-rerank": 1.2,
    "multivector-search": 1.2,
    "vlm-verify": 1.5,
}


def fuse_results(
    results: list[EvidenceResult],
    source_weights: dict[str, float] | None = None,
    top_k: int | None = None,
    mode: str = "weighted_sum",
) -> list[EvidenceResult]:
    weights = _source_weights(source_weights, mode)
    mode = mode.lower().replace("-", "_")
    if mode == "sum":
        mode = "weighted_sum"
    if mode == "rrf":
        return _fuse_rrf(results, weights=weights, top_k=top_k)

    max_by_source: dict[str, float] = defaultdict(float)
    for result in results:
        max_by_source[result.source_action] = max(max_by_source[result.source_action], result.score)

    fused: dict[tuple[str, str], EvidenceResult] = {}
    for result in results:
        norm = result.score / max(max_by_source[result.source_action], 1e-9)
        weighted = norm * weights.get(result.source_action, 1.0)
        key = (result.query_id, result.unit_id)
        if key not in fused:
            fused[key] = EvidenceResult(
                query_id=result.query_id,
                unit_id=result.unit_id,
                score=0.0,
                source_action="fusion",
                metadata={"sources": []},
            )
        if mode == "max_norm":
            fused[key].score = max(fused[key].score, weighted)
        else:
            fused[key].score += weighted
        fused[key].metadata.setdefault("sources", []).append(result.source_action)

    ordered = sorted(fused.values(), key=lambda r: r.score, reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]
    for idx, result in enumerate(ordered, start=1):
        result.rank = idx
    return ordered


def rankings_from_results(results: list[EvidenceResult]) -> dict[str, list[str]]:
    grouped: dict[str, list[EvidenceResult]] = defaultdict(list)
    for result in results:
        grouped[result.query_id].append(result)
    rankings: dict[str, list[str]] = {}
    for query_id, group in grouped.items():
        ordered = sorted(group, key=lambda r: (r.rank if r.rank is not None else 999999, -r.score))
        rankings[query_id] = [r.unit_id for r in ordered]
    return rankings


def _source_weights(source_weights: dict[str, float] | None, mode: str) -> dict[str, float]:
    weights = dict(DEFAULT_SOURCE_WEIGHTS)
    if source_weights:
        weights.update(source_weights)
    normalized_mode = mode.lower().replace("-", "_")
    if normalized_mode == "visual_dominant":
        for source in ["bm25", "dense_text", "text-search", "dense-search"]:
            weights[source] = weights.get(source, 1.0) * 0.85
        for source in ["colpali", "multivector-rerank", "multivector-search"]:
            weights[source] = weights.get(source, 1.0) * 1.35
    return weights


def _fuse_rrf(
    results: list[EvidenceResult],
    weights: dict[str, float],
    top_k: int | None,
    rrf_k: int = 60,
) -> list[EvidenceResult]:
    by_source_query: dict[tuple[str, str], list[EvidenceResult]] = defaultdict(list)
    for result in results:
        by_source_query[(result.query_id, result.source_action)].append(result)

    fused: dict[tuple[str, str], EvidenceResult] = {}
    for (_query_id, source), group in by_source_query.items():
        ordered = sorted(group, key=lambda r: (r.rank if r.rank is not None else 999999, -r.score))
        for rank, result in enumerate(ordered, start=1):
            key = (result.query_id, result.unit_id)
            if key not in fused:
                fused[key] = EvidenceResult(
                    query_id=result.query_id,
                    unit_id=result.unit_id,
                    score=0.0,
                    source_action="fusion",
                    metadata={"sources": []},
                )
            fused[key].score += weights.get(source, 1.0) / float(rrf_k + rank)
            fused[key].metadata.setdefault("sources", []).append(source)

    ordered = sorted(fused.values(), key=lambda r: r.score, reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]
    for idx, result in enumerate(ordered, start=1):
        result.rank = idx
    return ordered
