from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from aaber.actions.base import RetrievalAction
from aaber.data.schema import ActionCandidate, EvidenceResult, EvidenceUnit, Query, RouteTrace
from aaber.router.fusion import fuse_results
from aaber.router.label_builder import add_action_features
from aaber.router.selector import greedy_select, used_budget

QUERY_LEVEL_ACTIONS = {
    "text-search",
    "dense-search",
    "image-search",
    "multivector-rerank",
    "multivector-search",
    "cross-rerank",
}
QUERY_LEVEL_UNIT_ID = "__query__"


class GainRouter(Protocol):
    def predict_gain(self, row: dict) -> float:
        ...


def build_action_candidates(
    feature_rows: list[dict],
    router: GainRouter,
    action_names: list[str],
    action_costs: dict[str, float] | None = None,
    selection_score: str = "gain_per_cost",
) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    enriched_rows: list[dict] = []

    query_row = _query_level_feature_row(feature_rows)
    for action_name in action_names:
        if action_name not in QUERY_LEVEL_ACTIONS:
            continue
        enriched = add_action_features(query_row, action_name, action_costs=action_costs)
        enriched["unit_id"] = QUERY_LEVEL_UNIT_ID
        enriched["action_scope"] = "query"
        enriched["action_scope_id"] = 1.0
        enriched["is_query_level_action"] = 1.0
        enriched["is_unit_level_action"] = 0.0
        enriched_rows.append(enriched)

    for row in feature_rows:
        for action_name in action_names:
            if action_name in QUERY_LEVEL_ACTIONS:
                continue
            enriched = add_action_features(row, action_name, action_costs=action_costs)
            enriched["action_scope"] = "unit"
            enriched["action_scope_id"] = 0.0
            enriched["is_query_level_action"] = 0.0
            enriched["is_unit_level_action"] = 1.0
            enriched_rows.append(enriched)

    gains = _predict_gains(router, enriched_rows)
    for enriched, gain in zip(enriched_rows, gains):
        cost = float(enriched["action_cost"])
        priority = gain if selection_score == "gain" else gain / max(cost, 1e-9)
        candidates.append(
            ActionCandidate(
                query_id=str(enriched["query_id"]),
                unit_id=str(enriched["unit_id"]),
                action_name=str(enriched["action_name"]),
                predicted_gain=gain,
                estimated_cost=cost,
                priority=priority,
                metadata={
                    "action_scope": str(enriched.get("action_scope", "unit")),
                    "features": _reason_features(enriched),
                },
            )
        )
    return candidates


def run_budgeted_routing(
    queries: list[Query],
    units: list[EvidenceUnit],
    feature_rows: list[dict],
    base_candidates: list[EvidenceResult],
    actions: dict[str, RetrievalAction],
    router: GainRouter,
    budget: float,
    top_k: int = 10,
    max_per_action: dict[str, int] | None = None,
    min_gain: float = 0.0,
    min_gain_by_action: dict[str, float] | None = None,
    suppress_image_if_text_gain_above: float | None = None,
    fusion_mode: str = "weighted_sum",
    action_costs: dict[str, float] | None = None,
    selection_score: str = "gain_per_cost",
) -> tuple[list[EvidenceResult], list[RouteTrace]]:
    unit_by_id = {unit.unit_id: unit for unit in units}
    rows_by_query: dict[str, list[dict]] = defaultdict(list)
    base_by_query: dict[str, list[EvidenceResult]] = defaultdict(list)
    for row in feature_rows:
        rows_by_query[str(row["query_id"])].append(row)
    for result in base_candidates:
        base_by_query[result.query_id].append(result)

    all_results: list[EvidenceResult] = []
    traces: list[RouteTrace] = []
    action_names = list(actions)
    min_gain_by_action = min_gain_by_action or {}
    for query in queries:
        action_candidates = build_action_candidates(
            rows_by_query.get(query.query_id, []),
            router,
            action_names,
            action_costs=action_costs,
            selection_score=selection_score,
        )
        action_candidates = _suppress_image_when_text_is_confident(
            action_candidates,
            threshold=suppress_image_if_text_gain_above,
        )
        selected = greedy_select(
            action_candidates,
            budget=budget,
            max_per_action=max_per_action,
            min_gain=min_gain,
            min_gain_by_action=min_gain_by_action,
            selection_score=selection_score,
        )
        selected_results: list[EvidenceResult] = list(base_by_query.get(query.query_id, []))
        query_action_cache: dict[str, list[EvidenceResult]] = {}
        added_global_actions: set[str] = set()
        for candidate in selected:
            action = actions[candidate.action_name]
            if candidate.metadata.get("action_scope") == "query" or candidate.unit_id == QUERY_LEVEL_UNIT_ID:
                cache_key = candidate.action_name
                if cache_key not in query_action_cache:
                    if candidate.action_name in {"multivector-rerank", "cross-rerank"}:
                        seed_ids = [r.unit_id for r in base_by_query.get(query.query_id, [])]
                        seed_units = [unit_by_id[x] for x in seed_ids if x in unit_by_id]
                        query_action_cache[cache_key] = action.execute(query, seed_units, top_k=top_k)
                    else:
                        query_action_cache[cache_key] = action.execute(query, [], top_k=top_k)
                if cache_key not in added_global_actions:
                    selected_results.extend(query_action_cache[cache_key])
                    added_global_actions.add(cache_key)
                continue

            seed = unit_by_id.get(candidate.unit_id)
            if seed is None:
                continue
            seed_units = [seed]
            selected_results.extend(action.execute(query, seed_units, top_k=top_k))
        fused = fuse_results(selected_results, top_k=top_k, mode=fusion_mode)
        all_results.extend(fused)
        traces.append(
            RouteTrace(
                query_id=query.query_id,
                budget=budget,
                used_budget=used_budget(selected),
                selected_actions=[
                    {
                        "unit_id": c.unit_id,
                        "action": c.action_name,
                        "action_scope": c.metadata.get("action_scope", "unit"),
                        "predicted_gain": c.predicted_gain,
                        "estimated_cost": c.estimated_cost,
                        "priority": c.priority,
                        "reason_features": c.metadata.get("features", {}),
                    }
                    for c in selected
                ],
                metadata={
                    "min_gain": min_gain,
                    "min_gain_by_action": dict(min_gain_by_action),
                    "suppress_image_if_text_gain_above": suppress_image_if_text_gain_above,
                    "fusion_mode": fusion_mode,
                    "selection_score": selection_score,
                },
            )
        )
    return all_results, traces


def _suppress_image_when_text_is_confident(
    candidates: list[ActionCandidate],
    threshold: float | None,
) -> list[ActionCandidate]:
    if threshold is None:
        return candidates
    text_gains = [
        candidate.predicted_gain
        for candidate in candidates
        if candidate.action_name == "text-search"
        and (candidate.metadata.get("action_scope") == "query" or candidate.unit_id == QUERY_LEVEL_UNIT_ID)
    ]
    if not text_gains or max(text_gains) < threshold:
        return candidates
    return [candidate for candidate in candidates if candidate.action_name != "image-search"]


def _seed_units_by_query(
    queries: list[Query],
    base_by_query: dict[str, list[EvidenceResult]],
    unit_by_id: dict[str, EvidenceUnit],
) -> dict[str, list[EvidenceUnit]]:
    output: dict[str, list[EvidenceUnit]] = {}
    for query in queries:
        seen: set[str] = set()
        units: list[EvidenceUnit] = []
        ordered = sorted(base_by_query.get(query.query_id, []), key=lambda row: (row.rank or 999999, -row.score))
        for result in ordered:
            if result.unit_id in seen or result.unit_id not in unit_by_id:
                continue
            seen.add(result.unit_id)
            units.append(unit_by_id[result.unit_id])
        output[query.query_id] = units
    return output


def _reason_features(row: dict) -> dict[str, float]:
    keys = [
        "query_text_sim",
        "query_caption_sim",
        "query_unit_sim",
        "keyword_overlap",
        "has_chart_keyword",
        "has_table_keyword",
        "has_figure_keyword",
        "relation_degree",
        "has_reference_link",
        "ocr_confidence",
        "top1_top2_margin",
    ]
    return {key: row[key] for key in keys if key in row}


def _predict_gains(router: GainRouter, rows: list[dict]) -> list[float]:
    predict_many = getattr(router, "predict_gains", None)
    if callable(predict_many):
        return [float(value) for value in predict_many(rows)]
    return [float(router.predict_gain(row)) for row in rows]


def _query_level_feature_row(feature_rows: list[dict]) -> dict:
    if not feature_rows:
        return {
            "query_id": "",
            "unit_id": QUERY_LEVEL_UNIT_ID,
            "candidate_sources": "query-level",
            "query_candidate_count": 0.0,
            "action_scope": "query",
            "action_scope_id": 1.0,
            "is_query_level_action": 1.0,
            "is_unit_level_action": 0.0,
        }

    base = dict(feature_rows[0])
    numeric_keys = {
        key
        for row in feature_rows
        for key, value in row.items()
        if isinstance(value, (int, float))
    }
    for key in numeric_keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in feature_rows]
        if key.endswith("_rank"):
            base[key] = min(values)
        elif key in {"top1_top2_margin", "score_entropy", "retrieval_score_variance", "query_length"}:
            base[key] = sum(values) / max(1, len(values))
        else:
            base[key] = max(values)
    base["unit_id"] = QUERY_LEVEL_UNIT_ID
    base["candidate_sources"] = "query-level"
    base["query_candidate_count"] = float(len(feature_rows))
    base["action_scope"] = "query"
    base["action_scope_id"] = 1.0
    base["is_query_level_action"] = 1.0
    base["is_unit_level_action"] = 0.0
    return base
