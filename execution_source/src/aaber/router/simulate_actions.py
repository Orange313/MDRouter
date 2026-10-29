from __future__ import annotations

from collections import defaultdict

from aaber.actions.base import RetrievalAction
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query
from aaber.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from aaber.router.fusion import fuse_results
from aaber.router.label_builder import ACTION_COSTS, add_action_features

QUERY_LEVEL_ACTIONS = {
    "text-search",
    "dense-search",
    "image-search",
    "multivector-rerank",
    "multivector-search",
    "cross-rerank",
}
QUERY_LEVEL_UNIT_ID = "__query__"


def simulate_action_labels(
    queries: list[Query],
    units: list[EvidenceUnit],
    qrels: dict[str, set[str]],
    feature_rows: list[dict],
    base_candidates: list[EvidenceResult],
    actions: list[RetrievalAction],
    top_k: int = 10,
    gain_metric: str = "ndcg",
    fusion_mode: str = "weighted_sum",
) -> list[dict]:
    unit_by_id = {unit.unit_id: unit for unit in units}
    features_by_key = {(row["query_id"], row["unit_id"]): row for row in feature_rows}
    features_by_query: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        features_by_query[str(row["query_id"])].append(row)
    base_by_query: dict[str, list[EvidenceResult]] = defaultdict(list)
    for result in base_candidates:
        base_by_query[result.query_id].append(result)

    batched_action_results: dict[str, dict[str, list[EvidenceResult]]] = {}
    for action in actions:
        execute_batch_candidates = getattr(action, "execute_batch_candidates", None)
        if callable(execute_batch_candidates):
            seed_units_by_query = _seed_units_by_query(queries, base_by_query, unit_by_id)
            batched_action_results[action.name] = execute_batch_candidates(queries, seed_units_by_query, top_k=top_k)
        elif hasattr(action, "execute_batch"):
            batched_action_results[action.name] = action.execute_batch(queries, top_k=top_k)  # type: ignore[attr-defined]

    rows: list[dict] = []
    for query in queries:
        positives = qrels.get(query.query_id, set())
        base_rank = fuse_results(base_by_query.get(query.query_id, []), top_k=top_k, mode=fusion_mode)
        ids_before = [r.unit_id for r in base_rank]
        metrics_before = _ranking_metrics(ids_before, positives, top_k)
        candidate_unit_ids = _candidate_unit_ids_by_rank(base_by_query.get(query.query_id, []))
        query_feature_row = _query_level_feature_row(
            query=query,
            feature_rows=features_by_query.get(query.query_id, []),
        )
        query_action_cache: dict[str, list[EvidenceResult]] = {}

        for action in actions:
            if action.name not in QUERY_LEVEL_ACTIONS:
                continue
            action_results = _execute_query_level_action(
                action=action,
                query=query,
                candidate_unit_ids=candidate_unit_ids,
                unit_by_id=unit_by_id,
                query_action_cache=query_action_cache,
                batched_action_results=batched_action_results,
                top_k=top_k,
            )
            rows.append(
                _make_label_row(
                    feature_row=query_feature_row,
                    action=action,
                    query=query,
                    unit=None,
                    unit_id=QUERY_LEVEL_UNIT_ID,
                    positives=positives,
                    metrics_before=metrics_before,
                    base_results=base_by_query.get(query.query_id, []),
                    action_results=action_results,
                    top_k=top_k,
                    gain_metric=gain_metric,
                    action_scope="query",
                    fusion_mode=fusion_mode,
                )
            )

        for unit_id in candidate_unit_ids:
            unit = unit_by_id.get(unit_id)
            if unit is None:
                continue
            feature_row = features_by_key.get((query.query_id, unit_id))
            if feature_row is None:
                continue
            for action in actions:
                if action.name in QUERY_LEVEL_ACTIONS:
                    continue
                seed_units = [unit]
                action_results = action.execute(query, seed_units, top_k=top_k)
                rows.append(
                    _make_label_row(
                        feature_row=_unit_level_feature_row(feature_row),
                        action=action,
                        query=query,
                        unit=unit,
                        unit_id=unit_id,
                        positives=positives,
                        metrics_before=metrics_before,
                        base_results=base_by_query.get(query.query_id, []),
                        action_results=action_results,
                        top_k=top_k,
                        gain_metric=gain_metric,
                        action_scope="unit",
                        fusion_mode=fusion_mode,
                    )
                )
    return rows


def action_cost(action_name: str) -> float:
    return ACTION_COSTS.get(action_name, 1.0)


def _ranking_metrics(ranked_ids: list[str], positives: set[str], top_k: int) -> dict[str, float]:
    return {
        "recall": recall_at_k(ranked_ids, positives, top_k),
        "mrr": mrr_at_k(ranked_ids, positives, top_k),
        "ndcg": ndcg_at_k(ranked_ids, positives, top_k),
    }


def _candidate_unit_ids_by_rank(results: list[EvidenceResult]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    ordered = sorted(results, key=lambda row: (row.rank or 999999, -row.score))
    for result in ordered:
        if result.unit_id in seen:
            continue
        seen.add(result.unit_id)
        output.append(result.unit_id)
    return output


def _seed_units_by_query(
    queries: list[Query],
    base_by_query: dict[str, list[EvidenceResult]],
    unit_by_id: dict[str, EvidenceUnit],
) -> dict[str, list[EvidenceUnit]]:
    output: dict[str, list[EvidenceUnit]] = {}
    for query in queries:
        unit_ids = _candidate_unit_ids_by_rank(base_by_query.get(query.query_id, []))
        output[query.query_id] = [unit_by_id[x] for x in unit_ids if x in unit_by_id]
    return output


def _execute_query_level_action(
    action: RetrievalAction,
    query: Query,
    candidate_unit_ids: list[str],
    unit_by_id: dict[str, EvidenceUnit],
    query_action_cache: dict[str, list[EvidenceResult]],
    batched_action_results: dict[str, dict[str, list[EvidenceResult]]],
    top_k: int,
) -> list[EvidenceResult]:
    cache_key = action.name
    if cache_key in query_action_cache:
        return query_action_cache[cache_key]
    if cache_key in batched_action_results:
        query_action_cache[cache_key] = batched_action_results[cache_key].get(query.query_id, [])
    elif action.name in {"multivector-rerank", "cross-rerank"}:
        seed_units = [unit_by_id[x] for x in candidate_unit_ids if x in unit_by_id]
        query_action_cache[cache_key] = action.execute(query, seed_units, top_k=top_k)
    else:
        query_action_cache[cache_key] = action.execute(query, [], top_k=top_k)
    return query_action_cache[cache_key]


def _make_label_row(
    feature_row: dict,
    action: RetrievalAction,
    query: Query,
    unit: EvidenceUnit | None,
    unit_id: str,
    positives: set[str],
    metrics_before: dict[str, float],
    base_results: list[EvidenceResult],
    action_results: list[EvidenceResult],
    top_k: int,
    gain_metric: str,
    action_scope: str,
    fusion_mode: str,
) -> dict:
    rank_after = fuse_results([*base_results, *action_results], top_k=top_k, mode=fusion_mode)
    ids_after = [r.unit_id for r in rank_after]
    metrics_after = _ranking_metrics(ids_after, positives, top_k)
    recall_gain = metrics_after["recall"] - metrics_before["recall"]
    mrr_gain = metrics_after["mrr"] - metrics_before["mrr"]
    ndcg_gain = metrics_after["ndcg"] - metrics_before["ndcg"]
    gain = {
        "recall": recall_gain,
        "mrr": mrr_gain,
        "ndcg": ndcg_gain,
    }.get(gain_metric, ndcg_gain)
    cost = action.estimate_cost(query, unit)
    normalized_gain = gain / max(cost, 1e-9)
    label = add_action_features(feature_row, action.name, cost)
    label.update(
        {
            "unit_id": unit_id,
            "action_scope": action_scope,
            "action_scope_id": 1.0 if action_scope == "query" else 0.0,
            "is_query_level_action": 1.0 if action_scope == "query" else 0.0,
            "is_unit_level_action": 1.0 if action_scope == "unit" else 0.0,
            "metric_before": metrics_before.get(gain_metric, metrics_before["ndcg"]),
            "metric_after": metrics_after.get(gain_metric, metrics_after["ndcg"]),
            "gain": gain,
            "recall_before": metrics_before["recall"],
            "recall_after": metrics_after["recall"],
            "recall_gain": recall_gain,
            "mrr_before": metrics_before["mrr"],
            "mrr_after": metrics_after["mrr"],
            "mrr_gain": mrr_gain,
            "ndcg_before": metrics_before["ndcg"],
            "ndcg_after": metrics_after["ndcg"],
            "ndcg_gain": ndcg_gain,
            "gain_metric": gain_metric,
            "fusion_mode": fusion_mode,
            "normalized_gain": normalized_gain,
            "value_gain": gain * cost,
            "positive_value_gain": max(0.0, gain) * cost,
            "net_utility_l001": gain - 0.001 * cost,
            "net_utility_l005": gain - 0.005 * cost,
            "net_utility_l01": gain - 0.01 * cost,
            "latency_ms": cost,
            "gpu_ms": 0.0,
            "token_cost": 0.0,
            "is_positive_unit": float(unit_id in positives),
        }
    )
    return label


def _unit_level_feature_row(row: dict) -> dict:
    result = dict(row)
    result["action_scope"] = "unit"
    result["action_scope_id"] = 0.0
    result["is_query_level_action"] = 0.0
    result["is_unit_level_action"] = 1.0
    return result


def _query_level_feature_row(query: Query, feature_rows: list[dict]) -> dict:
    if not feature_rows:
        return {
            "query_id": query.query_id,
            "unit_id": QUERY_LEVEL_UNIT_ID,
            "query_type": query.query_type or "",
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
