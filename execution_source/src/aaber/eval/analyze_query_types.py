from __future__ import annotations

from collections import defaultdict


def group_metrics_by_query_type(query_rows: list[dict], per_query_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for query in query_rows:
        query_id = str(query["query_id"])
        query_type = str(query.get("query_type") or "unknown")
        if query_id in per_query_metrics:
            grouped[query_type].append(per_query_metrics[query_id])
    out: dict[str, dict[str, float]] = {}
    for query_type, rows in grouped.items():
        keys = sorted({key for row in rows for key in row})
        out[query_type] = {
            key: sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))
            for key in keys
        }
    return out

