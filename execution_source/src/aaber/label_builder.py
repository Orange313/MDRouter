from __future__ import annotations

ACTION_IDS = {
    "skip": 0,
    "text-search": 1,
    "image-search": 2,
    "relation-expand": 3,
    "multivector-rerank": 4,
    "vlm-verify": 5,
    "cross-rerank": 6,
    "multivector-search": 7,
    "dense-search": 8,
}

ACTION_COSTS = {
    "skip": 0.0,
    "text-search": 1.0,
    "image-search": 3.0,
    # Frozen from the five-action server benchmark: 0.021 / 14.363 ~= 0.0015,
    # rounded to 0.001 because one relation expansion uses one coarse seed.
    "relation-expand": 0.001,
    "multivector-rerank": 8.0,
    # Keep the prior main-protocol visual budget for comparability; measured ratio is reported separately.
    "multivector-search": 8.0,
    "vlm-verify": 20.0,
    # Frozen from the five-action server benchmark: 80.083 / 14.363 ~= 5.576.
    "cross-rerank": 5.6,
    # Keep the prior main-protocol budget axis for comparability; measured ratio is reported separately.
    "dense-search": 0.65,
}


def add_action_features(
    row: dict,
    action_name: str,
    cost: float | None = None,
    action_costs: dict[str, float] | None = None,
) -> dict:
    enriched = dict(row)
    enriched["action_name"] = action_name
    enriched["action_id"] = float(ACTION_IDS.get(action_name, -1))
    resolved_cost = ACTION_COSTS.get(action_name, 1.0) if cost is None else cost
    if cost is None and action_costs is not None and action_name in action_costs:
        resolved_cost = action_costs[action_name]
    enriched["action_cost"] = float(resolved_cost)
    for name, action_id in ACTION_IDS.items():
        enriched[f"action_is_{name.replace('-', '_')}"] = float(action_id == ACTION_IDS.get(action_name))
    enriched.update(_action_interaction_features(enriched, action_name))
    return enriched


def _action_interaction_features(row: dict, action_name: str) -> dict[str, float]:
    visual_need = float(row.get("query_visual_need_score", 0.0) or 0.0)
    textual_need = float(row.get("query_textual_need_score", 0.0) or 0.0)
    image_gap = float(row.get("image_vs_base_top_score_gap", 0.0) or 0.0)
    image_overlap = float(row.get("image_base_overlap_at10", 0.0) or 0.0)
    image_has_candidates = float(row.get("image_has_candidates", 0.0) or 0.0)
    is_image = float(action_name == "image-search")
    is_text = float(action_name == "text-search")
    is_dense = float(action_name == "dense-search")
    is_cross = float(action_name == "cross-rerank")
    is_multivector = float(action_name in {"multivector-rerank", "multivector-search"})
    return {
        "action_visual_need": max(is_image, is_multivector) * visual_need,
        "action_textual_need": max(is_text, is_cross) * textual_need,
        "action_dense_textual_need": is_dense * textual_need,
        "action_image_score_gap": is_image * image_gap,
        "action_image_low_overlap": is_image * max(0.0, 1.0 - image_overlap),
        "action_image_has_candidates": is_image * image_has_candidates,
        "action_cross_textual_need": is_cross * textual_need,
        "action_cross_base_uncertainty": is_cross * float(row.get("score_entropy", 0.0) or 0.0),
        "action_multivector_visual_need": is_multivector * visual_need,
        "action_multivector_base_uncertainty": is_multivector * float(row.get("score_entropy", 0.0) or 0.0),
    }
