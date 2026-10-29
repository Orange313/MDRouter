from __future__ import annotations

from aaber.common.text import has_any, tokenize
from aaber.data.schema import Query

TABLE_TERMS = ["table", "row", "column", "cell", "tabular", "spreadsheet", "表"]
FIGURE_TERMS = ["figure", "fig", "image", "diagram", "picture", "photo", "shown", "logo", "brand", "图"]
CHART_TERMS = ["chart", "plot", "trend", "bar", "line", "axis", "x axis", "y axis", "graph"]
LAYOUT_TERMS = ["where", "position", "layout", "top", "bottom", "left", "right"]
COUNT_TERMS = ["how many", "number of", "count", "total", "percentage", "percent", "%"]
FIELD_TERMS = ["name", "date", "title", "order no", "log-in", "no.", "field", "subheading"]
VISUAL_VERB_TERMS = ["shown", "pictured", "advertised", "plotted", "displayed", "drawn"]


def build_query_features(query: Query) -> dict[str, float | str]:
    tokens = tokenize(query.question)
    has_table = float(has_any(query.question, TABLE_TERMS))
    has_figure = float(has_any(query.question, FIGURE_TERMS))
    has_chart = float(has_any(query.question, CHART_TERMS))
    has_layout = float(has_any(query.question, LAYOUT_TERMS))
    has_count = float(has_any(query.question, COUNT_TERMS))
    has_field = float(has_any(query.question, FIELD_TERMS))
    has_visual_verb = float(has_any(query.question, VISUAL_VERB_TERMS))
    visual_need = min(1.0, 0.35 * has_figure + 0.35 * has_chart + 0.15 * has_table + 0.15 * has_visual_verb)
    return {
        "query_length": float(len(tokens)),
        "has_table_keyword": has_table,
        "has_figure_keyword": has_figure,
        "has_chart_keyword": has_chart,
        "has_layout_keyword": has_layout,
        "has_count_keyword": has_count,
        "has_field_keyword": has_field,
        "has_visual_verb_keyword": has_visual_verb,
        "query_visual_need_score": visual_need,
        "query_textual_need_score": max(0.0, 1.0 - visual_need + 0.2 * has_field),
        "query_starts_with_what": float((query.question or "").strip().lower().startswith("what")),
        "query_starts_with_which": float((query.question or "").strip().lower().startswith("which")),
        "query_starts_with_how_many": float((query.question or "").strip().lower().startswith("how many")),
        "query_type": query.query_type or infer_query_type(query.question),
    }


def infer_query_type(question: str) -> str:
    if has_any(question, TABLE_TERMS):
        return "table-centric"
    if has_any(question, CHART_TERMS):
        return "chart-centric"
    if has_any(question, FIGURE_TERMS):
        return "image-centric"
    if has_any(question, LAYOUT_TERMS):
        return "layout-centric"
    return "text-centric"
