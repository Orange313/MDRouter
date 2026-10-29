from __future__ import annotations

from aaber.data.schema import EvidenceUnit

LAYOUT_IDS = {
    "unknown": 0,
    "page": 1,
    "text": 2,
    "figure": 3,
    "table": 4,
    "chart": 5,
    "caption": 6,
}


def build_layout_features(unit: EvidenceUnit) -> dict[str, float]:
    bbox = unit.bbox or (0.0, 0.0, 0.0, 0.0)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    return {
        "page_id_norm": unit.page_id / 1000.0,
        "bbox_area": area,
        "bbox_area_ratio": min(1.0, area / max(1.0, 1000.0 * 1400.0)),
        "bbox_y_norm": y1 / 1400.0 if y1 else 0.0,
        "layout_type_id": float(LAYOUT_IDS.get(unit.layout_type, LAYOUT_IDS.get(unit.unit_type, 0))),
    }

