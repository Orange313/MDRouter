from __future__ import annotations

from aaber.data.schema import EvidenceUnit


def build_quality_features(unit: EvidenceUnit) -> dict[str, float]:
    quality = unit.quality or {}
    has_text = bool(unit.text or unit.context_text or unit.caption)
    has_image = bool(unit.image_path)
    completeness = (float(has_text) + float(has_image)) / 2.0
    return {
        "ocr_confidence": float(quality.get("ocr_confidence", 0.0) or 0.0),
        "image_resolution": float(quality.get("image_resolution", 0.0) or 0.0),
        "blur_score": float(quality.get("blur_score", 0.0) or 0.0),
        "modality_completeness": completeness,
        "duplicate_score": float(quality.get("duplicate_score", 0.0) or 0.0),
    }

