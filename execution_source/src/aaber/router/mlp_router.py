from __future__ import annotations

from pathlib import Path
from typing import Any

from aaber.router.lgbm_router import LightGBMActionGainRouter


class MLPActionGainRouter(LightGBMActionGainRouter):
    """MLP-compatible router placeholder.

    The interface is intentionally identical to the LightGBM router. A torch
    implementation can replace the inherited fallback after baseline results
    are stable.
    """

    name = "mlp"

    @classmethod
    def load(cls, path: str | Path) -> "MLPActionGainRouter":
        base = LightGBMActionGainRouter.load(path)
        router = cls()
        router.model = base.model
        return router

