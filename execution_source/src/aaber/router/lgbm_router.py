from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from aaber.common.io import read_json, write_json

IGNORED_COLUMNS = {
    "query_id",
    "unit_id",
    "candidate_sources",
    "query_type",
    "action_name",
    "action_scope",
    "gain",
    "recall_before",
    "recall_after",
    "recall_gain",
    "mrr_before",
    "mrr_after",
    "mrr_gain",
    "ndcg_before",
    "ndcg_after",
    "ndcg_gain",
    "gain_metric",
    "normalized_gain",
    "value_gain",
    "positive_value_gain",
    "net_utility_l001",
    "net_utility_l005",
    "net_utility_l01",
    "metric_before",
    "metric_after",
    "latency_ms",
    "gpu_ms",
    "token_cost",
    "is_positive_unit",
    # Target-side utilities and post-prediction quantities must never be
    # inferred as router features.  Keeping this guard in the shared helper
    # protects legacy callers as well as the binary-routing feasibility script.
    "text_ndcg",
    "visual_ndcg",
    "signed_regret",
    "positive_regret",
    "predicted_regret",
    "upper_risk",
}


class SimpleGainRegressor:
    """Dependency-free gain regressor used as the default router backend."""

    def __init__(self, epochs: int = 80, lr: float = 0.03, l2: float = 1e-4) -> None:
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.columns: list[str] = []
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.weights: dict[str, float] = {}
        self.bias = 0.0

    def fit(self, rows: list[dict[str, Any]], target: str = "gain") -> None:
        self.columns = infer_numeric_columns(rows)
        if not rows or not self.columns:
            self.bias = 0.0
            return
        self._fit_normalizer(rows)
        self.weights = {col: 0.0 for col in self.columns}
        self.bias = sum(float(row.get(target, 0.0) or 0.0) for row in rows) / len(rows)
        for _ in range(self.epochs):
            for row in rows:
                y = float(row.get(target, 0.0) or 0.0)
                pred = self.predict(row)
                err = pred - y
                self.bias -= self.lr * err
                for col in self.columns:
                    x = self._norm(col, row.get(col, 0.0))
                    grad = err * x + self.l2 * self.weights[col]
                    self.weights[col] -= self.lr * grad

    def predict(self, row: dict[str, Any]) -> float:
        score = self.bias
        for col in self.columns:
            score += self.weights.get(col, 0.0) * self._norm(col, row.get(col, 0.0))
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return score

    def predict_many(self, rows: list[dict[str, Any]]) -> list[float]:
        return [self.predict(row) for row in rows]

    def _fit_normalizer(self, rows: list[dict[str, Any]]) -> None:
        for col in self.columns:
            vals = [float(row.get(col, 0.0) or 0.0) for row in rows]
            mean = sum(vals) / max(1, len(vals))
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
            self.means[col] = mean
            self.stds[col] = math.sqrt(var) or 1.0

    def _norm(self, col: str, value: Any) -> float:
        try:
            numeric = float(value or 0.0)
        except (TypeError, ValueError):
            numeric = 0.0
        return (numeric - self.means.get(col, 0.0)) / max(self.stds.get(col, 1.0), 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "simple_linear",
            "columns": self.columns,
            "means": self.means,
            "stds": self.stds,
            "weights": self.weights,
            "bias": self.bias,
            "epochs": self.epochs,
            "lr": self.lr,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimpleGainRegressor":
        model = cls(
            epochs=int(data.get("epochs", 80)),
            lr=float(data.get("lr", 0.03)),
            l2=float(data.get("l2", 1e-4)),
        )
        model.columns = list(data.get("columns", []))
        model.means = {str(k): float(v) for k, v in data.get("means", {}).items()}
        model.stds = {str(k): float(v) for k, v in data.get("stds", {}).items()}
        model.weights = {str(k): float(v) for k, v in data.get("weights", {}).items()}
        model.bias = float(data.get("bias", 0.0))
        return model


class LightGBMRegressorBackend:
    """Real LightGBM backend used when the optional dependency is installed."""

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.model_str: str | None = None
        self._model: Any | None = None

    def fit(self, rows: list[dict[str, Any]], target: str = "gain") -> None:
        import lightgbm as lgb

        self.columns = infer_numeric_columns(rows)
        if not rows or not self.columns:
            return

        x = [[_to_float(row.get(col, 0.0)) for col in self.columns] for row in rows]
        y = [_to_float(row.get(target, 0.0)) for row in rows]
        weights = [8.0 if val > 0 else 1.0 for val in y]

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=120,
            learning_rate=0.06,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.85,
            colsample_bytree=0.85,
            max_bin=63,
            reg_lambda=1.0,
            random_state=13,
            n_jobs=4,
            verbose=-1,
            force_col_wise=True,
        )
        model.fit(x, y, sample_weight=weights)
        self._model = model
        self.model_str = model.booster_.model_to_string()

    def predict(self, row: dict[str, Any]) -> float:
        model = self._ensure_model()
        if model is None or not self.columns:
            return 0.0
        x = [[_to_float(row.get(col, 0.0)) for col in self.columns]]
        # Tiny online batches are dominated by LightGBM's thread-pool overhead.
        # A single prediction thread is deterministic and substantially faster
        # for the router's one-query/three-action inference path.
        return float(model.predict(x, num_threads=1)[0])

    def predict_many(self, rows: list[dict[str, Any]]) -> list[float]:
        model = self._ensure_model()
        if model is None or not self.columns:
            return [0.0 for _ in rows]
        x = [[_to_float(row.get(col, 0.0)) for col in self.columns] for row in rows]
        return [float(value) for value in model.predict(x, num_threads=1)]

    def _ensure_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if not self.model_str:
            return None
        import lightgbm as lgb

        self._model = lgb.Booster(model_str=self.model_str)
        return self._model

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "lightgbm",
            "columns": self.columns,
            "model_str": self.model_str,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LightGBMRegressorBackend":
        backend = cls()
        backend.columns = list(data.get("columns", []))
        backend.model_str = data.get("model_str")
        return backend


class LightGBMActionGainRouter:
    """Action-gain router.

    Uses real LightGBM when installed; otherwise falls back to a dependency-free
    linear model so the pipeline remains runnable.
    """

    name = "lgbm"

    def __init__(self) -> None:
        self.model = _default_backend()

    def fit(self, rows: list[dict[str, Any]], target: str = "gain") -> None:
        self.model.fit(rows, target=target)

    def predict_gain(self, row: dict[str, Any]) -> float:
        return self.model.predict(row)

    def predict_gains(self, rows: list[dict[str, Any]]) -> list[float]:
        predict_many = getattr(self.model, "predict_many", None)
        if callable(predict_many):
            return predict_many(rows)
        return [self.predict_gain(row) for row in rows]

    def save(self, path: str | Path) -> None:
        write_json(path, {"name": self.name, "model": self.model.to_dict()})

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMActionGainRouter":
        data = read_json(path)
        router = cls()
        model_data = data["model"]
        if model_data.get("backend") == "lightgbm":
            router.model = LightGBMRegressorBackend.from_dict(model_data)
        else:
            router.model = SimpleGainRegressor.from_dict(model_data)
        return router


def infer_numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    cols: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in IGNORED_COLUMNS:
                continue
            if isinstance(value, (int, float)):
                cols.add(key)
    return sorted(cols)


def _default_backend():
    try:
        import lightgbm  # noqa: F401

        return LightGBMRegressorBackend()
    except Exception:
        return SimpleGainRegressor()


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
