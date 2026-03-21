"""XGBoost-based forecasting model using lag/rolling features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecastModel, ForecastResult

logger = logging.getLogger(__name__)


class XGBoostModel(BaseForecastModel):
    """
    Gradient-boosted tree forecaster.
    Uses lag and rolling-window features derived from the target series.
    """

    def __init__(
        self,
        horizon: int = 24,
        confidence_interval: float = 0.95,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        lookback: int = 48,
        quantile_alpha: float = 0.05,
    ) -> None:
        super().__init__(horizon, confidence_interval)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.lookback = lookback
        self.quantile_alpha = quantile_alpha
        self._model: Any = None
        self._lower_model: Any = None
        self._upper_model: Any = None
        self._target: str = "value"
        self._last_values: np.ndarray = np.array([])

    def fit(self, data: pd.Series | pd.DataFrame, target: str | None = None, **_: Any) -> "XGBoostModel":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required: pip install xgboost") from exc

        series = self._extract_series(data, target)
        X, y = self._build_features(series)

        common = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=42,
            n_jobs=-1,
        )
        self._model = XGBRegressor(**common, objective="reg:squarederror")
        self._model.fit(X, y)

        self._last_values = series.values[-self.lookback:]
        self._fitted = True
        logger.info("XGBoostModel fitted: %d samples, %d features", len(y), X.shape[1])
        return self

    def predict(self, steps: int | None = None, **_: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        steps = steps or self.horizon
        history = list(self._last_values)
        preds: list[float] = []

        for _ in range(steps):
            features = self._row_features(history)
            p = float(self._model.predict(features)[0])
            preds.append(p)
            history.append(p)

        idx = pd.date_range(start=pd.Timestamp.now().floor("h"), periods=steps, freq="1h")
        predictions = pd.Series(preds, index=idx, name=self._target)

        # Naive confidence interval via prediction std (bootstrap-lite)
        std = np.std(preds)
        z = 1.96
        lower = predictions - z * std
        upper = predictions + z * std

        return ForecastResult(
            target=self._target,
            horizon=steps,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            model_name="XGBoost",
        )

    def _build_features(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        values = series.values
        X, y = [], []
        for i in range(self.lookback, len(values)):
            X.append(self._feature_vector(values[i - self.lookback: i]))
            y.append(values[i])
        return np.array(X), np.array(y)

    def _row_features(self, history: list[float]) -> np.ndarray:
        window = np.array(history[-self.lookback:])
        return self._feature_vector(window).reshape(1, -1)

    @staticmethod
    def _feature_vector(window: np.ndarray) -> np.ndarray:
        lags = window[-1:-25:-1][:24]  # last 24 lags
        return np.concatenate([
            lags,
            [window.mean(), window.std(), window.min(), window.max()],
            [window[-6:].mean(), window[-12:].mean()],
        ])

    def _extract_series(self, data: pd.Series | pd.DataFrame, target: str | None) -> pd.Series:
        if isinstance(data, pd.DataFrame):
            if target is None:
                raise ValueError("Provide `target` column name when passing a DataFrame")
            self._target = target
            return data[target].dropna()
        self._target = str(data.name or "value")
        return data.dropna()
