"""
EnsembleForecaster: combines multiple models via averaging or stacking.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecastModel, ForecastResult

logger = logging.getLogger(__name__)


class EnsembleForecaster(BaseForecastModel):
    """
    Ensemble of several forecasting models.
    Supports simple averaging and weighted averaging.
    Weights default to equal; pass custom weights to favour better models.
    """

    def __init__(
        self,
        models: list[BaseForecastModel],
        weights: list[float] | None = None,
        horizon: int = 24,
        confidence_interval: float = 0.95,
    ) -> None:
        super().__init__(horizon, confidence_interval)
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("weights length must match number of models")
        if abs(sum(self.weights) - 1.0) > 1e-6:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def fit(self, data: pd.Series | pd.DataFrame, **kwargs: Any) -> "EnsembleForecaster":
        for model in self.models:
            model.fit(data, **kwargs)
        self._fitted = True
        logger.info("EnsembleForecaster fitted: %d models", len(self.models))
        return self

    def predict(self, steps: int | None = None, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        steps = steps or self.horizon
        all_preds: list[np.ndarray] = []

        for model in self.models:
            result = model.predict(steps=steps, **kwargs)
            all_preds.append(result.predictions.values)

        weighted = np.average(np.stack(all_preds, axis=0), axis=0, weights=self.weights)
        std = np.std(np.stack(all_preds, axis=0), axis=0)

        idx = self.models[0].predict(steps=steps).predictions.index
        predictions = pd.Series(weighted, index=idx[:steps], name="ensemble")
        z = 1.96
        lower = predictions - z * std
        upper = predictions + z * std

        return ForecastResult(
            target="ensemble",
            horizon=steps,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            model_name=f"Ensemble({', '.join(type(m).__name__ for m in self.models)})",
        )
