"""
Base classes for all forecasting models.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Stores predictions, confidence intervals, and evaluation metrics."""

    target: str
    horizon: int
    predictions: pd.Series
    lower_bound: pd.Series | None = None
    upper_bound: pd.Series | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    model_name: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"forecast": self.predictions})
        if self.lower_bound is not None:
            df["lower"] = self.lower_bound
        if self.upper_bound is not None:
            df["upper"] = self.upper_bound
        return df

    def summary(self) -> str:
        lines = [
            f"Model   : {self.model_name}",
            f"Target  : {self.target}",
            f"Horizon : {self.horizon} steps",
        ]
        for k, v in self.metrics.items():
            lines.append(f"{k:8s}: {v:.4f}")
        return "\n".join(lines)


class BaseForecastModel(ABC):
    """Abstract base class for all gas forecasting models."""

    def __init__(self, horizon: int = 24, confidence_interval: float = 0.95) -> None:
        self.horizon = horizon
        self.confidence_interval = confidence_interval
        self._fitted = False

    @abstractmethod
    def fit(self, data: pd.Series | pd.DataFrame, **kwargs: Any) -> "BaseForecastModel":
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, steps: int | None = None, **kwargs: Any) -> ForecastResult:
        """Generate forecasts for the next `steps` time steps."""

    def evaluate(self, actuals: pd.Series, predictions: pd.Series) -> dict[str, float]:
        """Compute standard regression metrics."""
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = float(np.mean(np.abs((actuals - predictions) / np.clip(actuals, 1e-8, None)))) * 100
        r2 = r2_score(actuals, predictions)
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    def cross_validate(
        self,
        data: pd.Series,
        n_splits: int = 5,
        test_size: int | None = None,
    ) -> pd.DataFrame:
        """Time-series walk-forward cross-validation."""
        test_size = test_size or self.horizon
        n = len(data)
        min_train = n - n_splits * test_size

        if min_train < 2 * test_size:
            raise ValueError("Not enough data for the requested cross-validation splits")

        records = []
        for fold in range(n_splits):
            split = min_train + fold * test_size
            train = data.iloc[:split]
            test = data.iloc[split: split + test_size]

            self.fit(train)
            result = self.predict(steps=len(test))
            metrics = self.evaluate(test, result.predictions.values[: len(test)])
            metrics["fold"] = fold + 1
            records.append(metrics)
            logger.info("CV fold %d/%d: RMSE=%.4f", fold + 1, n_splits, metrics["RMSE"])

        return pd.DataFrame(records).set_index("fold")
