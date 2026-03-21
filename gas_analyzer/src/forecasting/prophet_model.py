"""Prophet-based forecasting model for gas time series."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .base import BaseForecastModel, ForecastResult

logger = logging.getLogger(__name__)


class ProphetModel(BaseForecastModel):
    """
    Wraps Facebook Prophet for gas measurement forecasting.
    Best suited for data with strong seasonality (daily, weekly, yearly).
    """

    def __init__(
        self,
        horizon: int = 24,
        confidence_interval: float = 0.95,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
        freq: str = "1h",
    ) -> None:
        super().__init__(horizon, confidence_interval)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.freq = freq
        self._model: Any = None
        self._target: str = "y"

    def fit(self, data: pd.Series | pd.DataFrame, target: str | None = None, **_: Any) -> "ProphetModel":
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError("prophet is required: pip install prophet") from exc

        if isinstance(data, pd.DataFrame):
            if target is None:
                raise ValueError("Provide `target` column name when passing a DataFrame")
            series = data[target]
            self._target = target
        else:
            series = data
            self._target = str(series.name or "y")

        prophet_df = pd.DataFrame({"ds": series.index, "y": series.values})

        self._model = Prophet(
            interval_width=self.confidence_interval,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        self._model.fit(prophet_df)
        self._fitted = True
        logger.info("ProphetModel fitted on %d observations", len(series))
        return self

    def predict(self, steps: int | None = None, **_: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        steps = steps or self.horizon
        future = self._model.make_future_dataframe(periods=steps, freq=self.freq)
        forecast_df = self._model.predict(future)
        future_rows = forecast_df.tail(steps)

        idx = pd.DatetimeIndex(future_rows["ds"])
        predictions = pd.Series(future_rows["yhat"].values, index=idx, name=self._target)
        lower = pd.Series(future_rows["yhat_lower"].values, index=idx)
        upper = pd.Series(future_rows["yhat_upper"].values, index=idx)

        return ForecastResult(
            target=self._target,
            horizon=steps,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            model_name="Prophet",
        )
