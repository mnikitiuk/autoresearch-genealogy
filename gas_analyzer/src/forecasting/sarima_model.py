"""SARIMA-based forecasting model for stationary / seasonal gas time series."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .base import BaseForecastModel, ForecastResult

logger = logging.getLogger(__name__)


class SARIMAModel(BaseForecastModel):
    """
    Seasonal ARIMA (SARIMA) forecasting.
    Use auto_order=True for automatic parameter selection (requires pmdarima).
    """

    def __init__(
        self,
        horizon: int = 24,
        confidence_interval: float = 0.95,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
        auto_order: bool = False,
    ) -> None:
        super().__init__(horizon, confidence_interval)
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self._model_fit: Any = None
        self._target: str = "value"

    def fit(self, data: pd.Series | pd.DataFrame, target: str | None = None, **_: Any) -> "SARIMAModel":
        series = self._extract_series(data, target)

        if self.auto_order:
            try:
                import pmdarima as pm
                auto = pm.auto_arima(
                    series,
                    seasonal=True,
                    m=self.seasonal_order[3],
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                )
                self._model_fit = auto
                logger.info(
                    "Auto ARIMA order=%s seasonal_order=%s",
                    auto.order,
                    auto.seasonal_order,
                )
            except ImportError as exc:
                raise ImportError("pmdarima is required for auto_order=True") from exc
        else:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._model_fit = model.fit(disp=False)
            logger.info("SARIMA fitted: order=%s seasonal_order=%s", self.order, self.seasonal_order)

        self._fitted = True
        return self

    def predict(self, steps: int | None = None, **_: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        steps = steps or self.horizon
        alpha = 1 - self.confidence_interval

        if self.auto_order:
            preds, conf = self._model_fit.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
            idx = pd.date_range(
                self._model_fit.arima_res_.fittedvalues.index[-1],
                periods=steps + 1,
                freq=pd.infer_freq(self._model_fit.arima_res_.fittedvalues.index) or "1h",
            )[1:]
            predictions = pd.Series(preds, index=idx, name=self._target)
            lower = pd.Series(conf[:, 0], index=idx)
            upper = pd.Series(conf[:, 1], index=idx)
        else:
            forecast = self._model_fit.get_forecast(steps=steps)
            predictions = forecast.predicted_mean
            ci = forecast.conf_int(alpha=alpha)
            lower = ci.iloc[:, 0]
            upper = ci.iloc[:, 1]
            predictions.name = self._target

        return ForecastResult(
            target=self._target,
            horizon=steps,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            model_name="SARIMA",
        )

    def _extract_series(self, data: pd.Series | pd.DataFrame, target: str | None) -> pd.Series:
        if isinstance(data, pd.DataFrame):
            if target is None:
                raise ValueError("Provide `target` column name when passing a DataFrame")
            self._target = target
            return data[target].dropna()
        self._target = str(data.name or "value")
        return data.dropna()
