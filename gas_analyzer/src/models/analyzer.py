"""
GasAnalyzer: time-series decomposition and statistical analysis
of gas composition measurements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DecompMethod = Literal["additive", "multiplicative"]


@dataclass
class DecompositionResult:
    component: str
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    period: int
    method: str


@dataclass
class AnalysisReport:
    descriptive_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    decompositions: dict[str, DecompositionResult] = field(default_factory=dict)
    stationarity: dict[str, dict] = field(default_factory=dict)
    autocorrelations: dict[str, pd.Series] = field(default_factory=dict)


class GasAnalyzer:
    """Statistical analysis of gas measurement time series."""

    def __init__(
        self,
        decomposition_period: int = 24,
        decomposition_method: DecompMethod = "additive",
        acf_lags: int = 48,
    ) -> None:
        self.decomposition_period = decomposition_period
        self.decomposition_method = decomposition_method
        self.acf_lags = acf_lags

    def analyze(self, data: pd.DataFrame, columns: list[str] | None = None) -> AnalysisReport:
        """Run full statistical analysis on selected columns."""
        cols = columns or list(data.select_dtypes(include=[np.number]).columns)
        report = AnalysisReport()

        report.descriptive_stats = self._descriptive_stats(data[cols])

        for col in cols:
            series = data[col].dropna()
            if len(series) < 2 * self.decomposition_period:
                logger.warning("Skipping decomposition for '%s': too few data points", col)
                continue

            report.decompositions[col] = self._decompose(series, col)
            report.stationarity[col] = self._adf_test(series)
            report.autocorrelations[col] = self._acf(series)

        return report

    # ------------------------------------------------------------------

    def _descriptive_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        stats = data.describe().T
        stats["skewness"] = data.skew()
        stats["kurtosis"] = data.kurtosis()
        stats["missing_pct"] = data.isnull().mean() * 100
        return stats

    def _decompose(self, series: pd.Series, name: str) -> DecompositionResult:
        from statsmodels.tsa.seasonal import seasonal_decompose

        result = seasonal_decompose(
            series,
            model=self.decomposition_method,
            period=self.decomposition_period,
            extrapolate_trend="freq",
        )
        logger.info("Decomposed '%s' (%s, period=%d)", name, self.decomposition_method, self.decomposition_period)
        return DecompositionResult(
            component=name,
            trend=result.trend,
            seasonal=result.seasonal,
            residual=result.resid,
            period=self.decomposition_period,
            method=self.decomposition_method,
        )

    def _adf_test(self, series: pd.Series) -> dict:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "n_lags": result[2],
            "n_obs": result[3],
            "is_stationary": result[1] < 0.05,
        }

    def _acf(self, series: pd.Series) -> pd.Series:
        from statsmodels.tsa.stattools import acf

        values = acf(series.dropna(), nlags=self.acf_lags, fft=True)
        return pd.Series(values, name=str(series.name))
