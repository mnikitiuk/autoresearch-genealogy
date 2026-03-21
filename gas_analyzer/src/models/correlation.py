"""
CorrelationAnalyzer: computes pairwise and cross-correlations
between gas components and operational parameters.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CorrMethod = Literal["pearson", "spearman", "kendall"]


class CorrelationAnalyzer:
    """Analyse linear and rank correlations between gas measurement channels."""

    def __init__(self, method: CorrMethod = "pearson") -> None:
        self.method = method

    def correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the pairwise correlation matrix."""
        numeric = data.select_dtypes(include=[np.number])
        matrix = numeric.corr(method=self.method)
        logger.info("Computed %s correlation matrix (%dx%d)", self.method, *matrix.shape)
        return matrix

    def top_correlations(
        self,
        data: pd.DataFrame,
        target: str,
        n: int = 10,
        threshold: float = 0.1,
    ) -> pd.Series:
        """Return the n features most correlated with a target column."""
        if target not in data.columns:
            raise KeyError(f"Target column '{target}' not found")
        matrix = self.correlation_matrix(data)
        corr = matrix[target].drop(labels=[target])
        corr = corr[corr.abs() >= threshold]
        return corr.abs().sort_values(ascending=False).head(n)

    def cross_correlation(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        max_lag: int = 48,
    ) -> pd.Series:
        """Compute cross-correlation between two series across ±max_lag lags."""
        a = (series_a - series_a.mean()) / series_a.std()
        b = (series_b - series_b.mean()) / series_b.std()
        lags = range(-max_lag, max_lag + 1)
        values = [a.corr(b.shift(lag)) for lag in lags]
        return pd.Series(values, index=list(lags), name=f"xcorr({series_a.name},{series_b.name})")

    def rolling_correlation(
        self,
        data: pd.DataFrame,
        col_a: str,
        col_b: str,
        window: int = 24,
    ) -> pd.Series:
        """Compute rolling pairwise correlation between two columns."""
        return data[col_a].rolling(window).corr(data[col_b])
