"""
DataCleaner: handles outlier detection, missing-value imputation,
resampling, and normalisation of gas measurement time series.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

OutlierMethod = Literal["iqr", "zscore", "isolation_forest"]
NormMethod = Literal["standard", "minmax", "robust", "none"]


class DataCleaner:
    """Clean and normalise raw gas measurement DataFrames."""

    def __init__(
        self,
        outlier_method: OutlierMethod = "iqr",
        outlier_threshold: float = 3.0,
        interpolation_method: str = "linear",
        resampling_freq: str | None = "1h",
        normalization: NormMethod = "standard",
    ) -> None:
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.interpolation_method = interpolation_method
        self.resampling_freq = resampling_freq
        self.normalization = normalization
        self._scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None
        self._feature_columns: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean, resample, and normalise; fit the scaler in the process."""
        data = self._remove_duplicates(data)
        data = self._remove_outliers(data)
        data = self._impute_missing(data)
        if self.resampling_freq:
            data = self._resample(data)
        data = self._normalize(data, fit=True)
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the already-fitted pipeline to new data."""
        data = self._remove_duplicates(data)
        data = self._remove_outliers(data)
        data = self._impute_missing(data)
        if self.resampling_freq:
            data = self._resample(data)
        data = self._normalize(data, fit=False)
        return data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reverse normalisation to restore original scale."""
        if self._scaler is None:
            return data
        numeric = data[self._feature_columns]
        original = self._scaler.inverse_transform(numeric)
        result = data.copy()
        result[self._feature_columns] = original
        return result

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        n_before = len(data)
        data = data[~data.index.duplicated(keep="first")]
        removed = n_before - len(data)
        if removed:
            logger.info("Removed %d duplicate rows", removed)
        return data

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        numeric = data.select_dtypes(include=[np.number])
        mask = pd.Series(True, index=data.index)

        if self.outlier_method == "iqr":
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - self.outlier_threshold * iqr, q3 + self.outlier_threshold * iqr
            mask = ((numeric >= lower) & (numeric <= upper)).all(axis=1)

        elif self.outlier_method == "zscore":
            z = (numeric - numeric.mean()) / numeric.std()
            mask = (z.abs() <= self.outlier_threshold).all(axis=1)

        elif self.outlier_method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.05, random_state=42)
            preds = clf.fit_predict(numeric.fillna(numeric.median()))
            mask = pd.Series(preds == 1, index=data.index)

        n_removed = (~mask).sum()
        if n_removed:
            logger.info("Outlier removal (%s): %d rows flagged", self.outlier_method, n_removed)
        return data[mask]

    def _impute_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        missing_before = data.isnull().sum().sum()
        if missing_before == 0:
            return data

        if self.interpolation_method == "forward_fill":
            data = data.ffill().bfill()
        else:
            data = data.interpolate(method=self.interpolation_method).ffill().bfill()

        logger.info("Imputed %d missing value(s) via '%s'", missing_before, self.interpolation_method)
        return data

    def _resample(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.resample(self.resampling_freq).mean()
        logger.info("Resampled to '%s': %d rows", self.resampling_freq, len(data))
        return data

    def _normalize(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if self.normalization == "none":
            return data

        self._feature_columns = list(data.select_dtypes(include=[np.number]).columns)
        numeric = data[self._feature_columns]

        if fit:
            scaler_cls: type = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
                "robust": RobustScaler,
            }.get(self.normalization, StandardScaler)
            self._scaler = scaler_cls()
            scaled = self._scaler.fit_transform(numeric)
        else:
            if self._scaler is None:
                raise RuntimeError("Scaler not fitted. Call fit_transform first.")
            scaled = self._scaler.transform(numeric)

        result = data.copy()
        result[self._feature_columns] = scaled
        return result
