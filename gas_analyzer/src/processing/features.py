"""
FeatureEngineer: derives time, lag, rolling, and gas-quality features
from cleaned measurement DataFrames.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate derived features for gas analysis and forecasting models."""

    def __init__(
        self,
        lag_periods: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        add_time_features: bool = True,
        add_quality_features: bool = True,
    ) -> None:
        self.lag_periods = lag_periods or [1, 3, 6, 12, 24]
        self.rolling_windows = rolling_windows or [6, 12, 24, 48]
        self.add_time_features = add_time_features
        self.add_quality_features = add_quality_features

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all configured feature groups to the DataFrame."""
        df = data.copy()

        if self.add_time_features:
            df = self._add_time_features(df)

        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)

        if self.add_quality_features:
            df = self._add_quality_features(df)

        initial_nulls = df.isnull().sum().sum()
        df = df.dropna()
        if initial_nulls:
            logger.info("Dropped %d rows with NaN after feature generation", initial_nulls)

        logger.info("Feature engineering complete: %d columns, %d rows", df.shape[1], df.shape[0])
        return df

    # ------------------------------------------------------------------

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        df["hour"] = idx.hour
        df["day_of_week"] = idx.dayofweek
        df["day_of_month"] = idx.day
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

        # Cyclical encoding to preserve periodicity
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        base_cols = [c for c in numeric_cols if not any(c.endswith(s) for s in ("_sin", "_cos"))]
        for col in base_cols:
            for lag in self.lag_periods:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        base_cols = [
            c for c in numeric_cols
            if not any(c.endswith(s) for s in ("_sin", "_cos")) and "_lag" not in c
        ]
        for col in base_cols:
            for window in self.rolling_windows:
                df[f"{col}_roll_mean_{window}h"] = df[col].rolling(window).mean()
                df[f"{col}_roll_std_{window}h"] = df[col].rolling(window).std()
        return df

    def _add_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derived gas quality indicators if the required columns exist."""

        # Higher heating value approximation (MJ/m3) from dry-basis composition
        if all(c in df.columns for c in ("CH4", "C2H6")):
            df["HHV_approx"] = df["CH4"] * 0.3726 + df["C2H6"] * 0.6636

        # Wobbe Index approximation (HHV / sqrt(SG))
        # Specific gravity approximation from N2 and CO2 dilution
        if all(c in df.columns for c in ("CH4", "N2", "CO2")):
            df["SG_approx"] = 1.0 - 0.004 * df["CH4"] + 0.02 * df["N2"] + 0.015 * df["CO2"]
            if "HHV_approx" in df.columns:
                df["Wobbe_approx"] = df["HHV_approx"] / np.sqrt(df["SG_approx"].clip(lower=0.01))

        # Hydrocarbon dew-point proxy
        if "C2H6" in df.columns and "C3H8" in df.columns:
            df["HC_dewpoint_proxy"] = df["C2H6"] + 2.0 * df["C3H8"]

        return df
