"""
AnomalyDetector: identifies unusual gas measurements using
statistical and machine-learning approaches.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AnomalyMethod = Literal["isolation_forest", "lof", "zscore", "iqr", "autoencoder"]


class AnomalyDetector:
    """Detect anomalies in gas measurement time series."""

    def __init__(
        self,
        method: AnomalyMethod = "isolation_forest",
        contamination: float = 0.05,
        threshold: float = 3.0,
    ) -> None:
        self.method = method
        self.contamination = contamination
        self.threshold = threshold
        self._model: object | None = None

    def fit(self, data: pd.DataFrame) -> "AnomalyDetector":
        """Fit the anomaly detection model on training data."""
        numeric = data.select_dtypes(include=[np.number]).fillna(0)

        if self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            self._model = IsolationForest(
                contamination=self.contamination, random_state=42, n_jobs=-1
            )
            self._model.fit(numeric)

        elif self.method == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            self._model = LocalOutlierFactor(
                contamination=self.contamination, novelty=True, n_jobs=-1
            )
            self._model.fit(numeric)

        elif self.method == "autoencoder":
            self._model = self._build_autoencoder(numeric.shape[1])
            self._model.fit(
                numeric, numeric,
                epochs=50, batch_size=32, verbose=0,
                validation_split=0.1,
            )

        logger.info("AnomalyDetector fitted: method=%s", self.method)
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Return a boolean Series: True = anomaly."""
        numeric = data.select_dtypes(include=[np.number]).fillna(0)

        if self.method in ("isolation_forest", "lof"):
            labels = self._model.predict(numeric)
            anomalies = pd.Series(labels == -1, index=data.index, name="anomaly")

        elif self.method == "zscore":
            z = (numeric - numeric.mean()) / numeric.std()
            anomalies = (z.abs() > self.threshold).any(axis=1).rename("anomaly")

        elif self.method == "iqr":
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
            anomalies = ((numeric < lower) | (numeric > upper)).any(axis=1).rename("anomaly")

        elif self.method == "autoencoder":
            reconstructed = self._model.predict(numeric, verbose=0)
            mse = np.mean((numeric.values - reconstructed) ** 2, axis=1)
            cutoff = np.percentile(mse, (1 - self.contamination) * 100)
            anomalies = pd.Series(mse > cutoff, index=data.index, name="anomaly")

        else:
            raise ValueError(f"Unknown anomaly method: {self.method}")

        n_anomalies = int(anomalies.sum())
        logger.info("Detected %d anomalies (%.1f%%)", n_anomalies, 100 * n_anomalies / len(data))
        return anomalies

    def fit_predict(self, data: pd.DataFrame) -> pd.Series:
        return self.fit(data).predict(data)

    @staticmethod
    def _build_autoencoder(n_features: int):
        try:
            import tensorflow as tf
            from tensorflow import keras

            enc_dim = max(2, n_features // 2)
            inputs = keras.Input(shape=(n_features,))
            encoded = keras.layers.Dense(enc_dim, activation="relu")(inputs)
            decoded = keras.layers.Dense(n_features, activation="linear")(encoded)
            model = keras.Model(inputs, decoded)
            model.compile(optimizer="adam", loss="mse")
            return model
        except ImportError as exc:
            raise ImportError("tensorflow is required for the autoencoder method") from exc
