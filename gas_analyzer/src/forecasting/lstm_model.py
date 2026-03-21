"""LSTM-based deep learning forecasting model for gas time series."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecastModel, ForecastResult

logger = logging.getLogger(__name__)


class LSTMModel(BaseForecastModel):
    """
    LSTM recurrent neural network for multi-step gas forecasting.
    Requires PyTorch (default) or can be adapted to TensorFlow.
    """

    def __init__(
        self,
        horizon: int = 24,
        confidence_interval: float = 0.95,
        lookback: int = 48,
        hidden_units: list[int] | None = None,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 10,
    ) -> None:
        super().__init__(horizon, confidence_interval)
        self.lookback = lookback
        self.hidden_units = hidden_units or [64, 32]
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self._model: Any = None
        self._target: str = "value"
        self._mean: float = 0.0
        self._std: float = 1.0
        self._last_seq: np.ndarray = np.array([])

    def fit(self, data: pd.Series | pd.DataFrame, target: str | None = None, **_: Any) -> "LSTMModel":
        series = self._extract_series(data, target)
        values = series.values.astype(float)

        # Normalise
        self._mean = float(values.mean())
        self._std = float(values.std()) or 1.0
        normed = (values - self._mean) / self._std

        X, y = self._build_sequences(normed)
        self._model = self._build_and_train(X, y)
        self._last_seq = normed[-self.lookback:]
        self._fitted = True
        logger.info("LSTMModel fitted: %d sequences, lookback=%d", len(y), self.lookback)
        return self

    def predict(self, steps: int | None = None, **_: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        steps = steps or self.horizon
        history = list(self._last_seq)
        preds: list[float] = []

        import torch
        self._model.eval()
        with torch.no_grad():
            for _ in range(steps):
                seq = torch.tensor(history[-self.lookback:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                p = float(self._model(seq).item())
                preds.append(p)
                history.append(p)

        preds_orig = np.array(preds) * self._std + self._mean
        idx = pd.date_range(start=pd.Timestamp.now().floor("h"), periods=steps, freq="1h")
        predictions = pd.Series(preds_orig, index=idx, name=self._target)

        std = float(np.std(preds_orig))
        z = 1.96
        return ForecastResult(
            target=self._target,
            horizon=steps,
            predictions=predictions,
            lower_bound=predictions - z * std,
            upper_bound=predictions + z * std,
            model_name="LSTM",
        )

    def _build_sequences(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.lookback, len(values)):
            X.append(values[i - self.lookback: i])
            y.append(values[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _build_and_train(self, X: np.ndarray, y: np.ndarray) -> Any:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("torch is required: pip install torch") from exc

        class LSTMNet(nn.Module):
            def __init__(self, hidden: list[int], dropout: float) -> None:
                super().__init__()
                self.lstm1 = nn.LSTM(1, hidden[0], batch_first=True)
                self.drop1 = nn.Dropout(dropout)
                self.lstm2 = nn.LSTM(hidden[0], hidden[1], batch_first=True) if len(hidden) > 1 else None
                self.drop2 = nn.Dropout(dropout) if len(hidden) > 1 else None
                self.fc = nn.Linear(hidden[-1], 1)

            def forward(self, x: Any) -> Any:
                out, _ = self.lstm1(x)
                out = self.drop1(out[:, -1, :].unsqueeze(1))
                if self.lstm2:
                    out, _ = self.lstm2(out)
                    out = self.drop2(out[:, -1, :])
                else:
                    out = out.squeeze(1)
                return self.fc(out).squeeze(-1)

        model = LSTMNet(self.hidden_units, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X).unsqueeze(-1)
        y_t = torch.tensor(y)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        best_loss, patience_count = float("inf"), 0
        best_state = None

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info("Early stopping at epoch %d (loss=%.6f)", epoch + 1, best_loss)
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model

    def _extract_series(self, data: pd.Series | pd.DataFrame, target: str | None) -> pd.Series:
        if isinstance(data, pd.DataFrame):
            if target is None:
                raise ValueError("Provide `target` column name when passing a DataFrame")
            self._target = target
            return data[target].dropna()
        self._target = str(data.name or "value")
        return data.dropna()
