"""Tests for forecasting models."""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.loader import DataLoader
from src.forecasting.base import ForecastResult
from src.forecasting.xgboost_model import XGBoostModel
from src.forecasting.ensemble import EnsembleForecaster


@pytest.fixture
def ch4_series():
    data = DataLoader.load_sample()
    return data["CH4"].head(500)


def test_xgboost_fit_predict(ch4_series):
    model = XGBoostModel(horizon=12, lookback=24)
    model.fit(ch4_series[:-12])
    result = model.predict(steps=12)

    assert isinstance(result, ForecastResult)
    assert len(result.predictions) == 12
    assert result.lower_bound is not None
    assert result.upper_bound is not None


def test_xgboost_metrics(ch4_series):
    model = XGBoostModel(horizon=12, lookback=24)
    train = ch4_series[:-12]
    test = ch4_series[-12:]
    model.fit(train)
    result = model.predict(steps=12)
    metrics = model.evaluate(test.values, result.predictions.values)

    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert metrics["RMSE"] >= 0


def test_ensemble_forecaster(ch4_series):
    m1 = XGBoostModel(horizon=6, lookback=24, n_estimators=50)
    m2 = XGBoostModel(horizon=6, lookback=24, n_estimators=50)
    ensemble = EnsembleForecaster([m1, m2], weights=[0.6, 0.4], horizon=6)
    ensemble.fit(ch4_series[:-6])
    result = ensemble.predict(steps=6)

    assert len(result.predictions) == 6
    assert "Ensemble" in result.model_name


def test_forecast_result_to_dataframe(ch4_series):
    model = XGBoostModel(horizon=6, lookback=24, n_estimators=50)
    model.fit(ch4_series[:-6])
    result = model.predict(steps=6)
    df = result.to_dataframe()

    assert "forecast" in df.columns
    assert len(df) == 6
