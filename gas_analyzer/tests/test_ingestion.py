"""Tests for data ingestion modules."""

import pandas as pd
import pytest

from src.ingestion.loader import DataLoader


def test_load_sample_returns_dataframe():
    data = DataLoader.load_sample()
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert isinstance(data.index, pd.DatetimeIndex)


def test_load_sample_has_expected_columns():
    data = DataLoader.load_sample()
    expected = {"CH4", "C2H6", "N2", "CO2", "H2S_ppm", "pressure_bar", "temperature_C", "flow_rate_m3h"}
    assert expected.issubset(set(data.columns))


def test_load_sample_value_ranges():
    data = DataLoader.load_sample()
    assert data["CH4"].between(70, 100).all()
    assert data["N2"].ge(0).all()
    assert data["flow_rate_m3h"].ge(0).all()
