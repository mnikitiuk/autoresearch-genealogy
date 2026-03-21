"""Tests for data cleaning and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.loader import DataLoader
from src.processing.cleaner import DataCleaner
from src.processing.features import FeatureEngineer


@pytest.fixture
def sample_data():
    return DataLoader.load_sample().head(500)


def test_cleaner_removes_duplicates(sample_data):
    duplicated = pd.concat([sample_data, sample_data.head(10)])
    cleaner = DataCleaner(normalization="none", resampling_freq=None)
    cleaned = cleaner.fit_transform(duplicated)
    assert cleaned.index.is_unique


def test_cleaner_imputes_missing(sample_data):
    data = sample_data.copy()
    data.iloc[5:10, 0] = np.nan
    cleaner = DataCleaner(normalization="none", resampling_freq=None)
    cleaned = cleaner.fit_transform(data)
    assert cleaned.isnull().sum().sum() == 0


def test_cleaner_normalizes_standard(sample_data):
    cleaner = DataCleaner(normalization="standard", resampling_freq=None)
    cleaned = cleaner.fit_transform(sample_data)
    means = cleaned.mean()
    assert (means.abs() < 0.5).all(), "Standard scaled means should be near zero"


def test_feature_engineer_adds_time_features(sample_data):
    cleaner = DataCleaner(normalization="none", resampling_freq=None)
    cleaned = cleaner.fit_transform(sample_data)
    fe = FeatureEngineer(lag_periods=[1], rolling_windows=[6], add_time_features=True)
    features = fe.transform(cleaned)
    assert "hour" in features.columns
    assert "hour_sin" in features.columns
    assert "month_cos" in features.columns


def test_feature_engineer_adds_lag_features(sample_data):
    cleaner = DataCleaner(normalization="none", resampling_freq=None)
    cleaned = cleaner.fit_transform(sample_data)
    fe = FeatureEngineer(lag_periods=[1, 6], rolling_windows=[6], add_time_features=False)
    features = fe.transform(cleaned)
    assert any("_lag1" in c for c in features.columns)
    assert any("_lag6" in c for c in features.columns)


def test_feature_engineer_no_nan_after_dropna(sample_data):
    cleaner = DataCleaner(normalization="none", resampling_freq=None)
    cleaned = cleaner.fit_transform(sample_data)
    fe = FeatureEngineer(lag_periods=[1], rolling_windows=[6])
    features = fe.transform(cleaned)
    assert features.isnull().sum().sum() == 0
