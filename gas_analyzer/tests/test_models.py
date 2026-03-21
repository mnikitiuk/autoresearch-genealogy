"""Tests for analysis models: GasAnalyzer, AnomalyDetector, CorrelationAnalyzer."""

import pandas as pd
import pytest

from src.ingestion.loader import DataLoader
from src.models.analyzer import GasAnalyzer
from src.models.anomaly import AnomalyDetector
from src.models.correlation import CorrelationAnalyzer


@pytest.fixture
def clean_data():
    return DataLoader.load_sample().head(300)


def test_gas_analyzer_descriptive_stats(clean_data):
    analyzer = GasAnalyzer(decomposition_period=24)
    report = analyzer.analyze(clean_data, columns=["CH4", "N2"])
    assert "CH4" in report.descriptive_stats.index
    assert "mean" in report.descriptive_stats.columns


def test_gas_analyzer_decomposition(clean_data):
    analyzer = GasAnalyzer(decomposition_period=24)
    report = analyzer.analyze(clean_data, columns=["CH4"])
    assert "CH4" in report.decompositions
    decomp = report.decompositions["CH4"]
    assert len(decomp.trend) > 0
    assert len(decomp.seasonal) > 0


def test_gas_analyzer_stationarity(clean_data):
    analyzer = GasAnalyzer(decomposition_period=24)
    report = analyzer.analyze(clean_data, columns=["CH4"])
    assert "CH4" in report.stationarity
    stats = report.stationarity["CH4"]
    assert "p_value" in stats
    assert "is_stationary" in stats


def test_anomaly_detector_isolation_forest(clean_data):
    detector = AnomalyDetector(method="isolation_forest", contamination=0.05)
    anomalies = detector.fit_predict(clean_data)
    assert len(anomalies) == len(clean_data)
    assert anomalies.dtype == bool


def test_anomaly_detector_zscore(clean_data):
    detector = AnomalyDetector(method="zscore", threshold=3.0)
    anomalies = detector.fit_predict(clean_data)
    assert anomalies.sum() < len(clean_data)


def test_correlation_matrix_shape(clean_data):
    analyzer = CorrelationAnalyzer(method="pearson")
    matrix = analyzer.correlation_matrix(clean_data)
    n = clean_data.select_dtypes("number").shape[1]
    assert matrix.shape == (n, n)


def test_correlation_top(clean_data):
    analyzer = CorrelationAnalyzer()
    top = analyzer.top_correlations(clean_data, target="CH4", n=5)
    assert len(top) <= 5
    assert all(top.values <= 1.0)
