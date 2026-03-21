"""
DataLoader: unified entry point for gas measurement data ingestion.
Selects the appropriate connector based on project configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .connectors import BaseConnector, CSVConnector, DatabaseConnector

logger = logging.getLogger(__name__)


class DataLoader:
    """High-level data loader that reads configuration and routes to connectors."""

    def __init__(self, config_path: str | Path = "config/settings.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._connector: BaseConnector | None = None

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _build_connector(self) -> BaseConnector:
        sources = self.config.get("data_sources", {})

        if sources.get("database", {}).get("enabled"):
            db = sources["database"]
            conn_str = (
                f"postgresql+psycopg2://{db.get('user', 'postgres')}:"
                f"{db.get('password', '')}@{db['host']}:{db['port']}/{db['name']}"
            )
            return DatabaseConnector(conn_str, table=db.get("table", "measurements"))

        if sources.get("csv", {}).get("enabled", True):
            csv = sources.get("csv", {})
            return CSVConnector(
                path=csv.get("path", "data/raw"),
                date_column=csv.get("date_column", "timestamp"),
                date_format=csv.get("date_format"),
                separator=csv.get("separator", ","),
                encoding=csv.get("encoding", "utf-8"),
            )

        raise ValueError("No enabled data source found in configuration")

    def load(self, **kwargs: Any) -> pd.DataFrame:
        """Load data using the configured connector."""
        self._connector = self._build_connector()
        with self._connector:
            return self._connector.read(**kwargs)

    @staticmethod
    def load_sample() -> pd.DataFrame:
        """Return a synthetic sample dataset for testing."""
        import numpy as np

        rng = np.random.default_rng(seed=42)
        idx = pd.date_range("2024-01-01", periods=8760, freq="1h")
        n = len(idx)

        hourly = np.sin(2 * np.pi * np.arange(n) / 24)
        weekly = np.sin(2 * np.pi * np.arange(n) / (24 * 7))
        trend = np.linspace(0, 0.5, n)

        ch4 = 90.0 + 2.0 * hourly + 1.0 * weekly - trend + rng.normal(0, 0.3, n)
        c2h6 = 3.0 - 0.5 * hourly + rng.normal(0, 0.05, n)
        n2 = 2.0 + 0.3 * hourly + trend + rng.normal(0, 0.1, n)
        co2 = 0.5 + 0.1 * weekly + rng.normal(0, 0.02, n)
        h2s = np.abs(2.0 + rng.normal(0, 0.5, n))
        pressure = 50.0 + 5.0 * hourly + rng.normal(0, 0.5, n)
        temperature = 15.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / (24 * 365)) + rng.normal(0, 0.5, n)
        flow_rate = 5000 + 1000 * hourly + 500 * weekly + rng.normal(0, 50, n)

        return pd.DataFrame(
            {
                "CH4": np.clip(ch4, 80, 99),
                "C2H6": np.clip(c2h6, 0, 6),
                "N2": np.clip(n2, 0, 8),
                "CO2": np.clip(co2, 0, 3),
                "H2S_ppm": np.clip(h2s, 0, 20),
                "pressure_bar": np.clip(pressure, 30, 70),
                "temperature_C": temperature,
                "flow_rate_m3h": np.clip(flow_rate, 0, None),
            },
            index=idx,
        )
