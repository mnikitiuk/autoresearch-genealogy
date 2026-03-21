"""
DataLoader: unified entry point for gas measurement data ingestion.

DataLoader     — single-source loader (original, backwards-compatible)
MultiSourceLoader — loads ALL enabled sources in parallel and merges results
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
    """High-level data loader that reads configuration and routes to one connector."""

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
        ).rename_axis("timestamp")


class MultiSourceLoader:
    """
    Loads data from ALL enabled sources simultaneously and merges into one DataFrame.

    Merge strategies:
      - "outer"  : keep every timestamp from every source (fills missing with NaN)
      - "inner"  : keep only timestamps present in ALL sources
      - "newest" : per-column, prefer the source with the most recent data
    """

    MERGE_STRATEGIES = ("outer", "inner", "newest")

    def __init__(
        self,
        config_path: str | Path = "config/settings.yaml",
        merge_strategy: str = "outer",
        resample_freq: str = "1h",
    ) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        if merge_strategy not in self.MERGE_STRATEGIES:
            raise ValueError(f"merge_strategy must be one of {self.MERGE_STRATEGIES}")
        self.merge_strategy = merge_strategy
        self.resample_freq = resample_freq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, extra_files: list[str | Path] | None = None) -> pd.DataFrame:
        """
        Load from all enabled sources + any ad-hoc extra_files paths.
        Returns a single merged, time-sorted DataFrame.

        Parameters
        ----------
        extra_files : list of file paths (CSV/XLSX) to include on top of config sources
        """
        frames: list[tuple[str, pd.DataFrame]] = []

        frames.extend(self._load_csv_sources())
        frames.extend(self._load_database_sources())
        frames.extend(self._load_extra_files(extra_files or []))

        if not frames:
            raise ValueError("No data loaded — check that at least one source is enabled.")

        logger.info("Loaded %d source(s): %s", len(frames), [name for name, _ in frames])

        merged = self._merge(frames)
        merged = self._align(merged)

        if isinstance(merged.index, pd.DatetimeIndex):
            date_range = f"[{merged.index[0].date()} – {merged.index[-1].date()}]"
        else:
            date_range = f"[{merged.index[0]} – {merged.index[-1]}]"
        logger.info("Merged result: %d rows x %d cols  %s", len(merged), merged.shape[1], date_range)
        return merged

    def source_summary(self, extra_files: list[str | Path] | None = None) -> pd.DataFrame:
        """Return a summary table of each source: rows, columns, date range."""
        frames = self._load_csv_sources() + self._load_database_sources()
        frames += self._load_extra_files(extra_files or [])

        rows = []
        for name, df in frames:
            rows.append({
                "source": name,
                "rows": len(df),
                "columns": list(df.columns),
                "start": df.index.min(),
                "end": df.index.max(),
                "missing_pct": round(df.isnull().mean().mean() * 100, 1),
            })
        return pd.DataFrame(rows).set_index("source")

    # ------------------------------------------------------------------
    # Source loaders
    # ------------------------------------------------------------------

    def _load_csv_sources(self) -> list[tuple[str, pd.DataFrame]]:
        """Load every enabled CSV data_sources entry from config."""
        results: list[tuple[str, pd.DataFrame]] = []
        sources = self.config.get("data_sources", {})

        # Primary CSV source
        csv_cfg = sources.get("csv", {})
        if csv_cfg.get("enabled", True):
            path = Path(csv_cfg.get("path", "data/raw"))
            if path.exists() and any(path.glob("*.csv")):
                connector = CSVConnector(
                    path=path,
                    date_column=csv_cfg.get("date_column", "timestamp"),
                    date_format=csv_cfg.get("date_format"),
                    separator=csv_cfg.get("separator", ","),
                    encoding=csv_cfg.get("encoding", "utf-8"),
                )
                try:
                    with connector:
                        df = connector.read()
                    results.append(("csv:primary", df))
                    logger.info("csv:primary — %d rows from %s", len(df), path)
                except Exception as exc:
                    logger.warning("csv:primary failed: %s", exc)
            else:
                logger.debug("csv:primary path empty or not found: %s", path)

        # Additional named CSV sources (optional, defined under data_sources.extra_csv)
        for entry in sources.get("extra_csv", []):
            if not entry.get("enabled", True):
                continue
            path = Path(entry["path"])
            name = entry.get("name", f"csv:{path.stem}")
            if not path.exists():
                logger.warning("%s path not found: %s", name, path)
                continue
            connector = CSVConnector(
                path=path,
                date_column=entry.get("date_column", "timestamp"),
                date_format=entry.get("date_format"),
                separator=entry.get("separator", ","),
                encoding=entry.get("encoding", "utf-8"),
            )
            try:
                with connector:
                    df = connector.read()
                results.append((name, df))
                logger.info("%s — %d rows", name, len(df))
            except Exception as exc:
                logger.warning("%s failed: %s", name, exc)

        return results

    def _load_database_sources(self) -> list[tuple[str, pd.DataFrame]]:
        sources = self.config.get("data_sources", {})
        db_cfg = sources.get("database", {})
        if not db_cfg.get("enabled"):
            return []

        db = db_cfg
        conn_str = (
            f"postgresql+psycopg2://{db.get('user', 'postgres')}:"
            f"{db.get('password', '')}@{db['host']}:{db['port']}/{db['name']}"
        )
        connector = DatabaseConnector(conn_str, table=db.get("table", "measurements"))
        try:
            with connector:
                df = connector.read()
            logger.info("database:primary — %d rows", len(df))
            return [("database:primary", df)]
        except Exception as exc:
            logger.warning("database:primary failed: %s", exc)
            return []

    def _load_extra_files(self, paths: list[str | Path]) -> list[tuple[str, pd.DataFrame]]:
        """Load ad-hoc files passed at runtime (e.g. dropped into data/raw/ by the watcher)."""
        results: list[tuple[str, pd.DataFrame]] = []
        csv_cfg = self.config.get("data_sources", {}).get("csv", {})

        # Resolve primary CSV directory so we don't load same files twice
        primary_path = Path(
            self.config.get("data_sources", {}).get("csv", {}).get("path", "data/raw")
        ).resolve()

        for p in paths:
            p = Path(p).resolve()
            if not p.exists():
                logger.warning("Extra file not found: %s", p)
                continue
            # Skip if this file is already covered by the primary CSV source
            if p.parent == primary_path:
                logger.debug("Skipping %s — already loaded by csv:primary", p.name)
                continue
            connector = CSVConnector(
                path=p,
                date_column=csv_cfg.get("date_column", "timestamp"),
                date_format=csv_cfg.get("date_format"),
                separator=csv_cfg.get("separator", ","),
            )
            try:
                with connector:
                    df = connector.read(pattern=p.name)
                results.append((f"file:{p.stem}", df))
                logger.info("file:%s — %d rows", p.stem, len(df))
            except Exception as exc:
                logger.warning("file:%s failed: %s", p.stem, exc)

        return results

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    def _merge(self, frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        if len(frames) == 1:
            return frames[0][1].copy()

        if self.merge_strategy in ("outer", "inner"):
            how = self.merge_strategy
            # Stack all frames into one DataFrame, then coalesce per column
            all_dfs = [df.copy() for _, df in frames]

            # Collect all unique column names across sources
            all_cols: list[str] = []
            for df in all_dfs:
                for c in df.columns:
                    if c not in all_cols:
                        all_cols.append(c)

            # For each column, concat across sources and take first non-null per timestamp
            series_map: dict[str, pd.Series] = {}
            for col in all_cols:
                parts = [df[col] for df in all_dfs if col in df.columns]
                if len(parts) == 1:
                    series_map[col] = parts[0]
                else:
                    combined = pd.concat(parts, axis=1)
                    combined.columns = range(len(parts))
                    series_map[col] = combined.bfill(axis=1).iloc[:, 0]

            merged = pd.DataFrame(series_map)
            if how == "inner":
                # Keep only timestamps present in ALL sources
                common_idx = all_dfs[0].index
                for df in all_dfs[1:]:
                    common_idx = common_idx.intersection(df.index)
                merged = merged.loc[common_idx]

        elif self.merge_strategy == "newest":
            # For each column, prefer the source with the latest end date
            base = pd.concat([df for _, df in frames], axis=0)
            merged = base.groupby(base.index).last()

        else:
            merged = frames[0][1].copy()

        return merged

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort, drop duplicate index entries, optionally resample."""
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if self.resample_freq and isinstance(df.index, pd.DatetimeIndex):
            df = df.resample(self.resample_freq).mean()
        return df

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
