"""
Data source connectors for gas measurement ingestion.
Supports CSV files, relational databases, serial ports, and MQTT brokers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Abstract base class for all data source connectors."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    def read(self, **kwargs: Any) -> pd.DataFrame:
        """Read data from the source and return a DataFrame."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""

    def __enter__(self) -> "BaseConnector":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()


class CSVConnector(BaseConnector):
    """Read gas measurement data from CSV / Excel files."""

    def __init__(
        self,
        path: str | Path,
        date_column: str = "timestamp",
        date_format: str | None = None,
        separator: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.date_column = date_column
        self.date_format = date_format
        self.separator = separator
        self.encoding = encoding

    def connect(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Data path not found: {self.path}")
        logger.info("CSV connector ready: %s", self.path)

    def read(self, pattern: str = "*.csv", **kwargs: Any) -> pd.DataFrame:
        """Read all matching files in the directory (or a single file)."""
        if self.path.is_file():
            files = [self.path]
        else:
            files = sorted(self.path.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' in {self.path}")

        frames = []
        for f in files:
            logger.info("Reading %s", f)
            if f.suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(f, **kwargs)
            else:
                df = pd.read_csv(f, sep=self.separator, encoding=self.encoding, **kwargs)
            frames.append(df)

        data = pd.concat(frames, ignore_index=True)

        # Find datetime column: configured name, or first column that parses as datetime
        date_col = None
        if self.date_column in data.columns:
            date_col = self.date_column
        else:
            # Auto-detect: try the first column
            first = data.columns[0]
            try:
                pd.to_datetime(data[first].head(5), format=self.date_format)
                date_col = first
                logger.debug("Auto-detected datetime column: '%s'", first)
            except Exception:
                pass

        if date_col is not None:
            data[date_col] = pd.to_datetime(data[date_col], format=self.date_format)
            data = data.set_index(date_col).sort_index()
            data.index.name = "timestamp"

        logger.info("Loaded %d rows from %d file(s)", len(data), len(files))
        return data

    def disconnect(self) -> None:
        pass  # Stateless connector


class DatabaseConnector(BaseConnector):
    """Read gas measurements from a relational database via SQLAlchemy."""

    def __init__(
        self,
        connection_string: str,
        table: str = "measurements",
        schema: str | None = None,
    ) -> None:
        self.connection_string = connection_string
        self.table = table
        self.schema = schema
        self._engine: Any = None

    def connect(self) -> None:
        try:
            from sqlalchemy import create_engine
            self._engine = create_engine(self.connection_string)
            logger.info("Database connection established: %s", self.table)
        except ImportError as exc:
            raise ImportError("sqlalchemy is required for DatabaseConnector") from exc

    def read(
        self,
        start: str | None = None,
        end: str | None = None,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        col_expr = ", ".join(columns) if columns else "*"
        table_ref = f"{self.schema}.{self.table}" if self.schema else self.table
        query = f"SELECT {col_expr} FROM {table_ref}"  # noqa: S608

        conditions = []
        if start:
            conditions.append(f"timestamp >= '{start}'")
        if end:
            conditions.append(f"timestamp <= '{end}'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        data = pd.read_sql(query, self._engine, parse_dates=["timestamp"], **kwargs)
        if "timestamp" in data.columns:
            data = data.set_index("timestamp").sort_index()
        logger.info("Loaded %d rows from database", len(data))
        return data

    def disconnect(self) -> None:
        if self._engine:
            self._engine.dispose()
            logger.info("Database connection closed")


class SerialConnector(BaseConnector):
    """Stream real-time gas measurements from a serial port (RS-232/RS-485)."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        timeout: float = 5.0,
        parser_func: Any = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.parser_func = parser_func or self._default_parser
        self._serial: Any = None

    def connect(self) -> None:
        try:
            import serial
            self._serial = serial.Serial(
                self.port, baudrate=self.baudrate, timeout=self.timeout
            )
            logger.info("Serial port opened: %s @ %d baud", self.port, self.baudrate)
        except ImportError as exc:
            raise ImportError("pyserial is required for SerialConnector") from exc

    def read(self, n_samples: int = 100, **_: Any) -> pd.DataFrame:
        """Collect n_samples lines from the serial port and parse them."""
        rows = []
        for _ in range(n_samples):
            line = self._serial.readline().decode("utf-8", errors="replace").strip()
            if line:
                rows.append(self.parser_func(line))
        return pd.DataFrame(rows)

    def disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("Serial port closed: %s", self.port)

    @staticmethod
    def _default_parser(line: str) -> dict:
        """Naive comma-separated parser: timestamp,component,value."""
        parts = line.split(",")
        if len(parts) >= 3:
            return {"timestamp": parts[0], "component": parts[1], "value": float(parts[2])}
        return {"raw": line}


class MQTTConnector(BaseConnector):
    """Subscribe to an MQTT broker and buffer incoming gas measurements."""

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "gas/measurements/#",
        client_id: str = "gas_analyzer",
        buffer_size: int = 1000,
    ) -> None:
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.buffer_size = buffer_size
        self._client: Any = None
        self._buffer: list[dict] = []

    def connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(client_id=self.client_id)
            self._client.on_message = self._on_message
            self._client.connect(self.broker, self.port)
            self._client.subscribe(self.topic)
            self._client.loop_start()
            logger.info("MQTT connected: %s:%d topic=%s", self.broker, self.port, self.topic)
        except ImportError as exc:
            raise ImportError("paho-mqtt is required for MQTTConnector") from exc

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        import json
        try:
            payload = json.loads(message.payload.decode())
            if len(self._buffer) < self.buffer_size:
                self._buffer.append(payload)
        except Exception as exc:
            logger.warning("Failed to parse MQTT message: %s", exc)

    def read(self, flush: bool = True, **_: Any) -> pd.DataFrame:
        """Return buffered messages as a DataFrame."""
        data = pd.DataFrame(self._buffer)
        if flush:
            self._buffer.clear()
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.set_index("timestamp").sort_index()
        return data

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            logger.info("MQTT disconnected")
