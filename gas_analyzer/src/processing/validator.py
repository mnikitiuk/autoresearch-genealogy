"""
DataValidator: checks that incoming gas measurements conform to
physical constraints and configured alarm thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    n_rows: int = 0
    missing_columns: list[str] = field(default_factory=list)
    out_of_range: dict[str, int] = field(default_factory=dict)
    alarm_breaches: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    is_valid: bool = True

    def summary(self) -> str:
        lines = [
            f"Rows checked : {self.n_rows}",
            f"Missing cols : {self.missing_columns or 'none'}",
            f"Out-of-range : {self.out_of_range or 'none'}",
            f"Alarms       : {self.alarm_breaches or 'none'}",
            f"Duplicates   : {self.duplicate_rows}",
            f"Valid        : {self.is_valid}",
        ]
        return "\n".join(lines)


class DataValidator:
    """Validate gas measurement DataFrames against configured thresholds."""

    COMPONENT_COLUMN_MAP = {
        "methane": "CH4",
        "ethane": "C2H6",
        "propane": "C3H8",
        "butane": "C4H10",
        "nitrogen": "N2",
        "carbon_dioxide": "CO2",
        "hydrogen_sulfide": "H2S_ppm",
        "oxygen": "O2",
    }

    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self._gas_components: list[dict[str, Any]] = cfg.get("gas_components", [])
        self._quality_params: list[dict[str, Any]] = cfg.get("quality_parameters", [])

    def validate(self, data: pd.DataFrame, strict: bool = False) -> ValidationReport:
        report = ValidationReport(n_rows=len(data))

        # Duplicate index check
        report.duplicate_rows = int(data.index.duplicated().sum())

        # Per-component range checks
        for comp in self._gas_components:
            col = self.COMPONENT_COLUMN_MAP.get(comp["name"])
            if col is None or col not in data.columns:
                if col:
                    report.missing_columns.append(col)
                continue

            lo, hi = comp["normal_range"]
            alarm = comp.get("alarm_threshold")

            out = int(((data[col] < lo) | (data[col] > hi)).sum())
            if out:
                report.out_of_range[col] = out

            if alarm is not None:
                breaches = int((data[col] > alarm).sum())
                if breaches:
                    report.alarm_breaches[col] = breaches
                    logger.warning("%s alarm threshold (%.1f) breached %d time(s)", col, alarm, breaches)

        if report.missing_columns or report.alarm_breaches:
            report.is_valid = False
        if strict and report.out_of_range:
            report.is_valid = False

        logger.info("Validation complete. Valid=%s", report.is_valid)
        return report
