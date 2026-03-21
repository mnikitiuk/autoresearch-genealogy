"""
ReportGenerator: produces HTML and CSV summary reports from
analysis and forecasting results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate structured reports from gas analysis and forecast results."""

    def __init__(self, output_dir: str | Path = "outputs/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analysis_report(
        self,
        report: Any,
        filename: str | None = None,
    ) -> Path:
        """Write analysis report to an HTML file."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / (filename or f"analysis_{ts}.html")

        sections: list[str] = [self._html_header("Gas Analysis Report")]

        sections.append("<h2>Descriptive Statistics</h2>")
        sections.append(report.descriptive_stats.to_html(classes="table", border=0, float_format="{:.4f}".format))

        sections.append("<h2>Stationarity Tests (ADF)</h2>")
        for col, stats in report.stationarity.items():
            status = "Stationary" if stats.get("is_stationary") else "Non-Stationary"
            sections.append(
                f"<p><strong>{col}</strong>: {status} "
                f"(p-value={stats.get('p_value', 'n/a'):.4f})</p>"
            )

        sections.append(self._html_footer())
        out.write_text("\n".join(sections), encoding="utf-8")
        logger.info("Analysis report saved: %s", out)
        return out

    def forecast_report(
        self,
        results: list[Any],
        filename: str | None = None,
    ) -> Path:
        """Write forecast comparison report to HTML."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / (filename or f"forecast_{ts}.html")

        sections: list[str] = [self._html_header("Gas Forecast Report")]

        for r in results:
            sections.append(f"<h2>{r.model_name}: {r.target}</h2>")
            sections.append(f"<p>Horizon: {r.horizon} steps</p>")
            if r.metrics:
                sections.append("<h3>Metrics</h3>")
                metrics_df = pd.DataFrame([r.metrics])
                sections.append(metrics_df.to_html(classes="table", border=0, index=False, float_format="{:.4f}".format))
            sections.append("<h3>Forecast Values</h3>")
            sections.append(r.to_dataframe().head(24).to_html(classes="table", border=0, float_format="{:.4f}".format))

        sections.append(self._html_footer())
        out.write_text("\n".join(sections), encoding="utf-8")
        logger.info("Forecast report saved: %s", out)
        return out

    def export_csv(self, data: pd.DataFrame, filename: str) -> Path:
        out = self.output_dir / filename
        data.to_csv(out)
        logger.info("CSV exported: %s", out)
        return out

    @staticmethod
    def _html_header(title: str) -> str:
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ color: #16213e; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  .table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  .table td, .table th {{ border: 1px solid #ddd; padding: 6px 10px; }}
  .table tr:nth-child(even) {{ background-color: #f9f9f9; }}
  .table th {{ background-color: #1a1a2e; color: white; }}
</style>
</head><body>
<h1>{title}</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"""

    @staticmethod
    def _html_footer() -> str:
        return "</body></html>"
