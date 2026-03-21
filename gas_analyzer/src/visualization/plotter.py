"""
GasPlotter: matplotlib/plotly visualisations for gas measurements,
decompositions, anomalies, correlations, and forecasts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


class GasPlotter:
    """Generate and optionally save publication-quality charts."""

    def __init__(self, output_dir: str | Path = "outputs/plots", fmt: str = "png", dpi: int = 150) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Time series
    # ------------------------------------------------------------------

    def plot_time_series(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        title: str = "Gas Measurements Over Time",
        save_as: str | None = None,
    ) -> plt.Figure:
        cols = columns or list(data.select_dtypes(include=[np.number]).columns)
        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, cols):
            ax.plot(data.index, data[col], linewidth=0.8)
            ax.set_ylabel(col, fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[0].set_title(title, fontsize=12, fontweight="bold")
        axes[-1].set_xlabel("Time")
        fig.tight_layout()
        return self._save(fig, save_as or "time_series")

    # ------------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix",
        save_as: str | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            square=True,
            linewidths=0.5,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, save_as or "correlation_heatmap")

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def plot_decomposition(self, decomp: Any, save_as: str | None = None) -> plt.Figure:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Seasonal Decomposition: {decomp.component}", fontsize=12, fontweight="bold")

        data_series = decomp.trend + decomp.seasonal + decomp.residual
        for ax, (label, series) in zip(
            axes,
            [
                ("Observed", data_series),
                ("Trend", decomp.trend),
                ("Seasonal", decomp.seasonal),
                ("Residual", decomp.residual),
            ],
        ):
            ax.plot(series.index, series, linewidth=0.8)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time")
        fig.tight_layout()
        return self._save(fig, save_as or f"decomposition_{decomp.component}")

    # ------------------------------------------------------------------
    # Anomalies
    # ------------------------------------------------------------------

    def plot_anomalies(
        self,
        data: pd.DataFrame,
        anomalies: pd.Series,
        column: str,
        title: str = "Anomaly Detection",
        save_as: str | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(data.index, data[column], linewidth=0.8, label="Normal", color="steelblue")
        ax.scatter(
            data.index[anomalies],
            data[column][anomalies],
            color="red",
            s=20,
            label=f"Anomalies ({anomalies.sum()})",
            zorder=5,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._save(fig, save_as or f"anomalies_{column}")

    # ------------------------------------------------------------------
    # Forecasts
    # ------------------------------------------------------------------

    def plot_forecast(
        self,
        historical: pd.Series,
        forecast_result: Any,
        n_history: int = 168,
        title: str | None = None,
        save_as: str | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 5))

        hist_plot = historical.tail(n_history)
        ax.plot(hist_plot.index, hist_plot.values, linewidth=0.9, label="Historical", color="steelblue")
        ax.plot(
            forecast_result.predictions.index,
            forecast_result.predictions.values,
            linewidth=1.2,
            label="Forecast",
            color="darkorange",
        )

        if forecast_result.lower_bound is not None and forecast_result.upper_bound is not None:
            ax.fill_between(
                forecast_result.predictions.index,
                forecast_result.lower_bound,
                forecast_result.upper_bound,
                alpha=0.25,
                color="darkorange",
                label="Confidence Interval",
            )

        ax.axvline(x=historical.index[-1], color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(
            title or f"Forecast: {forecast_result.target} ({forecast_result.model_name})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel(forecast_result.target)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._save(fig, save_as or f"forecast_{forecast_result.target}")

    # ------------------------------------------------------------------
    # Distributions
    # ------------------------------------------------------------------

    def plot_distributions(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        save_as: str | None = None,
    ) -> plt.Figure:
        cols = columns or list(data.select_dtypes(include=[np.number]).columns)[:8]
        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for ax, col in zip(axes, cols):
            sns.histplot(data[col].dropna(), kde=True, ax=ax, color="steelblue")
            ax.set_title(col, fontsize=9)
        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle("Gas Component Distributions", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, save_as or "distributions")

    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, name: str) -> plt.Figure:
        path = self.output_dir / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        logger.info("Plot saved: %s", path)
        return fig
