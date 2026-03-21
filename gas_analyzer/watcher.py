"""
Gas Analyzer — File Watcher

Monitors one or more directories for new/modified data files (CSV, XLSX).
When a change is detected the full analysis pipeline reruns automatically
and outputs fresh plots + reports.

Usage:
    python watcher.py                          # watch data/raw/ with defaults
    python watcher.py --watch data/raw data/extra --target CH4 --horizon 48
    python watcher.py --once                   # run once and exit (no watching)

Requirements:
    pip install watchdog
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger("gas_watcher")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# ──────────────────────────────────────────────────────────────────────
# Pipeline runner (called each time a file event fires)
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    target: str,
    horizon: int,
    config_path: str,
    merge_strategy: str,
    no_plots: bool,
    extra_files: list[Path],
) -> None:
    """Execute full analysis + forecast pipeline and save outputs."""

    import warnings
    warnings.filterwarnings("ignore")

    from src.ingestion.loader import MultiSourceLoader
    from src.processing.cleaner import DataCleaner
    from src.processing.validator import DataValidator
    from src.models.analyzer import GasAnalyzer
    from src.models.anomaly import AnomalyDetector
    from src.models.correlation import CorrelationAnalyzer
    from src.forecasting.xgboost_model import XGBoostModel
    from src.forecasting.sarima_model import SARIMAModel
    from src.forecasting.ensemble import EnsembleForecaster
    from src.visualization.plotter import GasPlotter
    from src.utils.report import ReportGenerator
    from src.utils.config import load_config

    cfg = load_config(config_path)
    ts = time.strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  Pipeline triggered at {ts}")
    print(f"{'='*60}")

    # 1. Load — all enabled sources + any extra files detected by watcher
    print("[1/5] Loading data from all sources ...")
    loader = MultiSourceLoader(
        config_path=config_path,
        merge_strategy=merge_strategy,
        resample_freq=cfg.get("preprocessing", {}).get("resampling_freq", "1h"),
    )
    try:
        data = loader.load(extra_files=extra_files)
        summary = loader.source_summary(extra_files=extra_files)
        print(summary.to_string())
        print(f"  Merged: {data.shape[0]:,} rows x {data.shape[1]} columns")
    except ValueError as exc:
        logger.warning("No data available yet: %s", exc)
        return

    if len(data) < 48:
        logger.warning("Too few rows (%d) for meaningful analysis — skipping.", len(data))
        return

    # 2. Validate
    print("[2/5] Validating ...")
    try:
        validator = DataValidator(config_path=config_path)
        report = validator.validate(data)
        print(f"  Valid={report.is_valid}  Alarms={report.alarm_breaches or 'none'}")
    except Exception as exc:
        logger.warning("Validation skipped: %s", exc)

    # 3. Clean
    print("[3/5] Cleaning ...")
    preproc = cfg.get("preprocessing", {})
    cleaner = DataCleaner(
        outlier_method=preproc.get("outlier_method", "iqr"),
        outlier_threshold=float(preproc.get("outlier_threshold", 3.0)),
        interpolation_method=preproc.get("interpolation_method", "linear"),
        resampling_freq=preproc.get("resampling_freq", "1h"),
        normalization="none",
    )
    data_clean = cleaner.fit_transform(data)
    print(f"  Clean: {data_clean.shape}")

    # 4. Analyse
    print("[4/5] Analysing ...")
    numeric_cols = list(data_clean.select_dtypes("number").columns)
    analysis_cols = [c for c in numeric_cols if c in data_clean.columns][:6]

    analyzer = GasAnalyzer(
        decomposition_period=cfg.get("analysis", {}).get("decomposition_period", 24)
    )
    try:
        analysis_report = analyzer.analyze(data_clean, columns=analysis_cols)
    except Exception as exc:
        logger.warning("Decomposition skipped: %s", exc)
        analysis_report = None

    corr_matrix = CorrelationAnalyzer().correlation_matrix(data_clean)

    detector = AnomalyDetector(method="isolation_forest", contamination=0.05)
    anomalies = detector.fit_predict(data_clean)
    n_anom = int(anomalies.sum())
    print(f"  Anomalies: {n_anom} ({100*n_anom/len(anomalies):.1f}%)")

    # 5. Forecast
    print(f"[5/5] Forecasting '{target}' {horizon}h ...")
    if target not in data_clean.columns:
        alt = numeric_cols[0] if numeric_cols else None
        if alt:
            logger.warning("'%s' not found, using '%s'", target, alt)
            target = alt
        else:
            logger.error("No numeric columns available for forecasting.")
            return

    series = data_clean[target].dropna()
    if len(series) < horizon * 2:
        logger.warning("Not enough data for forecasting (%d rows).", len(series))
        return

    train = series.iloc[:-horizon]
    test  = series.iloc[-horizon:]
    results = []

    xgb = XGBoostModel(horizon=horizon, n_estimators=200)
    xgb.fit(train)
    xgb_r = xgb.predict(steps=horizon)
    xgb_r.metrics = xgb.evaluate(test.values, xgb_r.predictions.values[:len(test)])
    results.append(xgb_r)
    print(f"  XGBoost  RMSE={xgb_r.metrics['RMSE']:.4f}  MAE={xgb_r.metrics['MAE']:.4f}")

    try:
        sarima = SARIMAModel(horizon=horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        sarima.fit(train)
        sar_r = sarima.predict(steps=horizon)
        sar_r.metrics = sarima.evaluate(test.values, sar_r.predictions.values[:len(test)])
        results.append(sar_r)
        print(f"  SARIMA   RMSE={sar_r.metrics['RMSE']:.4f}  MAE={sar_r.metrics['MAE']:.4f}")
    except Exception as exc:
        logger.warning("SARIMA skipped: %s", exc)

    if len(results) > 1:
        ens = EnsembleForecaster(
            [xgb, sarima], weights=[0.6, 0.4], horizon=horizon
        )
        ens.fit(train)
        ens_r = ens.predict(steps=horizon)
        ens_r.metrics = xgb.evaluate(test.values, ens_r.predictions.values[:len(test)])
        results.append(ens_r)
        best = ens_r
        print(f"  Ensemble RMSE={ens_r.metrics['RMSE']:.4f}  MAE={ens_r.metrics['MAE']:.4f}")
    else:
        best = results[0]

    # Plots
    if not no_plots:
        plotter = GasPlotter()
        plotter.plot_time_series(data_clean, columns=analysis_cols,
                                 save_as="auto_time_series")
        plotter.plot_correlation_heatmap(corr_matrix, save_as="auto_correlation")
        plotter.plot_anomalies(data_clean, anomalies, column=target,
                               save_as=f"auto_anomalies_{target}")
        plotter.plot_forecast(series, best,
                              save_as=f"auto_forecast_{target}")
        plotter.plot_distributions(data_clean, save_as="auto_distributions")

    # Reports
    reporter = ReportGenerator()
    if analysis_report:
        reporter.analysis_report(analysis_report, filename="auto_analysis.html")
    reporter.forecast_report(results, filename="auto_forecast.html")

    print(f"\n  Done. Outputs updated in outputs/")
    print(f"  Next update on file change.\n")


# ──────────────────────────────────────────────────────────────────────
# File event handler
# ──────────────────────────────────────────────────────────────────────

def _make_handler(pipeline_args: dict, debounce_sec: float = 3.0):
    """Returns a watchdog event handler that debounces rapid file events."""
    try:
        from watchdog.events import FileSystemEventHandler
    except ImportError as exc:
        raise ImportError("watchdog is required: pip install watchdog") from exc

    class _Handler(FileSystemEventHandler):
        def __init__(self) -> None:
            super().__init__()
            self._last_run: float = 0.0
            self._pending_files: set[Path] = set()

        def on_created(self, event) -> None:
            self._on_event(event)

        def on_modified(self, event) -> None:
            self._on_event(event)

        def _on_event(self, event) -> None:
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                return

            self._pending_files.add(path)
            now = time.monotonic()

            # Debounce: wait a bit before triggering (rapid saves / partial writes)
            if now - self._last_run < debounce_sec:
                logger.debug("Debouncing — holding off for %.0fs", debounce_sec)
                return

            self._last_run = now
            detected = sorted(self._pending_files)
            self._pending_files.clear()
            logger.info("Detected %d new/changed file(s): %s",
                        len(detected), [f.name for f in detected])

            run_pipeline(**pipeline_args, extra_files=detected)

    return _Handler()


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gas Analyzer — File Watcher")
    p.add_argument("--watch", nargs="+", default=["data/raw"],
                   help="Directories to watch (default: data/raw)")
    p.add_argument("--target", default="CH4", help="Column to forecast")
    p.add_argument("--horizon", type=int, default=24, help="Forecast horizon (hours)")
    p.add_argument("--config", default="config/settings.yaml")
    p.add_argument("--merge", default="outer",
                   choices=["outer", "inner", "newest"],
                   help="Multi-source merge strategy")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--once", action="store_true",
                   help="Run pipeline once with existing files and exit")
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="Watchdog polling interval in seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pipeline_args = dict(
        target=args.target,
        horizon=args.horizon,
        config_path=args.config,
        merge_strategy=args.merge,
        no_plots=args.no_plots,
    )

    watch_dirs = [Path(d) for d in args.watch]
    for d in watch_dirs:
        d.mkdir(parents=True, exist_ok=True)

    if args.once:
        # Collect all existing files and run once
        existing: list[Path] = []
        for d in watch_dirs:
            existing.extend(d.glob("*.csv"))
            existing.extend(d.glob("*.xlsx"))
        logger.info("--once mode: found %d file(s)", len(existing))
        run_pipeline(**pipeline_args, extra_files=existing)
        return

    try:
        from watchdog.observers import Observer
    except ImportError:
        logger.error("watchdog is not installed. Run: pip install watchdog")
        sys.exit(1)

    handler = _make_handler(pipeline_args)
    observer = Observer()
    for d in watch_dirs:
        observer.schedule(handler, str(d), recursive=False)
        logger.info("Watching: %s", d.resolve())

    observer.start()
    logger.info(
        "Gas Analyzer Watcher started.\n"
        "  Drop CSV/XLSX files into %s to trigger the pipeline.\n"
        "  Press Ctrl+C to stop.",
        [str(d) for d in watch_dirs],
    )

    try:
        while True:
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("Stopping watcher ...")
        observer.stop()

    observer.join()
    logger.info("Watcher stopped.")


if __name__ == "__main__":
    main()
