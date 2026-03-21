"""
Gas Analyzer: main pipeline entry point.

Usage:
    python main.py                        # run full pipeline on sample data
    python main.py --config config/settings.yaml --target CH4
    python main.py --source csv --data-path data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.loader import DataLoader
from src.processing.cleaner import DataCleaner
from src.processing.features import FeatureEngineer
from src.processing.validator import DataValidator
from src.models.analyzer import GasAnalyzer
from src.models.anomaly import AnomalyDetector
from src.models.correlation import CorrelationAnalyzer
from src.forecasting.prophet_model import ProphetModel
from src.forecasting.xgboost_model import XGBoostModel
from src.forecasting.ensemble import EnsembleForecaster
from src.visualization.plotter import GasPlotter
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.report import ReportGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gas Analyzer Pipeline")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML")
    parser.add_argument("--target", default="CH4", help="Target column to forecast")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (hours)")
    parser.add_argument("--sample", action="store_true", default=True, help="Use synthetic sample data")
    parser.add_argument("--data-path", default=None, help="Override CSV data path")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    setup_logger(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_file=cfg.get("logging", {}).get("file"),
    )

    print("=" * 60)
    print("  Gas Analyzer Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data ...")
    if args.sample:
        data_raw = DataLoader.load_sample()
        print(f"  Sample data: {data_raw.shape[0]:,} rows x {data_raw.shape[1]} columns")
    else:
        loader = DataLoader(config_path=args.config)
        data_raw = loader.load()
        print(f"  Loaded: {data_raw.shape[0]:,} rows x {data_raw.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. Validate
    # ------------------------------------------------------------------
    print("\n[2/6] Validating ...")
    validator = DataValidator(config_path=args.config)
    validation = validator.validate(data_raw)
    print(validation.summary())

    # ------------------------------------------------------------------
    # 3. Clean & preprocess
    # ------------------------------------------------------------------
    print("\n[3/6] Cleaning & preprocessing ...")
    preproc_cfg = cfg.get("preprocessing", {})
    cleaner = DataCleaner(
        outlier_method=preproc_cfg.get("outlier_method", "iqr"),
        outlier_threshold=float(preproc_cfg.get("outlier_threshold", 3.0)),
        interpolation_method=preproc_cfg.get("interpolation_method", "linear"),
        resampling_freq=preproc_cfg.get("resampling_freq", "1h"),
        normalization="none",  # keep original scale for analysis
    )
    data_clean = cleaner.fit_transform(data_raw)
    print(f"  Clean data: {data_clean.shape}")

    # ------------------------------------------------------------------
    # 4. Analyse
    # ------------------------------------------------------------------
    print("\n[4/6] Running analysis ...")
    analyzer = GasAnalyzer(
        decomposition_period=cfg.get("analysis", {}).get("decomposition_period", 24)
    )
    analysis_cols = [c for c in ["CH4", "C2H6", "N2", "pressure_bar", "flow_rate_m3h"] if c in data_clean.columns]
    report = analyzer.analyze(data_clean, columns=analysis_cols)

    corr_analyzer = CorrelationAnalyzer(method=cfg.get("analysis", {}).get("correlation_method", "pearson"))
    corr_matrix = corr_analyzer.correlation_matrix(data_clean)
    print("  Correlation matrix computed.")

    anomaly_cfg = cfg.get("analysis", {}).get("anomaly_detection", {})
    detector = AnomalyDetector(
        method=anomaly_cfg.get("method", "isolation_forest"),
        contamination=float(anomaly_cfg.get("contamination", 0.05)),
    )
    anomalies = detector.fit_predict(data_clean)
    n_anom = int(anomalies.sum())
    print(f"  Anomalies detected: {n_anom} ({100 * n_anom / len(anomalies):.1f}%)")

    # ------------------------------------------------------------------
    # 5. Forecast
    # ------------------------------------------------------------------
    print(f"\n[5/6] Forecasting '{args.target}' ({args.horizon}h horizon) ...")
    if args.target not in data_clean.columns:
        print(f"  Warning: '{args.target}' not found. Using first numeric column.")
        args.target = list(data_clean.select_dtypes("number").columns)[0]

    target_series = data_clean[args.target]
    train = target_series.iloc[:-args.horizon]
    test = target_series.iloc[-args.horizon:]

    fc_cfg = cfg.get("forecasting", {})
    prophet = ProphetModel(
        horizon=args.horizon,
        changepoint_prior_scale=float(fc_cfg.get("prophet", {}).get("changepoint_prior_scale", 0.05)),
    )
    xgb = XGBoostModel(
        horizon=args.horizon,
        n_estimators=int(fc_cfg.get("xgboost", {}).get("n_estimators", 200)),
    )

    prophet.fit(train)
    xgb.fit(train)

    prophet_result = prophet.predict(steps=args.horizon)
    xgb_result = xgb.predict(steps=args.horizon)

    ensemble = EnsembleForecaster([prophet, xgb], weights=[0.5, 0.5], horizon=args.horizon)
    ensemble.fit(train)
    ensemble_result = ensemble.predict(steps=args.horizon)

    # Evaluate on test set
    try:
        xgb_metrics = xgb.evaluate(test, xgb_result.predictions.values[: len(test)])
        xgb_result.metrics = xgb_metrics
        print(f"  XGBoost RMSE = {xgb_metrics['RMSE']:.4f}, MAE = {xgb_metrics['MAE']:.4f}")
    except Exception as e:
        print(f"  Metrics not available: {e}")

    # ------------------------------------------------------------------
    # 6. Report & visualise
    # ------------------------------------------------------------------
    print("\n[6/6] Generating reports and plots ...")
    reporter = ReportGenerator()
    reporter.analysis_report(report)
    reporter.forecast_report([prophet_result, xgb_result, ensemble_result])

    if not args.no_plots:
        plotter = GasPlotter()
        plotter.plot_time_series(data_clean, columns=analysis_cols, save_as="time_series_overview")
        plotter.plot_correlation_heatmap(corr_matrix, save_as="correlation_heatmap")
        if args.target in anomalies.index.name or True:
            plotter.plot_anomalies(data_clean, anomalies, column=args.target, save_as=f"anomalies_{args.target}")
        plotter.plot_forecast(target_series, ensemble_result, save_as=f"forecast_{args.target}_ensemble")
        plotter.plot_distributions(data_clean, save_as="distributions")
        if args.target in report.decompositions:
            plotter.plot_decomposition(report.decompositions[args.target], save_as=f"decomp_{args.target}")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Reports : outputs/reports/")
    print(f"  Plots   : outputs/plots/")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline(parse_args())
