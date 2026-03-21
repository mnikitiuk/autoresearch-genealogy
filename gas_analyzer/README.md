# Gas Analyzer

A Python framework for gas composition analysis and multi-model time-series forecasting.

## Features

- **Ingestion**: CSV files, relational databases (SQLAlchemy), serial ports (RS-232/RS-485), MQTT brokers.
- **Validation**: per-component range checks and configurable alarm thresholds.
- **Preprocessing**: outlier removal (IQR, Z-score, Isolation Forest), interpolation, resampling, normalisation.
- **Feature engineering**: time encodings, lag features, rolling statistics, gas quality indicators (HHV, Wobbe Index, SG).
- **Analysis**: seasonal decomposition, ADF stationarity tests, autocorrelation, correlation matrices, anomaly detection.
- **Forecasting**: Prophet, SARIMA, XGBoost, LSTM, weighted Ensemble with cross-validation.
- **Visualisation**: time-series plots, heatmaps, decomposition charts, anomaly overlays, forecast bands.
- **Reports**: HTML summary reports, CSV exports.

## Project Structure

```
gas_analyzer/
    config/
        settings.yaml          Main configuration
    src/
        ingestion/
            connectors.py      CSV, DB, Serial, MQTT connectors
            loader.py          DataLoader entry point
        processing/
            cleaner.py         Outlier removal, imputation, normalisation
            features.py        Feature engineering
            validator.py       Schema and threshold validation
        models/
            analyzer.py        Decomposition, stationarity, ACF
            anomaly.py         Anomaly detection
            correlation.py     Correlation analysis
        forecasting/
            base.py            Abstract model + ForecastResult
            prophet_model.py   Facebook Prophet
            sarima_model.py    SARIMA / Auto-ARIMA
            xgboost_model.py   XGBoost with lag features
            lstm_model.py      PyTorch LSTM
            ensemble.py        Weighted ensemble
        visualization/
            plotter.py         matplotlib / seaborn charts
        utils/
            config.py          YAML loader with env-var overrides
            logger.py          loguru setup
            report.py          HTML / CSV reports
    data/
        raw/                   Raw input files (not committed)
        processed/             Preprocessed cache (not committed)
        samples/               Synthetic sample data
    notebooks/
        01_exploration.ipynb   Data loading, cleaning, visualisation
        02_forecasting.ipynb   Model comparison and cross-validation
    tests/
        test_ingestion.py
        test_processing.py
        test_models.py
        test_forecasting.py
    outputs/
        plots/
        reports/
        exports/
    main.py                    Full pipeline entry point
    setup.py
    requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline on synthetic sample data
python main.py --sample

# Run pipeline on your CSV files
python main.py --data-path data/raw --target CH4 --horizon 24
```

## Configuration

Edit `config/settings.yaml` to configure:

- Gas components and alarm thresholds.
- Data sources (CSV, DB, Serial, MQTT).
- Preprocessing options.
- Forecasting models and hyperparameters.
- Output paths and formats.

Environment variables override YAML values:
```
GAS_ANALYZER__FORECASTING__HORIZON=48
GAS_ANALYZER__PREPROCESSING__RESAMPLING_FREQ=30min
```

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Notebooks

Open Jupyter and navigate to `notebooks/`:

- `01_exploration.ipynb`: data loading, validation, visualisation, anomaly detection.
- `02_forecasting.ipynb`: model training, forecast comparison, cross-validation.

## Forecasting Models

| Model    | Best For                              | Notes                                   |
|----------|---------------------------------------|-----------------------------------------|
| Prophet  | Strong daily/weekly/yearly seasonality| Auto changepoint detection              |
| SARIMA   | Stationary or differenced series      | Auto-order via pmdarima optional        |
| XGBoost  | Non-linear patterns, fast training    | Lag + rolling features                  |
| LSTM     | Long-range dependencies               | Requires PyTorch                        |
| Ensemble | Robust production use                 | Weighted combination of any models      |
