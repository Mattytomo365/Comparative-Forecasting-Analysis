# Comparative Evalutation of Forecasting Approaches for Hospitality Sales

---

## Overview

This is my BSc Computer Science capstone project. The aim is to establish an accurate and efficient approach to time-series forecasting for hospitality sales. 

Three models - SARIMAX, XGBoost, Lasso - will all be trained and tested on the same cleaned and preprocessed dataset. The dataset contains exogenous factors (e.g. weather/events/holidays), added using publicly available sources, and engineered features (Fourier/cyclical features and lags/rolling statistics) to establish more meaningful relationships between catgeories and model seasonality throughout.

Onehot schema generated during data encoding will be persisted and re-used, minimising drifts in data between train/test and inference. Models will also be tested using rolling-origin backtesting, reducing overfitting of models and the leakage of data, and models will be cross-validated with a time-aware expanding/rolling window approach, paired with hyperparameter tuning to establish parameter configurations which yeild the best accuracy.

Performance metrics such as MASE, MAE, and RMSE are reported both before and after model tuning, and across all models for comparison. The Lasso approach will hold a strong baseline for the other more complex models to be compared against. Evaluation interpretations and findings will be presented in both tabular and graphical formats to convey findings of model performance and EDA performed on the historical dataset effectively, with modelling decisions guided by EDA diagnostics e.g., autocorrelation.

Additional experiments analyse the effects of decisions made throughout the pipeline, such as missing-value imputation analysis, traning optimisation analysis, and feature-contribution analysis.

---

## Setup

This project runs as a plain Python package from the repository root. The commands below assume you are in the project directory.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation on your machine, you can still run commands with `.\.venv\Scripts\python.exe` instead of activating the environment first.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install statsmodels sktime
```

`requirements.txt` covers most packages used by the project. `statsmodels` is required for SARIMAX modelling and `sktime` is required for MASE evaluation.

### 3. Check the input data

The repository already includes the CSV files needed for the default pipeline under `data/`, so you can run the project without downloading anything else:

- `data/sales_daily.csv`
- `data/weather_daily.csv`
- `data/holidays.csv`
- `data/events.csv`

---

## Running The Project

To see all supported commands:

```powershell
python -m scripts --help
```

### Run the full pipeline

```powershell
python -m scripts all
```

This runs the project in pipeline order:

1. `run_preprocessing`
2. `run_data_analysis`
3. `run_modelling`
4. `run_comparison`
5. `run_experiments`

### Run individual stages

```powershell
python -m scripts run_preprocessing
python -m scripts run_data_analysis
python -m scripts run_modelling
python -m scripts run_comparison
python -m scripts run_experiments
```

Most later stages depend on files produced by earlier ones, so `run_preprocessing` should be run before the analysis or modelling steps, and `run_modelling` should be run before `run_comparison` or `run_experiments` unless those artefacts already exist.

### Optional arguments

You can override the default dataset paths or target column if needed:

```powershell
python -m scripts all --raw-data data/sales_daily.csv --data data/sales_daily_processed.csv --target sales
```

### Outputs

Running the pipeline writes artefacts to these folders:

- `data/` for processed datasets and the persisted one-hot schema
- `results/` for prediction files, metrics, and summary tables
- `figures/` for generated plots
- `model_info/` for saved manifests and tuned parameter files

`run_modelling` and `run_experiments` are the heaviest stages because they perform backtesting and hyperparameter search, so the full pipeline can take a while to finish.



