# Comparative Evalutation of Forecasting Approaches for Hospitality Sales

---

## Overview
BSc Computer Science Capstone Project. Conducted a comparative evaluation of machine-learning and statistical approaches for forecasting hospitality sales, including SARIMAX, XGBoost and Lasso regression.

Cleaned and preprocessed time-series sales data supplied by a local hospitality establishment, combining it with external explanatory variables from publicly available sources, including weather, events and public holidays. Engineered calendar and seasonal features, including cyclical/Fourier-based variables, alongside lagged and rolling statistical features.

Performed exploratory time-series analysis to identify trend and seasonality patterns and inform modelling decisions, using autocorrelation diagnostics and Fourier visualisations.

Trained and evaluated models through a strict time-aware expanding-window validation approach, both before and after hyperparameter optimisation. Particular attention was given to preventing temporal data leakage, with lagged and rolling features generated recursively at each forecasting point.

Compared model performance using MAE, MASE and RMSE, supported by graphical and tabular visualisations designed to make findings interpretable. Additional experiments assessed the impact of feature sets, training optimisation strategies and missing-sales imputation methods on predictive performance.

I plan to extend the work into a full-stack forecasting platform using Angular, ASP.NET Web API and a Flask-based REST API, giving hospitality users accessible visibility into forecasts and the factors influencing them.


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



