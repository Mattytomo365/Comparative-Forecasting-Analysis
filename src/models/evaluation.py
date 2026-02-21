import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from typing import Mapping, Any
from src.models.training import train_model
from src.models.tuning import rolling_splits
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
'''
Model performance evaluation using regression metrics and visualisations
'''

def calculate_metrics(y_test: pd.Series, 
                      y_train: pd.Series, 
                      y_pred: np.ndarray) -> dict[str, float]:
    '''
    Calculates MAE, RMSE, and MASE for supplied testing data and corresponding predictions
    '''
    # mean absolute error
    mae = float(mean_absolute_error(y_test, y_pred))
    # root mean squared error
    rmse = float(root_mean_squared_error(y_test, y_pred))
    # mean absolute scaled error
    mase = mean_absolute_scaled_error(y_test, y_pred, y_train=y_train)

    return {"MAE": mae, "MASE": mase, "RMSE": rmse}


def naive_forecast(y_train: pd.Series, 
                   y_test: pd.Series, 
                   kind: str, 
                   test: pd.DataFrame) -> None:
    '''
    Implements naive baseline for forecasting performance benchmark
    '''
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    preds = np.empty_like(y_test, dtype=float)
    preds[0] = y_train[-1] # predict first test point using last train value
    preds[1:] = y_test[:-1] # then predict using previous actual (1-step naive)

    oos_all, metrics_all = [], []
    oos = test[["date"]].copy()
    oos["Actual data"] = y_test
    oos["Forecasted data"] = preds
    oos["model"] = kind
    metrics = calculate_metrics(y_test, y_train, preds)
    metrics["model"] = kind
    metrics_all.append(metrics)
    oos = pd.concat(oos_all, ignore_index=True)
    metrics = pd.DataFrame(metrics_all)
    oos.name = f"{kind}_predictions_baseline"
    metrics.name = f"{kind}_metrics_baseline"
    save_oos(oos, oos.name)
    save_metrics(metrics, metrics.name)


def backtest(df: pd.DataFrame, 
             kind: str, 
             features: list[str], 
             params: Mapping[str, Any], 
             target: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    '''
    Orchestrate rolling-origin/expanding-window backtesting using established outer windows
    '''
    oos_all, metrics_all = [], []
    window = 1

    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])

    for train_mask, test_mask in rolling_splits(dates, 28, 308): # rolling out-of-sample windows
        train, test = df.loc[train_mask], df.loc[test_mask]
        oos, metrics, model = train_model(train, test, kind, features, target, params)
        oos["model"] = kind
        oos["window"] = window
        metrics["model"] = kind
        metrics["window"] = window

        oos_all.append(oos)
        metrics_all.append(metrics)
        window = window + 1

    return pd.concat(oos_all, ignore_index=True), pd.DataFrame(metrics_all), model


def save_oos(oos: pd.DataFrame, name: str) -> str:
    '''
    Save oos dataframe as csv and return path for model registry
    '''
    path_csv = f"results/{name}.csv"
    oos.to_csv(path_csv, index=False)
    return path_csv


def save_metrics(metrics: pd.DataFrame, name: str) -> str:
    '''
    Save metrics dataframe as csv and return path for model registry
    '''
    path = f"results/{name}.csv"
    metrics.to_csv(path, index=False)
    return path






