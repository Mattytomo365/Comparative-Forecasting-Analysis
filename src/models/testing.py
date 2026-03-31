import numpy as np, pandas as pd
from typing import Mapping, Any
from src.models.training import train_predict
from src.models.tuning import expanding_splits
from src.models.metrics import calculate_metrics, save_metrics, save_oos

def backtest(df: pd.DataFrame, 
             kind: str,
             params: Mapping[str, Any], 
             target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Orchestrate expanding-window backtesting using established outer windows
    '''
    oos_all, metrics_all = [], []
    window = 1

    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    horizon_days = 28
    min_training_days = 308

    # expanding out-of-sample windows
    for train_mask, test_mask in expanding_splits(dates, horizon_days, min_training_days): 
        train, test = df.loc[train_mask], df.loc[test_mask]
        oos, metrics = train_predict(train, test, kind, target, params)
        oos["model"] = kind
        oos["window"] = window
        metrics["model"] = kind
        metrics["window"] = window

        oos_all.append(oos)
        metrics_all.append(metrics)
        window = window + 1

    return pd.concat(oos_all, ignore_index=True), pd.DataFrame(metrics_all)

def naive_forecast(y_train: pd.Series, 
                   y_test: pd.Series, 
                   kind: str, 
                   test: pd.DataFrame) -> None:
    '''
    Implements seasonal naive baseline for forecasting performance benchmark
    '''
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    # OR repeat last 7 days pattern across horizon:
    pattern = y_train.iloc[-7:].to_numpy()
    preds = np.resize(pattern, len(y_test))

    oos = test[["date"]].copy()
    oos["Actual data"] = y_test
    oos["Forecasted data"] = preds
    oos["model"] = kind

    metrics = calculate_metrics(y_test, y_train, preds)
    metrics["model"] = kind
    metrics = pd.DataFrame([metrics])

    oos.name = f"{kind}_predictions_baseline"
    metrics.name = f"{kind}_metrics_baseline"

    save_oos(oos, oos.name)
    save_metrics(metrics, metrics.name)