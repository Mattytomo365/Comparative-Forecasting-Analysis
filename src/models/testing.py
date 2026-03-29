import numpy as np, pandas as pd
from typing import Mapping, Any
from src.models.training import train_predict
from src.models.tuning import expanding_splits

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