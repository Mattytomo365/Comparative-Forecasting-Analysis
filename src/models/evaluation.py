import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from .training import train_model
from .model_factory import rolling_splits
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
'''
Model performance evaluation using regression metrics and visualisations
'''

def calculate_metrics(y_test, y_train, pred):

    # mean absolute error
    mae = float(mean_absolute_error(y_test, pred))
    # root mean squared error
    rmse = float(root_mean_squared_error(y_test, pred))
    # mean absolute scaled error
    mase = mean_absolute_scaled_error(y_test, pred, y_train=y_train)

    return {"MAE": mae, "MASE": mase, "RMSE": rmse}

# implements naive baseline for forecasting performance benchmark
def naive_forecast(y_train, y_test):
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    preds = np.empty_like(y_test, dtype=float)
    preds[0] = y_train[-1] # predict first test point using last train value
    preds[1:] = y_test[:-1] # then predict using previous actual (1-step naive)

    metrics = calculate_metrics(y_test, y_train, preds)
    return metrics, preds

# orchestrate rolling-origin/expanding-window backtesting using established outer windows
def backtest(df, kind, features, parameters, target):
    oos_all, metrics_all = [], []
    window = 1

    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])

    for train_mask, test_mask in rolling_splits(dates, 28, 180): # rolling outer windows using suitable minimum training and horizon days
        train, test = df.loc[train_mask], df.loc[test_mask]
        oos, metric, model = train_model(train, test, kind, features, target, parameters) # train model
        oos["model"] = kind
        oos["window"] = window
        metric["model"] = kind
        metric["window"] = window

        oos_all.append(oos) # contains all outer window predictions
        metrics_all.append(metric) # contains metrics for all outer window predictions
        window = window + 1

    return pd.concat(oos_all, ignore_index=True), pd.DataFrame(metrics_all), model

# save oos dataframe as csv and return path for model registry
def save_oos(oos, name):
    path_csv = f"results/{name}.csv"
    oos.to_csv(path_csv, index=False)
    return path_csv

# save metrics dataframe as csv and return path for model registry
def save_metrics(metrics, name):
    path = f"results/{name}.csv"
    metrics.to_csv(path, index=False)
    return path





