import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
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






