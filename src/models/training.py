from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Mapping, Any
import pandas as pd
'''
Orchestrates ml model training, tuning, testing, and saving using preprocessed data
'''

def time_split(df: pd.DataFrame, days=56) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Create a single holdout using time-aware split
    '''
    if df is None or len(df) == 0: # enforce datetime
        raise ValueError("Empty dataframe provided to time_split")
    cutoff = df["date"].max() - pd.Timedelta(days)
    train = df[df["date"] <= cutoff].reset_index(drop=True)
    holdout = df[df["date"] >= cutoff].reset_index(drop=True)
    return train, holdout


def make_estimator(X_train: pd.DataFrame, 
                   y_train: pd.Series, 
                   kind: str, 
                   params: Mapping[str, Any]) -> Any:
    '''
    Creates a machine learning model constructors used to fit onto training data based on the kind of model provided
    '''
    # Lasso
    if kind == "lasso":
         lasso_pipe = Pipeline([ # lasso pipeline
               ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
               ("scaler", StandardScaler()),
               ("model", Lasso(max_iter=2000, selection="random", random_state=42, **params))
          ])
         return lasso_pipe
    # XGBoost
    if kind == "xgboost":
         return XGBRegressor(n_estimators=4000, random_state=42, n_jobs=-1, **params, early_stopping_rounds=10)
    # SARIMAX
    if kind == "sarimax":
          drop_cols = [c for c in X_train.columns if c.startswith("sales_lag") or c.startswith("sales_roll")]
          X_train_sarimax = X_train.drop(columns=drop_cols)
          return SARIMAX(endog=y_train, exog=X_train_sarimax, **params)
    raise ValueError(f"Unkown kind: {kind}")


def train_model(train: pd.DataFrame, 
                test: pd.DataFrame, 
                kind: str, 
                features: list[str], 
                target: str, 
                params: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, float], Any]:
    '''
    Trains and tests specified model with specified parameters for the specified window of data
    '''
    if len(features) == 0:
        raise ValueError("No training features found after applying exclude set")

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    model = make_estimator(X_train, y_train, kind, params)

    if kind == "sarimax":
        results = model.fit()
        drop_cols = [c for c in X_test.columns if c.startswith("sales_lag") or c.startswith("sales_roll")]
        X_test_sarimax = X_test.drop(columns=drop_cols)
        preds = results.get_forecast(steps=len(X_test_sarimax), exog=X_test_sarimax).predicted_mean

    elif kind == "xgboost": # early stopping
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        preds = model.predict(X_test)

    elif kind == "lasso":
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    oos = test[["date"]].copy() # out-of-sample predictions
    oos["actual data"] = y_test
    oos["forecasted data"] = preds
    from src.models.evaluation import calculate_metrics
    metrics = calculate_metrics(y_test, y_train, preds) # calculate metrics for current window predicted
    return oos, metrics, model
