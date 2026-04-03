from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from typing import Mapping, Any
import warnings
import pandas as pd
import numpy as np
from src.models.metrics import calculate_metrics
from src.preprocessing.features import add_train_lag_roll, add_test_lag_roll
'''
Orchestrates ml model training, tuning, testing, and saving using preprocessed data
'''

EXCLUDE = {"date","sales","internal_events","external_events","holiday"}

def time_split(df: pd.DataFrame, days: int = 56) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Create a single holdout using time-aware split
    '''
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) <= days:
        raise ValueError("Not enough rows to create holdout")

    train = df.iloc[:-days].reset_index(drop=True)
    test = df.iloc[-days:].reset_index(drop=True)
    return train, test


def feature_cols(df: pd.DataFrame) -> list[str]:
     '''
     Chooses appropriate model input features
     '''
     return [c for c in df.columns if (c not in EXCLUDE)]


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
            ("model", Lasso(max_iter=15000, tol=1e-3, selection="cyclic", random_state=42, **params))
        ])
        return lasso_pipe
    # XGBoost
    if kind == "xgboost":
        return XGBRegressor(n_estimators=4000, random_state=42, n_jobs=-1, **params)
    # SARIMAX
    if kind == "sarimax":
        drop_cols = [c for c in X_train.columns if c.startswith("sales_lag") or c.startswith("sales_roll")]
        X_train_sarimax = X_train.drop(columns=drop_cols)
        sarimax_params = {"enforce_stationarity": False, "enforce_invertibility": False, **params}
        return SARIMAX(endog=y_train, exog=X_train_sarimax, **sarimax_params)
    
    raise ValueError(f"Unkown kind: {kind}")


def sarimax_fit_diagnostics(results: Any) -> dict[str, Any]:
    '''
    Extract key convergence diagnostics from fitted SARIMAX results.
    '''
    retvals = getattr(results, "mle_retvals", {}) or {}
    return {k: retvals.get(k) for k in ("converged", "warnflag", "iterations", "fcalls")}


def fit_sarimax_model(model: Any) -> tuple[Any, dict[str, Any]]:
    '''
    Fit SARIMAX with optimizer retries and return best available result + diagnostics.
    '''
    attempts = [
        {"method": "lbfgs", "maxiter": 500, "maxfun": 50000, "disp": False},
        {"method": "lbfgs", "maxiter": 1200, "maxfun": 120000, "disp": False},
        {"method": "powell", "maxiter": 600, "disp": False},
    ]
    fitted_results: list[Any] = []

    for attempt in attempts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            results = model.fit(**attempt)
        diagnostics = sarimax_fit_diagnostics(results) # pulls key optimiser status fields
        diagnostics["method"] = attempt["method"]
        fitted_results.append(results)
        if diagnostics.get("converged"):
            return results, diagnostics

    # fallback: return best likelihood fit even if not fully converged
    best_results = max(fitted_results, key=lambda fit: fit.llf)
    best_diagnostics = sarimax_fit_diagnostics(best_results)
    best_diagnostics["method"] = "best_llf_fallback"
    return best_results, best_diagnostics

def fit_model(train: pd.DataFrame,  
            kind: str, 
            features: list[str], 
            target: str, 
            params: Mapping[str, Any]) -> Any:
    '''
    Fits given model to training dataset
    '''
    if not features:
        raise ValueError("fit_model() received an empty feature list")

    X_train, y_train = train[features], train[target] # manual split
    model = make_estimator(X_train, y_train, kind, params)

    if kind == "sarimax":
        fitted_model, diagnostics = fit_sarimax_model(model)  # data is passed when model is constructed
        print(diagnostics)

    elif kind == "xgboost":
        fitted_model = model.fit(X_train, y_train, verbose=False) # fit on remaining training data

    elif kind == "lasso":
        fitted_model = model.fit(X_train, y_train)
    
    return fitted_model, y_train


def predict_model(X_test: pd.Series,
                  kind: str,
                  fitted_model: Any) -> np.ndarray:
    '''
    Forms sales predictions based off fitted model passed
    '''
    if kind == "sarimax":
        preds = fitted_model.get_forecast(steps=len(X_test), exog=X_test).predicted_mean

    elif kind == "xgboost" or kind == "lasso":
        preds = fitted_model.predict(X_test)

    return preds

def score_model(test: pd.DataFrame,
                y_train: pd.DataFrame,
                y_test: pd.Series,
                preds: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    '''
    Scores model based off predictions provided
    '''
    oos = test[["date"]].copy() # out-of-sample predictions
    oos["actual data"] = y_test
    oos["forecasted data"] = preds
    metrics = calculate_metrics(y_test, y_train, preds) # calculate metrics for current window predicted

    return oos, metrics

def train_predict(train: pd.DataFrame, 
                test: pd.DataFrame,
                kind: str,
                target: str, 
                params: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    '''
    Wraps fitting, predicting, and scoring into one centralised function
    '''
    train = add_train_lag_roll(train)
    features = feature_cols(train)


    fitted_model, y_train = fit_model(train, kind, features, target, params)

    # store sales data history for leakage-free computation of lags and rolls
    history = train.sort_values("date").set_index("date")[target].copy()
    preds = []

    # builds feature vector for each day in the test horizon
    if kind == "lasso" or kind == "xgboost":
        for _, row in test.iterrows():
            row_dict = row.to_dict()
            row_dict.update(add_test_lag_roll(history))
            X_test = pd.DataFrame([row_dict])[features] # the row
            pred = float(predict_model(X_test, kind, fitted_model)[0])
            preds.append(pred)
            history = pd.concat([history, pd.Series([pred])], ignore_index=True)

    else:
        preds = predict_model(test[feature_cols(test)], kind, fitted_model) # SARIMAX

    oos, metrics = score_model(test, y_train, test[target], preds)

    return oos, metrics
