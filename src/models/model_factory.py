from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np, pandas as pd
'''
A factory returning the right estimator for a specified ML algorithm
'''
EXCLUDE = {"date","sales","covers","weather","internal_events","external_events","holiday"}

# chooses appropriate model input features
def feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE]

# calculates mean absolute percentage error metric for fold parameter combinations???
def mae(test, pred):
     return float(mean_absolute_error(test, pred))

def make_estimator(X_train, y_train, kind, params):
    # pipelines used with linear models to fit StandardScaler (scales data) before fitting the estimator
    # Ridge
    if kind == "lasso":
         return make_pipeline(StandardScaler(), Lasso(random_state=42, **params)) # sets higher penality strength for more bias and less variance
    # XGBoost
    if kind == "xgb":
         return XGBRegressor(random_state=42, n_jobs=-1, **params)
    # SARIMAX - seasonal extension of ARIMA with exogenous factors included
    if kind == "sarimax":
         return SARIMAX(endog=y_train, exog=X_train, **params) # parameters informed by ACF/PACF seasonal/non-seasonal plots, endogenous and exogenous factors included
    raise ValueError(f"Unkown kind: {kind}")


# splits data over specified dates using a moving train/test set
def rolling_splits(d, horizon_days, min_train_days):
     last_date = d.iloc[-1]

     # first origin with enough history
     first_train_end = d.iloc[0] + pd.Timedelta(days=min_train_days - 1) # the last day of the initial traning window
     first_test_start = first_train_end + pd.Timedelta(days=1) # first test window starts the next day after initial training window
     first_test_end = first_test_start + pd.Timedelta(days=horizon_days - 1) # first test window ends after all horizon days have passed

     span_days = (last_date - first_test_end).days # excludes the initial window to ensure it always has sufficient data

     if span_days < 0:
          raise ValueError("Not enough data for min_train_days + horizon_days")

     n_folds = (span_days + 1) // horizon_days # computes correct number of folds given the amount of data provided

     for k in range(n_folds): # forms train/validation windows forward in time using k as the fold counter
          test_start = first_test_start + pd.Timedelta(days=(k) * horizon_days) # start of kth validation window
          test_end = test_start + pd.Timedelta(days=horizon_days - 1) # prevents overlapping folds
          train_end = test_start - pd.Timedelta(days=1)

          train_mask = d <= train_end # boolean mask which selects training portion of fold, an expanding window from d
          test_mask = d.between(test_start, test_end) # boolean mask which selects testing portion of fold

          if train_mask.sum() == 0 or test_mask.sum() == 0: # train/test validation
               raise ValueError("Train or test split has zero rows")
          
          yield train_mask, test_mask # returns each fold until loop ends

# determine optimal hyper-parameter combinations using folds returned from rolling_splits
def grid_search(train, features, target, kind, param_grid):
     train = train.sort_values("date").reset_index(drop=True)
     dates = pd.to_datetime(train["date"])
     X = train[features] # feature matrix
     y = train[target] # target vector
     best_score, best_params = np.inf, None # keeps track of lowest average validation error and its parameters

     for params in ParameterGrid(param_grid): # iterates over each hyper-parameter combination manually
          fold_scores = []
          for train_mask, test_mask in rolling_splits(dates, 7, 28): # rolling windows across training data only using suitable minimum training and horizon days
               model = make_estimator(X.loc[train_mask], y.loc[train_mask], kind, params)
               if kind == "sarimax": # sarimax doesn't use the sklearn interface
                    results = model.fit()
                    prediction = results.get_forecast(steps=len(X.loc[test_mask]), exog=X.loc[test_mask]).predicted_mean

               if kind == "xgboost": # early stopping implementation
                    model.fit(X.loc[train_mask], y.loc[train_mask], early_stopping_rounds=10, eval_metric="mae", eval_set=[(X.loc[test_mask], y.loc[test_mask])], verbose=False)
                    prediction = model.predict(X.loc[test_mask])

               else:
                    model.fit(X.loc[train_mask], y.loc[train_mask]) # training data within training window of fold
                    prediction = model.predict(X.loc[test_mask]) # predict on validation window for fold
               fold_scores.append(mae(y.loc[test_mask], prediction)) # computes MAE for current fold
          avg = float(np.mean(fold_scores)) # uses average metric across folds
          if avg < best_score:
               best_score, best_params = avg, params
               
     return {"kind": kind, "score": best_score, "params": best_params}