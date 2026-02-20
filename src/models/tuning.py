from sklearn.model_selection import ParameterGrid
from typing import Iterator, Mapping, Any
from sklearn.metrics import mean_absolute_error
from src.models.training import make_estimator
import numpy as np, pandas as pd

'''
Defines rolling origin protocol and cross-validation implementation
'''
EXCLUDE = {"date","sales","covers","weather","internal_events","external_events","holiday"}


def feature_cols(df: pd.DataFrame) -> list[str]:
     '''
     Chooses appropriate model input features
     '''
     return [c for c in df.columns if c not in EXCLUDE]


def mae(y_test: pd.Series, 
        y_pred: np.ndarray) -> float:
     '''
     Fold-wise MAE calculation for cross-validation
     '''
     return float(mean_absolute_error(y_test, y_pred))



def rolling_splits(d: pd.Series, 
                   horizon_days: int, 
                   min_train_days: int) -> Iterator[tuple[pd.Series, pd.Series]]:
     '''
     Splits data over specified min training and horizon days using a moving train/test set
     '''
     last_date = d.iloc[-1]

     # first origin with enough history
     first_train_end = d.iloc[0] + pd.Timedelta(days=min_train_days - 1)
     first_test_start = first_train_end + pd.Timedelta(days=1)

     span_days = (last_date - first_test_start).days # exclude initial window to ensure sufficient data

     if span_days < 0:
          raise ValueError("Not enough data for min_train_days + horizon_days")

     n_folds = (span_days + 1) // horizon_days

     for k in range(n_folds): # form train/validation windows forward in time
          test_start = first_test_start + pd.Timedelta(days=(k) * horizon_days)
          test_end = test_start + pd.Timedelta(days=horizon_days - 1) # prevent overlapping folds
          train_end = test_start - pd.Timedelta(days=1)

          train_mask = d <= train_end # boolean mask
          test_mask = d.between(test_start, test_end)

          if train_mask.sum() == 0 or test_mask.sum() == 0:
               raise ValueError("Train or test split has zero rows")
          
          yield train_mask, test_mask # returns each fold until loop ends


def grid_search(train: pd.DataFrame, 
                features: list[str], 
                target: str, 
                kind: str, 
                param_grid: Mapping[str, list[Any]]) -> dict[str, Any]:
     '''
     Determine optimal hyper-parameter combinations using folds returned from rolling_splits
     '''
     train = train.sort_values("date").reset_index(drop=True)
     dates = pd.to_datetime(train["date"])
     X = train[features]
     y = train[target]
     best_score, best_params = np.inf, None

     for params in ParameterGrid(param_grid):
          fold_scores = []
          # cycle through folds
          # set suitable min_training_days and horizon_days
          for train_mask, test_mask in rolling_splits(dates, 28, 197): 
               model = make_estimator(X.loc[train_mask], y.loc[train_mask], kind, params)
               if kind == "sarimax":
                    results = model.fit()
                    drop_cols = [c for c in X.columns if c.startswith("sales_lag") or c.startswith("sales_roll")] # SARIMAX has built-in exog
                    X_sarimax = X.drop(columns=drop_cols)
                    prediction = results.get_forecast(steps=len(X_sarimax.loc[test_mask]), exog=X_sarimax.loc[test_mask]).predicted_mean

               elif kind == "xgboost": # early stopping
                    model.fit(X.loc[train_mask], y.loc[train_mask], eval_set=[(X.loc[test_mask], y.loc[test_mask])], verbose=False)
                    prediction = model.predict(X.loc[test_mask])

               elif kind == "lasso":
                    model.fit(X.loc[train_mask], y.loc[train_mask])
                    prediction = model.predict(X.loc[test_mask])
               fold_scores.append(mae(y.loc[test_mask], prediction))
          avg = float(np.mean(fold_scores))
          if avg < best_score:
               best_score, best_params = avg, params
               
     return {"kind": kind, "score": best_score, "params": best_params}