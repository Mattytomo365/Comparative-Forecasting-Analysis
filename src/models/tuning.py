from sklearn.model_selection import ParameterGrid
from pathlib import Path
from typing import Iterator, Mapping, Any
from src.models.training import train_predict
import numpy as np, pandas as pd
import json

'''
Defines rolling origin protocol and cross-validation implementation
'''

def expanding_splits(dates: pd.Series, 
                   horizon_days: int, 
                   min_train_days: int) -> Iterator[tuple[pd.Series, pd.Series]]:
     '''
     Splits data over specified min training and horizon days using a moving train/test set
     '''
     last_date = dates.iloc[-1]

     # first origin with enough history
     first_train_end = dates.iloc[0] + pd.Timedelta(days=min_train_days - 1)
     first_test_start = first_train_end + pd.Timedelta(days=1)

     span_days = (last_date - first_test_start).days # exclude initial window to ensure sufficient data

     if span_days < 0:
          raise ValueError("Not enough data for min_train_days + horizon_days")

     n_folds = (span_days + 1) // horizon_days

     for k in range(n_folds): # form train/validation windows forward in time
          test_start = first_test_start + pd.Timedelta(days=(k) * horizon_days)
          test_end = test_start + pd.Timedelta(days=horizon_days - 1) # prevent overlapping folds
          train_end = test_start - pd.Timedelta(days=1)

          train_mask = dates <= train_end # boolean mask
          test_mask = dates.between(test_start, test_end)

          if train_mask.sum() == 0 or test_mask.sum() == 0:
               raise ValueError("Train or test split has zero rows")
          
          yield train_mask, test_mask # returns each fold until loop ends


def grid_search(train: pd.DataFrame, 
                target: str, 
                kind: str, 
                param_grid: Mapping[str, list[Any]]) -> dict[str, Any]:
     '''
     Determine optimal hyper-parameter combinations using folds returned from rolling_splits
     '''
     train = train.sort_values("date").reset_index(drop=True)
     dates = pd.to_datetime(train["date"])
     best_score, best_params = np.inf, None

     for params in ParameterGrid(param_grid):
          fold_scores = []
          # cycle through folds
          # set suitable min_training_days and horizon_days
          horizon_days = 28
          min_training_days = 197
          
          for train_mask, test_mask in expanding_splits(dates, horizon_days, min_training_days):
               train_fold = train.loc[train_mask].copy()
               test_fold = train.loc[test_mask].copy()
               _, metrics = train_predict(train_fold, test_fold, kind, target, params)
               fold_scores.append(metrics["MAE"])

          avg = float(np.mean(fold_scores))

          if avg < best_score:
               best_score, best_params = avg, params
               
     return {"kind": kind, "score": best_score, "params": best_params}

def save_configuration(params: dict[str, float, float]) -> None: 
    '''
    Writes tuned model hyperparameter configurations to assist with imputation analysis
    '''
    kind = params["kind"]
    configuration = params["params"]
    
    with open((f"model_info/{kind}_best_params.json"), "w") as f:
        json.dump(configuration, f, indent=2)

def read_configuration(kind: str) -> dict[str, float, float]:
    '''
    Reads manifest JSON file - used for displaying model information to user
    '''
    path = Path(f"model_info/{kind}_best_params.json")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise (f"Invalid JSON in {path}: {e}") from e
    return data
