from src.models.tuning import grid_search, feature_cols
from src.models.evaluation import backtest, save_metrics, save_oos
from src.dataset.load_save import load_csv
from src.models.registry import save_manifest
from src.models.training import time_split

'''
Module for model tuning, testing, and training functions
'''

def run(data_path="data/sales_daily_processed.csv", target="sales"):
    df = load_csv(data_path)
    features = feature_cols(df)
    train_df, holdout_df = time_split(df) # simple holdout splitter
    

    # train on suitable default parameter combinations first to give performance baselines
    for kind, default_params in[
        ("lasso", {"alpha": 1.0}), # Lasso
        ("sarimax", {"order": (0, 1, 1), "seasonal_order": (0, 1, 1, 7)}), # SARIMA
        ("xgboost", {"max_depth": 4, "learning_rate": 0.05}) # XGBoost
    ]:
        oos, metrics, model = backtest(df, kind, features, default_params, target)
        oos.name = f"{kind}_predictions_baseline"
        metrics.name = f"{kind}_metrics_baseline"
        oos_path = save_oos(oos, oos.name)
        metrics_path = save_metrics(metrics, metrics.name)
        save_manifest(kind, "baseline", target, features, default_params, oos_path, metrics_path, model) # baseline model manifests

    # define param grids for each model type
    Grids = {
        "lasso": { # Lasso
            "alpha": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], # controls penalty strength
        },

        "sarimax": { # SARIMAX
            "order": [(0, 1, 1), (1, 1, 1)],
            "seasonal_order": [(0,1,1,7), (1,1,1,7)],
        },

        "xgboost": { # XGBoost
            "learning_rate": [0.03, 0.08],
            "max_depth": [3, 5],
            "min_child_weight": [1, 10],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1.0, 10.0],
        },
    }

    for kind, grid in Grids.items():
        # tune on train only
        best = grid_search(train_df, features, target, kind, param_grid=grid)
        print("Best CV: ", best)

        # refit on train with best hyper-parameters
        oos, metrics, model = backtest(df, kind, features, best["params"], target)
        oos.name = f"{kind}_predictions_tuned"
        metrics.name = f"{kind}_metrics_tuned"
        oos_path = save_oos(oos, oos.name)
        metrics_path = save_metrics(metrics, metrics.name)
        save_manifest(kind, "tuned", target, features, best["params"], oos_path, metrics_path, model) # tuned model manifests

        print("Model saved")



if __name__ == "__main__":
    run()