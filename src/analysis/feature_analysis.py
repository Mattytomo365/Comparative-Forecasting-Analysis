import pandas as pd
import matplotlib.pyplot as plt
from figures.save_figure import save_figure
from src.preprocessing.features import add_test_lag_roll, add_train_lag_roll
from src.models.training import feature_cols, fit_model
from sklearn.inspection import permutation_importance
from typing import Any, Mapping
'''
Explores uplift and feature importance metrics over different periods for different metrics
'''

UPLIFT_COLS = ["holiday", "internal_event", "external_event"] # skips numerical columns for sales uplift calculations
OUT_COLS = ["tag", "n", "avg", "uplift"]

def ablation_plots(metrics_tuned: list[pd.DataFrame],
                ablation_metrics: list[pd.DataFrame],
                models: list[str], 
                folder: str,
                group_name: str) -> None:
    '''
    Visualises differences in MAE between tuned and ablated configurations via dumbell plots
    '''
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, model, ablation_df, tuned_df in zip(axes, models, ablation_metrics, metrics_tuned):
        # retrieve all mae metrics from all folds on tuned and ablated configurations
        fold_1_ablation = ablation_df.loc[(ablation_df["window"] == 1) & (ablation_df["model"] == model), "MAE"]
        fold_1_tuned = tuned_df.loc[(tuned_df["window"] == 1) & (tuned_df["model"] == model), "MAE"]
        fold_2_ablation = ablation_df.loc[(ablation_df["window"] == 2) & (ablation_df["model"] == model), "MAE"]
        fold_2_tuned = tuned_df.loc[(tuned_df["window"] == 2) & (tuned_df["model"] == model), "MAE"]

        ax.bar("fold 1", fold_1_tuned - fold_1_ablation, label="fold 1 delta")
        ax.bar("fold 2", fold_2_tuned - fold_2_ablation, label="fold 2 delta")

        ax.set_title(model)
        ax.set_ylabel("MAE difference")
        ax.set_xlabel("fold")
        ax.axhline(0, color="black", linewidth=1)

    fig.suptitle(f"{group_name} ablation MAE delta bar chart")
    fig.tight_layout()
    save_figure(fig, f"{group_name}_delta_plot", folder)

def permutation_preparation(train: pd.DataFrame, 
                     test: pd.DataFrame, 
                     kind: str, 
                     target: str, 
                     params: Mapping[str, Any]):
    '''
    Ensures lag-aware models are fitted for PFI experiments
    '''
    train_feat = add_train_lag_roll(train.copy())
    features = feature_cols(train_feat)
    fitted_model, _ = fit_model(train_feat, kind, features, target, params)

    history = train.sort_values("date").set_index("date")[target].copy()
    rows = []

    for _, row in test.iterrows():
        row_dict = row.to_dict()
        row_dict.update(add_test_lag_roll(history))
        rows.append(pd.DataFrame([row_dict])[features])
        history = pd.concat([history, pd.Series([row[target]])], ignore_index=True)

    X_test = pd.concat(rows, ignore_index=True)
    y_test = test[target].reset_index(drop=True)
    return fitted_model, X_test, y_test


def permutation_values(fitted_model: Any,
                       X_test: pd.Series,
                       y_test: pd.Series) -> pd.DataFrame:
    '''
    Calculates permutation feature importance values for all features
    '''
    
    r = permutation_importance(fitted_model, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=20, random_state=42, n_jobs=-1)

    mae_increase_mean = -r.importances_mean
    mae_increase_std = r.importances_std

    out = pd.DataFrame({
        "feature": X_test.columns,
        "mae_increase_mean": mae_increase_mean,
        "mae_increase_std": mae_increase_std,
    }).sort_values("mae_increase_mean", ascending=False).reset_index(drop=True)

    return out


def permutation_plot(permutation_values: pd.DataFrame,
                     folder: str,
                     model: str,
                     top_k: int=15) -> None:
    '''
    Places PFI values into a ranked bar chart
    '''
    # take the top 15 features
    permutation_values = permutation_values.sort_values("mae_increase_mean", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(permutation_values["feature"][::-1], permutation_values["mae_increase_mean"][::-1],
            xerr=(permutation_values["mae_increase_std"][::-1]))
    
    ax.set_title(f"PFI mean and std MAE increase ranked bar chart ({model})")
    ax.set_xlabel("Increase in MAE after permutation")
    ax.grid()
    fig.tight_layout()
    save_figure(fig, f"PFI_plot_{model}", folder)