import pandas as pd
import matplotlib.pyplot as plt
from figures.save_figure import save_figure
from sklearn.inspection import permutation_importance
from typing import Any
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


    

def uplifts(df: pd.DataFrame, 
            factor: str, 
            month: str,
            metric: str, 
            sep=";") -> pd.DataFrame:
    '''
    Compute uplift of specified factors against specified metric
    '''
    d = df["date"]
    m = d.dt.month.eq(int(month)) # boolean mask

    if not m.any(): # fallback
        return pd.DataFrame(columns=OUT_COLS)
    
    sub = df.loc[m].copy()
    s = sub[factor].fillna("").astype(str).str.strip().str.lower()

    if factor in UPLIFT_COLS:
        tags_list = s.apply(lambda val: [tag.strip() for tag in val.split(sep) if tag.strip() and tag.strip() != "none"]) # split multi-tag strings into list
        base_mask = tags_list.str.len().eq(0) # marks baseline days as days without tags
        baseline = (sub.loc[base_mask])[metric].mean() # calculates baseline from marked days

        sub["tags"] = tags_list

        # explode tags to individual rows
        sub = (sub.explode("tags").rename(columns={"tags": "tag"}))
        sub = sub[["date", "tag", metric]].drop_duplicates(subset=["date", "tag"])

        sub["baseline"] = baseline
        sub = sub.loc[~base_mask & sub["baseline"].notna()]

        if sub.empty or pd.isna(baseline) or baseline == 0: # guards uplift calculation against NaN and 0
            return pd.DataFrame(columns=OUT_COLS)
        
    else:
        return pd.DataFrame(columns=OUT_COLS)
        
    # calculate percentage uplift per row against baseline
    sub["uplift_row"] = 100.0 * (sub[metric] - sub["baseline"]) / sub["baseline"]

    # aggregate per tag
    tab = (sub.groupby("tag")
        .agg(n=("date", "nunique") # unique days tag occurs
             , avg=(metric, "mean") # mean metric on tagged days
             , uplift=("uplift_row", "mean")) # mean % uplift across occurrences
        .reset_index().sort_values("avg", ascending=False))
    
    return pd.DataFrame(tab[OUT_COLS])

# def plot_all(df: pd.DataFrame) -> pd.DataFrame:
#     '''
#     Percentage uplift for specified factor, month, and metric
#     '''
#     all_uplifts = []
#     for factor in UPLIFT_COLS:
#         for month in range(1, 13):
#             uplift = uplifts(df, factor, month, "sales")
#             uplift["factor"] = factor
#             uplift["month"] = month
#             all_uplifts.append(uplift)
#     return pd.concat(all_uplifts, ignore_index=True)