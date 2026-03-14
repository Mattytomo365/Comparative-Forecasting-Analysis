import pandas as pd
import matplotlib.pyplot as plt
from figures.save_figure import save_figure

'''
Evaluate whether hyperparameter tuning yeilds better forecasting accuracy over default configurations for each model family. 
Produce tuned vs. default difference in MAE/RMSE/MASE per fold and averaged table
'''

def delta_plots(metrics_baselines: list[pd.DataFrame], 
                metrics_tuned: list[pd.DataFrame], 
                models: list[str], 
                folder: str) -> None:
    '''
    Visualises differences in MAE between tuned and baseline configurations via dumbell plots
    '''
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, model, baseline_df, tuned_df in zip(axes, models, metrics_baselines, metrics_tuned):
        fold_1_baseline = baseline_df.loc[(baseline_df["window"] == 1) & (baseline_df["model"] == model), "MAE"]
        fold_1_tuned = tuned_df.loc[(tuned_df["window"] == 1) & (tuned_df["model"] == model), "MAE"]
        fold_2_baseline = baseline_df.loc[(baseline_df["window"] == 2) & (baseline_df["model"] == model), "MAE"]
        fold_2_tuned = tuned_df.loc[(tuned_df["window"] == 2) & (tuned_df["model"] == model), "MAE"]

        ax.bar("fold 1", fold_1_baseline - fold_1_tuned, label="fold 1 delta")
        ax.bar("fold 2", fold_2_baseline - fold_2_tuned, label="fold 2 delta")

        ax.set_title(model)
        ax.set_ylabel("MAE difference")
        ax.set_xlabel("fold")
        ax.axhline(0, color="black", linewidth=1)

    fig.suptitle(f"MAE delta bar chart across all models and folds")
    fig.tight_layout()
    save_figure(fig, f"delta_plot", folder)

def tuning_analysis_plots(df: pd.DataFrame, 
                          metrics_baselines: list[pd.DataFrame], 
                          metrics_tuned: list[pd.DataFrame], 
                          models: list[str]) -> None:
    '''
    Centralise the plotting of all tuning analysis related visualisations
    '''
    delta_plots(metrics_baselines, metrics_tuned, models, "tuning_analysis_figures")

