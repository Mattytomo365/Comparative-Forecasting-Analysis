import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Any
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from figures.save_figure import save_figure
from src.dataset.load_save import load_csv, load_metrics
'''
Responsible for plotting and handling residual plots across all models
Compares plots
'''
def daily_labels(ax) -> None:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", rotation=30)

def plot_residuals(oos_baseline: pd.DataFrame, oos_tuned: pd.DataFrame, model: str) -> None:
    '''
    Plot model residuals to analyse accuracy across folds
    '''
    fig, ax = plt.subplots()
    ax.scatter(oos_baseline["date"], oos_baseline["actual data"] - oos_baseline["forecasted data"], label="baseline") # baseline residuals
    ax.scatter(oos_tuned["date"], oos_tuned["actual data"] - oos_tuned["forecasted data"], label="tuned") # tuned residuals

    ax.set_title(f"OOS residuals over time - {model}")
    ax.set_ylabel("actual values - forecasted values")
    ax.set_xlabel("date")
    ax.legend()
    daily_labels(ax)
    fig.tight_layout(pad=1.2)
    save_figure(fig, f"residual_{model}", "evaluation_figures")

def forecast_plot(oos_list: list[pd.DataFrame], models: list[str], stage: str) -> None:
    '''
    Plot forecasted data vs. actual data
    '''
    fig, ax = plt.subplots()
    ref = oos_list[0].sort_values("date")
    ax.plot(ref["date"], ref["actual data"], label ='actual', color="black", linewidth=2)
    for model, oos in zip(models, oos_list) :
        ax.plot(oos["date"], oos["forecasted data"], '-.', label =f"forecasted - {model}")

    ax.set_ylabel("sales")
    ax.set_xlabel("date")
    ax.legend()
    ax.set_title(f"forecast vs. actual data ({stage})")
    daily_labels(ax)
    fig.tight_layout(pad=1.2)
    save_figure(fig, f"forecast_vs_actual_{stage}", "evaluation_figures")

def metrics_plots(metrics_list: list[pd.DataFrame], models: list[str], stage: str) -> None:
    '''
    Plot baseline and tuned evaluation metrics against eachother for each model
    '''
    metric_cols = ["MAE", "RMSE", "MASE"]

    if len(metrics_list) != len(models):
        raise ValueError("metrics_list and models must have same length")
    
    x = np.arange(len(models)) # 2 folds per metric
    width = 0.25
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, metric in zip(axes, metric_cols):
        fold_1 = [df.loc[df["window"] == 1, metric].iloc[0] for df in metrics_list]
        fold_2 = [df.loc[df["window"] == 2, metric].iloc[0] for df in metrics_list]
        ax.bar(x - width / 2, fold_1, width, label="fold 1")
        ax.bar(x + width / 2, fold_2, width, label="fold 2")

        ax.set_title(metric)
        ax.set_ylabel("value")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_xlabel("model")
        ax.legend()

    fig.suptitle(f"Metrics for all {stage} models across both folds")
    fig.tight_layout()
    save_figure(fig, f"metrics_plot_{stage}", "evaluation_figures")


def absolute_error_plot(oos_list: list[pd.DataFrame], models: list[str], stage: str) -> None:
    '''
    Plot absolute error over time for visualisation of magnitude
    '''
    fig, ax = plt.subplots()
    for model, oos in zip(models, oos_list) :
        oos["error"] = oos["actual data"] - oos["forecasted data"]
        oos["abs_error"] = oos["error"].abs()
        ax.plot(oos["date"], oos["abs_error"], '-.', label =f"forecasted - {model}")

    ax.set_ylabel("magnitude")
    ax.set_xlabel("date")
    ax.legend()
    ax.set_title(f"Absolute error over time ({stage})")
    daily_labels(ax)
    fig.tight_layout(pad=1.2)
    save_figure(fig, f"absolute_error_{stage}", "evaluation_figures")


def ranked_summary(oos_list: list[pd.DataFrame], models: list[str], primary: str = "mae") -> pd.DataFrame:
    '''
    Generate a ranked table for summary of findings
    '''
    # calculate overall metrics for entire holdout
    rows = []
    for oos, model in zip(oos_list, models):
        y_test = oos["actual data"]
        y_pred = oos["forecasted data"]

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        # me = float(np.mean(y_true - y_pred))  # bias
        
        rows.append({
            "model": model,
            "mae": mae,
            "rmse": rmse,
            #"me": me
        })

    summary = pd.DataFrame(rows)

    summary["rank"] = summary[primary].rank(method="dense", ascending=True).astype(int)

    return summary.sort_values(["rank", "model"]).reset_index(drop=True)

def ranked_table(summary: pd.DataFrame, stage: str) -> None:
    '''
    Visualises ranked table
    ''' 
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.table(cellText=summary.values, colLabels=summary.columns, loc="center")
    ax.set_title(f"Ranked metrics table ({stage})")
    fig.tight_layout()
    save_figure(fig, f"metrics_table_{stage}", "evaluation_figures")


def plot_all() -> None:
    '''
    Centralises execution of all comparative experiments
    '''
    models = ["lasso", "sarimax", "xgboost"]
    oos_baselines = []
    oos_tuned = []
    metrics_baselines = []
    metrics_tuned = []

    for model in models:

        oos_baseline = load_csv(f"results/{model}_predictions_baseline.csv")
        oos_tune = load_csv(f"results/{model}_predictions_tuned.csv")
        oos_baselines.append(oos_baseline)
        oos_tuned.append(oos_tune)
        plot_residuals(oos_baseline, oos_tune, model)

        metrics_baseline = load_metrics(f"results/{model}_metrics_baseline.csv")
        metrics_tune = load_metrics(f"results/{model}_metrics_tuned.csv")
        metrics_baselines.append(metrics_baseline)
        metrics_tuned.append(metrics_tune)


    forecast_plot(oos_baselines, models, "baselines")
    forecast_plot(oos_tuned, models, "tuned")

    metrics_plots(metrics_baselines, models, "baseline")
    metrics_plots(metrics_tuned, models, "tuned")

    absolute_error_plot(oos_baselines, models, "baselines")
    absolute_error_plot(oos_tuned, models, "tuned")

    baselines_ranked = ranked_summary(oos_baselines, models)
    tuned_ranked = ranked_summary(oos_tuned, models)

    ranked_table(baselines_ranked, "baselines")
    ranked_table(tuned_ranked, "tuned")

