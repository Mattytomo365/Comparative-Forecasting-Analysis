import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Any
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
    save_figure(fig, f"residual_{model}")

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
    save_figure(fig, f"forecast_vs_actual_{stage}")

def metrics_plots(metrics_list: pd.DataFrame, models: list[str], stage: str) -> None:
    '''
    Plot baseline and tuned evaluation metrics against eachother for each model
    '''
    fig, ax = plt.subplots()

    for metrics, model in zip(metrics_list, models):
        metric_1 = metrics[metrics["window"] == 1]
        metric_2 = metrics[metrics["window"] == 2]
        ax.bar(metric_1["MAE"], metric_1["RMSE"], metric_1["MASE"], label=f"window 1 {model} ({stage})")
        ax.bar(metric_2["MAE"], metric_2["RMSE"], metric_2["MASE"], label=f"window 2 {model} ({stage})")

    ax.set_title(f"Metrics for all {stage} models across all folds")
    ax.set_ylabel("value")
    ax.set_xlabel("metric, stage, window")
    ax.legend()
    save_figure(fig, f"metrics_plot_{stage}")

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

