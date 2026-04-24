import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

def residual_plots(oos_baseline: pd.DataFrame, 
                   oos_tuned: pd.DataFrame, 
                   model: str, 
                   folder: str) -> None:
    '''
    Plot model residuals to analyse accuracy across tuned and baseline configurations
    '''
    fig, ax = plt.subplots()
    ax.scatter(oos_baseline["date"], oos_baseline["actual data"] - oos_baseline["forecasted data"], label="baseline") # baseline residuals
    ax.scatter(oos_tuned["date"], oos_tuned["actual data"] - oos_tuned["forecasted data"], label="tuned") # tuned residuals
    ax.axhline(0, color="black", linewidth=1)

    ax.set_title(f"OOS residuals over time - {model}")
    ax.set_ylabel("actual values - forecasted values")
    ax.set_xlabel("date")
    ax.legend()
    daily_labels(ax)
    fig.tight_layout(pad=1.2)
    save_figure(fig, f"residual_{model}", folder)

def forecast_plot(oos_list: list[pd.DataFrame], 
                  models: list[str], 
                  stage: str, 
                  folder: str) -> None:
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
    save_figure(fig, f"forecast_vs_actual_{stage}", folder)

def metrics_plots(metrics_list: list[pd.DataFrame], 
                  models: list[str], 
                  title: str,
                  file_name: str, 
                  folder: str) -> None:
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

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, file_name, folder)


def absolute_error_plot(oos_list: list[pd.DataFrame], 
                        models: list[str], 
                        title: str,
                        file_name: str, 
                        folder: str) -> None:
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
    ax.set_title(title)
    daily_labels(ax)
    fig.tight_layout(pad=1.2)
    save_figure(fig, file_name, folder)


def ranked_summary(metrics_list: list[pd.DataFrame], 
                models: list[str], 
                primary: str = "mae") -> pd.DataFrame:
    '''
    Generate a ranked table for summary of findings
    '''
    metric_cols = ["MAE", "RMSE", "MASE"]
    
    # Average fold-wise metrics
    rows = []
    for model, df in zip(models, metrics_list):
        model_metrics = df.loc[df["model"] == model]
        rows.append(
            {
                "model": model,
                "mae": model_metrics["MAE"].mean(),
                "rmse": model_metrics["RMSE"].mean(),
                "mase": model_metrics["MASE"].mean(),
            }
        )

    summary = pd.DataFrame(rows)
    summary["rank"] = summary[primary].rank(method="dense", ascending=True).astype(int)

    return summary.sort_values(["rank", "model"]).reset_index(drop=True)

def ranked_table(summary: pd.DataFrame, 
                 title: str,
                 file_name: str, 
                 folder: str) -> None:
    '''
    Visualises ranked table
    ''' 
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.table(cellText=summary.values, colLabels=summary.columns, loc="center")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, file_name, folder)


def plot_all() -> None:
    '''
    Centralises execution of all comparative experiments
    '''
    all_models = ["naive", "lasso", "sarimax", "xgboost"]
    oos_baselines = []
    oos_tuned = []
    metrics_baselines = []
    metrics_tuned = []

    for model in all_models:
        if model == "naive":
            oos_naive = load_csv(f"results/{model}_predictions_baseline.csv")
            metrics_naive = load_metrics(f"results/{model}_metrics_baseline.csv")
            oos_baselines.append(oos_naive)
            oos_tuned.append(oos_naive)
            metrics_baselines.append(metrics_naive)
            metrics_tuned.append(metrics_naive)
        
        else:
            oos_baseline = load_csv(f"results/{model}_predictions_baseline.csv")
            oos_baselines.append(oos_baseline)
            oos_tune = load_csv(f"results/{model}_predictions_tuned.csv")
            oos_tuned.append(oos_tune)

            metrics_baseline = load_metrics(f"results/{model}_metrics_baseline.csv")
            metrics_baselines.append(metrics_baseline)
            metrics_tune = load_metrics(f"results/{model}_metrics_tuned.csv")
            metrics_tuned.append(metrics_tune)
            
            residual_plots(oos_baseline, oos_tune, model, "evaluation_figures")

    

    forecast_plot(oos_baselines, all_models, "baselines", "evaluation_figures")
    forecast_plot(oos_tuned, all_models, "tuned", "evaluation_figures")

    metrics_plots(metrics_baselines, all_models, "Metrics for all baseline models", "metrics_plot_baseline", "evaluation_figures")
    metrics_plots(metrics_tuned, all_models, "Metrics for all tuned models", "metrics_plot_tuned", "evaluation_figures")

    absolute_error_plot(oos_baselines, all_models, "Absolute error over time for all baseline models", "absolute_error_baseline", "evaluation_figures")
    absolute_error_plot(oos_tuned, all_models, "Absolute error over time for all tuned models", "absolute_error_tuned", "evaluation_figures")

    baselines_ranked = ranked_summary(metrics_baselines, all_models)
    tuned_ranked = ranked_summary(metrics_tuned, all_models)

    ranked_table(baselines_ranked, "Ranked metrics table (baselines)", "metrics_table_baseline", "evaluation_figures")
    ranked_table(tuned_ranked, "Ranked metrics table (tuned)", "metrics_table_tuned", "evaluation_figures")

