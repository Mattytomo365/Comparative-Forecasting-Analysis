import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from figures.save_figure import save_figure
'''
Responsible for plotting and handling residual plots across all models
Compares plots
'''


def plot_residuals(oos: pd.DataFrame, model: Any, out: pd.DataFrame) -> None:
    '''
    Plot model residuals to analyse accuracy across folds
    '''
    # plot using a dataframe of all predicted data across folds and unseen test data
    df = oos[oos["model"]==model].copy()
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["residual"])
    ax.set_title(f"OOS residuals over time - {model}")
    ax.set_ylabel = ("actual values - forecasted values")
    ax.set_xlabel = ("date")
    save_figure(fig, f"residual_{model}")
    plt.close(fig)