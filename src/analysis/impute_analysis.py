import pandas as pd
from src.analysis.performance_analysis import metrics_plots, absolute_error_plot, ranked_summary, ranked_table
'''
Determines whether imputing missing sales values improves forecast accuracy and stability compared to dropping missing observations. 
Reports difference in MAE/RMSE/MASE across rolling-origin folds between imputation and row removal.
'''

def impute_analysis_plots(oos_list: list[pd.DataFrame], 
                          metrics_list: list[pd.DataFrame],
                          models: list[str]) -> None:
    '''
    Centrlalise plotting of metrics plots for analysis of imputation strategy
    '''
    metrics_plots(metrics_list, models, "tuned", "imputation_analysis_figures")

    absolute_error_plot(oos_list, models, "tuned", "imputation_analysis_figures")

    tuned_ranked = ranked_summary(oos_list, models)

    ranked_table(tuned_ranked, "tuned", "imputation_analysis_figures")
