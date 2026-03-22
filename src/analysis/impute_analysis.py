import pandas as pd
from src.analysis.performance_analysis import metrics_plots, absolute_error_plot, ranked_summary, ranked_table
from src.analysis.eda import seasonal_curve
'''
Determines whether imputing missing sales values improves forecast accuracy and stability compared to dropping missing observations. 
Reports difference in MAE/RMSE/MASE across rolling-origin folds between imputation and row removal.
'''

def impute_analysis_plots(impute_df: pd.DataFrame,
                        oos_list: list[pd.DataFrame], 
                        metrics_list: list[pd.DataFrame],
                        models: list[str]) -> None:
    '''
    Centrlalise plotting of metrics plots for analysis of imputation strategy
    '''
    metrics_plots(metrics_list, models, "tuned", "imputation_analysis_figures")

    absolute_error_plot(oos_list, models, "tuned", "imputation_analysis_figures")

    tuned_ranked = ranked_summary(oos_list, models)

    ranked_table(tuned_ranked, "tuned", "imputation_analysis_figures")

    seasonal_curve(impute_df, "sales", 2, "Sales seasonal curve (global median imputation)", "fourier_seasonal_imputation", "imputation_analysis_figures")
