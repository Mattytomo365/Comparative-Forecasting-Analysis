import pandas as pd
from src.analysis.performance_analysis import metrics_plots, absolute_error_plot, ranked_summary, ranked_table
from src.analysis.eda import seasonal_curve, acf_plots
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
    metrics_plots(metrics_list, models, "imputation", "imputation_analysis_figures")

    absolute_error_plot(oos_list, models, "imputation", "imputation_analysis_figures")

    tuned_ranked = ranked_summary(oos_list, models)

    ranked_table(tuned_ranked, "imputation", "imputation_analysis_figures")

    seasonal_curve(impute_df, "sales", 2, "Sales seasonal curve (different imputation strategy)", "fourier_seasonal_imputation", "imputation_analysis_figures")

    acf_plots(impute_df, "sales", 1, "acf_sales_imputation", "1st Order Non-Seasonal Differencing (different imputation strategy)", "imputation_analysis_figures")
    acf_plots(impute_df, "sales", 7, "acf_sales_seasonal_imputation", "Seasonal Differencing with Period 7 (different imputation strategy)", "imputation_analysis_figures")
