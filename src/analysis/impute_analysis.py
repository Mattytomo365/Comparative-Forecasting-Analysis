import pandas as pd
from src.analysis.performance_analysis import metrics_plots, absolute_error_plot, ranked_summary, ranked_table
from src.analysis.eda import seasonal_curve, acf_plots
'''
Determines whether imputing missing sales values improves forecast accuracy and stability compared to dropping missing observations. 
Reports difference in MAE/RMSE/MASE across rolling-origin folds between imputation and row removal.
'''

def impute_analysis_plots(global_median_df: pd.DataFrame, 
                        dow_mean_df: pd.DataFrame, 
                        median_oos_list: list[pd.DataFrame], 
                        median_metrics_list: list[pd.DataFrame], 
                        mean_oos_list: list[pd.DataFrame], 
                        mean_metrics_list: list[pd.DataFrame],
                        models: list[str]) -> None:
    '''
    Centrlalise plotting of metrics plots for analysis of imputation strategy
    '''
    metrics_plots(median_metrics_list, models, "Metrics plots for tuned models with global median imputation", "metrics_plots_median_impute", "imputation_analysis_figures")
    metrics_plots(mean_metrics_list, models, "Metriccs plots for tuned models with dow mean imputation", "metrics_plots_mean_impute", "imputation_analysis_figures")

    absolute_error_plot(median_oos_list, models, "Absolute error over time for tuned models with global median imputation", "absolute_error_median_impute", "imputation_analysis_figures")
    absolute_error_plot(mean_oos_list, models, "Absolute error over time for tuned models with dow mean imputation", "absolute_error_mean_impute", "imputation_analysis_figures")

    median_ranked = ranked_summary(median_metrics_list, models)
    mean_ranked = ranked_summary(mean_metrics_list, models)

    ranked_table(median_ranked, "Ranked metrics table (tuned models with global median imputation)", "metrics_table_median_impute", "imputation_analysis_figures")
    ranked_table(mean_ranked, "Ranked metrics table (tuned models with dow mean imputation)", "metrics_table_mean_impute", "imputation_analysis_figures")

    seasonal_curve(global_median_df, "sales", 2, "Sales seasonal curve (global median imputation)", "fourier_seasonal_median_impute", "imputation_analysis_figures")
    seasonal_curve(dow_mean_df, "sales", 2, "Sales seasonal curve (dow mean imputation)", "fourier_seasonal_mean_impute", "imputation_analysis_figures")

    acf_plots(global_median_df, "sales", 1, "acf_sales_median_impute", "1st Order Non-Seasonal Differencing (global median imputation)", "imputation_analysis_figures")
    acf_plots(dow_mean_df, "sales", 1, "acf_sales_mean_impute", "1st Order Non-Seasonal Differencing (dow mean imputation)", "imputation_analysis_figures")