from src.dataset.load_save import load_csv
from src.analysis.feature_analysis import plot_all
from results.save_results import save_results
from src.analysis.impute_analysis import impute_analysis_plots
'''
Module for feature analysis, data preprocessing analysis, and tuning analysis
Produces figures and metrics regarding featrure importance, missing value imputation impact, and tuning impact
'''

def run(impute_analysis_path="data/sales_globally_imputed.csv", data_path="data/sales_daily_processed.csv", target=None):
    # missing value analysis
    impute_df = load_csv(impute_analysis_path)
    impute_analysis_plots(impute_df)

    # feature analysis
    df = load_csv(data_path)
    uplifts = plot_all(df)
    save_results(uplifts, "sales uplifts")

    # tuning analysis
