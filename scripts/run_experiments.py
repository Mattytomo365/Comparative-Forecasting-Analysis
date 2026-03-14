from src.dataset.load_save import load_csv, load_metrics
from src.analysis.feature_analysis import plot_all
from results.save_results import save_results
from src.analysis.impute_analysis import impute_analysis_plots
from src.analysis.tuning_analysis import tuning_analysis_plots
from src.models.tuning import read_configuration, feature_cols
from src.models.testing import backtest
'''
Module for feature analysis, data preprocessing analysis, and tuning analysis
Produces figures and metrics regarding featrure importance, missing value imputation impact, and tuning impact
'''

def run(impute_analysis_path="data/sales_globally_imputed.csv", data_path="data/sales_daily_processed.csv", target="sales"):
    # OOS. metrics, and dataframe set up
    impute_df = load_csv(impute_analysis_path)
    df = load_csv(data_path)
    features = feature_cols(impute_df)
    impute_oos_list = []
    impute_metrics_list = []
    metrics_baselines = []
    metrics_tuned = []
    models = ["lasso", "sarimax", "xgboost"]
    for model in models:
        params = read_configuration(model)
        impute_oos, impute_metrics = backtest(impute_df, model, features, params, target)
        impute_oos_list.append(impute_oos)
        impute_metrics_list.append(impute_metrics)

        metrics_baseline = load_metrics(f"results/{model}_metrics_baseline.csv")
        metrics_tune = load_metrics(f"results/{model}_metrics_tuned.csv")
        metrics_baselines.append(metrics_baseline)
        metrics_tuned.append(metrics_tune)

    # missing value analysis
    impute_analysis_plots(impute_df, impute_oos_list, impute_metrics_list, models)

    # tuning analysis
    tuning_analysis_plots(df, metrics_baselines, metrics_tuned, models)


    # feature analysis
    # df = load_csv(data_path)
    # uplifts = plot_all(df)
    # save_results(uplifts, "sales uplifts")

if __name__ == "__main__":
    run()
