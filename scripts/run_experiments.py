from src.dataset.load_save import load_csv, load_metrics
from src.analysis.feature_analysis import ablation_plots, permutation_values, permutation_plot, permutation_preparation
from src.analysis.impute_analysis import impute_analysis_plots
from src.analysis.tuning_analysis import delta_plots, tuning_residuals
from src.models.tuning import read_configuration
from src.models.testing import backtest
from src.models.training import time_split, fit_model, feature_cols
from results.save_results import save_results
'''
Module for feature analysis, data preprocessing analysis, and tuning analysis
Produces figures and metrics regarding featrure importance, missing value imputation impact, and tuning impact
'''

def run(data_path="data/sales_daily_processed.csv", target="sales"):
    # OOS, metrics, and feature configuration set up
    global_median_df = load_csv("data/sales_globally_imputed.csv")
    dow_mean_df = load_csv("data/sales_dow_mean_imputed.csv")
    df = load_csv(data_path)
    features = feature_cols(df)


    # oos predictions + metrics for impute analysis
    median_oos_list = []
    mean_oos_list = []
    median_metrics_list = []
    mean_metrics_list = []

    # oos predictions + metrics for tuning analysis
    metrics_baselines = []
    metrics_tuned = []

    # metrics and feature groups for feature analysis
    ABLATION_FEATURE_GROUPS = { # feature groups for ablation experiments
    "calendar": [c for c in features if c.startswith(("dow_", "month_", "doy_"))],
    "events": [c for c in features if c.startswith(("internal_events_", "internal_event_", "external_events_", "external_event_"))],
    "holidays": [c for c in features if c.startswith("holiday__")],
    "weather": [c for c in features if c.startswith(("precipitation_", "temperature_", "wind_"))],
    }
    
    metrics_ablations = {group_name: [] for group_name in ABLATION_FEATURE_GROUPS}


    # re-train, and evaluate models for each experiment using persisted optimal parameter configurations
    models = ["lasso", "sarimax", "xgboost"]
    for kind in models:
        params = read_configuration(kind)

        # impute analysis
        median_oos, median_metrics = backtest(global_median_df, kind, target, params)
        median_oos_list.append(median_oos)
        median_metrics_list.append(median_metrics)

        mean_oos, mean_metrics = backtest(dow_mean_df, kind, target, params)
        mean_oos_list.append(mean_oos)
        mean_metrics_list.append(mean_metrics)

        # metrics and oos predictions for tuning and feature analysis
        metrics_baseline = load_metrics(f"results/{kind}_metrics_baseline.csv")
        metrics_tune = load_metrics(f"results/{kind}_metrics_tuned.csv")
        metrics_baselines.append(metrics_baseline)
        metrics_tuned.append(metrics_tune)
        oos_baseline = load_csv(f"results/{kind}_predictions_baseline.csv")
        oos_tune = load_csv(f"results/{kind}_predictions_tuned.csv")
        tuning_residuals(oos_baseline, oos_tune, kind, "tuning_analysis_figures")

        # feature analysis
        # grouped ablation experiments
        for group_name, feature_group in ABLATION_FEATURE_GROUPS.items():
            df_ablated = df.drop(columns=feature_group)
            _, ablation_metrics = backtest(df_ablated, kind, target, params)
            metrics_ablations[group_name].append(ablation_metrics)

        # grouped PFI experiments
        train, test = time_split(df)
        if kind == "lasso" or kind == "xgboost":
            fitted_model, X_test, y_test = permutation_preparation(train, test, kind, target, params)
            permutation_table = permutation_values(fitted_model, X_test, y_test)
            save_results(permutation_table, f"permutation_values_{kind}")
            permutation_plot(permutation_table, "feature_analysis_figures", kind)
        else:
            pass
            


    # impute analysis figures
    impute_analysis_plots(global_median_df, dow_mean_df, median_oos_list, median_metrics_list, mean_oos_list, mean_metrics_list, models)

    # tuning analysis delta plot
    delta_plots(metrics_baselines, metrics_tuned, models, "tuning_analysis_figures")

    # feature analysis results and figures
    for group_name, group_metrics in metrics_ablations.items():
        ablation_plots(metrics_tuned, group_metrics, models, "feature_analysis_figures", group_name)

if __name__ == "__main__":
    run()
