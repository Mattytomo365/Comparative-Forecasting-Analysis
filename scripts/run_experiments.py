from src.dataset.load_save import load_csv, load_metrics
from src.analysis.feature_analysis import ablation_plots, uplifts, permutation_values, permutation_plot
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

def run(impute_analysis_path="data/sales_globally_imputed.csv", data_path="data/sales_daily_processed.csv", target="sales"):
    # OOS, metrics, and feature configuration set up
    impute_df = load_csv(impute_analysis_path)
    df = load_csv(data_path)
    features = feature_cols(df)


    # oos predictions + metrics for impute analysis
    impute_oos_list = []
    impute_metrics_list = []

    # oos predictions + metrics for tuning analysis
    metrics_baselines = []
    metrics_tuned = []

    # metrics and feature groups for feature analysis
    ABLATION_FEATURE_GROUPS = { # feature groups for ablation experiments
    "lags": [c for c in features if c.startswith("sales_lag_")],
    "rolls": [c for c in features if c.startswith("sales_roll")],
    "calendar": [c for c in features if c.startswith(("dow_", "month_", "doy_"))],
    "events": [c for c in features if c.startswith(("internal_events_", "internal_event_", "external_events_", "external_event_"))],
    "holidays": [c for c in features if c.startswith("holiday__")],
    "weather": [c for c in features if c.startswith(("precipitation_", "temperature_", "wind_"))],
    }
    
    metrics_ablations = {group_name: [] for group_name in ABLATION_FEATURE_GROUPS}


    # re-train, tune, and evaluate models for each experiment
    models = ["lasso", "sarimax", "xgboost"]
    for kind in models:
        params = read_configuration(kind)

        # impute analysis
        impute_oos, impute_metrics = backtest(impute_df, kind, features, params, target)
        impute_oos_list.append(impute_oos)
        impute_metrics_list.append(impute_metrics)

        # tuning analysis
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
            ablated_features = [c for c in features if c not in feature_group]
            _, ablation_metrics = backtest(df, kind, ablated_features, params, target)
            metrics_ablations[group_name].append(ablation_metrics)

        # grouped PFI experiments
        train, test = time_split(df)
        fitted_model, _, X_test, y_test = fit_model(train, test, kind, features, target, params)
        if kind == "lasso" or kind == "xgboost":
            permutation_table = permutation_values(fitted_model, X_test, y_test)
            save_results(permutation_table, "permutation_values")
            permutation_plot(permutation_table, "feature_analysis_figures", kind)
        else:
            pass
            


    # impute analysis figures
    impute_analysis_plots(impute_oos_list, impute_metrics_list, models)

    # tuning analysis delta plot
    delta_plots(metrics_baselines, metrics_tuned, models, "tuning_analysis_figures")

    # feature analysis results and figures
    for group_name, group_metrics in metrics_ablations.items():
        ablation_plots(metrics_tuned, group_metrics, models, "feature_analysis_figures", group_name)

    # uplifts = uplifts(df)
    # save_results(uplifts, "sales uplifts")

if __name__ == "__main__":
    run()
