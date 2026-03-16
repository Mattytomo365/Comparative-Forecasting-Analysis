from src.dataset.load_save import load_csv, load_metrics
from src.analysis.feature_analysis import ablation_plots, uplifts
from results.save_results import save_results
from src.analysis.impute_analysis import impute_analysis_plots
from src.analysis.tuning_analysis import delta_plots
from src.models.tuning import read_configuration, feature_cols
from src.models.testing import backtest
'''
Module for feature analysis, data preprocessing analysis, and tuning analysis
Produces figures and metrics regarding featrure importance, missing value imputation impact, and tuning impact
'''

def run(impute_analysis_path="data/sales_globally_imputed.csv", data_path="data/sales_daily_processed.csv", target="sales"):
    # OOS, metrics, and feature configuration set up
    impute_df = load_csv(impute_analysis_path)
    df = load_csv(data_path)
    features = feature_cols(df)

    FEATURE_GROUPS = { # feature groups for ablation experiments
    "calendar": [c for c in features if not c.startswith(("dow_", "month_"))],
    "events": [c for c in features if not c.startswith(("internal_events_", "internal_event_", "external_events_", "external_event_"))],
    "holidays": [c for c in features if not c.startswith(("holidays_", "holidays_"))],
    "weather": [c for c in features if not c.startswith(("precipitation_", "temperature_", "wind_"))],
    }

    # oos predictions + metrics for impute analysis
    impute_oos_list = []
    impute_metrics_list = []

    # metrics for tuning analysis
    metrics_baselines = []
    metrics_tuned = []

    # metrics for feature analysis
    metrics_ablations = {group_name: [] for group_name in FEATURE_GROUPS}


    # re-train, tune, and evaluate models for each experiment
    models = ["lasso", "sarimax", "xgboost"]
    for model in models:
        params = read_configuration(model)

        # impute analysis
        impute_oos, impute_metrics = backtest(impute_df, model, features, params, target)
        impute_oos_list.append(impute_oos)
        impute_metrics_list.append(impute_metrics)

        # tuning analysis
        metrics_baseline = load_metrics(f"results/{model}_metrics_baseline.csv")
        metrics_tune = load_metrics(f"results/{model}_metrics_tuned.csv")
        metrics_baselines.append(metrics_baseline)
        metrics_tuned.append(metrics_tune)

        # ablation tests (feature analysis)
        for group_name, feature_group in FEATURE_GROUPS.items():
            _, ablation_metrics = backtest(df, model, feature_group, params, target)
            metrics_ablations[group_name].append(ablation_metrics)
            


    # impute analysis figures
    impute_analysis_plots(impute_oos_list, impute_metrics_list, models)

    # tuning analysis figures
    delta_plots(metrics_baselines, metrics_tuned, models, "tuning_analysis_figures")

    # feature analysis results and figures
    for group_name, group_metrics in metrics_ablations.items():
        ablation_plots(metrics_tuned, group_metrics, models, "feature_analysis_figures", group_name)

    # uplifts = uplifts(df)
    # save_results(uplifts, "sales uplifts")

if __name__ == "__main__":
    run()
