import pandas as pd
from src.dataset.load_save import load_csv
from src.preprocessing.cleaning import clean_data
from src.preprocessing.encoding import fit_onehot_schema, save_onehot_schema, apply_onehot_schema
from src.preprocessing.features import add_cyclical
from src.dataset.prepare import merge_data
from src.models.training import time_split
from src.preprocessing.auditing import audit_stage
from pprint import pprint

'''
Module for data ingestion, cleaning, and transformation functions
'''

def run(raw_path="data/sales_daily.csv", out_path="data/sales_daily_processed.csv"):

    # data loading
    df = load_csv(raw_path)
    print(df)

    # data preparation/merging
    df_merged = merge_data(df)
    print(df_merged.head())


    # data cleaning
    train, _ = time_split(df)
    df_mean_dow, df_med_dow, df_med_global, dow_mean_summary, dow_median_summary, global_median_summary, dow_median_outlier_summary = clean_data(df_merged, train)
    pprint(dow_median_summary, sort_dicts=False)
    pprint(dow_mean_summary, sort_dicts=False)
    pprint(global_median_summary, sort_dicts=False)

    # data encoding
    train_med_dow, _ = time_split(df_med_dow)
    schema = fit_onehot_schema(train_med_dow)
    save_onehot_schema(schema, "data/onehot_schema.json")

    df_encoded = apply_onehot_schema(df_mean_dow, schema, drop_original=True)
    df_med_dow = apply_onehot_schema(df_med_dow, schema, drop_original=True)
    df_med_global = apply_onehot_schema(df_med_global, schema, drop_original=True)
    print(df_encoded.head())

    # feature engineering
    df_engineered = add_cyclical(df_encoded)
    df_med_dow = add_cyclical(df_med_dow)
    df_med_global = add_cyclical(df_med_global)
    print(df_engineered.head())

    audit = [
        audit_stage(df, "raw"),
        audit_stage(df_merged, "merged"),
        audit_stage(
            df_med_dow,
            "cleaned",
            extreme_outliers_flagged=dow_median_outlier_summary["extreme_outliers_flagged"],
            extreme_outliers_removed=dow_median_outlier_summary["extreme_outliers_removed"],
        ),
        audit_stage(df_encoded, "encoded"),
        audit_stage(df_engineered, "feature_engineered"),
    ]
    print(pd.DataFrame(audit).to_string(index=False))

    df_engineered.to_csv(out_path, index=False, mode="w")
    df_mean_dow.to_csv("data/sales_dow_mean_imputed.csv", index=False, mode="w") # enables imputation analysis
    df_med_global.to_csv("data/sales_globally_imputed.csv", index=False, mode="w")



if __name__ == "__main__": # used for running script outside of vscode, add argparsing to complete configuration
    run()
