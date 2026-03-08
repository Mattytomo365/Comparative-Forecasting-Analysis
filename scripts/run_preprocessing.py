from src.dataset.load_save import load_csv
from src.preprocessing.cleaning import clean_data
from src.preprocessing.encoding import fit_onehot_schema, save_onehot_schema, apply_onehot_schema
from src.preprocessing.features import add_all_features
from src.dataset.prepare import merge_data
from src.models.training import time_split

'''
Module for data ingestion, cleaning, and transformation functions
'''

def run(raw_path="data/sales_daily.csv", out_path="data/sales_daily_processed.csv"):

    # data loading
    df = load_csv(raw_path)
    print(df)

    # data preparation/merging
    df_merged = merge_data(df)
    print(df_merged)


    # data cleaning
    train, test = time_split(df)
    df_med_dow, df_med_global, report = clean_data(df_merged, train)
    print(df_med_dow)
    print(report)

    # data encoding
    schema = fit_onehot_schema(df_med_dow)
    save_onehot_schema(schema, "data/onehot_schema.json")

    df_med_dow = apply_onehot_schema(df_med_dow, schema, drop_original=True)
    df_med_global = apply_onehot_schema(df_med_global, schema, drop_original=True)
    print(df_med_dow)

    # feature engineering
    df_med_dow = add_all_features(df_med_dow)
    df_med_global = add_all_features(df_med_global)
    print(df_med_dow)
    df_med_dow.to_csv(out_path, index=False, mode="w")
    df_med_global.to_csv("data/sales_globally_imputed.csv", index=False, mode="w") # enables imputation analysis



if __name__ == "__main__": # used for running script outside of vscode, add argparsing to complete configuration
    run()
