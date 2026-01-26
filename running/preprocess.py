import pandas as pd
from src.dataset.load_save import load_csv # absolute imports
from src.preprocessing.cleaning import clean_data
from src.preprocessing.encoding import fit_onehot_schema, save_onehot_schema, apply_onehot_schema
from src.preprocessing.features import add_all_features
from src.dataset.prepare import merge_data

'''
Module for data ingestion, cleaning, and transformation functions
'''

def run(raw_path="data/sales_daily.csv", out_path="data/sales_daily_processed.csv"):

    # data cleaning
    df = load_csv(raw_path)
    print(df)

    # prepare data first
    df_merged = merge_data(df)
    print(df_merged)

    df_clean, report = clean_data(df_merged)
    print(df_clean)
    print(report)

    # data encoding
    schema = fit_onehot_schema(df_clean)
    save_onehot_schema(schema, "registry/onehot_schema.json")

    df_onehot = apply_onehot_schema(df_clean, schema, drop_original=False)
    print(df_onehot)

    # feature engineering
    df_feature = add_all_features(df_onehot)
    print(df_feature)
    df_feature.to_csv(out_path, index=False, mode="w")
    

# do i still need this???
if __name__ == "__main__": # used for running script outside of vscode, add argparsing to complete configuration
    run()
