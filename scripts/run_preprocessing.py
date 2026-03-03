from src.dataset.load_save import load_csv
from src.preprocessing.cleaning import clean_data
from src.preprocessing.encoding import fit_onehot_schema, save_onehot_schema, apply_onehot_schema
from src.preprocessing.features import add_all_features
from src.dataset.prepare import merge_data
from src.models.training import time_split

from src.models.tuning import feature_cols

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
    df_clean, report = clean_data(df_merged, train)
    print(df_clean)
    print(report)

    # data encoding
    schema = fit_onehot_schema(df_clean)
    save_onehot_schema(schema, "registry/onehot_schema.json")

    df_onehot = apply_onehot_schema(df_clean, schema, drop_original=True)
    print(df_onehot)

    # feature engineering
    df_feature = add_all_features(df_onehot)
    print(df_feature)
    df_feature.to_csv(out_path, index=False, mode="w")

    train, test = time_split(df_feature)
    features = feature_cols(df_feature)
    target = "sales"
    X_train, y_train = train[features], train[target] # manual split
    X_test, y_test = test[features], test[target]

    train.to_csv("data/train.csv", index=False, mode="w")
    test.to_csv("data/test.csv", index=False, mode="w")
    X_train.to_csv("data/X_train.csv", index=False, mode="w")
    X_test.to_csv("data/X_test.csv", index=False, mode="w")
    y_train.to_csv("data/y_train.csv", index=False, mode="w")
    y_test.to_csv("data/y_test.csv", index=False, mode="w")


if __name__ == "__main__": # used for running script outside of vscode, add argparsing to complete configuration
    run()
