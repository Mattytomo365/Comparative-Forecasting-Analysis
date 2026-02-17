import pandas as pd
from src.models.model_factory import make_estimator
'''
Orchestrates ml model training, tuning, testing, and saving using preprocessed data
'''
# forms train and test datasets using time-aware split, creating a single holdout
def time_split(df, days=56):
    if df is None or len(df) == 0: # enforcing datetime
        raise ValueError("Empty dataframe provided to time_split")
    cutoff = df["date"].max() - pd.Timedelta(days) # splitting date
    train = df[df["date"] <= cutoff].reset_index(drop=True)
    test = df[df["date"] > cutoff].reset_index(drop=True)
    return train, test

# trains and tests specified model with specified parameters for the specified window of data
def train_model(train, test, kind, features, target, params):
    if len(features) == 0: # feature validation
        raise ValueError("No training features found after applying exclude set")

    X_train, y_train = train[features], train[target] # feature matrix and target vector
    X_test, y_test = test[features], test[target] # unseen testing data
    model = make_estimator(X_train, y_train, kind, params)

    if kind == "sarimax": # sarimax doesn't use the sklearn interface
        results = model.fit()
        drop_cols = [c for c in X_test.columns if c.startswith("sales_lag") or c.startswith("sales_roll")] # sarimax doesnt require these features
        X_test_sarimax = X_test.drop(columns=drop_cols)
        preds = results.get_forecast(steps=len(X_test), exog=X_test_sarimax).predicted_mean

    elif kind == "xgboost": # early stopping implementation
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        preds = model.predict(X_test)

    elif kind == "lasso":
        model.fit(X_train, y_train) # fitted to all available data up to the current origin
        preds = model.predict(X_test)
    
    oos = test[["date"]].copy() # out of sample dataset containing all unseen data and the corresponding predictions
    oos["Actual data"] = y_test
    oos["Forecasted data"] = preds
    from src.models.evaluation import calculate_metrics
    metrics = calculate_metrics(y_test, y_train, preds) # calculate metrics for current window predicted
    return oos, metrics, model
