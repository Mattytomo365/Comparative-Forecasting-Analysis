import numpy as np, pandas as pd
'''
Engineering additional features to produce more meaningful data
'''

# generates seasonal Fourier features
# adds cyclical features for day of week and day of year to capture and help models understand periodic data
def add_cyclical(df):
    out = df.copy()
    d = out["date"]

    # day of week (7 day cycle)
    dow = d.dt.weekday.astype(float) # day of week represented as float
    out["dow_sin"] = np.sin((2 * np.pi * dow) / 7.0).astype("float32") # y-coordinate on circle
    out["dow_cos"] = np.cos((2 * np.pi * dow) / 7.0).astype("float32") # x-coordinate on circle

    # day of year (~365 day cycle)
    doy = d.dt.dayofyear.astype(float) # day of year represented as float
    denom = np.where(d.dt.is_leap_year, 366.0, 365.0) # adds leap-year precision
    out["doy_sin"] = np.sin((2 * np.pi * (doy - 1.0)) / denom).astype("float32") # (doy - 1) so Jan 1st is angle 0
    out["doy_cos"] = np.cos((2 * np.pi * (doy - 1.0)) / denom).astype("float32")
    out["doy_sin_2"] = np.sin(2 * (2 * np.pi * (doy - 1.0)) / denom).astype("float32") # ONLY KEEP IF BACKTESTS IMPROVE MAE/RMSE/MAPE etc...
    out["doy_cos_2"] = np.cos(2 * (2 * np.pi * (doy - 1.0)) / denom).astype("float32")

    return out

# adds lags to capture short-term momentum and weekday repitition
def add_lags(df):
    lags = (1, 7, 14, 21) # informed by acf plots
    out = df.copy()

    for lag in lags:
        out[f"sales_lag_{lag}"] = out["sales"].shift(lag).fillna("") # shifts past values from the same day into new columns
    return out

# adds rolling statistics to summarise information over a specific period of time, giving a broader perspective
def add_rolls(df):
    windows = (7, 14, 28) # weekly windows to get weekly patterns
    out = df.copy()

    past = out["sales"].shift(1) # keeps past only
    for window in windows:
        # calculate mean and standard deviation over corresponding window
        out[f"sales_rollmean{window}"] = past.rolling(window).mean().fillna("")
        out[f"sales_rollstd{window}"] = past.rolling(window).std().fillna("")
    return out
    

# bringing all feature engineering methods together
def add_all_features(df):
    df = (df
        .pipe(add_cyclical)
        .pipe(add_lags)
        .pipe(add_rolls))
    return df