import numpy as np, pandas as pd
'''
Engineering additional features to produce more meaningful data
'''


def add_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Generates seasonal Fourier features
    Adds cyclical features for day of week and day of year to capture and help models understand periodic data
    '''
    out = df.copy()
    d = out["date"]

    # day of week (7 day cycle)
    dow = d.dt.weekday.astype(float) # day of week represented as float
    out["dow_sin"] = np.sin((2 * np.pi * dow) / 7.0).astype("float32") # y-coordinate
    out["dow_cos"] = np.cos((2 * np.pi * dow) / 7.0).astype("float32") # x-coordinate

    # day of year (~365 day cycle)
    doy = d.dt.dayofyear.astype(float) # day of year represented as float
    denom = np.where(d.dt.is_leap_year, 366.0, 365.0) # leap-year precision
    out["doy_sin"] = np.sin((2 * np.pi * (doy - 1.0)) / denom).astype("float32") # (doy - 1) so Jan 1st is angle 0
    out["doy_cos"] = np.cos((2 * np.pi * (doy - 1.0)) / denom).astype("float32")
    out["doy_sin_2"] = np.sin(2 * (2 * np.pi * (doy - 1.0)) / denom).astype("float32") # ONLY KEEP IF BACKTESTS IMPROVE MAE/RMSE/MAPE etc...
    out["doy_cos_2"] = np.cos(2 * (2 * np.pi * (doy - 1.0)) / denom).astype("float32")

    return out


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds lags to capture short-term momentum and weekday repitition
    '''
    lags = (1, 7, 14, 21)
    out = df.copy()

    for lag in lags:
        out[f"sales_lag_{lag}"] = out["sales"].shift(lag).fillna("")
    return out


def add_rolls(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds rolling statistics to summarise information over a specific period of time, giving a broader perspective
    '''
    windows = (7, 14, 21) # weekly windows
    out = df.copy()

    past = out["sales"].shift(1) # past only
    for window in windows:
        out[f"sales_roll_mean_{window}"] = past.rolling(window).mean()
        out[f"sales_roll_std_{window}"] = past.rolling(window).std()
    return out
    


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Bringing all feature engineering methods together
    '''
    df = (df
        .pipe(add_cyclical)
        .pipe(add_lags)
        .pipe(add_rolls))
    return df