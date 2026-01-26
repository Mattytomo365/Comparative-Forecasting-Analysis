import pandas as pd, hashlib, datetime as dt

'''
Loading raw data & saving to csv
'''

def load_csv(path: str):
    return pd.read_csv(path, parse_dates=["date"], dayfirst=True)

def save_csv(df, path:str):
    df = save_csv(path, index=False)