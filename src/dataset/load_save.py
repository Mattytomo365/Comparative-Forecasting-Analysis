import pandas as pd
from typing import Any

'''
Loading raw data & saving to csv
'''

def load_csv(path: str) -> Any:
    return pd.read_csv(path, parse_dates=["date"], dayfirst=True)

def save_csv(df, path:str) -> None:
    df = save_csv(path, index=False)