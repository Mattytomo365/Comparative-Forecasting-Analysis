from .load_save import load_csv
import pandas as pd
'''
Joins exogenous factor datasets to sales datasets by date
Full, complete and well-explained data
'''

def merge_data(sales: pd.DataFrame) -> pd.DataFrame:
    weather = load_csv('./data/weather_daily.csv')
    holidays = load_csv('./data/holidays.csv')
    events = load_csv('./data/events.csv')
    merged = sales.merge(weather, on="date", how="left")  # left join keeps only sales dates
    merged = merged.merge(holidays, on="date", how="left")
    merged = merged.merge(events, on="date", how="left")
    return merged
