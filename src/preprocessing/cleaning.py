import pandas as pd, hashlib, datetime as dt, numpy as np
import re
'''
Data pre-processing & cleaning
'''

# data normalisation to adhere to naming conventions
def normalise_headers(df):
    out = df.copy() # returning new dataframes avoids input mutation
    out.columns = [col.strip().lower().replace(" ", "_") for col in out.columns]
    return out

# cleanup of string-based columns, keeping values readable and canonical
def standardise_strings(df):
    out = df.copy()
    for col in ["internal_events", "external_events", "holiday"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str).str.strip().str.lower() # fills null values with space and removes leading/trailing spaces
            s = out[col]
            if col == "holiday" or col == "external_events":
                s = (s.str.replace(r"[|,/]", ";", regex=True) # normalize delimiters to ';' 
                        .str.replace(r"\s*;\s*", ";", regex=True) # trim around ;
                        .str.replace(r";{2,}", ";", regex=True) # collapse ;;
                        .str.strip(";"))
            else:
                s = s.str.replace(r"\s+", " ", regex=True) # collapse internal spaces to single space
    return out

# parsing dates to datetime objects
def parse_dates(df):
    out = df.copy()
    # Accept both ISO (YYYY-MM-DD) and day-first formats (DD/MM/YYYY) without hard-coding a single format.
    out["date"] = pd.to_datetime(out["date"].astype(str), dayfirst=True, errors="raise")
    return out

# ensures necessary columns are numerical in type and erronous data is surfaced
def coerce_numeric(df):
    out = df.copy()
    s = out["sales"] # series of values within column
    s = s.where(~s.apply(lambda x: isinstance(x, type)), np.nan) # replaces stray Python type objects with NaN
    s = s.astype(str).str.strip()
    s = s.str.replace("£,", "", regex=True) # removes currency symbols
    s = s.replace({"": np.nan, "None": np.nan, "N/A": np.nan, "-": np.nan}) # treats comman placeholders as NaN
    out["sales"] = pd.to_numeric(s, errors="coerce") # converts back to numeric, erronous values are nulled
    return out

# handles missing values according to column
def handle_missing(df):
    out = df.copy()
    out["day_of_week"] = out["date"].dt.day_name() # adds day-of-week feature for day-specific medians
    m_sales = out["sales"].isna()
    missing_sum = int(m_sales)
    m_holiday = out["holiday"].notna() & out["holiday"].ne("")

    # set 0 for missing sales on holidays due to closures
    out.loc[m_sales & m_holiday, "sales"] = 0

    med_global = (out.loc[~m_sales, "sales"]).median() # median sales for fallback use
    med_dow = ( # median sales for certain weekdays
        (out.loc[~m_sales])
        .groupby("day_of_week")["sales"]
        .median()
    )

    # impute remaining missing sales
    m_sales = out["sales"].isna() # recompute after holiday fill
    if m_sales.any():
        out.loc[m_sales, "sales"] = (out.loc[m_sales, "day_of_week"].map(med_dow).fillna(med_global).round(0)) # maps imputable sales rows through day-specific median using global median as fallback


    summary = { # summry report of imput & drop sums
        "datast cleaned: "
        "imputed_sales_rows": missing_sum,
        "med_global": float(med_global),
    }     

    return out.drop(columns=["day_of_week"]), summary

# drop all duplicate dates
def handle_duplicates(df):
    out = df.drop_duplicates(subset=["date"]).copy() # uses subset to treat rows with the same dates as duplicates
    return out

# handles outliers carefully using threshold-style rules (z-score)
def handle_outliers(df, z=5.0):
    out = df.copy()
    keep = out["holiday"] != ""
    col = "sales"
    mu, sd = out[col].mean(), out[col].std(ddof=0)
    within = (out[col] - mu).abs() <= z * sd # reveals values with deviations from the mean outside the threshold
    out = out[within | keep].copy() # ignores rows during holiday periods
    return out.reset_index(drop=True)

# centralises cleaning & pre-processing, imported into clean_data.py script
def clean_data(df):
    df = (df
            .pipe(normalise_headers)
            .pipe(standardise_strings)
            .pipe(parse_dates)
            .pipe(coerce_numeric))
    
    df, report = handle_missing(df)

    df = (df
            .pipe(handle_duplicates)
            .pipe(handle_outliers))
    
    return df, report
