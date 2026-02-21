import pandas as pd, numpy as np
'''
Data cleaning
'''


def normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Data normalisation to adhere to naming conventions
    '''
    out = df.copy()
    out.columns = [col.strip().lower().replace(" ", "_") for col in out.columns]
    return out


def standardise_strings(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Cleanup of string-based columns, keeping values readable and canonical
    '''
    out = df.copy()
    for col in ["internal_events", "external_events", "holiday"]:
        if col in out.columns:
            s = out[col].fillna("").astype(str).str.strip().str.lower()

            if col == "holiday" or col == "external_events":
                s = (s.str.replace(r"[|,/]", ";", regex=True)  
                        .str.replace(r"\s*;\s*", ";", regex=True) 
                        .str.replace(r";{2,}", ";", regex=True) 
                        .str.strip(";"))
                out[col] = s
            else:
                s = s.str.replace(r"\s+", " ", regex=True)
                out[col] = s
    return out


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Ensures necessary columns are numerical in type and erronous data is surfaced
    '''
    out = df.copy()
    s = out["sales"]
    s = s.astype(str).str.strip()
    s = s.astype(str).str.strip()
    s = s.str.replace("£", "", regex=False).str.replace(",", "", regex=False)
    s = s.replace({"": np.nan, "None": np.nan, "N/A": np.nan, "-": np.nan})
    out["sales"] = pd.to_numeric(s, errors="coerce") # errors coerced/nulled
    return out


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Handles missing values according to column
    '''
    out = df.copy()
    out["day_of_week"] = out["date"].dt.day_name() # add time-related features
    m_sales = out["sales"].isna()
    missing_sum = int(m_sales.sum())
    m_holiday = out["holiday"].notna() & out["holiday"].ne("")

    # set 0 for missing sales on holidays due to closures
    out.loc[m_sales & m_holiday, "sales"] = 0
    out.loc[m_sales & m_holiday, "closed"] = 1 # explainability

    med_global = (out.loc[~m_sales, "sales"]).median() # fallback
    med_dow = ( # dow based
        (out.loc[~m_sales])
        .groupby("day_of_week")["sales"]
        .median()
    )

    # impute remaining missing sales
    m_sales = out["sales"].isna()
    if m_sales.any():
        out.loc[m_sales, "sales"] = (out.loc[m_sales, "day_of_week"].map(med_dow).fillna(med_global).round(0)) # maps imputable sales rows through day-specific median using global median as fallback


    summary = { # summry report
        "datast cleaned: "
        "imputed_sales_rows": missing_sum,
        "med_global": float(med_global),
    }     

    return out.drop(columns=["day_of_week"]), summary


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Drop all duplicate dates
    '''
    out = df.drop_duplicates(subset=["date"]).copy()
    return out


def handle_outliers(df, z=5.0):
    '''
    Handles outliers carefully using threshold-style rules (z-score)
    '''
    out = df.copy()
    keep = out["holiday"].fillna("").ne("")
    col = "sales"
    mu, sd = out[col].mean(), out[col].std(ddof=0)
    within = (out[col] - mu).abs() <= z * sd # reveal values outside the threshold
    out = out[within | keep].copy()
    return out.reset_index(drop=True)


def clean_data(df: pd.DataFrame):
    '''
    Centralises cleaning & pre-processing
    '''
    df = (df
            .pipe(normalise_headers)
            .pipe(standardise_strings)
            .pipe(coerce_numeric))
    
    df, report = handle_missing(df)

    df = (df
            .pipe(handle_duplicates)
            .pipe(handle_outliers))
    
    return df, report
