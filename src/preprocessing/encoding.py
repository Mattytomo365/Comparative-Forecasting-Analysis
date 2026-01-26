import re
import json
from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
'''
Multi-label and singular one-hot encoding for string based fields
'''

# standardises category names, allowing for safe onehot column suffixes
def standardise_categories(c):
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
    return c or "unknown"

# adds and fills one-hot cols for a specified col with a fixed category list
def onehot_single(df, col, categories:list[str]): # passes in categories from fitted schema
    col_norm = df[col].fillna("unknown").map(standardise_categories) # replaces NaN and standardises categories to match schema
    cat_dtype = CategoricalDtype(categories=categories, ordered=False) # allows for casts to categorial dtype to filter out categories not in schema
    dummies = pd.get_dummies(col_norm.astype(cat_dtype), prefix=col, prefix_sep="__", dtype="int32") # onehot matrix (0/1 values)
    expected_cols = [f"{col}__{c}" for c in categories]
    dummies = dummies.reindex(columns=expected_cols, fill_value=0) # adds missing onehots filled with 0 from precomputed list
    return dummies

# multi-label onehot columns from separated category tags
def onehot_multi(df, col, categories:list[str], sep=";"):
    out = pd.DataFrame(index=df.index)
    tags = ( # series of sets containing individual category tags present for each row
        df[col].apply(lambda val: {standardise_categories(tag) for tag in val.split(sep) if tag.strip()})
    )
    # checks categories against each tag set in tags
    for c in categories:
        out[f"{col}__{c}"] = tags.apply(lambda set: int(c in set)).astype("int32") # convert to 0/1 matrix
    return out

# discover categories in historical dataset, returns schema to apply onehot
def fit_onehot_schema(df, weather_col="weather", internal_col="internal_events", external_col="external_events", holiday_col="holiday", dow_col="day_of_week"):
    schema = {}

    # weather / internal events (single)
    for col, key in [(weather_col, "weather"), (internal_col, "internal_events")]:
        if col in df.columns:
            s = df[col]
            categories = s[s.ne("")].unique().tolist() # determines unqiue categories listed not empty
            categories = {standardise_categories(c) for c in categories}
            schema[f"{key}"] = sorted(categories) # stable order for consistency
    
    # external events/ holiday (multi-label)
    for col, key in [(external_col, "external_events"), (holiday_col, "holiday")]:
        if col in df.columns:
            categories = set() # using set removes duplicates
            for val in df[col]:
                for category in [tag.strip() for tag in val.split(";") if tag.strip()]: # separates into individual categories
                    categories.add(standardise_categories(category))
            schema[f"{key}"] = sorted(categories)
    
    # # day of week (single)
    # if dow_col in df.columns:
    #     ordered = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    #     days_present = set(df[dow_col].str[:3]) # abbreviate category names
    #     days = [d for d in ordered if d in days_present] # avoids errenous data
    #     days = [standardise_categories(d) for d in days]
    #     schema["day_of_week"] = days
    
    return schema

# apply fitted schema to onehot methods
def apply_onehot_schema(df, schema, drop_original=False):
    onehot_cols = [df.reset_index(drop=True)] # what does this mean?

    # weather
    if "weather" in schema and "weather" in df.columns:
        onehot_w = onehot_single(df, "weather", schema["weather"])
        onehot_cols.append(onehot_w)

    # internal events
    if "internal_events" in schema and "internal_events" in df.columns:
        onehot_i = onehot_single(df, "internal_events", schema["internal_events"])
        onehot_cols.append(onehot_i)
    
    # external events
    if "external_events" in schema and "external_events" in df.columns:
        onehot_e = onehot_multi(df, "external_events", schema["external_events"])
        onehot_cols.append(onehot_e)
    
    # holidays
    if "holiday" in schema and "holiday" in df.columns:
        onehot_h = onehot_multi(df, "holiday", schema["holiday"])
        onehot_cols.append(onehot_h)
    
    # # day of week
    # if "day_of_week" in schema and "day_of_week" in df.columns:
    #     onehot_d = onehot_single(df.assign(day_of_week=df["day_of_week"].str[:3]), "day_of_week", schema["day_of_week"])
    #     onehot_cols.append(onehot_d)
    
    out = pd.concat(onehot_cols, axis=1)

    if drop_original:
        out = out.drop(columns=[col for col in ["weather","internal_events","external_events","holiday","day_of_week"] if col in out ], errors="ignore")
    
    return out

# saves onehot schema to json for reusability and stability between training and serving models
def save_onehot_schema(schema, path):
    Path(path).write_text(json.dumps(schema, indent=2))

# reads and returns saved onehot schema from json for additional encoding
def load_onehot_schema(path):
    return json.loads(Path(path).read_text())