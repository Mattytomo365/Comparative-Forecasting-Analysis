import re
import json
from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
'''
Multi-label and singular one-hot encoding for string based fields
'''


def standardise_categories(c: str) -> str:
    '''
    Standardises category names, allowing for safe onehot column suffixes
    '''
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
    return c or "unknown"


def split_categories(val: object) -> list[str]:
    '''
    Split multi-label values across common separators.
    '''
    return [tag.strip() for tag in re.split(r"[;,|]", str(val)) if tag and tag.strip()]


def onehot_single(df: pd.DataFrame,
                  col: str,
                  categories: list[str]) -> pd.DataFrame:
    '''
    Adds and fills one-hot cols for a specified col with a fixed category list
    '''
    col_norm = df[col].fillna("unknown").map(standardise_categories)
    cat_dtype = CategoricalDtype(categories=categories, ordered=False)  # cast to categorical dtype to filter out categories not in schema
    dummies = pd.get_dummies(col_norm.astype(cat_dtype), prefix=col, prefix_sep="__", dtype="int32")  # onehot matrix
    expected_cols = [f"{col}__{c}" for c in categories]
    dummies = dummies.reindex(columns=expected_cols, fill_value=0)  # fallback for missing dummies
    return dummies


def onehot_multi(df: pd.DataFrame,
                 col: str,
                 categories: list[str]) -> pd.DataFrame:
    '''
    Multi-label onehot columns from separated category tags
    '''
    out = pd.DataFrame(index=df.index)
    tags = (
        df[col]
        .fillna("")
        .astype(str)
        .apply(lambda val: {standardise_categories(tag) for tag in split_categories(val)})
    )
    for c in categories:
        out[f"{col}__{c}"] = tags.apply(lambda tag_set: int(c in tag_set)).astype("int32")
    return out


def fit_onehot_schema(df: pd.DataFrame,
                      internal_col="internal_events",
                      external_col="external_events",
                      holiday_col="holiday") -> dict[str, list[str]]:
    '''
    Discover categories in historical dataset, returns schema to apply onehot
    '''
    schema: dict[str, list[str]] = {}

    # internal events/ external events/ holiday (multi-label)
    for col, key in [(external_col, "external_events"), (holiday_col, "holiday"), (internal_col, "internal_events")]:
        if col in df.columns:
            categories = set()
            for val in df[col].fillna("").astype(str):
                for category in split_categories(val):
                    categories.add(standardise_categories(category))
            schema[key] = sorted(categories)
    return schema


def apply_onehot_schema(df: pd.DataFrame, schema: dict[str, list[str]], drop_original=False) -> pd.DataFrame:
    '''
    Apply fitted schema to onehot methods
    '''
    onehot_cols = [df.reset_index(drop=True)]

    # internal events (multi-label)
    if "internal_events" in schema and "internal_events" in df.columns:
        onehot_i = onehot_multi(df, "internal_events", schema["internal_events"])
        onehot_cols.append(onehot_i)

    # external events
    if "external_events" in schema and "external_events" in df.columns:
        onehot_e = onehot_multi(df, "external_events", schema["external_events"])
        onehot_cols.append(onehot_e)

    # holidays
    if "holiday" in schema and "holiday" in df.columns:
        onehot_h = onehot_multi(df, "holiday", schema["holiday"])
        onehot_cols.append(onehot_h)

    out = pd.concat(onehot_cols, axis=1)

    if drop_original:
        out = out.drop(columns=[col for col in ["internal_events", "external_events", "holiday"] if col in out.columns], errors="ignore")

    return out


def save_onehot_schema(schema: dict[str, list[str]], path: str) -> Path:
    '''
    Saves onehot schema to json for reusability and stability between training and serving models
    '''
    path_obj = Path(path)
    path_obj.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return path_obj


def load_onehot_schema(path: str | Path) -> dict[str, list[str]]:
    '''
    Reads and returns saved onehot schema from json for additional encoding
    '''
    return json.loads(Path(path).read_text(encoding="utf-8"))

