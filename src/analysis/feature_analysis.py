import pandas as pd
'''
Explores uplift and feature importance metrics over different periods for different metrics
'''

UPLIFT_COLS = ["holiday", "internal_event", "external_event"] # skips numerical columns for sales uplift calculations

def run_ablation():
    pass

def permutation_importance():
    pass


def uplifts(df: pd.DataFrame, 
            factor: str, 
            month: str,
            metric: str, 
            sep=";") -> pd.DataFrame:
    '''
    Compute uplift of specified factors against specified metric
    '''
    OUT_COLS = ["tag", "n", "avg", "uplift"]
    d = df["date"]
    m = d.dt.month.eq(int(month)) # boolean mask

    if not m.any(): # fallback
        return pd.DataFrame(columns=OUT_COLS)
    
    sub = df.loc[m].copy()
    s = sub[factor].fillna("").astype(str).str.strip().str.lower()

    if factor in UPLIFT_COLS:
        tags_list = s.apply(lambda val: [tag.strip() for tag in val.split(sep) if tag.strip() and tag.strip() != "none"]) # split multi-tag strings into list
        base_mask = tags_list.str.len().eq(0) # marks baseline days as days without tags
        baseline = (sub.loc[base_mask])[metric].mean() # calculates baseline from marked days

        sub["tags"] = tags_list

        # explode tags to individual rows
        sub = (sub.explode("tags").rename(columns={"tags": "tag"}))
        sub = sub[["date", "tag", metric]].drop_duplicates(subset=["date", "tag"])

        sub["baseline"] = baseline
        sub = sub.loc[~base_mask & sub["baseline"].notna()]

        if sub.empty or pd.isna(baseline) or baseline == 0: # guards uplift calculation against NaN and 0
            return pd.DataFrame(columns=OUT_COLS)
        
    else:
        return pd.DataFrame(columns=OUT_COLS)
        
    # calculate percentage uplift per row against baseline
    sub["uplift_row"] = 100.0 * (sub[metric] - sub["baseline"]) / sub["baseline"]

    # aggregate per tag
    tab = (sub.groupby("tag")
        .agg(n=("date", "nunique") # unique days tag occurs
             , avg=(metric, "mean") # mean metric on tagged days
             , uplift=("uplift_row", "mean")) # mean % uplift across occurrences
        .reset_index().sort_values("avg", ascending=False))
    
    return pd.DataFrame(tab[OUT_COLS])

def plot_all(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Percentage uplift for specified factor, month, and metric
    '''
    all_uplifts = []
    for factor in UPLIFT_COLS:
        for month in range(1, 13):
            uplift = uplifts(df, factor, month, "sales")
            uplift["factor"] = factor
            uplift["month"] = month
            all_uplifts.append(uplift)

    run_ablation()
    return pd.concat(all_uplifts, ignore_index=True)