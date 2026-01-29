import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from figures.save_figure import save_figure
'''
User-facing eda on historical data to uncover trends and patterns
'''

DOW_ORDER = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"] # tuple [] for a set order
UPLIFT_COLS = ["holiday", "internal_event", "external_event"] # skips numerical columns for sales uplift calculations

# mean metric per calendar month
def monthly_avg(df, metric):
    d = df["date"]
    month = d.dt.to_period("M") # returns year-month keys to group by year-month
    month_m = (df.groupby(month)[metric].agg(value="mean", n_days="count") # aggregating groups to compute mean and n_days 
               .rename_axis("month") # renames index to 'month'
               .reset_index()) # brings the renamed index into a new column

    # additional columns for user interface refinement
    month_m["label"] = month_m["month"].dt.strftime("%b %Y")
    month_m["value"] = month_m["value"].astype(float).round(2) # rounding
    month_m["n_days"] = month_m["n_days"].astype(int)
    return pd.DataFrame(month_m[["month", "label", "value", "n_days"]])

# mean metric by weekday for a specified month
def weekday_avg(df, month, metric):
    d = df["date"]
    m = d.dt.month.eq(int(month)) # locates month within dataset

    if not m.any():  # fallback
        return pd.DataFrame({"dow": DOW_ORDER, "value": [0]*7}) # => [0, 0, 0, 0, 0, 0, 0]
    
    dow = d.dt.day_name().str[:3] # returns "Mon", "Tue" etc
    day_m = (df.loc[m].groupby(dow)[metric].agg(value="mean") # calculates weekday-based average for metric and names column 'value'
            .reindex(DOW_ORDER).fillna(0.0) # maintains fixed order with no missing days
            .rename_axis("dow") # renames index to 'dow'
            .reset_index()) 
    
    day_m["value"] = day_m["value"].astype(float).round(2) # rounding for user interface
    return pd.DataFrame(day_m[["dow", "value"]])


# compute uplift of specified factors against specified metric
def uplifts(df, factor, month, metric, sep=";"):

    OUT_COLS = ["tag", "n", "avg", "uplift"]
    d = df["date"]
    m = d.dt.month.eq(int(month)) # builds boolean mask for selected month

    if not m.any(): # fallback if selected month doesnt exist
        return pd.DataFrame(columns=OUT_COLS)
    
    sub = df.loc[m].copy()
    s = sub[factor].fillna("").astype(str).str.strip().str.lower() # normalises factor column into clean strings

    if factor in UPLIFT_COLS:
        tags_list = s.apply(lambda val: [tag.strip() for tag in val.split(sep) if tag.strip() and tag.strip() != "none"]) # splits multi-tag strings into a list of tags
        base_mask = tags_list.str.len().eq(0) # marks baseline days as days without tags
        baseline = (sub.loc[base_mask])[metric].mean() # mean metric of days without events within the month

        # add helper columns 
        sub["tags"] = tags_list # attach tag if any

        # explode tags to individual rows
        sub = (sub.explode("tags").rename(columns={"tags": "tag"}))
        sub = sub[["date", "tag", metric]].drop_duplicates(subset=["date", "tag"])

        sub["baseline"] = baseline # sets baseline to month baseline for other factors

        # filter out rows with no event or an invalid baseline
        sub = sub.loc[~base_mask & sub["baseline"].notna()]

        # fallback
        if sub.empty or pd.isna(baseline) or baseline == 0: # guards uplift calculation against NaN and 0
            return pd.DataFrame(columns=OUT_COLS)
        
    else:
        return pd.DataFrame(columns=OUT_COLS)
        
    # calculate percentage uplift per row against baseline
    sub["uplift_row"] = 100.0 * (sub[metric] - sub["baseline"]) / sub["baseline"]

    # aggregate per tag - collapse daily rows into a few group-level numbers
    tab = (sub.groupby("tag")
        .agg(n=("date", "nunique") # unique days tag occurs
             , avg=(metric, "mean") # mean metric on tagged days
             , uplift=("uplift_row", "mean")) # mean % uplift across occurrences
        .reset_index().sort_values("avg", ascending=False))
    
    return pd.DataFrame(tab[OUT_COLS])

# helper function to improve visibility of values on x-axis
def monthly_labels(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.figure.autofmt_xdate()


# plots basis waves at varying harmonic levels visualising fourier features against time
def fourier_basis_wave(df, sin_col, cos_col, k, title, name):
    d = df["date"]
    s = df[sin_col].astype(float)
    c = df[cos_col].astype(float)
    fig, ax = plt.subplots()

    ax.plot(d, s, label=f"sin k={k}")
    ax.plot(d, c, label=f"cos k={k}")

    ax.set_xlabel("date")
    ax.set_ylabel("basis value")
    if title: ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    monthly_labels(ax)

    save_figure(fig, name)
    plt.close(fig)

# plots linear combination of baseline waves, visualises what linear models perform on yearly fourier features to understand seasonal shape
def seasonal_curve(df, target_col_name, k, title):
    d = df["date"]
    y = df[target_col_name].astype("float").values # the series which the seasonal shape is being visualised from
    if k == 2:
        s = df["doy_sin_2"].astype(float).values # Fourier features previously generated
        c = df["doy_cos_2"].astype(float).values
    else:
        s = df["doy_sin"].astype(float).values
        c = df["doy_cos"].astype(float).values

    # build a 2-D array of the features to make a small design matrix
    X = np.column_stack([np.ones_like(s), s, c]) # [1, sin, cos] so X has shape (num_days, 3)

    # find coefficients that minimise total squared error
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)    # beta = [β₀, a, b]

    # make a seasonal curve from calculated weights/coefficients
    y_fitted = X @ beta     # β₀ + a*sin + b*cos

    # plot actual vs fitted baseline
    fig, ax = plt.subplots()
    ax.plot(d, y, label="actual")
    ax.plot(d, y_fitted, "--", label="seasonal curve (from sin/cos)")
    if title: ax.set_title(title)
    ax.set_ylabel(target_col_name)
    ax.set_xlabel("date")
    ax.legend()
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    monthly_labels(ax)

    save_figure(fig, "fourier_seasonal")
    plt.close(fig)


# generates unit-circle plots of fourier features, helping visualise the circular pattern
def fourier_unit_circle(df, cos_col, sin_col, title, name):
    fig, ax = plt.subplots()
    df.plot.scatter(sin_col, cos_col, ax=ax)
    ax.set_aspect("equal", adjustable="box") # forms circle
    t = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(t), np.sin(t), linewidth=1) # unit-circle line
    if title: ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)

    save_figure(fig, name)
    plt.close(fig)


# centralised function
def plot_all(df):

    # fourier baseline waves
    fourier_basis_wave(df, "dow_sin", "dow_cos", 1, "Weekly Fourier basis", "fourier_basis_monthly")
    fourier_basis_wave(df, "doy_sin", "doy_cos", 1, "Yearly Fourier basis", "fourier_basis_yearly")
    fourier_basis_wave(df, "doy_sin_2", "doy_cos_2", 2, "Yearly Fourier basis", "fourier_basis_yearly_k2") # k=2 adds an extra Fourier harmonic so the model can capture more complex seasonality

    # seasonal curves
    seasonal_curve(df, "sales", 2, "Sales seasonal curve") # plot k=1?

    # fourier unit-circle plots
    fourier_unit_circle(df, "dow_cos", "dow_sin", "Cyclical day-of-week scatter plot", "fourier_unit_weekly")
    fourier_unit_circle(df, "doy_cos", "doy_sin", "Cyclical day-of-year scatter plot", "fourier_unit_daily")

    # monthly average for specified 
    monthly_averages = monthly_avg(df, "sales")

    # weekday average for specified metric
    weekday_averages = []
    for month in range(1, 13):
        average = weekday_avg(df, month, "sales")
        average["month"] = month
        weekday_averages.append(average)

    # percentage uplift for specified factor, month, and metric
    all_uplifts = []
    for factor in UPLIFT_COLS:
        for month in range(1, 13):
            uplift = uplifts(df, factor, month, "sales")
            uplift["factor"] = factor
            uplift["month"] = month
            all_uplifts.append(uplift)

    return pd.concat(weekday_averages, ignore_index=True), monthly_averages, pd.concat(all_uplifts, ignore_index=True)
