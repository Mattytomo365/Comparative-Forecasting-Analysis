import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from figures.save_figure import save_figure
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
'''
User-facing eda on historical data to uncover trends and patterns
'''

DOW_ORDER = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"] # set order


def monthly_avg(df: pd.DataFrame, 
                metric: str, 
                name: str) -> pd.DataFrame:
    '''
    Mean metric per calendar month
    '''
    d = df["date"]
    month = d.dt.to_period("M") # year-month keys for year-month grouping
    month_m = (df.groupby(month)[metric].agg(value="mean", n_days="count")
               .rename_axis("month")
               .reset_index())

    # additional labelling
    month_m["label"] = month_m["month"].dt.strftime("%b %Y")
    month_m["value"] = month_m["value"].astype(float).round(2)
    month_m["n_days"] = month_m["n_days"].astype(int)

    fig, ax = plt.subplots()
    m = month_m["month"].dt.to_timestamp()
    a = month_m["value"]

    ax.bar(m, a, width=20, alpha=0.35, label="daily avg")
    ax.plot(m, a, marker="o", linewidth=1.5, label="trend")

    ax.set_xlabel("month")
    ax.set_ylabel("average sales")
    ax.set_title("Average daily sales per month")
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    monthly_labels(ax)
    save_figure(fig, name)

    return pd.DataFrame(month_m[["month", "label", "value", "n_days"]])


def weekday_avg(df: pd.DataFrame, 
                month: str, 
                metric: str) -> pd.DataFrame:
    '''
    Mean metric by weekday for a specified month
    '''
    d = df["date"]
    m = d.dt.month.eq(int(month)) # locates month within dataset

    if not m.any(): # fallback
        return pd.DataFrame({"dow": DOW_ORDER, "value": [0]*7})
    
    dow = d.dt.day_name().str[:3] # day name abbreviation
    day_m = (df.loc[m].groupby(dow)[metric].agg(value="mean") # weekday-based average
            .reindex(DOW_ORDER).fillna(0.0) # maintain fixed order
            .rename_axis("dow")
            .reset_index()) 
    
    day_m["value"] = day_m["value"].astype(float).round(2)
    return pd.DataFrame(day_m[["dow", "value"]])


def weekday_avg_plot(weekday_averages: list[pd.DataFrame]) -> None:
    '''
    Visualisation of weekday averages across all months
    '''

    df = pd.concat(weekday_averages, ignore_index=True)

    fig, ax = plt.subplots()

    by_dow = df.groupby("dow")["value"].mean().reindex(DOW_ORDER)
    ax.bar(by_dow.index, by_dow.values, alpha=0.35, label="weekday avg")
    ax.plot(by_dow.index, by_dow.values, marker="o", linewidth=1.5, label="trend")
    ax.set_xlabel("weekday")
    ax.set_ylabel("average sales")
    ax.set_title("Average weekday sales across all months")
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    save_figure(fig, "weekday_average_total")



def sales_distribution(df: pd.DataFrame, name: str) -> None:
    '''
    Generates a box plot to visualise sales distribution
    '''
    pass


def monthly_labels(ax) -> None:
    '''
    Helper function to improve visibility of values on x-axis
    '''
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.figure.autofmt_xdate()



def fourier_basis_wave(df: pd.DataFrame, 
                       sin_col: str, 
                       cos_col: str, 
                       k: int, 
                       title: str, 
                       name: str) -> None:
    '''
    Plot basis waves at varying harmonic levels visualising fourier features against time
    '''
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


def seasonal_curve(df: pd.DataFrame, 
                   target: str, 
                   k: int, 
                   title: str) -> None:
    '''
    Plot linear combination of baseline waves, visualises what linear models perform on yearly fourier features to understand seasonal shape
    '''
    d = df["date"]
    y = df[target].astype("float").values
    if k == 2:
        s = df["doy_sin_2"].astype(float).values # Fourier features/terms
        c = df["doy_cos_2"].astype(float).values
    else:
        s = df["doy_sin"].astype(float).values
        c = df["doy_cos"].astype(float).values

    # build a 2-D array of the features to make a small design matrix
    X = np.column_stack([np.ones_like(s), s, c]) # [1, sin, cos] so X has shape (num_days, 3)

    # find coefficients that minimise total squared error
    beta, *_ = np.linalg.lstsq(X, y, rcond=None) # beta = [β₀, a, b]

    # make a seasonal curve from calculated weights/coefficients
    y_fitted = X @ beta  # β₀ + a*sin + b*cos

    # actual vs fitted
    fig, ax = plt.subplots()
    ax.plot(d, y, label="actual")
    ax.plot(d, y_fitted, "--", label="seasonal curve (from sin/cos)")
    if title: ax.set_title(title)
    ax.set_ylabel(target)
    ax.set_xlabel("date")
    ax.legend()
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    monthly_labels(ax)

    save_figure(fig, "fourier_seasonal")



def fourier_unit_circle(df: pd.DataFrame, 
                        cos_col: str, 
                        sin_col: str, 
                        title: str, 
                        name: str) -> None:
    '''
    Generates unit-circle plots of fourier features, helping visualise the circular pattern
    '''
    fig, ax = plt.subplots()
    df.plot.scatter(sin_col, cos_col, ax=ax)
    ax.set_aspect("equal", adjustable="box") # circle
    t = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(t), np.sin(t), linewidth=1) # unit-circle line
    if title: ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)

    save_figure(fig, name)



def acf_plots(df: pd.DataFrame, 
              col: str, 
              diff: int,
              name: str, 
              title: str) -> None:
    '''
    Generates autocorrelation plots
    '''
    series = df[col].astype(float)
    diff = series.diff(diff).dropna() # 1st order (non-seasonal/seasonal) differencing

    fig, axes = plt.subplots(3, 1)

    axes[0].plot(diff)
    axes[0].set_title(title)
    axes[0].grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    fig.tight_layout()

    plot_acf(diff, ax=axes[1])
    plot_pacf(diff, ax=axes[2])
    axes[1].set_title(f"ACF {title}")
    axes[2].set_title(f"PACF {title}")
    
    save_figure(fig, name)


def decomposition_plot(df: pd.DataFrame, name: str) -> None:
    '''
    Performs time-series seasonal decomposition
    '''
    fig, ax = plt.subplots()
    decomposition = seasonal_decompose(df["sales"], model='additive', period=7) # weekly seasonal decomposition of sales
    fig = decomposition.plot()
    ax.set_title("Seasonal decomposition plot")
    save_figure(fig, name)


def perform_adf(df: pd.DataFrame) -> None:
    '''
    Performs augmented dickey-fuller test
    '''
    adf_test = adfuller(df["sales"])
    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])



def plot_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Centralised function
    '''
    # fourier baseline waves
    fourier_basis_wave(df, "dow_sin", "dow_cos", 1, "Weekly Fourier basis", "fourier_basis_monthly")
    fourier_basis_wave(df, "doy_sin", "doy_cos", 1, "Yearly Fourier basis", "fourier_basis_yearly")
    fourier_basis_wave(df, "doy_sin_2", "doy_cos_2", 2, "Yearly Fourier basis 2nd Harmonic", "fourier_basis_yearly_k2") # k=2 adds an extra Fourier harmonic so the model can capture more complex seasonality

    # seasonal curves
    seasonal_curve(df, "sales", 2, "Sales seasonal curve")
    # seasonal_curve(df, "sales", 1, "Sales seasonal curve")

    # fourier unit-circle plots
    fourier_unit_circle(df, "dow_cos", "dow_sin", "Cyclical day-of-week scatter plot", "fourier_unit_weekly")
    fourier_unit_circle(df, "doy_cos", "doy_sin", "Cyclical day-of-year scatter plot", "fourier_unit_daily")

    # ACF plots seasonal/non-seasonal
    acf_plots(df, "sales", 1, "acf_sales", "1st Order Non-Seasonal Differencing")
    acf_plots(df, "sales", 7, "acf_sales_seasonal", "Seasonal Differencing with Period 7")

    # decomposition plot
    decomposition_plot(df, "seasonal_decompose")

    # monthly average for specified 
    monthly_averages = monthly_avg(df, "sales", "monthly_average")

    # weekday average for specified metric
    weekday_averages = []
    for month in range(1, 13):
        average = weekday_avg(df, month, "sales")
        average["month"] = month
        weekday_averages.append(average)

    weekday_avg_plot(weekday_averages)

    return pd.concat(weekday_averages, ignore_index=True), monthly_averages
