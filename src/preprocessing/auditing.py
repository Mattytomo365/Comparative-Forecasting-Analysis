import pandas as pd

def audit_stage(df, stage, **extra_metrics):
    summary = {
        "stage": stage,
        "rows": len(df),
        "cols": len(df.columns),
        "missing_sales": int(df["sales"].isna().sum()) if "sales" in df else None,
        "duplicate_dates": int(df["date"].duplicated().sum()) if "date" in df else None,
        "sales_mean": round(float(df["sales"].mean()), 2) if "sales" in df else None,
        "sales_median": round(float(df["sales"].median()), 2) if "sales" in df else None,
        "sales_std": round(float(df["sales"].std()), 2) if "sales" in df else None,
    }
    summary.update(extra_metrics)
    return summary
