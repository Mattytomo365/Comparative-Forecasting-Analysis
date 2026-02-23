from src.analysis.eda import plot_all
from src.dataset.load_save import load_csv
from results.save_results import save_results
'''
Module for data analysis performed on historical data
Produces figures and metrics which explain sales drivers and patterns
'''

def run(data_path="data/sales_daily_processed.csv"):
    df = load_csv(data_path)
    weekday_avg, monthly_avg = plot_all(df)

    save_results(weekday_avg, "weekday averages") # save results to /results for later use
    save_results(monthly_avg, "monthly averages")

if __name__ == "__main__":
    run()
