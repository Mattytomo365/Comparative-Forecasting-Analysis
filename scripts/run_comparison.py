from src.dataset.load_save import load_csv
from src.analysis.performance_analysis import plot_all

'''
Module for orchestration of model comparisons and residuals
'''

def run(data_path="data/sales_daily_processed.csv", target="sales"):
    plot_all()

if __name__ == "__main__":
    run()
