from src.dataset.load_save import load_csv
from src.analysis.performance_analysis import plot_all

'''
Module for orchestration of model comparisons and residuals
'''

def run(data_path="data/sales_daily_processed.csv", target="sales"):
    df = load_csv(data_path)
    plot_all()

# use data from all model results datasets (post and prior tuning) with included predictions, average metrics if necessary?
# produce ranked table
# produce residual graph with all models on
# error distribution comparison
# rolling error plots

if __name__ == "__main__":
    run()
