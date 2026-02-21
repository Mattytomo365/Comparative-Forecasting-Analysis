from src.dataset.load_save import load_csv
from src.analysis.feature_analysis import plot_all
from results.save_results import save_results
'''
Module for feature analysis, data preprocessing analysis, and tuning analysis
Produces figures and metrics regarding featrure importance, missing value imputation impact, and tuning impact
'''

def run(data_path=None, target=None):
    df = load_csv(data_path)
    uplifts = plot_all(df)
    save_results(uplifts, "sales uplifts")
