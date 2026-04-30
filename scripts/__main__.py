import argparse
import sys
from . import run_comparison, run_data_analysis, run_experiments, run_modelling, run_preprocessing
'''
A clean entry point to the 'running' package, orchestrating all individual modules/scripts within.
Works in terminal, CI, and allows subcommands to be built cleanly

Acts as a thin CLI to centralise the workflow in the correct pipeline order as follows:
1. Load dataset and preprocess data
2. Perform EDA on historical data
3. Run machine learning pipeline to product sales forecasts
7. Execute the comparative evaluation of forecasting approaches
8. Run additional analysis of features, imputation strategy, and training optimisation strategy
'''

def main(argv=None):
    ap = argparse.ArgumentParser(prog="restaurant_forecasting",
                                 description="One-click pipeline for restaurant sales forecasting")
    ap.add_argument("--data", default="data/sales_daily_processed.csv") # path to processed input data
    ap.add_argument("--raw-data", default="data/sales_daily.csv") # path to raw input data
    ap.add_argument("--target", default="sales") # target column name

    sub = ap.add_subparsers(dest="cmd", required=True) # create subcommands

    # `python -m … [parser]
    sub.add_parser("run_preprocessing")
    sub.add_parser("run_data_analysis")
    sub.add_parser("run_modelling")
    sub.add_parser("run_comparison")
    sub.add_parser("run_experiments")
    sub.add_parser("all")

    args = ap.parse_args(argv) # parse CLI arguments into an object

    # for each possible subcommand, call the corresponding stage function
    # using `args.cmd in ("stage","all")` lets "all" run every stage in order

    if args.cmd in ("run_preprocessing", "all"):
        run_preprocessing.run(args.raw_data, args.data)

    if args.cmd in ("run_data_analysis", "all"):
        run_data_analysis.run(args.data)

    if args.cmd in ("run_modelling", "all"):
        run_modelling.run(args.data, args.target)

    if args.cmd in ("run_comparison", "all"):
        run_comparison.run(args.data, args.target)   # inner CV, saves best params per model

    if args.cmd in ("run_experiments", "all"):
        run_experiments.run(args.data, args.target)  # outer rolling-origin, saves OOS + metrics + manifests



if __name__ == "__main__":
    sys.exit(main())
