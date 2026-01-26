import argparse
import sys
from . import preprocess, analyse, train, compare, analyse_features, report
'''
A clean entry point to the 'running' package, orchestrating all individual modules/scripts within.
Works in terminal, CI, and allows subcommands to be built cleanly

Acts as a thin CLI to centralise the workflow in the correct pipeline order as follows:
1. Load dataset and preprocess data
2. Perform EDA on historical data
3. Rolling-origin splits on data
4. Train models (Lasso, ARIMA, XGBoost)
5. Evaluate model performance and generate metrics
6. Evaluate feature importance and correlation to daily sales
7. Save plots/figures
8. Persist outputs into relevant artifacts (e.g. model registry, figures)
'''

def main(argv=None):
    ap = argparse.ArgumentParser(prog="restaurant_forecasting",
                                 description="One-click pipeline for restaurant sales forecasting")
    ap.add_argument("--data", default="data/restaurant_data_processed.csv") # path to processed input data
    ap.add_argument("--raw-data", default="data/sales_daily.csv") # path to raw input data
    ap.add_argument("--target", default="sales") # target column name

    sub = ap.add_subparsers(dest="cmd", required=True) # create subcommands

    sub.add_parser("preprocess") # `python -m … preprocess`
    sub.add_parser("analyse") # `python -m … analyse`
    sub.add_parser("train") # `python -m … train`
    sub.add_parser("compare") # `python -m … compare`
    sub.add_parser("analyse_features") # `python -m … analyse_features`
    sub.add_parser("report") # `python -m … report`
    sub.add_parser("all") # `python -m … all`

    args = ap.parse_args(argv) # parse CLI arguments into an object

    # For each possible subcommand, call the corresponding stage function.
    # Using `args.cmd in ("stage","all")` lets "all" run every stage in order.

    if args.cmd in ("preprocess", "all"):
        preprocess.run(args.raw_data, args.data)

    if args.cmd in ("analyse", "all"):
        analyse.run(args.data)

    if args.cmd in ("train", "all"):
        train.run(args.data, args.target)

    if args.cmd in ("compare", "all"):
        compare.run(args.data, args.target)   # inner CV, saves best params per model

    if args.cmd in ("analyse_features", "all"):
        analyse_features.run(args.data, args.target)  # outer rolling-origin, saves OOS + metrics + manifests

    if args.cmd in ("report", "all"):
        report.run_ablation(args.data, args.target)  # permutation/SHAP, saves figs/tables


if __name__ == "__main__":
    sys.exit(main())
