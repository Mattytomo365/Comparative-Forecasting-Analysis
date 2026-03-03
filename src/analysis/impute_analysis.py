import pandas as pd
'''
Determines whether imputing missing sales values improves forecast accuracy and stability compared to dropping missing observations. 
Reports difference in MAE/RMSE/MASE across rolling-origin folds between imputation and row removal.
'''

def impute_analysis_plots(df: pd.DataFrame) -> None:
    '''
    Centrlalise plotting of metrics plots for analysis of imputation strategy
    '''
    # train and predict using the other dataset
    # generate metrics plots
