# Experimental Evalutation of Forecasting Approaches to Restaurant Sales

## Overview

This is my BSc Computer Science capstone project. The aim is to establish an accurate and efficient approach to time-series forecasting for hospitality sales. 

Three models - ARIMA, XGBoost, Lasso - will all be trained and tested on the same cleaned and preprocessed dataset. The dataset contains exogenous factors (e.g. weather/events/holidays) and engineered features (Fourier/cyclical features and lags/rolling statistics) to establish more meaningful relationships between catgeories and model seasonality throughout.

Onehot schema generated during data encoding will be persisted and re-used, minimising drifts in data between train/test and inference. Models will also be tested using rolling-origin backtesting, reducing overfitting of models and the leakage of data, and models will be cross-validated with a time-aware expanding/rolling window approach, paired with hyperparameter tuning to establish parameter configurations which yeild the best accuracy.

Performance metrics such as MAPE, MAE, and RMSE will be reported both before and after model tuning, and across all models for comparison. The Lasso approach will hold a strong baseline for the other more complex models to be compared against. Evaluation interpretations and findings will be presented in both tabular and graphical formats to convey findings of model performance and EDA performed on the historical dataset effectively.

The top model will undergo further and more deep analysis to establish features with the most significant impact on sales, casuing fluctuations. Ablation tests and permutation importance will be carried out to establish this, with relevant visualisations being produced.

The approach which presents the best time-aware accuracy will be integrated into the full-stack `Hospitality-Sales-Forecasting-Platform` web-application, which will be resumed after this project.

