# Housing Price Prediction and Economic Analysis

This project performs housing price prediction using various regression methods and incorporates economic data from the Federal Reserve Economic Data (FRED) API to analyze macroeconomic factors affecting house prices.

---

## Overview

- Fetches multiple economic time series data from the [FRED API](https://fred.stlouisfed.org/) using the `pystlouisfed` Python package.
- Loads and explores a housing dataset (`kc_house_data.csv`) with features such as square footage, bedrooms, bathrooms, year built, and more.
- Conducts exploratory data analysis (EDA) including correlation heatmaps.
- Implements various linear regression models to predict house prices:
  - **Simple linear regression** using `statsmodels`, `scipy.stats`, `sklearn`, and manual calculations.
  - **Gradient descent optimization** for linear regression (both simple and multivariate).
  - **Multivariate linear regression** with feature normalization and standard scaling.
- Utilizes macroeconomic variables from FRED to predict national median house prices.
- Applies statistical tests (correlation and p-value) for feature selection.
- Visualizes regression fits and convergence of gradient descent.
  
---

## Data Sources

- **Housing Dataset:** `kc_house_data.csv` — includes house attributes and prices.
- **Macroeconomic Data:** FRED API series such as:
  - 30-Year Fixed Mortgage Rate (`MORTGAGE30US`)
  - Federal Funds Rate (`FEDFUNDS`)
  - Consumer Price Index (`CPIAUCSL`)
  - Unemployment Rate (`UNRATE`)
  - Job Openings (`JTSJOL`)
  - Treasury Yield Spreads (`T10Y3M`, `T10Y2Y`)
  - Inflation Expectations (`T10YIE`)
  - High Yield Bond Option Adjusted Spread (`BAMLH0A0HYM2`)
  - Credit Card Delinquencies (`DRCCLACBS`)
  - Home Price Index (`CSUSHPINSA`)
  - Median House Price (`MSPUS`)
  - Treasury Yields (2Y, 10Y, 30Y)
  - Producer Price Index (`PCUOMFGOMFG`)
  - AAA Corporate Bond Yield (`WAAA`)

---

## Key Functionalities

### Data Fetching & Preparation
- Downloads monthly economic indicators via the FRED API and resamples to monthly frequency.
- Loads and preprocesses housing data, dropping irrelevant columns.
- Computes correlation matrices to understand relationships among features.

### Regression Analysis

#### Simple Linear Regression
  - `statsmodels.OLS`
  - `scipy.stats.linregress`
  - `sklearn.linear_model.LinearRegression`
  - Manual calculation via formula
  - Gradient Descent optimization

#### Multivariate Linear Regression
- Applies feature normalization and scaling.
- Fits using gradient descent and sklearn's `LinearRegression`.
- Validates results with `statsmodels.OLS`.

### Economic Indicator Modeling
- Loads macroeconomic indicators.
- Performs feature scaling and normalization.
- Selects significant features based on correlation and p-values.
- Builds predictive models for national median house prices.

---

# Regression Diagnostics and Model Evaluation

Comprehensive diagnostics and evaluation of multiple linear regression models, with a focus on detecting issues such as multicollinearity, seasonality, outliers, autocorrelation, stationarity, normality, and heteroscedasticity. It also includes techniques for in-sample and out-of-sample prediction visualization, as well as Leave-One-Year-Out Cross Validation (LOYO-CV).

---

## Key Features

### 1. Multicollinearity Detection
- Uses **Variance Inflation Factor (VIF)** to detect highly correlated independent variables.
- VIF values greater than 5 indicate significant multicollinearity.
- Includes two methods for calculating VIF: via `statsmodels` and matrix inversion.

### 2. Goodness-of-Fit Statistics
- Computes and reports:
  - R-squared
  - Adjusted R-squared
  - Residual RMSE (Root Mean Squared Error)
  - Residual MAE (Mean Absolute Error)

### 3. Residual Analysis
- Summarizes residual statistics: mean, standard deviation, skewness, kurtosis, percentiles.
- Visualizes residuals and dependent variable seasonality through boxplots by quarter and month.

### 4. Seasonality Testing
- Encodes seasonality (quarterly and monthly) using one-hot encoding.
- Performs regression of dependent variable and residuals on seasonal dummies.
- Conducts F-tests on seasonal effects.

### 5. Outlier Influence Analysis
- Calculates Cook's Distance to identify influential observations.
- Plots Cook's Distance and annotates potential leverage points (Cook's D > 1).

### 6. Autocorrelation and Stationarity Tests
- Plots Autocorrelation Function (ACF) and Partial ACF (PACF).
- Performs Durbin-Watson test for autocorrelation.
- Conducts stationarity tests: Augmented Dickey-Fuller (ADF) and KPSS tests.

### 7. Normality Tests of Residuals
- Generates QQ plots and kernel density estimates.
- Performs Shapiro-Wilk, Anderson-Darling, Jarque-Bera, and Kolmogorov-Smirnov tests for normality.

### 8. Heteroscedasticity Tests
- Conducts Breusch-Pagan, White’s test, and ARCH test for heteroscedasticity.

### 9. Model Prediction Visualization
- In-sample prediction plots.
- Out-of-sample prediction plots over specified date ranges.

### 10. Leave-One-Year-Out Cross Validation (LOYO-CV)
- Performs cross-validation by leaving out one year of data at a time.
- Computes RMSE and MAE for LOYO-CV vs. full model.
- Visualizes actual vs predicted and their differences.

---

## Usage

1. Fit a regression model using `statsmodels.OLS`.
2. Use provided functions to:
   - Check multicollinearity with `colinearality_test()`.
   - Analyze residuals and seasonality.
   - Test for autocorrelation, stationarity, normality, and heteroscedasticity.
   - Plot in-sample and out-of-sample predictions.
   - Perform LOYO-CV for robust model validation.

---

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`
- `sklearn`

---

Feel free to explore the code for deeper insights into each diagnostic step and adapt it for your own regression model analysis.
