# Daily Temperature By Town - Two Sigma
# You are given the daily temperature reading of P towns and the daily temperature reading of New York City (NYC) for N days. Write a function that returns the answer to the following five questions: 
# • Q1: The name of the place (either a town or NYC) with the largest variation in the daily temperature. Use the standard deviation to measure the variation. 
# • Q2: The median daily temperature of NYC when the daily temperature of Town2 is between 90 and 100 degrees (inclusive of 90 and 100). Round your answer to the nearest integer. 
# • Q3: Fit P simple linear models with intercept using least squares to predict the daily temperature of NYC given each individual town. Find the sum of the absolute values of the regression coefficients, rounded to the nearest integer.
# • Q4: For the given data, find the town that is most predictive of the daily temperature of NYC. By most predictive, we mean the town that leads to the lowest mean squared error (MSE) on the given data when fit using a linear model with intercept 
# • Q5: For the given data, find the two towns that are jointly most predictive of the daily temperature of NYC. As before, we mean the two towns that lead to the lowest MSE when fit using a linear model with intercept.
# full description with input examples: https://www.chegg.com/homework-help/questions-and-answers/daily-temperature-town-given-daily-temperature-reading-p-towns-daily-temperature-reading-n-q54885306
import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools
input_df = pd.DataFrame({'Town1': [70,65,87,67],'Town2': [95,88,91,101],'Town3': [34,45,23,34],
                         'Town4': [46,24,35,55],'Town5': [10,32,10,15],'NYC': [50,51,78,88,]})
def daily_temperature(input_df):
    res = []
    # Q1
    largest_variation = input_df.std(axis=0).idxmax()
    res.append(largest_variation)
    # Q2
    nyc_median = np.median(input_df.loc[(input_df['Town2'] >= 90) & (input_df['Town2'] <= 100)].NYC)
    res.append(str(round(nyc_median)))
    # Q3 & Q4
    min_mse, min_mse_town = float('inf'), None
    coeff_sum = 0
    for col in input_df.columns[:-1]:
        dep_var = input_df.NYC
        ind_var = input_df[col]
        ind_var = sm.add_constant(ind_var)
        reg = sm.OLS(dep_var, ind_var).fit()
        coeff_sum += sum([abs(x) for x in list(reg.params[1:])]) # not include coeff for constant
        mse = reg.mse_resid
        if mse < min_mse:
            min_mse = mse
            min_mse_town = col
    res.extend([str(round(coeff_sum)), min_mse_town])
    # Q5
    min_mse2, min_mse_town2 = float('inf'), None
    for combo in itertools.combinations(input_df.columns[:-1], 2):
        dep_var = input_df.NYC
        ind_var = input_df[list(combo)]
        ind_var = sm.add_constant(ind_var)
        reg = sm.OLS(dep_var, ind_var).fit()
        mse = reg.mse_resid
        if mse < min_mse2:
            min_mse2 = mse
            min_mse_town2 = list(combo)
    res.extend(min_mse_town2)
    return res
print(daily_temperature(input_df)) #['NYC', '64', '5', 'Town2', 'Town1', 'Town2']

n = 5
x_knots = [-2.0, -1.0, 0.0, 1.0, 2.0]
y_knots = [0.0, 10.0, 15.0, 0.0, 5.0]
x_input = -0.3
import bisect
def linear_interpolate(n, x_knots, y_knots, x_input):
    if n < 2: return
    if n == 2:
        if x_knots[0] == x_knots[1]:
            return 
        else:
            slope = (y_knots[1]-y_knots[0])/(x_knots[1]-x_knots[0])
            intercept = y_knots[1]-x_knots[1]*slope
            return x_input*slope+intercept
    pos = bisect.bisect_left(x_knots, x_input)
    if 1 <= pos <= n-1:
        x1, y1 = x_knots[pos-1], y_knots[pos-1]
        x2, y2 = x_knots[pos], y_knots[pos]
    elif pos == 0:
        x1, y1 = x_knots[0], y_knots[0]
        x2, y2 = x_knots[1], y_knots[1]
    else:
        x1, y1 = x_knots[n-2], y_knots[n-2]
        x2, y2 = x_knots[n-1], y_knots[n-1]
    if x1 == x2: return
    slope = (y2-y1)/(x2-x1)
    intercept = y1 - x1*slope
    return x_input*slope+intercept

print(linear_interpolate(n, x_knots, y_knots, x_input)) #13.5
