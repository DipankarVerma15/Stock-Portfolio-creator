#!/usr/bin/env python
# coding: utf-8

# In[175]:


import time
import datetime
import pandas as pd
import numpy as np
from scipy.optimize import linprog

#Training data
tickers = stock_tickers = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS","INFY.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS",
    "BAJFINANCE.NS","LICNFNHGP.NS","LT.NS","KOTAKBANK.NS","HCLTECH.NS","ASIANPAINT.NS","ADANIENT.NS", "AXISBANK.NS",
    "MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","BAJAJFINSV.NS","DMART.NS","ONGC.NS","WIPRO.NS","NESTLEIND.NS",
    "NTPC.NS","TATAMOTORS.NS","M&M.NS","JSWSTEEL.NS","ADANIPORTS.NS","POWERGRID.NS","ADANIGREEN.NS","TATASTEEL.NS",
    "COALINDIA.NS","HDFCLIFE.NS","HINDZINC.NS","BAJAJ-AUTO.NS","IOC.NS","SIEMENS.NS", "SBILIFE.NS",
    "HAL.NS","PIDILITIND.NS","TECHM.NS","GRASIM.NS","ADANIPOWER.NS","DLF.NS","VBL.NS","BRITANNIA.NS"]
interval = '1d'
period1 = int(time.mktime(datetime.datetime(2018, 8, 21, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2021, 8, 21, 23, 59).timetuple()))

# Create a dictionary to hold dataframes for each ticker
dataframes = {}

for ticker in tickers:
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    dataframes[ticker] = df['Adj Close']  # Save only the 'Adj Close' column
# Create a combined DataFrame with all the Adj Close prices

combined_df = pd.DataFrame(dataframes)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('C:\\Users\\user\\Downloads\\historical_prices.csv')


x = combined_df.dropna()

# Display the shape of the cleaned DataFrame
print("Original Shape:", combined_df.shape)
print("Cleaned Shape:", x.shape)

T,n=x.shape
returnT = np.zeros((T, n), dtype=float)

for i in range(1, T):
    for j in range(1, n):
        a = x.iloc[i, j]
        b = x.iloc[i - 1, j]
        returnT[i - 1, j - 1] = (a - b) / b

np.savetxt("C:\\Users\\user\\downloads\\Returns1.csv", returnT, delimiter=",")
Returns1 = np.loadtxt('C:\\Users\\user\\downloads\\Returns1.csv', delimiter=',')
Returns = Returns1[:T, 1:n]
n=n-1
ObjFunc = np.concatenate(((1 / T) * np.ones(T), np.zeros(n)))

AvgOfComp = (1 / T) * np.sum(Returns, axis=0)


T_plus_n = T + n
PositiveConst = np.zeros((T, T_plus_n))
NegativeConst = np.zeros((T, T_plus_n))
np.fill_diagonal(PositiveConst[:T, :], -1)
np.fill_diagonal(NegativeConst[:T, :], -1)

for i in range(T):
    for j in range(T, T + n):
        if j - T < n:  # Check if the index is within the valid range
            PositiveConst[i, j] = Returns[i, j - T] - AvgOfComp[j - T]
            NegativeConst[i, j] = -Returns[i, j - T] + AvgOfComp[j - T]


MeanConst = np.concatenate((np.zeros(T), -AvgOfComp))
A = np.vstack((PositiveConst, NegativeConst, MeanConst))
b = np.zeros(2 * T + 1)
b[-1] = -0.001

Aeq = np.zeros((1, T + n))
Aeq[0, T:] = 1.0  # Set the coefficients for the last n variables to 1.0
Beq = np.array([1.0])

lb = np.zeros(T + n)
ub = None

# Create bounds by specifying only lower bounds and setting ub to None for all variables
bounds = [(lb_val, ub) for lb_val in lb]

# Call linprog without specifying ub in bounds
result = linprog(c=ObjFunc, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=Beq, bounds=bounds)



if result.success:
    Weights = result.x[T:T + n+1]
    print("Optimal Weights:", Weights)
else:
    print("Optimization failed:", result.message)
    
#Testing data
tickers = stock_tickers = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS","INFY.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS",
    "BAJFINANCE.NS","LICNFNHGP.NS","LT.NS","KOTAKBANK.NS","HCLTECH.NS","ASIANPAINT.NS","ADANIENT.NS", "AXISBANK.NS",
    "MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","BAJAJFINSV.NS","DMART.NS","ONGC.NS","WIPRO.NS","NESTLEIND.NS",
    "NTPC.NS","TATAMOTORS.NS","M&M.NS","JSWSTEEL.NS","ADANIPORTS.NS","POWERGRID.NS","ADANIGREEN.NS","TATASTEEL.NS",
    "COALINDIA.NS","HDFCLIFE.NS","HINDZINC.NS","BAJAJ-AUTO.NS","IOC.NS","SIEMENS.NS", "SBILIFE.NS",
    "HAL.NS","PIDILITIND.NS","TECHM.NS","GRASIM.NS","ADANIPOWER.NS","DLF.NS","VBL.NS","BRITANNIA.NS"]
interval = '1d'
period1 = int(time.mktime(datetime.datetime(2021, 8, 21, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2023, 8, 21, 23, 59).timetuple()))

# Create a dictionary to hold dataframes for each ticker
dataframes = {}

for ticker in tickers:
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    dataframes[ticker] = df['Adj Close']  # Save only the 'Adj Close' column
# Create a combined DataFrame with all the Adj Close prices

new_combined_df = pd.DataFrame(dataframes)

# Save the combined DataFrame to a CSV file
new_combined_df.to_csv('C:\\Users\\user\\Downloads\\Testing_data.csv')

x_new = new_combined_df.dropna()

# Display the shape of the cleaned DataFrame
print("Original Shape:", new_combined_df.shape)
print("Cleaned Shape:", x_new.shape)


T,n=x_new.shape
new_returnT = np.zeros((T, n), dtype=float)

for i in range(1, T):
    for j in range(1, n):
        a = x.iloc[i, j]
        b = x.iloc[i - 1, j]
        new_returnT[i - 1, j - 1] = (a - b) / b

np.savetxt("C:\\Users\\user\\downloads\\Returns1.csv", new_returnT, delimiter=",")
new_Returns1 = np.loadtxt('C:\\Users\\user\\downloads\\Returns1.csv', delimiter=',')
new_Returns = new_Returns1[:T, 1:n]

T, n = new_Returns.shape 
Optimized_portfolio = np.dot(new_Returns, Weights)

# Finding portfolio statistics
LeftoverData = new_Returns
PortfolioReturns = np.dot(LeftoverData, Weights)

MeanPortfolioReturns = np.mean(PortfolioReturns)
StdDev = np.std(PortfolioReturns, ddof=0)
SharpeRatio = MeanPortfolioReturns / StdDev


# Naive Portfolio
EqualWts = np.ones(n) / n
NaivePortfolioReturns = np.dot(LeftoverData, EqualWts)

MeanNaive = np.mean(NaivePortfolioReturns)
StdDevNaive = np.std(NaivePortfolioReturns, ddof=0)
SharpeRatioNaive = MeanNaive / StdDevNaive

Portfolios = ["Using LINPROG", "Using NAIVE"]
Means = [MeanPortfolioReturns, MeanNaive]
StandardDev = [StdDev, StdDevNaive]
SharpeRat = [SharpeRatio, SharpeRatioNaive]

Comparison = {"Portfolios": Portfolios, "Means": Means, "StandardDev": StandardDev, 
              "SharpeRat": SharpeRat}
for i,j in Comparison.items():
    print(i,':-',j)

