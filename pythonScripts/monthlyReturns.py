# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:31:46 2018

@author: febner
"""

"""Compute daily returns."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from panda_utils import get_data, plot_data

riskFreeRate = 0.02123 # 3 month T bill 8/28/2018

os.chdir("C:\\Users\\febner\\Documents\\CourseraDataScience\\quant\\udacity-machine-learning-for-trading-master\\udacity-machine-learning-for-trading-master")
def compute_daily_returns(df):
    """Compute and return the daily return values."""
    df = df / df.shift(1) - 1
    df[df.isnull()] = 0
    return df

def fill_missing_values(df):
    df.fillna(method='ffill', inplace=True) # fill in missing values
    df.fillna(method='bfill', inplace=True)
    return df

def compute_portfolio_value(df, alloc, initial):
    normed = df/df.iloc[0]
    alloced = normed*alloc
    pos_vals = alloced * initial
    port_val_daily = pos_vals.sum(axis=1)
    return port_val_daily

def compute_portfolio_stats(port, daily_rets):
    daily_rets = daily_rets[1:]
    cum_ret = port[-1]/port[0] - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    daily_riskFreeRate = pow(1+riskFreeRate,1/252.0) - 1
    print "daily risk free rate ",daily_riskFreeRate
    k = pow(252, 0.5) # daily k 
    sharpe_ratio = k*(daily_rets - daily_riskFreeRate).mean() / \
                    daily_rets.std()
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def test_run():
    # Read data
    dates = pd.date_range('2000-01-01', '2012-12-31', freq='BM') 
    symbols = ['SPY']
    df = get_data(symbols, dates)
    df = fill_missing_values(df)
    monthly = compute_daily_returns(df) #its monthly
    plt.plot(monthly)
    #monthly.groupby(monthly.index.month).boxplot()

    monthlyWithMonth = monthly.copy()
    monthlyWithMonth['Month'] = monthly.index.month
    monthlyWithMonth.boxplot(by='Month')
    plt.suptitle("")
    plt.show()
    #avg = monthly.groupby(monthly.index.month).mean()
    #std = monthly.groupby(monthly.index.month).std()
    #tot = monthly.groupby(monthly.index.month).sum()
    midtermYearsMonthly = monthlyWithMonth.copy()
    midtermYearsMonthly['Year'] = monthly.index.year
    midtermYearsMonthly = midtermYearsMonthly.loc[midtermYearsMonthly['Year'].isin([2002,2006,2010])]
    midtermYearsMonthly[['SPY','Month']].boxplot(by='Month')
    plt.title("boxplot of midterm years")
    plt.suptitle("")
    plt.show()
    
    electionYearsMonthly = monthlyWithMonth.copy()
    electionYearsMonthly['Year'] = monthly.index.year
    electionYearsMonthly = electionYearsMonthly.loc[electionYearsMonthly['Year'].isin([2000,2004,2008,2012])]
    electionYearsMonthly[['SPY','Month']].boxplot(by='Month')
    plt.title("boxplot of election years")
    plt.suptitle("")
    plt.show()
    
if __name__ == "__main__":
    test_run()
