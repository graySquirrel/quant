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
import copy
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
    dates = pd.date_range('2010-07-01', '2010-12-31') 
    symbols = ['GOOG','AAPL',  'GLD', 'XOM']
    df1 = get_data(symbols, dates, addSPY=True)
    spy = get_data([],dates)
    df1 = fill_missing_values(df1)
    rel1 = df1/df1.iloc[0]
    plt.plot(rel1)
    plt.legend(rel1)
    plt.show()
    
    
    df = get_data(symbols, dates, addSPY=False)
    df = fill_missing_values(df)

    start_val = 1000000
    alloc=[0.0, 0.3, 0.7, 0.0]
    if sum(alloc) != 1.0:
        print "allocations must sum to 1"
        raise
    portfolio = compute_portfolio_value(df, alloc, start_val)
    daily = compute_daily_returns(portfolio)
    cumulative, avg, risk, sharpe = compute_portfolio_stats(portfolio, daily)
    print "cumulative return ",cumulative
    print "average daily return ", avg
    print "risk (std of daily return) ", risk
    print "sharpe ratio ", sharpe
    # plot relative performance of portfolio vs SPY
    spy['portfolio'] = portfolio
    rel = spy/spy.iloc[0]
    plt.plot(rel)
    plt.title('portfolio return vs. SPY')
    plt.legend(rel)
    plt.show()

if __name__ == "__main__":
    test_run()
