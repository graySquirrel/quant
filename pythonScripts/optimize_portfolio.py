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
import scipy.optimize as spo
from panda_utils import get_data, plot_data
import datetime as dt

#riskFreeRate = 0.02123 # 3 month T bill 8/28/2018
riskFreeRate = 0.0 

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
    #print "daily risk free rate ",daily_riskFreeRate
    k = pow(252, 0.5) # daily k 
    sharpe_ratio = k*(daily_rets - daily_riskFreeRate).mean() / \
                    daily_rets.std()
    #print "risk ", std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def error_calc(alloc, data, opt):
    """ Compute error (as risk) of a given alloc on a given set of stocks
    Parameters
    ----------
    alloc: array of allocations, each alloc [0-1], must sum to 1
    data: dataframe of stocks
    
    Returns error (risk) as a single real value.
    """
    # calculate portfolio, initial starting point is 1.0
    portfolio = compute_portfolio_value(data, alloc, 1)
    daily = compute_daily_returns(portfolio)
    cumulative, avg, risk, sharpe = compute_portfolio_stats(portfolio, daily)
    if opt=='risk':
        return risk
    elif opt =='cumulative':
        return -cumulative
    elif opt =='sharpe':
        return -sharpe
    
def fit_portfolio_alloc(data, error_func, optimize):
    """ Find an alloc array that optimizes a certain aspect.
    
    Parameters
    ----------
    data: dataframe indexed by date, each col is a daily value of a stock
    error_func: function that computes the total risk of an alloc
    minimize: string enums for 'risk', 'cumulative', 'sharpe'
    
    Returns alloc that optimizes what you want.
    """
    num = data.shape[1]
    Initial = np.ones(num)/num
    lower = np.zeros(num)
    upper = np.ones(num)
    mybounds = spo.Bounds(lower, upper)
    result = spo.minimize(error_calc, Initial, args=(data,optimize,), method='SLSQP', \
                          bounds=mybounds, \
                          constraints=({ 'type': 'eq', 'fun': lambda inputs: 1-np.sum(inputs) }),\
                          options={'disp':False})
    return result.x

def optimize_portfolio(sd, ed, syms, opt, gen_plot=False):
    dates = pd.date_range(sd, ed)
    df = get_data(syms, dates, addSPY=False)
    df = fill_missing_values(df)
    spy = get_data([],dates)
    alloc = fit_portfolio_alloc(df, error_calc, optimize=opt)
    portfolio = compute_portfolio_value(df, alloc, 1)
    daily = compute_daily_returns(portfolio)
    cumulative, avg, risk, sharpe = compute_portfolio_stats(portfolio, daily)
    if gen_plot:
        spy['portfolio'] = portfolio
        rel = spy/spy.iloc[0]
        plt.plot(rel)
        tit = "portfolio return optimizing "+opt+" vs. SPY"
        plt.title(tit)
        plt.legend(rel)
        plt.show()
    return alloc, cumulative, avg, risk, sharpe
    
def test_run():
    opts = ['risk','cumulative','sharpe']
    syms=['GOOG','AAPL','GLD','XOM']
    for opt in opts:
        allocs, cr, adr, sddr, sr = \
        optimize_portfolio(sd=dt.datetime(2010,1,1), ed=dt.datetime(2010,3,31), \
                           syms=syms, opt=opt, gen_plot=True)
        dr = pd.DataFrame(allocs).T
        dr.columns = syms
        print "allocations for opt",opt,"\n", dr.to_string(index=False)
        print "sanity sum", sum(allocs)
        print "cumulative return", cr
        print "average daily return", adr
        print "risk",sddr
        print "sharpe ratio", sr
    
if __name__ == "__main__":
    test_run()
    
#def test_run_bak():
#    # Read data
#    dates = pd.date_range('2010-01-01', '2010-12-31') 
#    symbols = ['AAPL', 'XOM', 'GOOG', 'GLD']
#    df = get_data(symbols, dates, addSPY=False)
#    spy = get_data([],dates)
#    df = fill_missing_values(df)
#    start_val = 1000000
#    alloc=[0.3, 0.1, 0.2, 0.4]
#    if sum(alloc) != 1.0:
#        print "allocations must sum to 1"
#        raise
#    portfolio = compute_portfolio_value(df, alloc, start_val)
#    daily = compute_daily_returns(portfolio)
#    cumulative, avg, risk, sharpe = compute_portfolio_stats(portfolio, daily)
#    print "cumulative return ",cumulative
#    print "average daily return ", avg
#    print "risk (std of daily return) ", risk
#    print "sharpe ratio ", sharpe
#    # plot relative performance of portfolio vs SPY
#    spy['portfolio'] = portfolio
#    rel = spy/spy.iloc[0]
#    plt.plot(rel)
#    plt.title('portfolio return vs. SPY')
#    plt.legend(rel)
#    plt.show()