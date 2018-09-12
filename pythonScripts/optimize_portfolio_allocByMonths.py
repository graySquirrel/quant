# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:31:46 2018
Create an 'optimal' allocation of assets for a given portfolio, based on 
minimizing sets of contiguous months of returns. Apply the 'optimal' 
allocation and test if its a good strategy.

The idea is as follows: we have historical data from months previous and
some months are (posited to be) more important predicters of future response.
I want the allocation for Month N below (today is, say first day of Month N).
So, take contiguous dates [M-3, M-2,M-1] and [M-14, M-13, M-12], maybe take
more in previous years (if we think that month-anality is good predicter).
For a set of contiguous date ranges, get an allocation prediction for Month N.
THEN, iterate over lots of months, and save the monthly 'optimized' allocs.
THEN, simulate performance of a portfolio that used the 'optimized' allocs.
How well does it perform?

    M1   M2   M3   M4   M5   M6   M7   M8   M9   M10  M11  M12
Y1                 M-14 M-13 M-12
Y2            M-3  M-2  M-1  N
Y3
@author: febner
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import panda_utils as pu 
import datetime as dt
from datetime import date
import time
import datetimeManip as dtm
from dateutil import relativedelta

os.chdir("C:\\Users\\febner\\Documents\\CourseraDataScience\\quant\\udacity-machine-learning-for-trading-master\\udacity-machine-learning-for-trading-master")
"""
take a list of previous contiguous months, and return a date range that 
covers that month

currentMonth: a date in the month that we make date ranges relative to.
monthRange: a 2-tuple of negative numbers, e.g. (-2,-1) which denotes
  a window of previous months.
"""
def create_date_range(currentMonth, monthRange):
    mr = sorted(monthRange) # make sure it goes from least to most
    #print mr
    cm = dtm.mkFirstOfMonth(currentMonth)
    sd = cm + relativedelta.relativedelta(months=mr[0])
    ed = dtm.mkLastOfMonth(cm + relativedelta.relativedelta(months=mr[1]))
    #print sd, ed
    return sd, ed

def compute_cr(portfolio, daily, rfr):
    c,a,r,s=pu.compute_portfolio_stats(portfolio, daily, rfr)
    return c

"""
compute_returns_from_allocs
takes allocDf
returns cr for each month in new col of allocDf
and returns cumulative return for the period
and graphs the portfolio value
"""
def compute_returns_from_allocs(allocDf):
    riskFreeRate = 0.0 
    dfAll = allocDf.copy()
    syms = dfAll.columns
    monthly = []
    for index, row in dfAll.iterrows():
        dates = pd.date_range(dtm.mkFirstOfMonth(index), dtm.mkLastOfMonth(index))
        df = pu.get_data(syms, dates, addSPY=False)
        df = pu.fill_missing_values(df)
        portfolio = pu.compute_portfolio_value(df, row, 1)
        daily = pu.compute_daily_returns(portfolio)
        monthly.append(compute_cr(portfolio, daily, riskFreeRate))
    dfAll['monthly'] = monthly
    dfAll['cumulative'] = (dfAll['monthly'] + 1).cumprod()    
    return dfAll
    
def test_run():
    #opts = ['risk','cumulative','sharpe']
    opt = 'cumulative'
    #syms=['GOOG','AAPL','GLD','XOM']
    syms=['$DJI','$SPX','GLD']
    monthsList=[(-3,-1), (-14, -12)]
    #dates = pd.date_range('2002-01-01', '2012-08-31', freq='BM') 
    dates = pd.date_range('2006-01-01', '2012-08-31', freq='BM') 
    allocsByMonth=[]
    for curMon in dates:
        allocsList=[]
        for tup in monthsList:
            #print tup
            #curMon = dt.datetime(2008,1,1)
            sd, ed = create_date_range(curMon, tup)
            #print sd, ed
            allocs, cr, adr, sddr, sr = \
            pu.optimize_portfolio(sd=sd, ed=ed, \
                                  syms=syms, opt=opt, gen_plot=False)
            dr = pd.DataFrame(allocs).T
            dr.columns = syms
            #print "allocations for opt",opt,"\n", dr.to_string(index=False)
            #print "sanity sum", sum(allocs)
            #print "cumulative return", cr
            #print "average daily return", adr
            #print "risk",sddr
            #print "sharpe ratio", sr
            allocsList.append(allocs)
        allocArr = pd.DataFrame(allocsList)
        allocArr.columns = syms
        #print curMon
        #print allocArr
        #print allocArr.mean()
        allocsByMonth.append(allocArr.mean())
    #print allocsByMonth
    allocDf = pd.DataFrame(allocsByMonth, index=dates)
    res = compute_returns_from_allocs(allocDf)
    print res.round(3)
    spy = pu.get_data([],dates)
    spy['portfolio'] = res.cumulative
    spy = pu.normalize_data(spy)
    pu.plot_data(spy)
    
if __name__ == "__main__":
    test_run()
    