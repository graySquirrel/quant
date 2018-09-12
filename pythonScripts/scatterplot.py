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

os.chdir("C:\\Users\\febner\\Documents\\CourseraDataScience\\quant\\udacity-machine-learning-for-trading-master\\udacity-machine-learning-for-trading-master")
def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # TODO: Your code here
    # Note: Returned DataFrame must have the same number of rows
    #daily_returns = df.copy()
    df = df / df.shift(1) - 1
    df[df.isnull()] = 0
    return df

def test_run():
    # Read data
    dates = pd.date_range('2009-01-01', '2012-12-31') 
    #symbols = ['SPY','XOM']
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)
    #df.fillna(method='ffill', inplace=True)
    #df.fillna(method='bfill', inplace=True)
    daily = compute_daily_returns(df)
    plot_data(df)
    plot_data(daily)
    daily.plot(kind='scatter',x='SPY',y='XOM')
    beta_XOM,alpha_XOM=np.polyfit(daily['SPY'],daily['XOM'],1)
    print "XOM b ",beta_XOM," alph ",alpha_XOM
    plt.plot(daily['SPY'],beta_XOM*daily['SPY']+alpha_XOM, '-',color='r')
    plt.show()
    daily.plot(kind='scatter',x='SPY',y='GLD')
    beta_GLD,alpha_GLD=np.polyfit(daily['SPY'],daily['GLD'],1)
    print "GLD b ",beta_GLD," alph ",alpha_GLD
    plt.plot(daily['SPY'],beta_GLD*daily['SPY']+alpha_GLD, '-',color='r')
    plt.show()
    print daily.corr(method='pearson')
if __name__ == "__main__":
    test_run()
