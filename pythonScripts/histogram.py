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
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, dates)
    #df.fillna(method='ffill', inplace=True)
    #df.fillna(method='bfill', inplace=True)
    daily = compute_daily_returns(df)
    plot_data(df)
    plot_data(daily)
    daily['SPY'].hist(bins=20,label="SPY")
    daily['XOM'].hist(bins=20,label="XOM")
    plt.legend(loc='upper right')
    mean = daily.mean()
    std = daily.std()
    k = daily.kurtosis()
    
    plt.axvline(mean['SPY'],color='w',linestyle='dashed',linewidth=2)
    plt.axvline(mean['XOM'],color='w',linestyle='dashed',linewidth=2)
    plt.axvline(std['SPY'],color='r',linestyle='dashed',linewidth=2)
    plt.axvline(-std['SPY'],color='r',linestyle='dashed',linewidth=2)
    plt.axvline(std['XOM'],color='r',linestyle='dashed',linewidth=2)
    plt.axvline(-std['XOM'],color='r',linestyle='dashed',linewidth=2)
    plt.show()
    print "mean is ",mean," and std is ",std
    print "kurtosos is ", k

if __name__ == "__main__":
    test_run()
