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
os.chdir("C:\\Users\\febner\\Documents\\CourseraDataScience\\quant\\udacity-machine-learning-for-trading-master\\udacity-machine-learning-for-trading-master")

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # TODO: Your code here
    # Note: Returned DataFrame must have the same number of rows
    #daily_returns = df.copy()
    df = df / df.shift(1) - 1
    df[df.isnull()] = 0
    return df
    #daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    #daily_returns.iloc[0, :] = 0
    #return daily_returns


def test_run():
    # Read data
    dates = pd.date_range('2005-12-31', '2014-12-07') 
    #symbols = ['SPY','XOM']
    symbols = ["JAVA","FAKE1","FAKE2"]
    df = get_data(symbols, dates)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    plot_data(df)

    # Compute daily returns
    #daily_returns = compute_daily_returns(df)
    #plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")


if __name__ == "__main__":
    test_run()
