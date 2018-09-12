# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:55:42 2018

@author: febner
"""

import os
import pandas as pd
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
        # TODO: Read and join data for each symbol
        df1 = pd.read_csv(symbol_to_path(symbol),index_col='Date',usecols=['Date','Adj Close'],parse_dates=True,na_values=['nan'])
        df1=df1.rename(columns={'Adj Close':symbol})
        df=df.join(df1,how="inner")

    return df

def plot_data(df,title="Stock Prices"):
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def test_run():
    # Define a date range
    dates = pd.date_range('2010-01-01', '2010-12-31')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']
    
    # Get stock data
    df = get_data(symbols, dates)
    df=df.sort_index()

    plot_data(df/df.iloc[0])
    
    print "mean data \n",df.mean()
    print "stdev data \n",df.std()
    print "stdev data norm\n",df.std()/df.mean()


if __name__ == "__main__":
    test_run()