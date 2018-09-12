# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:55:42 2018

@author: febner
"""

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

def get_rolling_mean(df, window):
    return df.rolling(window).mean()
    #return pd.DataFrame(df.rolling(window).mean())

def get_rolling_std(df, window):
    return df.rolling(window).std()
    #return pd.DataFrame(df.rolling(window).std())

def get_bollinger_bands(rm, rstd):
    #upper = np.sum(rm_SPY, 2*rstd_SPY)
    #NB HAVE TO INDEX DFs to get them to add and not cat...!!!!!!!!!
    #upper = pd.DataFrame(rm_SPY.iloc[:,0].add(2*rstd_SPY.iloc[:,0]))
    #lower = pd.DataFrame(rm_SPY.iloc[:,0].sub(2*rstd_SPY.iloc[:,0]))
    lower_band = rm - 2 * rstd
    upper_band = rm + 2 * rstd

    return lower_band, upper_band

def test_run():
    # Define a date range
    dates = pd.date_range('2012-01-01', '2012-12-31')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']
    #symbols = ['SPY']
    
    # Get stock data
    df = get_data(symbols, dates)
    
    ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')
    
    #rm_SPY = df.rolling(20).mean() # course uses pd.rolling_mean, but doesnt work
    rm_SPY = get_rolling_mean(df['SPY'], window=20)
    rm_SPY.columns=['Rolling mean']
    rstd_SPY = get_rolling_std(df['SPY'], window=20)
    lower_band, upper_band = get_bollinger_bands(rm_SPY, rstd_SPY)
    #upper_band.columns=['Upper band']
    #lower_band.columns=['Lower band']
    rm_SPY.plot(label='Rolling mean', ax=ax) # label doesnt work, have to name df col
    upper_band.plot(label="upper band", ax=ax)
    lower_band.plot(label="lower band", ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()
    
    
if __name__ == "__main__":
    test_run()