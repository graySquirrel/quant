# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

datadir = "C:\\Users\\febner\\Documents\\CourseraDataScience\\quant\\udacity-machine-learning-for-trading-master\\udacity-machine-learning-for-trading-master\\data"
def test_run():
    start_date='2010-01-22'
    end_date='2010-01-26'
    dates=pd.date_range(start_date,end_date)
    print(dates)
    df1=pd.DataFrame(index=dates)
    os.chdir(datadir)
    ##os.chdir("C:/Users/febner/Documents/CourseraDataScience/quant/")
    os.chdir("C:\\Users\\febner\\Documents\\CourseraDataScience\\quant\\udacity-machine-learning-for-trading-master\\udacity-machine-learning-for-trading-master")
    ##print(os.listdir('.'))
    os.getcwd()
    dates = pd.date_range('2010-01-22', '2010-01-26')
    print(dates)
    df = pd.DataFrame(index=dates)
    print(df)
    symbols = ['GOOG', 'IBM', 'GLD']
    f=symbol_to_path('GOOG')
    df1=pd.read_csv(f,index_col='Date',usecols=['Date','Adj Close'],parse_dates=True)
    df1=df1.rename(columns={'Adj Close':'GOOG'})
    #print(df1)
    plt.plot(df1)
    
if __name__ == "__main__":
    test_run()