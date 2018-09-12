import os
import pandas as pd
import matplotlib.pyplot as plt
import copy
import numpy as np
import scipy.optimize as spo

def symbol_to_path(symbol, base_dir='data'):
	return os.path.join(base_dir, "{}.csv".format(str(symbol)))
	
def get_data(sym, dates, addSPY=True):
   symbols = copy.copy(sym)
   dataframe = pd.DataFrame(index=dates)
   if 'SPY' not in symbols and addSPY is True:
		symbols.insert(0, 'SPY')
		
   for symbol in symbols:
		dataframe_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date', \
                               parse_dates=True, usecols=['Date', 'Adj Close'], \
                               na_values=['nan']).rename(columns = {'Adj Close' : symbol})
		if symbol == 'SPY':
			dataframe = dataframe.join(dataframe_temp, how='inner')
		else:
			dataframe = dataframe.join(dataframe_temp)
	
   return dataframe
	
def plot_data(dataframe, title="Stock prices", xlabel="Date", ylabel="Price"):
	ax = dataframe.plot(title = title, fontsize=8)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.show()
	
def plot_selected(dataframe, columns, start_index, end_index):
	plot_data(dataframe.ix[start_index:end_index, columns])
	
def normalize_data(dataframe):
	return dataframe/dataframe.ix[0]

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

def compute_portfolio_stats(port, daily_rets, riskFreeRate):
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

def error_calc(alloc, data, opt, riskFreeRate):
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
    cumulative, avg, risk, sharpe = compute_portfolio_stats(portfolio, daily, riskFreeRate)
    if opt=='risk':
        return risk
    elif opt =='cumulative':
        return -cumulative
    elif opt =='sharpe':
        return -sharpe
    
def fit_portfolio_alloc(data, error_func, optimize, riskFreeRate):
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
    result = spo.minimize(error_calc, Initial, args=(data,optimize,riskFreeRate,), method='SLSQP', \
                          bounds=mybounds, \
                          constraints=({ 'type': 'eq', 'fun': lambda inputs: 1-np.sum(inputs) }),\
                          options={'disp':False})
    return result.x

"""
Take a start and end date, list of symbols, optimization metric and 
return allocation percentages that optimize on that metric plus
other statistics over that time range, assuming constant allocations
"""
def optimize_portfolio(sd, ed, syms, opt, gen_plot=False):
    #riskFreeRate = 0.02123 # 3 month T bill 8/28/2018
    riskFreeRate = 0.0 
    dates = pd.date_range(sd, ed)
    df = get_data(syms, dates, addSPY=False)
    df = fill_missing_values(df)
    alloc = fit_portfolio_alloc(df, error_calc, optimize=opt, riskFreeRate=riskFreeRate)
    portfolio = compute_portfolio_value(df, alloc, 1)
    daily = compute_daily_returns(portfolio)
    cumulative, avg, risk, sharpe = compute_portfolio_stats(portfolio, daily, riskFreeRate)
    spy = get_data([],dates)
    if gen_plot:
        spy['portfolio'] = portfolio
        rel = spy/spy.iloc[0]
        plt.plot(rel)
        tit = "portfolio return optimizing "+opt+" vs. SPY"
        plt.title(tit)
        plt.legend(rel)
        plt.show()
    return alloc, cumulative, avg, risk, sharpe