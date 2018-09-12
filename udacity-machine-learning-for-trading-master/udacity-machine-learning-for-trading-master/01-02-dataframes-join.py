import pandas as pd

def test_run():
	start_date = '2010-01-22'
	end_date = '2010-01-26'
	dates = pd.date_range(start_date, end_date)
	
	#Create an empty dataframe
	dataframe1 = pd.DataFrame(index=dates)
	
	dataframeSPY = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan']) #Use dates as an index
	dataframeSPY = dataframeSPY.rename(columns = {'Adj Close':'SPY'})
	dataframe1 = dataframe1.join(dataframeSPY, how='inner')
	
	symbols = ['GOOG', 'IBM', 'GLD']
	for symbol in symbols:
		dataframe_temp = pd.read_csv("data/{}.csv".format(symbol), index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
		dataframe_temp = dataframe_temp.rename(columns = {'Adj Close':symbol})
		dataframe1 = dataframe1.join(dataframe_temp)
	
	print dataframe1
	
if __name__ == "__main__":
	test_run()