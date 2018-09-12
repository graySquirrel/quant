import os
import pandas as pd

def symbol_to_path(symbol, base_dir='data'):
	return os.path.join(base_dir, "{}.csv".format(str(symbol)))
	
def get_data(symbols, dates):
	dataframe = pd.DataFrame(index=dates)
	if 'SPY' not in symbols:
		symbols.insert(0, 'SPY')
		
	for symbol in symbols:
		dataframe_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan']).rename(columns = {'Adj Close' : symbol})
		if symbol == 'SPY':
			dataframe = dataframe.join(dataframe_temp, how='inner')
		else:
			dataframe = dataframe.join(dataframe_temp)
	
	return dataframe
	
def test_run():
	dates = pd.date_range('2010-01-22', '2010-01-26')
	symbols = ['GOOG', 'IBM', 'GLD']
	print get_data(symbols, dates)
	
if __name__ == "__main__":
	test_run()