import os
import pandas as pd
import matplotlib.pyplot as plt

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
	
def plot_data(dataframe, title="Stock prices", xlabel="Date", ylabel="Price"):
	ax = dataframe.plot(title = title, fontsize=8)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.show()
	
def plot_selected(dataframe, columns, start_index, end_index):
	plot_data(dataframe.ix[start_index:end_index, columns])
	
def normalize_data(dataframe):
	return dataframe/dataframe.ix[0]