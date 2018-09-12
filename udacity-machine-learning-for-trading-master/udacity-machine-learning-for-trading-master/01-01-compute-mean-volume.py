import pandas as pd

def get_mean_volume(dataframe, symbol):
	return dataframe['Volume'].mean()
	
def get_max_price(dataframe, symbol, pricetype):
	return dataframe[pricetype].max()
	
def get_min_price(dataframe, symbol, pricetype):
	return dataframe[pricetype].min()
	
def test_run():
	for symbol in ['AAPL', 'IBM']:
		dataframe = pd.read_csv("data/{}.csv".format(symbol))
		print "Mean Volume"
		print symbol, get_mean_volume(dataframe, symbol)
		print "Highest close price"
		print symbol, get_max_price(dataframe, symbol, 'Close')
		print "Lowest low price"
		print symbol, get_min_price(dataframe, symbol, 'Low')
		
if __name__ == "__main__":
	test_run()