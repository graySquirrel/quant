import os
import pandas as pd
import panda_utils as util
	
def test_run():
	dates = pd.date_range('2010-01-01', '2010-12-31')
	symbols = ['GOOG', 'IBM', 'GLD']
	dataframe = util.get_data(symbols, dates)
	print dataframe.ix['2010-01-31':'2010-01-01', ['GOOG', 'IBM']]
	
if __name__ == "__main__":
	test_run()