import pandas as pd
import panda_utils as util

def test_run():
	dates = pd.date_range('2010-01-01', '2010-01-30')
	symbols = ['GOOG', 'IBM', 'GLD']
	dataframe = util.get_data(symbols, dates)

	util.plot_selected(dataframe, ['GOOG', 'IBM'], '2010-01-30', '2010-01-01')
	util.plot_selected(util.normalize_data(dataframe), ['GOOG', 'IBM'], '2010-01-30', '2010-01-01')
	

if __name__ == "__main__":
	test_run()