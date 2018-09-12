import pandas as pd
import panda_utils as util

def test_run():
	dates = pd.date_range('2010-01-01', '2010-01-30')
	symbols = ['IBM', 'GOOG', 'GLD']
	dataframe = util.get_data(symbols, dates)
	print dataframe
	
	ndarray = dataframe.values
	print "[0,0]: " + str(ndarray[0,0])
	print "[3,2]: " + str(ndarray[3,2])
	print "[0:3,1:3]: " + str(ndarray[0:3,1:3])
	print "[:,3]: " + str(ndarray[:,3])
	print "[-1,1:3]: " + str(ndarray[-1,1:3])
	

if __name__ == "__main__":
	test_run()