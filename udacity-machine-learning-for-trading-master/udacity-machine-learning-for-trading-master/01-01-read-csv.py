import pandas as pd

def test_run():
	dataframe = pd.read_csv("data/AAPL.csv")
	print "DataFrame"
	print dataframe
	print "First 5 rows"
	print dataframe.head()
	print "Last 6 rows"
	print dataframe.tail(6)
	
if __name__ == "__main__":
	test_run()
	