import numpy as np

def test_run():
	print np.array([2, 3, 4])
	print np.array([(1,2,3), (2,1,3)])
	print np.empty(5)
	print np.empty((5,4))
	print np.ones((5,4))
	print np.zeros((5,4))
	
if __name__ == "__main__":
	test_run()