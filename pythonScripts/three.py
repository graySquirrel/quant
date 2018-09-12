# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 12:49:47 2018

@author: febner
"""

import numpy as np
import pandas as pa

def test_run():
    a = np.random.rand(5)
    print(a)
    indices = np.array([1,1,2,3])
    print a[indices]
    
def test_run2():
    a = np.array([(20,25,10,23,26,32,10,5,0),(0,2,50,20,0,1,28,5,0)])
    b = np.array([(0,2,50,20,0,1,28,5,0),(20,25,10,23,26,32,10,5,0)])
    mean = np.mean(a)
    print "mean is " + str(mean)
    print a
    #masking
    a[a<mean] = 1
    print a
    print "a times 2 is " , 2*a
    print "dims",a.shape
    print a/2
    print a/2.0
    dir(a)
    print "a plus b\n" , a+b
    print "a times  b\n", a*b
    b[b<=0] = 1
    print "a div  b\n", a/b
    id(a)
    globals()
    locals()
#    info(a)
#    a.info()
#    a.apply(lambda x: [x.unique()])
    
    
if __name__ == "__main__":
    test_run()
    test_run2()
        
