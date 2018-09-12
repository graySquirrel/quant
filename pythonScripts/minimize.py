# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:20:52 2018

@author: febner
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def f(X):
    Y = (X-1.5)**2 + 0.5
    print "X = {}, Y = {}".format(X, Y)
    return Y

def error(line, data):
    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1]))**2)
    return err

def error_poly(C, data):
    """ Compute error between given polynomial and observed data
    Parameters
    ----------
    C: numpy.poly1d object or equivalent array representing polynomial coefficients
    data: 2D array where each row is a point (x, y)
    
    Returns error as a single real value.
    """
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0]))**2)
    return err

def fit_line(data, error_func):
    l = np.float32([0, np.mean(data[:, 1])])
    x_ends = np.float32([0, 10])
    plt.plot(x_ends, l[0]*x_ends + l[1], 'm--',linewidth=2.0, label="initial guess")
    result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'disp':True})
    return result.x

def fit_poly(data, error_func, degree=3):
    """ Fit a polynomial to given data, using supplied error function.
    
    Parameters
    ----------
    data: 2D array where each row is a point (x, y)
    error_func: function that computes the error between polynomial and observed data
    
    Returns polynomial that minimizes the error function.
    """
    Cguess = np.poly1d(np.ones(degree+1, dtype=np.float32))
    
    x=np.linspace(-5, 5, 21)
    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="Initial guess")
    
    result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'disp':True})
    return np.poly1d(result.x) # convert optimal result to poly1d and return
    
def test_run():
    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp':True})
    print "Minima found at:"
    print "X = {}, Y = {}".format(min_result.x, min_result.fun)
    
    Xplot=np.linspace(0.5, 2.5, 21)
    Yplot=f(Xplot)
    plt.plot(Xplot, Yplot)
    plt.plot(min_result.x, min_result.fun, 'ro')
    plt.title("Minima of an objective function")
    plt.show()
    
def test_run_line():
    l_orig = np.float32([4, 2])
    print "Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1])
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")
    
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig+noise]).T
    plt.plot(data[:,0], data[:, 1], 'go', label='Data points')
    
    l_fit = fit_line(data, error)
    print "Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1])
    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', linewidth=2.0, label='fitted line')
    
def test_run_poly():
    Corig = np.poly1d([1.5, -10, -5, 60, 50])
    degree = 4
    print "Original polynomial: C \n{}".format(Corig)
    Xorig = np.linspace(-5, 5, 21)
    Yorig = np.polyval(Corig, Xorig)
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original polynomial")

    noise_sigma = 50.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape) #shape is samples
    data = np.asarray([Xorig, Yorig+noise]).T
    plt.plot(data[:,0], data[:, 1], 'go', label='Data points')

    p_fit = fit_poly(data, error_poly, degree=degree)
    print "Fitted poly: C \n{}".format(p_fit)
    plt.plot(data[:, 0], np.polyval(p_fit, Xorig), 'r--', linewidth=2.0, label='fitted line')
    plt.legend()
    return 0

if __name__ == "__main__":
    #test_run()
    #test_run_line()
    test_run_poly()