#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 2019

@author: talespadilha
"""
import numpy as np

from scipy.optimize import minimize
from math import pi, log


def loglike(params, series):
    """
    Calculates log-likelihood given parameters and data (series)
    """
    T = series.size
    # Calculating the unconditional variance:
    unc_var = series.var()
    # Calculating squared errors:
    e2 = series**2     
    # Setting the unconditional variance as the element in t=0 for e2:
    e2 = np.append(unc_var, e2)
    # Building the series of conditional variance based on GARCH(1,1)  
    # Using the unconditional variance as the element in t=0 for ht:    
    ht = np.array([unc_var])
    for t in range(1, T+1):
        ht = np.append(ht, params[0] + params[1]*e2[t-1] + params[2]*ht[t-1])
    # Calculating the log-likelihood:
    ls = 0.5*(log(2*pi)+np.log(ht[1:])+(e2[1:]/ht[1:]))
    log_likelihood = ls.sum()
    return log_likelihood
    

def main(series):
    """
    Estimates the parameters of a GARCH(1,1) model of Bollerslev(1986). 
    
    We estimate the following variance for the residuals(tex format):
        \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
        
    garch11_params: List of mean zero residuals (epsilons)
    Returns: The list of parameters param = [omega, alpha, beta]
    """
    # Making sure the series of residuals has mean zero:
    dm_series = series - series.mean()
    # Starting values for the parameters [omega, alpha, beta]:
    x0 = [dm_series.var(), 0.2, 0.2]
    # Estimating parameters using MLE:
    solution = minimize(loglike, x0, method = 'SLSQP', args = dm_series, bounds = [(0,None),]*3)
    # Selecting parameters:
    garch11_params = list(solution.x)
    return garch11_params