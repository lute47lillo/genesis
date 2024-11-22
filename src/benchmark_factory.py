"""
    Definition
    -----------
        Contains the Symbolic Regression Functions.
"""

import numpy as np
    
# -------------- Symbolic Regression Functions --------------- #

def nguyen1(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = x ** 3 + x ** 2 + x
    return y

def nguyen2(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = x ** 4 + x ** 3 + x ** 2 + x
    return y

def nguyen3(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = x ** 5 + x ** 4 + x ** 3 + x ** 2 + x
    return y

def nguyen4(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x
    return y

def nguyen5(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = np.sin(x**2) * np.cos(x) - 1
    return y

def nguyen6(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = np.sin(x) + np.sin(x + x ** 2)
    return y

def nguyen7(x):
    """
        Input E[0.0, 8.0] with step size of 0.1. Undefined for inputs smaller than -1
    """
    y = np.log(x+1) + np.log(x ** 2 + 1)
    return y

def nguyen8(x):
    """
        Input E[0.0, 8.0] with step size of 0.1. Undefined for negative inputs
    """
    y = np.sqrt(x)
    return y