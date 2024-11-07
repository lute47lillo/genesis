"""
    Definition
    -----------
        File containing helper functions for the tree evaluation in GP
"""
import numpy as np

def protected_divide(a, b):
    if b is None or b == 0.0:
        return 0.0
    if a is None:
        return 0.0
    return a / b
    
def protected_log(a):
    if a is None or a <= 0.0:
        return 0.0
    return np.log(a)
    
def protected_sin(a):
    if a is None or not np.isfinite(a):
        return 0.0
    return np.sin(a)

def protected_cos(a):
    if a is None or not np.isfinite(a):
        return 0.0
    return np.cos(a)

def protected_subtract(a, b):
    if b is None:
        return 0.0
    if a is None:
        return 0.0
    return a - b

def protected_sum(a, b):
    if b is None:
        return 0.0
    if a is None:
        return 0.0
    return a + b

def protected_mult(a, b):
    if b is None:
        return 0.0
    if a is None:
        return 0.0
    return a * b