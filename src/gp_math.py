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

def protected_divide_array(a, b):
    # Convert both inputs to arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # For b == 0.0 or non-finite, we set b to 1.0 to avoid division by zero
    b_invalid = (b == 0.0) | ~np.isfinite(b)
    b[b_invalid] = 1.0
    
    # For a not finite (inf or NaN), set a to 0.0
    a_invalid = ~np.isfinite(a)
    a[a_invalid] = 0.0
    
    return a / b
    
def protected_log(a):
    if a is None or a <= 0.0:
        return 0.0
    return np.log(a)

def protected_log_array(a):
    a = np.asarray(a, dtype=float)
    
    # For negative/zero or non-finite values, set them to 1.0 so log(1) = 0
    invalid = (a <= 0.0) | ~np.isfinite(a)
    a[invalid] = 1.0
    
    return np.log(a)
    
def protected_sin(a):
    if a is None or not np.isfinite(a):
        return 0.0
    return np.sin(a)

def protected_sin_array(a):
    a = np.asarray(a, dtype=float)
    
    # For non-finite values, set them to 0.0
    invalid = ~np.isfinite(a)
    a[invalid] = 0.0
    
    return np.sin(a)

def protected_cos_array(a):
    a = np.asarray(a, dtype=float)
    # For non-finite values, set them to 0.0
    invalid = ~np.isfinite(a)
    a[invalid] = 0.0
    
    return np.cos(a)

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