import numpy as np

def protected_divide(a, b):
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
    a = np.asarray(a, dtype=float)
    
    # For negative/zero or non-finite values, set them to 1.0 so log(1) = 0
    invalid = (a <= 0.0) | ~np.isfinite(a)
    a[invalid] = 1.0
    
    return np.log(a)

def protected_sin(a):
    a = np.asarray(a, dtype=float)
    # For non-finite values, set them to 0.0
    invalid = ~np.isfinite(a)
    a[invalid] = 0.0
    
    return np.sin(a)

def protected_cos(a):
    a = np.asarray(a, dtype=float)
    # For non-finite values, set them to 0.0
    invalid = ~np.isfinite(a)
    a[invalid] = 0.0
    
    return np.cos(a)

def protected_subtract(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # For any inf/NaN in a or b, replace them with 0
    a[~np.isfinite(a)] = 0.0
    b[~np.isfinite(b)] = 0.0
    
    return a - b

def protected_sum(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # For any inf/NaN in a or b, replace them with 0
    a[~np.isfinite(a)] = 0.0
    b[~np.isfinite(b)] = 0.0
    
    return a + b

def protected_mult(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # For any inf/NaN in a or b, replace them with 0
    a[~np.isfinite(a)] = 0.0
    b[~np.isfinite(b)] = 0.0
    
    return a * b
