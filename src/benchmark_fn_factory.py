import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import math



# Rastrigin Function
def rastrigin(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

def sphere_function(x):
    """
        [-5.12, 5.12]
    """
    return sum(xi ** 2 for xi in x)

def rosenbrock_function(x):
    """
        [-2.048, 2.048]
    """
    return sum(100 * (x[i+1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1))

def ackley_function(x):
    """
        # Parameters specific to Ackley Function
        bounds_ackley = (-32.768, 32.768)
    """
    n = len(x)
    sum_sq = sum(xi ** 2 for xi in x)
    sum_cos = sum(math.cos(2 * math.pi * xi) for xi in x)
    term1 = -20 * math.exp(-0.2 * math.sqrt(sum_sq / n))
    term2 = -math.exp(sum_cos / n)
    return term1 + term2 + 20 + math.e

def schwefel_function(x):
    """
        [-500, 500]
    """
    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)

def griewank_function(x):
    """
        [-600, 600]
    """
    sum_sq = sum(xi ** 2 for xi in x) / 4000
    prod_cos = np.prod([math.cos(xi / math.sqrt(i+1)) for i, xi in enumerate(x)])
    return sum_sq - prod_cos + 1