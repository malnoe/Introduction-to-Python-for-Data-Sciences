import numpy as np

from functions import Function1, Function2

def make_random_func1(n: int):
    """
    Random quadratic function (convex)
    """
    L = np.tril(np.random.randn(n, n)) / n
    A = L @ L.T + np.eye(n)
    b = np.random.randn(n)
    return Function1(A, b)

def make_random_func2(n: int, p: int):
    _theta = np.random.randn(p)
    X = np.random.randn(n, p)
    y = X @ _theta + np.random.randn(n)
    del _theta
    return Function2(X, y)
