import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    n = np.array(x)
    ans = 1 / (1 + np.e**(-1*n))
    return ans
    pass