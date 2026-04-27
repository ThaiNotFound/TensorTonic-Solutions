import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    n = np.array(x)
    ans = np.maximum(0,n)
    return ans
    pass