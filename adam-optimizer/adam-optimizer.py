import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param, grad, m, v = (np.array(x, dtype = np.float64) for x in (param, grad, m, v))
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_b = m / (1 - beta1**t)
    v_b = v / (1 - beta2**t)
    param_new = param - lr * (m_b / (np.sqrt(v_b) + eps))
    return param_new, m, v 
    pass