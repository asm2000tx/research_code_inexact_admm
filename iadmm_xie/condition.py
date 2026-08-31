import numpy as np

def cg_condition(C, x, b):
    num = np.linalg.norm(C @ x - b, ord=2)
    den = max(np.linalg.norm(b, ord=2), 1.0)
    return (num / den) < 1e-8