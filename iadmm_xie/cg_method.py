import numpy as np

from condition import cg_condition

class cg_class:
    def __init__(self, C, b):
        self.C = C
        self.b = b

    def cg_method(self, x_p): 
        r_curr = self.C @ x_p - self.b
        p = -r_curr

        x_curr = x_p
        while True:
            alpha = (r_curr.T @ r_curr) / (p.T @ self.C @ p)
            x_curr = x_curr + alpha * p
            r_prev = r_curr
            r_curr = r_prev + alpha * self.C @ p
            beta = (r_curr.T @ r_curr) / (r_prev.T @ r_prev)
            p = - r_curr + beta * p

            if cg_condition(self.C, x_curr, self.b): return x_curr