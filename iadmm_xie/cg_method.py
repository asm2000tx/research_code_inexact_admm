import numpy as np

from condition import cg_condition

class cg_class:
    def __init__(self, **kwargs):
        # Defining the preliminary data to execute the CG algorithm
        self.C_cg = kwargs["A"].T @ kwargs["A"] + kwargs["mu"] * kwargs["beta"] * kwargs["D"].T @ kwargs["D"]
        self.b_cg = kwargs["mu"] * kwargs["D"].T @ (kwargs["l"] * kwargs["y"]) + kwargs["A"].T @ kwargs["c"]

    def cg_method(self, x_p): 
        r_curr = self.C_cg @ x_p - self.b_cg
        p = -r_curr

        x_curr = x_p
        while True:
            alpha = (r_curr.T @ r_curr) / (p.T @ self.C_cg @ p)
            x_curr = x_curr + alpha * p
            r_prev = r_curr
            r_curr = r_prev + alpha * self.C_cg @ p
            beta = (r_curr.T @ r_curr) / (r_prev.T @ r_prev)
            p = - r_curr + beta * p

            if cg_condition(self.C_cg, x_curr, self.b_cg): return x_curr