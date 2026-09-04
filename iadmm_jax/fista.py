import numpy as np
from condition import dist_cond, approx_cond

class fista_const:
    def __init__(self, **kwargs):
        self.A = kwargs["A"]
        self.y_p = kwargs["y"]
        self.l_d = kwargs["l"]
        self.b = kwargs["b"]
        self.w_1 = kwargs["w_1"]
        self.beta = kwargs["beta"]
        self.sigma_1 = kwargs["sigma_1"]
        self.xi_2 = kwargs["xi_2"]

        # Second term inside the smooth function
        self.c_k = self.b + (1 / self.beta) * self.l_d - self.y_p

        # Lipschitz constant
        self.L = self.beta * np.linalg.norm(self.A, 2) ** 2

    # Gradient of smooth term of objective function
    def grad_func(self, curr_pt): return self.beta * self.A.T @ (self.A @ curr_pt - self.c_k)

    # Subgradient of L1-norm 
    def soft_shrinkage(self, grad_g): return np.sign(grad_g) * np.maximum(np.abs(grad_g) - (1/self.L), 0)

    def fista(self):
        _, n = self.A.shape

        # primal variables and step size from the FISTA algorithm
        x_curr = np.zeros(n)
        y_curr = np.zeros(n)
        t_curr = 1.0

        count = 1
        while True:
            # Gradient of smooth term at y_k with respect to x
            grad_y = self.grad_func(y_curr)
            x_next = self.soft_shrinkage(y_curr - (1/self.L) * grad_y)

            # Computing d_{1}^{k+1} from the FISTA subproblem
            grad_x = self.grad_func(x_next)
            d_step = grad_x - self.L * (x_next - y_curr) - grad_y
            
            # Checking approximation threshold
            c1_dict = {
                "A" : self.A, 
                "b" : self.b, 
                "x" : x_next, 
                "d" : d_step, 
                "y" : self.y_p, 
                "w_1": self.w_1, 
                "beta" : self.beta, 
                "sigma_1" : self.sigma_1, 
                "count" : count
            }
            subdiff = self.soft_shrinkage(grad_x)
            cond_1 = approx_cond(c1_dict) 
            cond_2 = dist_cond(subdiff, count, self.xi_2) 
            if cond_1 or cond_2: 
                print(f"Threshold is met! | Steps: {count}\n")
                return d_step, x_next

            # Condition(s) failed, computing the new local variables, t_next and y_next
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_curr ** 2))
            y_next = x_next + ((t_curr - 1.0) / t_next) * (x_next - x_curr)

            x_curr = x_next
            y_curr = y_next
            t_curr = t_next
            count += 1