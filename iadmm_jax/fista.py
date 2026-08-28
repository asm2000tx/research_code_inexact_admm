import numpy as np

class fista_const:
    def __init__(self, A, y_p, l_d, b, w_1, beta, sigma_1, xi_2):
        self.A = A
        self.y_p = y_p
        self.l_d = l_d
        self.b = b
        self.w_1 = w_1
        self.beta = beta
        self.sigma_1 = sigma_1
        self.xi_2 = xi_2

        # Second term inside the smooth function
        self.c_k = self.b + (1 / self.beta) * self.l_d - self.y_p

        # Lipschitz constant
        self.L = self.beta * np.linalg.norm(self.A, 2) ** 2

    # Gradient of smooth term of objective function
    def grad_func(self, curr_pt): return self.beta * self.A.T @ (self.A @ curr_pt - self.c_k)

    # Subgradient of L1-norm 
    def soft_shrinkage(self, grad_g): return np.sign(grad_g) * np.maximum(np.abs(grad_g) - (1/self.L), 0)

    def dist_cond(self, curr_pt, count): 
        grad_g = curr_pt - (1/self.L) * self.grad_func(curr_pt)
        curr_pt = self.soft_shrinkage(grad_g)
        dist = np.linalg.norm(curr_pt, ord=np.inf)
        if count % 250 == 0: print(f"||x_new||_inf | {dist}\n")
        return (dist < self.xi_2)

    def approx_cond(self, d_step, x_next, count):
        residual = self.A @ x_next + self.y_p - self.b

        lhs_term1 = (2.0 / self.beta) * np.abs((self.w_1 - x_next).T @ d_step)
        lhs_term2 = d_step.T @ d_step
        rhs_term = self.sigma_1 * (residual.T @ residual)
        if count % 250 == 0:
            print(f"Iteration #{count}:")
            print(f" LHS sum      | {lhs_term1 + lhs_term2}")
            print(f" RHS dist     | {rhs_term}")

        return (lhs_term1 + lhs_term2) <= rhs_term
    
    def prox_step(self):
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
            if self.approx_cond(d_step, x_next, count) or self.dist_cond(x_next, count): 
                print(f"Threshold is met! | Steps: {count}\n")
                return d_step, x_next

            # Condition(s) failed, computing the new local variables, t_next and y_next
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_curr ** 2))
            y_next = x_next + ((t_curr - 1.0) / t_next) * (x_next - x_curr)

            x_curr = x_next
            y_curr = y_next
            t_curr = t_next
            count += 1