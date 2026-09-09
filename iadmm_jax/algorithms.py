import numpy as np

# Classes
from fista import fista_const

# Helper functions
from conditions import check_fista_threshold

def admm_alg(A, AtA, b, beta, xi_1, xi_2, L, m, n, *, inexact=False, sigma_1=None, max_iter=None):
    if inexact and sigma_1 is None: raise ValueError("sigma_1 is required for an inexact ADMM update")

    ## mu - constant
    mu = np.sqrt(m) * np.linalg.norm(A.T @ b, ord=np.inf)

    ## Define x_p and y_p (primal) | l_d (dual)
    x_p, y_p, l_d = np.zeros(n), np.zeros(m), np.zeros(m)

    ## The inexact method carries the FISTA correction term w_1.
    w_1 = np.zeros(n) if inexact else None

    count = 1
    fista_args = {
        "A": A,
        "AtA": AtA,
        "y": y_p,
        "l": l_d,
        "b": b,
        "w_1": w_1,
        "beta": beta,
        "sigma_1": sigma_1 if inexact else None,
        "xi_2": xi_2,
        "L": L
    }

    ## ADMM iterations
    while True:
        print(f"Iteration #{count}:")

        ## Keeping the initial result of x_p
        x_prev =  x_p

        ## Solving x-subproblem to compute x^{k+1}
        fista_step = fista_const(**fista_args)
        if inexact: d_1, x_p = fista_step.fista(inexact=True)
        else: x_p = fista_step.fista(inexact=False)

        ## Saving the previous y_p to use for termination of algorithm 1.
        y_prev = y_p
        ## Using closed-form solution to compute y^{k+1} 
        z = b + (1.0 / beta) * l_d - (A @ x_p)
        y_curr = np.sign(z) * np.maximum(np.abs(z) - beta / mu, 0.0)

        ## Using a dictionary to contain the threshold arguments. 
        cond_args = {
            "A": A,
            "x_p": x_prev,
            "y_p": y_prev,
            "y_c": y_curr,
            "b": b,
            "beta": beta,
            "xi_1": xi_1
        }
        if check_fista_threshold(cond_args): break
        if max_iter is not None and count >= max_iter: break
        # print(f"Current count: {count}\n")

        ## Condition failed, updating l_d and w_1 variables. 
        y_p = y_curr
        l_d = l_d - beta * (A @ x_p + y_p - b)
        if inexact: w_1 = w_1 - beta * d_1

        ## Resetting arguments for FISTA algorithm
        count += 1
        fista_args["y"] = y_p
        fista_args["l"] = l_d
        fista_args["w_1"] = w_1

    print(f"Final Count: {count}\n")

    return x_p, count

def inexact_admm_alg(A, AtA, b, sigma_1, beta, xi_1, xi_2, L, m, n):
    return admm_alg(A, AtA, b, beta, xi_1, xi_2, L, m, n, inexact=True, sigma_1=sigma_1)

def classic_admm_alg(A, AtA, b, beta, delta, xi_1, xi_2, L, m, n):
    return admm_alg(A, AtA, b, beta, xi_1, xi_2, L, m, n, inexact=False)