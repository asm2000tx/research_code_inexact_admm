import numpy as np
import time

# Classes
from fista import fista_const

# Helper functions
from condition import threshold

## Comments:
## This code implements the inexact ADMM with relative error 
## criterion algorithm. The main loop is performing the minimization problem. 
## Author(s): Jiaxin Xie, Anping Liao, Xiaobo Yang

def iadmm_algorithm(A, b, sigma_1, beta, xi_1, xi_2, s, m, n):

    ## mu - constant
    mu = np.sqrt(m) * np.linalg.norm(A.T @ b, ord=np.inf)

    ## Define x_p and y_p (primal) | l_d (dual)
    x_p, y_p, l_d = np.zeros(n), np.zeros(m), np.zeros(m)

    ## Define w_1 - closed-form solution for y doesn't need w_2
    w_1 = np.zeros(n)

    count = 1
    fista_args = {
        "A": A,
        "y": y_p,
        "l": l_d,
        "b": b,
        "w_1": w_1,
        "beta": beta,
        "sigma_1": sigma_1,
        "xi_2": xi_2
    }

    ## Inexact ADMM algorithm
    while True:
        ## Solving x-subproblem to compute x^{k+1}
        fista_step = fista_const(**fista_args)
        d_1, x_p = fista_step.fista_inexact()

        ## Saving the previous y_p to use for termination of algorithm 1.
        y_prev = y_p

        ## Using closed-form solution to compute y^{k+1} 
        z = b + (1.0 / beta) * l_d - (A @ x_p)
        y_curr = np.sign(z) * np.maximum(np.abs(z) - mu / beta, 0.0)

        ## Using a dictionary to contain the threshold arguments. 
        cond_args = {
            "A": A,
            "x_c": x_p,
            "y_p": y_prev,
            "y_c": y_curr,
            "b": b,
            "beta": beta,
            "xi_1": xi_1
        }
        if threshold(cond_args): break
        # print(f"Current count: {count}\n")

        ## Condition failed, updating l_d and w_1 variables. 
        y_p = y_curr
        l_d = l_d - beta * (A @ x_p + y_p - b)
        w_1 = w_1 - beta * d_1

        ## Resetting arguments for FISTA algorithm
        count += 1
        fista_args["y"] = y_p
        fista_args["l"] = l_d
        fista_args["w_1"] = w_1

    return x_p, count

def admm_algorithm(A, beta, delta, xi_1, xi_2, s, m, n):
    pass

def main():
    ## Dimensions and sparsity constant
    s, m, n = 60, 1024, 4096

    ## Define the inexact solution
    beta = 1.5e3
    delta = 1e-3

    ## Defining xi_1 and xi_2
    xi_1 = 1e-4
    xi_2 = 1e-8

    ## Method 1 - Gaussian matrix
    A = np.random.randn(m, n)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    
    ## x_bar - sparse vector
    x_bar = np.zeros(n)
    s_bar = np.random.randint(1,s)
    x_bar[:s_bar] = np.random.randn(s_bar)
    np.random.shuffle(x_bar)

    ## eps - 'noise' vector
    eps = np.random.randn(m)

    ## b - inexact output 
    b = A @ x_bar + delta * eps

    print(f"Problem: m, n, s = {m}, {n}, {s}\n")

    x_sol, iter = iadmm_algorithm(A, b, 0.1, beta, xi_1, xi_2, s, m, n) 
    print(f"Final count: {iter}")
    print(f"Relative Error: {np.linalg.norm(x_bar - x_sol) / np.linalg.norm(x_bar)}")

    # print("Inexact ADMM with sigma_1 == 0.1")
    # iadmm_algorithm(A, 0.1, beta, delta, xi_1, xi_2, s, m, n) 
    # print("Inexact ADMM with sigma_1 == 0.5")
    # iadmm_algorithm(A, 0.5, beta, delta, xi_1, xi_2, s, m, n) 
    # print("Inexact ADMM with sigma_1 == 0.99")
    # iadmm_algorithm(A, 0.99, beta, delta, xi_1, xi_2, s, m, n)
    # admm_algorithm(A, beta, delta, xi_1, xi_2, s, m, n) 

    ## Method 2 - DCT matrix 
    # iadmm_algorithm(A, sigma_1, beta, delta, xi_1, xi_2, s, m, n)

if __name__ == '__main__':
	main()
