import numpy as np

# Classes
from fista import fista_const

# Helper functions
from condition import threshold

## Comments:
## This code is based on the inexact ADMM with relative error 
## criterion algorithm written by Jiaxin, Anping, and Xiaobo.

def main():
    m, n = 1024, 4096
    A = np.random.randn(m, n)
    beta = 1.5e3
    
    ## Define s-sparse vector x
    s = 15
    x_bar = np.zeros(n)
    x_bar[:s] = np.random.randn(s)
    np.random.shuffle(x_bar)

    ## Define the inexact solution
    delta = 1e-3
    eps = np.random.randn(m)
    b = A @ x_bar + delta * eps

    # mu variable 
    mu = np.sqrt(m) * np.linalg.norm(A.T @ b, ord=np.inf)

    # Define x_p, y_p, and l_d 
    x_p, y_p, l_d = np.zeros(n), np.zeros(m), np.zeros(m)

    # Define w_1, w_2
    w_1, w_2 = np.zeros(n), np.zeros(m)

    # Define random constants sigma_1 and sigma_2 whose sum is < 1
    sigma_1 = 0.5

    # Defining xi_1 and xi_2
    xi_1 = 1e-4
    xi_2 = 1e-8

    count = 0
    while True:
        # Solving x-subproblem to compute x^{k+1}
        fista_step = fista_const(A, y_p, l_d, b, w_1, beta, sigma_1, xi_2)
        d_1, x_p = fista_step.fista()

        # Saving the previous y_p to use for termination of algorithm 1.
        y_prev = y_p

        # Using closed-form solution to compute y^{k+1} 
        z = b + (1.0 / beta) * l_d - (A @ x_p)
        y_p = np.sign(z) * np.maximum(np.abs(z) - beta / mu, 0.0)

        if threshold(A, x_p, y_p, y_prev, b, beta, count, xi_1): break

        # Condition failed, incrementing count and updating l_d and w_1 variables. 
        count += 1
        l_d = l_d - beta * (A @ x_p + y_p - b)
        w_1 = w_1 - beta * d_1

if __name__ == '__main__':
	main()
