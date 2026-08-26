import numpy as np
from fista import fista_const

## Inexact ADMM with relative error criterion written by Jiaxin, Anping, and Xiaobo

def threshold():
    pass

def main():
    m, n = 512, 2048
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

    mu = np.sqrt(m) * np.linalg.norm(A.T @ b, ord=np.inf)

    # Define x_p, y_p, and l_d 
    x_p, y_p, l_d = np.zeros(n), np.zeros(m), np.zeros(m)

    # Define w_1, w_2
    w_1, w_2 = np.zeros(n), np.zeros(m)

    # Define random constants sigma_1 and sigma_2 whose sum is < 1
    sigma_1 = 0.01

    while True:
        # Solving x-subproblem to compute x^{k+1}
        fista_step = fista_const(A, y_p, l_d, b, w_1, beta, sigma_1)
        d_1, x_p = fista_step.prox_step()

        # Using closed-form solution to compute y^{k+1} 
        z = b + (1.0 / beta) * l_d - (A @ x_p)
        y_p = np.sign(z) * np.maximum(np.abs(z) - beta / mu, 0.0)

        break # Temporary line, delete when complete

if __name__ == '__main__':
	main()

