import numpy as np
from fista import fista_const

## Inexact ADMM with relative error criterion written by Jiaxin, Anping, and Xiaobo

def main():
    m, n = 250, 100
    A = np.random.randn(m, n)
    beta = 1.5e3
    
    ## Define s-sparse vector x
    s = 25
    x_bar = np.zeros(n)
    x_bar[:s] = np.random.randn(s)
    np.random.shuffle(x_bar)

    ## Define the inexact solution
    delta = 1e-3
    eps = np.random.randn(m)
    b = A @ x_bar + delta * eps

    # Define x_p, y_p, and l_d 
    x_p, y_p, l_d = np.zeros(n), np.zeros(m), np.zeros(m)

    # Define w_1, w_2
    w_1, w_2 = np.random.randn(n), np.random.randn(m)

    # Define random constants sigma_1 and sigma_2 whose sum is < 1
    sigma_1 = 0.9 
    sigma_2 = 0.01
    if sigma_1 + sigma_2 >= 1: sigma_2 = 1 - sigma_1 - 1e-3

    # Solve x-subproblem
    fista_step = fista_const(A, y_p, l_d, b, w_1, beta, sigma_1)
    d_step, x_step = fista_step.prox_step()
    
if __name__ == '__main__':
	main()

