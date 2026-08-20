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

    fista_step = fista_const(A, b, beta)

    count = 10
    while True:
        count += 1
        if count >= 10: break
    
if __name__ == '__main__':
	main()
