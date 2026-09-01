import numpy as np

from cg_method import cg_class

## Comments:
## This code implements the inexact ADMM with relative error 
## criterion algorithm. The main loop is performing the minimization problem. 
## Author(s): Jiaxin Xie

def main():
    m, n, s = 2048, 256, 512
    A, D = np.random.randn(m, n), np.random.randn(s, n)
    c = np.random.randn(m)

    # Defining the unknown variables
    x_p, y_p, l_d = np.zeros(n), np.random.randn(s), np.random.randn(s)

    # Defining the constants, mu, beta, and sigma
    mu, beta, sigma = 0.01, 25, 0.5

    cg_dict = {
        "A" : A,
        "D" : D,
        "y" : y_p,
        "l" : l_d,
        "c" : c,
        "mu" : mu,
        "beta" : beta, 
        "sigma" : sigma
    }

    # Executing the CG method
    conjugate_class = cg_class(**cg_dict)
    x_p = conjugate_class.cg_method(x_p)

if __name__ == '__main__':
	main()
