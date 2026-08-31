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

    # Defining the constants, mu and beta    
    mu, beta = 0.01, 25

    # Defining the preliminary data to execute the CG algorithm
    C_cg = A.T @ A + mu * beta * D.T @ D
    b_cg = mu * D.T @ (l_d * y_p) + A.T @ c

    # Executing the CG method
    conjugate_class = cg_class(C_cg, b_cg)
    x_p = conjugate_class.cg_method(x_p)

if __name__ == '__main__':
	main()
