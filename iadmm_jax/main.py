import numpy as np

## Inexact ADMM with relative error criterion written by Jiaxin, Anping, and Xiaobo

def main():
    m, n = 250, 100
    A = np.random.randn(m, n)
    
    ## Define s-sparse vector x
    s = 25
    x = np.zeros(n)
    x[:s] = np.random.randn(s)
    np.random.shuffle(x)

    ## Define the inexact solution
    delta = 1e-3
    eps = np.random.randn(m)
    b = A @ x + delta * eps

    
    
if __name__ == '__main__':
	main()
