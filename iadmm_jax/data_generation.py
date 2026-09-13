import numpy as np

def generate_data(delta, m, n, s):
    ## Method 1 - Gaussian matrix
    A = np.random.normal(loc=0.0, scale=1.0, size=(m,n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    
    ## x_bar - sparse vector
    x_bar = np.zeros(n)
    s_bar = np.random.randint(1,s+1)
    x_values = np.random.normal(loc=0.0, scale=1.0, size=n)
    x_bar[:s_bar] = x_values[:s_bar]
    np.random.shuffle(x_bar)

    ## eps - 'noise' vector
    eps = np.random.normal(loc=0.0, scale=1.0, size=m)
    
    ## b - inexact output 
    b = A @ x_bar + delta * eps

    ## Pre-computing large quantities being used every iteration
    AtA = A.T @ A

    return AtA, A, x_bar, b