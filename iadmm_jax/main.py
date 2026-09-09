import numpy as np
import time

# Classes
from fista import fista_const

# Helper functions
from conditions import check_fista_threshold

## Comments:
## This code implements the inexact ADMM with relative error 
## criterion algorithm. The main loop is performing the minimization problem. 
## Author(s): Jiaxin Xie, Anping Liao, Xiaobo Yang

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
        y_curr = np.sign(z) * np.maximum(np.abs(z) - mu / beta, 0.0)

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

def main():
    ## Dimensions and sparsity constant
    s, m, n = 10, 256, 1024

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

    ## Precomputing quantities used by the ADMM algorithms
    AtA = A.T @ A
    L = beta * np.linalg.norm(A, 2) ** 2

    print(f"Problem: m, n, s = {m}, {n}, {s}\n")

    # Keep all methods in one table so every configuration is tested identically.
    methods = [
        ("Inexact ADMM (sigma_1=0.1)",
         lambda: inexact_admm_alg(A, AtA, b, 0.1, beta, xi_1, xi_2, L, m, n)),
        ("Inexact ADMM (sigma_1=0.5)",
         lambda: inexact_admm_alg(A, AtA, b, 0.5, beta, xi_1, xi_2, L, m, n)),
        ("Inexact ADMM (sigma_1=0.99)",
         lambda: inexact_admm_alg(A, AtA, b, 0.99, beta, xi_1, xi_2, L, m, n)),
        ("Classic ADMM",
         lambda: classic_admm_alg(A, AtA, b, beta, delta, xi_1, xi_2, L, m, n)),
    ]
    results = {name: {"count": [], "error": [], "time": []} for name, _ in methods}

    for trial in range(1, 11):
        for name, solve in methods:
            print(f"Trial {trial}/10: Running {name}\n")
            start = time.perf_counter()
            x_sol, count = solve()
            elapsed = time.perf_counter() - start
            rel_error = np.linalg.norm(x_bar - x_sol) / np.linalg.norm(x_bar)

            results[name]["count"].append(count)
            results[name]["error"].append(rel_error)
            results[name]["time"].append(elapsed)
            print(f"{name}: m={m}, n={n}, s={s}, iterations={count}, "
                f"error={rel_error:.6e}, "
                  f"time={elapsed:.6f}s\n")

    print("\n" + "=" * 110)
    print("Summary of results (10 trials)")
    print(f"Problem dimensions: m={m}, n={n}, s={s}")
    print("=" * 110)
    print(f"{'Method':<34} "
        f"{'Iterations':>18} {'Relative error':>18} {'Time (s)':>12}")
    print("-" * 110)
    for name, _ in methods:
        method_results = results[name]
        print(
            f"{name:<34} "
            f"{np.mean(method_results['count']):>18.2f} "
            f"{np.mean(method_results['error']):>18.2e} "
            f"{np.mean(method_results['time']):>8.4f}"
        )
    print("=" * 110 + "\n")

    ## Method 2 - DCT matrix 
    # iadmm_algorithm(A, sigma_1, beta, delta, xi_1, xi_2, s, m, n)

if __name__ == '__main__':
	main()
