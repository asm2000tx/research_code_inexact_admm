import numpy as np
import time

# Helper functions
from algorithms import inexact_admm_alg, classic_admm_alg
from data_generation import generate_data

def main():
    ## Dimensions and sparsity constant
    s, m, n = 40, 256, 2048

    ## Define the inexact solution
    beta = 1.5e3
    delta = 1e-3

    ## Defining xi_1 and xi_2
    xi_1 = 1e-4
    xi_2 = 1e-8

    print(f"Problem: m, n, s = {m}, {n}, {s}\n")

    method_names = [
        "Inexact ADMM (sigma_1=0.1)",
        "Inexact ADMM (sigma_1=0.5)",
        "Inexact ADMM (sigma_1=0.99)",
        "Classic ADMM",
    ]
    results = {name: {"count": [], "error": [], "time": []}
               for name in method_names}

    for trial in range(1, 11):
        ## Method 1 - Gaussian matrix
        AtA, A, x_bar, b = generate_data(delta, m, n, s)

        # Keep all methods in one table so every configuration is tested identically.
        methods = [
            (method_names[0], lambda: inexact_admm_alg(A, AtA, b, 0.1, beta, xi_1, xi_2, m, n)),
            (method_names[1], lambda: inexact_admm_alg(A, AtA, b, 0.5, beta, xi_1, xi_2, m, n)),
            (method_names[2], lambda: inexact_admm_alg(A, AtA, b, 0.99, beta, xi_1, xi_2, m, n)),
            (method_names[3], lambda: classic_admm_alg(A, AtA, b, beta, delta, xi_1, xi_2, m, n)),
        ]
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
    for name in method_names:
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
