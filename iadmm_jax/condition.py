import numpy as np

def threshold(A, x_p, y_p, y_prev, b, beta, count, xi_1):
    stop_term1 = (np.linalg.norm(A @ x_p + y_p - b)) / (1.0 + np.linalg.norm(b))
    stop_term2 = (np.linalg.norm(beta * A.T @ (y_p - y_prev))) / (1.0 + np.linalg.norm(y_prev))

    stop_measure = max(stop_term1, stop_term2)
    if (count % 100) == 0: print(f"Status on max-norm: {stop_measure}")
    if stop_measure < xi_1: print(f"Condition met with stop_measure = {stop_measure}")

    return (stop_measure < xi_1)

def approx_cond(A, b, x_next, d_step, y_p, w_1, beta, sigma_1, count):
    residual = A @ x_next + y_p - b
    lhs_term1 = (2.0 / beta) * np.abs((w_1 - x_next).T @ d_step)
    lhs_term2 = d_step.T @ d_step
    rhs_term = sigma_1 * (residual.T @ residual)
    if count % 250 == 0:
        print(f"Iteration #{count}:")
        print(f" LHS sum      | {lhs_term1 + lhs_term2}")
        print(f" RHS dist     | {rhs_term}")
    return (lhs_term1 + lhs_term2) <= rhs_term

def dist_cond(subdiff, count, xi_2): 
    dist = np.linalg.norm(subdiff, ord=np.inf)
    if count % 250 == 0: print(f" ||x_new||    | {dist}\n")
    return (dist < xi_2)
