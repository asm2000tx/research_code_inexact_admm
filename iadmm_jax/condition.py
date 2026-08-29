import numpy as np

def threshold(t1_dict):
    stop_term1 = (np.linalg.norm(t1_dict["A"] @ t1_dict["x_p"] + t1_dict["y_p"] - t1_dict["b"])) / (1.0 + np.linalg.norm(t1_dict["b"]))
    stop_term2 = (np.linalg.norm(t1_dict["beta"] * t1_dict["A"].T @ (t1_dict["y_p"] - t1_dict["y_prev"]))) / (1.0 + np.linalg.norm(t1_dict["y_prev"]))

    stop_measure = max(stop_term1, stop_term2)
    if (t1_dict["count"] % 100) == 0: print(f"Status on max-norm: {stop_measure}")
    if stop_measure < t1_dict["xi_1"]: print(f"Condition met with stop_measure = {stop_measure}")

    return (stop_measure < t1_dict["xi_1"])

def approx_cond(c1_dict):
    residual = c1_dict["A"] @ c1_dict["x"] + c1_dict["y"] - c1_dict["b"]  # A @ x_next + y_p - b
    lhs_term1 = (2.0 / c1_dict["beta"]) * np.abs((c1_dict["w_1"] - c1_dict["x"]).T @ c1_dict["d"])  # (2.0 / beta) * np.abs((w_1 - x_next).T @ d_step)
    lhs_term2 = c1_dict["d"].T @ c1_dict["d"]  # d_step.T @ d_step
    rhs_term = c1_dict["sigma_1"] * (residual.T @ residual)  # sigma_1 * (residual.T @ residual)
    if c1_dict["count"] % 250 == 0:
        print(f"Iteration #{c1_dict['count']}:")
        print(f" LHS sum      | {lhs_term1 + lhs_term2}")
        print(f" RHS dist     | {rhs_term}")
    return (lhs_term1 + lhs_term2) <= rhs_term

def dist_cond(subdiff, count, xi_2): 
    dist = np.linalg.norm(subdiff, ord=np.inf)
    if count % 250 == 0: print(f" ||x_new||    | {dist}\n")
    return (dist < xi_2)
