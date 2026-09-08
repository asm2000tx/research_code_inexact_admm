import numpy as np

def check_fista_threshold(t1_dict):
    """Check FISTA convergence threshold."""
    residual_primal = t1_dict["A"] @ t1_dict["x_p"] + t1_dict["y_p"] - t1_dict["b"]
    residual_dual = t1_dict["beta"] * t1_dict["A"].T @ (t1_dict["y_c"] - t1_dict["y_p"])

    stop_term1 = np.linalg.norm(residual_primal) / (1.0 + np.linalg.norm(t1_dict["b"]))
    stop_term2 = np.linalg.norm(residual_dual) / (1.0 + np.linalg.norm(t1_dict["y_p"]))
    stop_measure = max(stop_term1, stop_term2)

    print("Checking Condition (35) (Classic and Inexact)")
    status = "Pass" if stop_measure < t1_dict["xi_1"] else "Fail"
    print(f"Status: {status} - max(stop_term1, stop_term2) = {stop_measure:.5e}\n")

    return stop_measure < t1_dict["xi_1"]

def check_approx_condition(c1_dict, inexact=False):
    """Check approximation condition for inexact ADMM."""
    residual = c1_dict["A"] @ c1_dict["x"] + c1_dict["y"] - c1_dict["b"]
    
    lhs_term1 = (2.0 / c1_dict["beta"]) * np.abs((c1_dict["w_1"] - c1_dict["x"]).T @ c1_dict["d"])
    lhs_term2 = c1_dict["d"].T @ c1_dict["d"]
    rhs_term = c1_dict["sigma_1"] * (residual.T @ residual)

    if c1_dict['count'] % 250 == 0 and inexact: print(f"Algorithm 1 - Condition (8) | count == {c1_dict['count']}")
    
    return (lhs_term1 + lhs_term2) <= rhs_term

def check_dist_condition(c2_dict, inexact=True):
    """Check distance condition for convergence."""
    c = c2_dict["b"] + c2_dict["l"] / c2_dict["beta"] - c2_dict["y"]
    q = c2_dict["beta"] * (
        c2_dict["A"].T @ (c2_dict["A"] @ c2_dict["x"] - c)
    )

    d = np.empty_like(c2_dict["x"])
    
    positive = c2_dict["x"] > 0
    negative = c2_dict["x"] < 0
    zero = ~(positive | negative)

    d[positive] = q[positive] + 1.0
    d[negative] = q[negative] - 1.0
    d[zero] = q[zero] - np.clip(q[zero], -1.0, 1.0)

    if c2_dict["count"] % 250 == 0 and not inexact:
        print(f"Iteration {c2_dict['count']} - Condition (36): norm(d) = {np.linalg.norm(d):.5e}")

    return np.linalg.norm(d) < c2_dict["xi_2"]
