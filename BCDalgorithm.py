import cvxpy as cp
import numpy as np
import time
import warnings
from constants import DEFAULT_EPSILON
# Define a small constant for numerical stability

# --- Objective Function Term Helpers (Matching Script Logic) ---

def term1(u, y, alpha_val, D_val, lambda_val, N, epsilon):
    """ Term 1: sum(alpha * D / (u * (1 - lambda * y))) """
    # u and y can be cp.Variable, cp.Parameter, or np.ndarray
    # Ensure inputs are CVXPY expressions if needed for graph
    u_expr = u if isinstance(u, cp.Expression) else cp.Constant(u)
    y_expr = y if isinstance(y, cp.Expression) else cp.Constant(y)
    denominator_inner = 1  - cp.multiply(lambda_val, y_expr)
    denominator = cp.multiply(u_expr, denominator_inner)
    term = cp.multiply(alpha_val * D_val, cp.inv_pos(denominator))
    return cp.sum(term)

def term2_y_step(u_val, y_var, y_th_val, N, alpha_val):
    """ Term 2 for y-step: y_th * max_n( u_n / y_n ) """
    # u_val is np.ndarray (fixed value), y_var is cp.Variable
    sqrt_u_val = np.sqrt(np.maximum(u_val, 0)) # Ensure non-negative before sqrt
    terms = [cp.quad_over_lin(sqrt_u_val[n], y_var[n]) for n in range(N)]
    if not terms: # Handle N=0 case
        return 0
    
    if alpha_val == 1:
        return 0
    return (1-alpha_val) * y_th_val * cp.max(cp.hstack(terms))


# --- Main BCD Solver Function ---

def solve_bcd_optimization_v2(
    N: int,
    K: float,
    alpha: float,
    D: np.ndarray,
    lambda_vec: np.ndarray,
    y_th: float,
    u_min: float,
    y_max_vec: np.ndarray, # Input arrays, might have y_min > y_max
    y_min_vec: np.ndarray, # Input arrays, might have y_min > y_max
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    initial_u: np.ndarray | None = None,
    initial_y: np.ndarray | None = None,
    solver: str = cp.SCS,
    verbose: bool = False,
    epsilon: float = DEFAULT_EPSILON
) -> dict:
    
    if verbose:
        print("--- Initializing BCD Solver (v2 Logic) ---")

    # --- 1. Input Validation & Preparation ---
    assert D.shape == (N,), "D must have shape (N,)"
    assert lambda_vec.shape == (N,), "lambda_vec must have shape (N,)"
    assert y_max_vec.shape == (N,), "y_max_vec must have shape (N,)"
    assert y_min_vec.shape == (N,), "y_min_vec must have shape (N,)"
    # assert u_min >= epsilon, f"u_min ({u_min}) should be >= epsilon ({epsilon}) for safety"
    assert K > 0, "K must be positive"
    assert alpha >= 0, "alpha must be non-negative"
    assert y_th >= 0, "y_th must be non-negative"

    # --- Adjust y_min/y_max (as per script logic) ---
    # Work on copies to avoid modifying input arrays
    y_min_adj = y_min_vec.copy()
    y_max_adj = y_max_vec.copy()


    # --- 2. CVXPY Variables ---
    u_var = cp.Variable(N, name="u")
    y_var = cp.Variable(N, name="y")

    # --- 3. CVXPY Parameters ---
    u_fixed = cp.Parameter(N, nonneg=True, name="u_fixed")
    y_fixed = cp.Parameter(N, nonneg=True, name="y_fixed")
    alpha_param = cp.Parameter(nonneg=True, name="alpha_param")
    alpha_param.value = alpha # Set once

    if initial_u is None:
        u_current = np.ones(N)
        u_sum = np.sum(u_current)
        if u_sum > K:
            u_current = u_current / u_sum * K # Scale down to meet sum constraint
        # Re-clip after scaling (important if K is small)
        u_current = np.clip(u_current, u_min, 1.0)
        # Final check on sum after clipping (might slightly exceed K if many hit u_min)
        u_sum = np.sum(u_current)
        if u_sum > K + epsilon:
            # Heuristic: slightly scale down again if significantly over K
             u_current = u_current * (K / u_sum) * 0.999
             u_current = np.maximum(u_current, u_min) # Ensure u_min holds

    if verbose: print(f"Initial u sum: {np.sum(u_current):.4f} (Constraint <= {K})")

    # Initialize y based on initial u and adjusted bounds (as per script)
    y_effective_lower_bound = y_min_adj # Use the adjusted lower bound
    y_effective_upper_bound = y_max_adj # Use the adjusted & lambda-limited upper bound

    # --- Check for feasible y range before random init ---
    if np.any(y_effective_lower_bound > y_effective_upper_bound + epsilon):
        print("ERROR: Adjusted y_min/y_max and lambda constraints lead to infeasible y bounds.")
        problematic_indices = np.where(y_effective_lower_bound > y_effective_upper_bound + epsilon)[0]
        print(f"Problematic indices (first 5): {problematic_indices[:5]}")
        print(f"Example (idx={problematic_indices[0]}): y_min_adj={y_effective_lower_bound[problematic_indices[0]]:.4f}, y_max_adj={y_effective_upper_bound[problematic_indices[0]]:.4f}")
        return {
            'status': 'infeasible_init', 'u': u_current, 'y': None,
            'objective': None, 'iterations': 0, 'objective_history': []
        }
    if verbose: print("Initial y bounds check passed.")

    if initial_y is None:
         # Random Initialization for y_current within effective bounds
         y_current = (y_effective_lower_bound + y_effective_upper_bound) / 2
         y_current = np.clip(y_current, y_min_adj, y_max_adj) # Clip just in case
         y_current = np.maximum(y_current, epsilon) # Ensure positivity
    else:
         assert initial_y.shape == (N,), "initial_y must have shape (N,)"
         y_current = np.clip(initial_y, y_min_adj, y_max_adj) # Clip to adjusted bounds
         y_current = np.maximum(y_current, epsilon)

    # Set initial parameter values
    u_fixed.value = u_current
    y_fixed.value = y_current

    # --- 5. BCD Iteration Loop ---
    objective_history = []
    status = "max_iterations_reached" # Default if loop finishes naturally

    if verbose: print("\n--- Starting BCD Iterations ---")

    for i in range(max_iterations):
        if verbose: print(f"\n--- Iteration {i + 1} ---")
        iteration_start_time = time.time()
        obj_prev = objective_history[-1] if objective_history else None

        # --- Step 1: Fix u, Solve for y ---
        if verbose: print("  Solving for y...")
        try:
            # Objective using fixed u_current value
            obj_y = term1(u_fixed.value, y_var, alpha_param.value, D, lambda_vec, N, epsilon) + term2_y_step(u_fixed.value, y_var, y_th, N, alpha_param.value)

            constraints_y = [
                y_var >= y_min_adj, # Use adjusted y_min
                y_var <= y_max_adj, # Use adjusted y_max
                1 - cp.multiply(lambda_vec, y_var) >= epsilon, # Denominator constraint
                cp.multiply(u_fixed.value, 1 - cp.multiply(lambda_vec, y_var)) >= epsilon, # Ensure denominator is positive
                y_var >= epsilon 
            ]
            problem_y = cp.Problem(cp.Minimize(obj_y), constraints_y)
            solve_start = time.time()
            problem_y.solve(solver=solver, verbose=False) # Suppress default solver verbosity
            solve_time = time.time() - solve_start

            if problem_y.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and y_var.value is not None:
                y_current = np.clip(y_var.value, y_min_adj, y_max_adj) # Clip to feasible range
                y_current = np.maximum(y_current, epsilon) # Ensure positivity
                y_fixed.value = y_current # Update parameter for the next u-step
                if verbose:
                    print(f"    y solved. Status: {problem_y.status}, Obj: {problem_y.value:.4f}, Time: {solve_time:.2f}s")
                if problem_y.status == cp.OPTIMAL_INACCURATE and verbose:
                    print("    WARNING: y subproblem solved inaccurately.")
            else:
                print(f"    ERROR: y optimization failed. Status: {problem_y.status}")
                status = "failed"
                break # Exit BCD loop

        except Exception as e:
            print(f"    ERROR during y step definition or solve: {e}")
            status = "failed"
            break # Exit BCD loop


        # --- Step 2: Fix y, Solve for u ---
        if verbose: print("  Solving for u...")
        try:
            # Pre-calculate constants based on fixed y_current (as per script)
            # Ensure denominator safety when calculating constants
            safe_y_fixed_val = np.maximum(y_fixed.value, epsilon)
            safe_denom_inner_y = np.maximum(1 - lambda_vec * safe_y_fixed_val, epsilon)

            term1_const = (alpha * D) / safe_denom_inner_y
            term2_const = (1-alpha) * y_th / safe_y_fixed_val

            # Define u-objective using pre-calculated constants
            obj_u = cp.sum(cp.multiply(term1_const, cp.inv_pos(u_var))) + cp.max(cp.multiply(u_var, term2_const))
            # obj_u = cp.sum(cp.multiply(term1_const, cp.inv_pos(u_var)))

            constraints_u = [
                u_var >= u_min,
                u_var <= 1.0,
                cp.sum(u_var) <= K
            ]
            problem_u = cp.Problem(cp.Minimize(obj_u), constraints_u)
            solve_start = time.time()
            problem_u.solve(solver=solver, verbose=False)
            solve_time = time.time() - solve_start

            if problem_u.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and u_var.value is not None:
                u_current = np.clip(u_var.value, u_min, 1.0) # Clip to bounds

                # Projection step: Ensure sum constraint holds strictly after solve
                current_sum_u = np.sum(u_current)
                if current_sum_u > K + epsilon: # Use epsilon for tolerance
                    if verbose: print(f"    Projecting u sum from {current_sum_u:.4f} back towards {K:.4f}")
                    u_current = u_current / current_sum_u * K
                    u_current = np.maximum(u_current, u_min) # Ensure u_min after projection
                    # Final check/clip if u_min enforcement pushed sum over K again
                    current_sum_u = np.sum(u_current)
                    if current_sum_u > K + epsilon:
                         u_current = u_current / current_sum_u * K
                         u_current = np.maximum(u_current, u_min)

                u_fixed.value = u_current # Update parameter for the next y-step

                # Use the objective value from the *u* subproblem for convergence check
                current_objective = problem_u.value
                objective_history.append(current_objective)

                if verbose:
                     print(f"    u solved. Status: {problem_u.status}, Obj: {current_objective:.4f}, Sum(u): {np.sum(u_current):.4f}, Time: {solve_time:.2f}s")
                if problem_u.status == cp.OPTIMAL_INACCURATE and verbose:
                    print("    WARNING: u subproblem solved inaccurately.")


                # --- Convergence Check ---
                if obj_prev is not None:
                    rel_change = abs(current_objective - obj_prev) / (abs(obj_prev) + epsilon)
                    if verbose:
                         print(f"    Relative objective change: {rel_change:.6f}")
                    if rel_change < tolerance:
                        if verbose: print(f"\nConvergence reached at iteration {i + 1}.")
                        status = "converged"
                        break # Exit BCD loop
                elif verbose:
                    print(f"    Objective value (u-step): {current_objective:.4f}")

            else:
                print(f"    ERROR: u optimization failed. Status: {problem_u.status}")
                status = "failed"
                break # Exit BCD loop

        except Exception as e:
            print(f"    ERROR during u step definition or solve: {e}")
            status = "failed"
            break # Exit BCD loop

        iteration_time = time.time() - iteration_start_time
        if verbose: print(f"    Iteration time: {iteration_time:.2f} sec")

    # --- 6. Final Status & Results ---
    # Status is already set correctly by loop breaks or default
    final_objective = objective_history[-1] if objective_history else None

    if verbose or status == 'failed': print("\n--- BCD Finished ---")
    if verbose:
        print(f"Status: {status}")
        print(f"Iterations: {len(objective_history)}")

    # Final constraint verification if solution obtained
    if status in ['converged', 'max_iterations_reached']:
        if verbose:
            if final_objective is not None:
                print(f"Final Objective (from u-step): {final_objective:.6f}")
            print(f"Final u (sum={np.sum(u_current):.4f}): min={np.min(u_current):.4f}, max={np.max(u_current):.4f}, mean={np.mean(u_current):.4f}")
            print(f"Final y: min={np.min(y_current):.4f}, max={np.max(y_current):.4f}, mean={np.mean(y_current):.4f}")

            tol_check = 1e-5
            print("\nConstraint Verification:")
            print(f"  u_min <= u <= 1: {np.all(u_current >= u_min - tol_check) and np.all(u_current <= 1.0 + tol_check)}")
            print(f"  sum(u) <= K: {np.sum(u_current) <= K + tol_check}")
            # Use adjusted bounds for y check
            print(f"  y_min_adj <= y <= y_max_adj: {np.all(y_current >= y_min_adj - tol_check) and np.all(y_current <= y_max_adj + tol_check)}")
            print(f"  y >= epsilon: {np.all(y_current >= epsilon - tol_check)}")
            print(f"  1 - lambda*y >= epsilon: {np.all(1 - lambda_vec * y_current >= epsilon - tol_check)}")
    elif final_objective is None and verbose:
         print("Optimization did not converge or failed before first u-step completion.")

    return {
        'status': status,
        'u': u_current,
        'y': y_current,
        'objective': final_objective,
        'iterations': len(objective_history),
        'objective_history': objective_history
    }