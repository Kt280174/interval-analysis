import numpy as np


def print_matrix(M, title=None, fmt="{:10.6f}"):
    if title:
        print(title)
    for row in M:
        print(" ".join(fmt.format(x) for x in row))
    print()

def print_interval_matrix(A0, eps, fmt="[{:.6f},{:.6f}]"):
    for row in A0:
        a, b = row
        print(fmt.format(a - eps, a + eps), fmt.format(b - eps, b + eps))
    print()

def debug_step(A0, eps, eps_range, step_id, phase=""):
    print(f"step = {step_id}" + (f"  ({phase})" if phase else ""))
    if eps_range is not None:
        l, r = eps_range
        print(f"interval epsilon = [{l:.6f}, {r:.6f}]")
    else:
        print("epsilon = (Not defined)")
    print(f"epsilon = {eps:.6f}")
    print("A(eps):")
    print_interval_matrix(A0, eps)

# ====== Interval helpers & collinearity parts 
def interval_div_positive(u_min, u_max, v_min, v_max):
    assert v_min > 0, "Знаменатель должна быть положительным."
    cands = [u_min / v_min, u_min / v_max, u_max / v_min, u_max / v_max]
    return min(cands), max(cands)

def interval_scale(lam, v_min, v_max):
    a = lam * v_min; b = lam * v_max
    return (min(a, b), max(a, b))

def interval_intersection(a_min, a_max, b_min, b_max, tol=0.0):
    lo = max(a_min, b_min); hi = min(a_max, b_max)
    if lo <= hi + tol: return (lo, hi)
    return None

def lambda_interval_for_eps(A0, eps, require_pos_den=True, verbose=False):
    a = A0[:, 0].astype(float)
    b = A0[:, 1].astype(float)
    if require_pos_den and np.any(b - eps <= 0):
        return None, []
    lam_l, lam_r = -np.inf, np.inf
    steps = []
    for i in range(len(a)):
        u_min, u_max = a[i] - eps, a[i] + eps
        v_min, v_max = b[i] - eps, b[i] + eps
        if require_pos_den:
            lo, hi = interval_div_positive(u_min, u_max, v_min, v_max)
        else:
            if v_min <= 0 <= v_max:
                return None, steps
            lo, hi = interval_div_positive(u_min, u_max, v_min, v_max)
        lam_l = max(lam_l, lo); lam_r = min(lam_r, hi)
        steps.append({"row": i, "U": (u_min, u_max), "V": (v_min, v_max),
                      "U_div_V": (lo, hi), "lam_running": (lam_l, lam_r)})
        if lam_l > lam_r:
            return None, steps
    return (lam_l, lam_r), steps

def construct_degenerate_matrix(A0, eps, lam, tol=1e-12, verbose=False):
    a = A0[:, 0].astype(float); b = A0[:, 1].astype(float)
    m = len(a); u = np.zeros(m); v = np.zeros(m)
    for i in range(m):
        U = (a[i] - eps, a[i] + eps)
        V = (b[i] - eps, b[i] + eps)
        WV = interval_scale(lam, V[0], V[1])
        I  = interval_intersection(U[0], U[1], WV[0], WV[1], tol=1e-12)
        if I is None: return None
        u_i = 0.5 * (I[0] + I[1])
        v_i = u_i / lam if abs(lam) > tol else 0.0
        v_i = min(max(v_i, V[0]), V[1])
        u[i], v[i] = u_i, v_i
    return np.column_stack([u, v])

def cols_are_collinear(A0, eps, verbose=False):
    lam_rng, steps = lambda_interval_for_eps(A0, eps, require_pos_den=True, verbose=verbose)
    if lam_rng is None: return False, None, steps
    return True, lam_rng, steps


def epsilon_star_3x2(A0, delta=1e-4, eps_init=0.05, eps_max=0.99, verbose=True):
    step_id = 0

    ok0, lam_rng0, _ = cols_are_collinear(A0, 0.0, verbose=verbose)
    if ok0:
        if verbose:
            debug_step(A0, 0.0, (0.0, 0.0), step_id, phase="initial")
        lam_star = lam_rng0[0] if lam_rng0 else 1.0
        M0 = construct_degenerate_matrix(A0, 0.0, lam_star, verbose=verbose)
        return 0.0, M0, lam_star

    left, right = 0.0, eps_init
    ok, lam_rng, _ = cols_are_collinear(A0, right, verbose=verbose)
    while not ok and right < eps_max:
        left = right
        right *= 2
        ok, lam_rng, _ = cols_are_collinear(A0, right, verbose=verbose)
        if verbose:
            step_id += 1
            debug_step(A0, right, (left, right), step_id, phase="expand")

    if not ok:
        return None, None, None

    while right - left > delta:
        mid = (left + right) / 2.0
        ok, lam_rng, _ = cols_are_collinear(A0, mid, verbose=verbose)
        if ok:
            right = mid
        else:
            left = mid
        if verbose:
            step_id += 1
            debug_step(A0, mid, (left, right), step_id, phase="bisection")

    lam_rng_final, _ = lambda_interval_for_eps(A0, right, verbose=verbose)
    lam_star = lam_rng_final[0]
    M_star = construct_degenerate_matrix(A0, right, lam_star, verbose=verbose)
    if M_star is None and lam_rng_final is not None:
        lam_star = lam_rng_final[1]
        M_star = construct_degenerate_matrix(A0, right, lam_star, verbose=verbose)

    return right, M_star, lam_star

# ===== Demo =====
if __name__ == "__main__":
    A0 = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.20, 1.00]
    ], dtype=float)

    eps_star, M_star, lam_star = epsilon_star_3x2(A0, delta=1e-5, eps_init=0.05, verbose=True)
    if eps_star is None:
        print("Не найден eps.")
    else:
        print(f"\nε* ≈ {eps_star:.6f},  λ* ≈ {lam_star:.6f}")
        print_interval_matrix(A0, eps_star)
        if M_star is not None:
            print_matrix(M_star, title="M*:")
