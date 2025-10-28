import numpy as np
import intvalpy as ip
def make_interval_matrix(mid, rad):
    return ip.Interval(np.stack((mid - rad, mid + rad), axis=-1))

def make_interval_vector(mid, rad):
    return ip.Interval(np.stack((mid - rad, mid + rad), axis=-1))

def tol_max(A, b):
    res = ip.linear.Tol.maximize(A, b)
    return float(res[1])

def optimal_Ab(midA, radA, midb, radb, w1=1.0, w2=1.0,
               alpha_grid=np.linspace(1.0, 0.1, 40),
               beta_grid=np.linspace(1.0, 2.0, 40)):
    best = None
    print("α     β     Tol     J")
    print("-------------------------")

    for α in alpha_grid:
        for β in beta_grid:
            A = make_interval_matrix(midA, radA * α)
            b = make_interval_vector(midb, radb * β)
            T = tol_max(A, b)
            J = w1 * (1 - α)**2 + w2 * (β - 1)**2
            #print(f"{α:.3f}  {β:.3f}  {T: .4f}  {J:.4f}")
            if T >= 0:
                if best is None or J < best[0]:
                    best = (J, α, β, T)
                break  

    if best:
        J_opt, α_opt, β_opt, T_opt = best
        print(f"\n Optimal: α*={α_opt:.3f}, β*={β_opt:.3f}, Tol={T_opt:.4f}, J={J_opt:.4f}")
        return α_opt, β_opt, T_opt
    else:
        print("\nnot found Tol≥0 .")
        return None, None, None
    
def to_bounds(lo, hi):
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    mid = (lo + hi)/2
    rad = (hi - lo)/2
    return mid, rad

# --- Hệ 1 ---
A1_lo = [[0.65, 0.70],
         [0.75, 0.70]]
A1_hi = [[1.25, 1.30],
         [1.35, 1.30]]
b1_lo = [2.75, 2.85]
b1_hi = [3.15, 3.25]
midA1, radA1 = to_bounds(A1_lo, A1_hi)
midb1, radb1 = to_bounds(b1_lo, b1_hi)

# --- Hệ 2 ---
A2_lo = [[0.65, 0.70],
         [0.75, 0.70],
         [0.80, 0.70]]
A2_hi = [[1.25, 1.30],
         [1.35, 1.30],
         [1.40, 1.30]]
b2_lo = [2.75, 2.85, 2.90]
b2_hi = [3.15, 3.25, 3.30]
midA2, radA2 = to_bounds(A2_lo, A2_hi)
midb2, radb2 = to_bounds(b2_lo, b2_hi)

# --- Hệ 3 ---
A3_lo = [[ 0.65, 0.70],
         [ 0.75, 0.70],
         [ 0.80, 0.70],
         [-0.30, 0.70]]
A3_hi = [[1.25, 1.30],
         [1.35, 1.30],
         [1.40, 1.30],
         [0.30, 1.30]]
b3_lo = [2.75, 2.85, 2.90, 1.80]
b3_hi = [3.15, 3.25, 3.30, 2.20]
midA3, radA3 = to_bounds(A3_lo, A3_hi)
midb3, radb3 = to_bounds(b3_lo, b3_hi)

# ===== ÁP DỤNG COMBINED AB-CORRECTION =====
for name, (midA, radA, midb, radb) in {
    "Variant 1 (A1,b1)": (midA1, radA1, midb1, radb1),
    "Variant 1 (A2,b2)": (midA2, radA2, midb2, radb2),
    "Variant 1 (A3,b3)": (midA3, radA3, midb3, radb3),
}.items():
    print(f"\n\n=== {name} ===")
    alpha_star, beta_star, Tol_star = optimal_Ab(midA, radA, midb, radb)

    if alpha_star is not None:
        print(f"alpha* (radA scale) = {alpha_star:.3f}")
        print(f"beta*  (radb scale) = {beta_star:.3f}")
        print(f"Tol_final           = {Tol_star:.4f}")
    else:
        print("⚠️  Không đạt Tol ≥ 0 trong miền tìm kiếm.")