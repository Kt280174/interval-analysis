import copy
import numpy as np
np.float_ = np.float64
import intvalpy as ip
import matplotlib.pyplot as plt
from tol import tolsolvty
import math
import os

ip.precision.extendedPrecisionQ = False


def is_empty(A, b):
    maxTol = ip.linear.Tol.maximize(A, b)
    return maxTol[1] < 0, maxTol[0], maxTol[1]


def b_correction(b, k):
    e = ip.Interval([[-k, k] for _ in range(len(b))])
    return b + e
def find_k_min(A, b, eps=1e-3, max_iter=100):
    """
    Tìm k nhỏ nhất sao cho hệ (A, b+[-k, k]) khả giải.
    - eps: độ chính xác
    - max_iter: giới hạn vòng lặp
    """
    # --- Bước 1: Kiểm tra hệ ban đầu ---
    emptiness, max_x, max_tol = is_empty(A, b)
    print(f"[Init] maxTol = {max_tol:.6g}")
    if not emptiness:
        print("Hệ đã có nghiệm dung sai, không cần hiệu chỉnh.")
        return 0.0

    # --- Bước 2: Dò mũ để tìm khoảng [low, high] ---
    low = 0.0
    high = None
    i = 0
    while i < max_iter:
        k_try = math.exp(i)           # tăng mũ: 1, e, e², ...
        b_try = b_correction(b, k_try)
        emptiness, max_x, max_tol = is_empty(A, b_try)
        print(f"[Grow] i={i}, k={k_try:.6g}, maxTol={max_tol:.6g}")
        if not emptiness:              # hệ đã khả giải
            high = k_try
            low = math.exp(i - 1)
            break
        i += 1

    if high is None:
        raise RuntimeError("Không tìm thấy K đủ lớn trong giới hạn cho phép")

    # --- Bước 3: Tìm nhị phân ---
    iteration = 0
    while abs(high - low) > eps and iteration < max_iter:
        mid = (low + high) / 2
        b_mid = b_correction(b, mid)
        emptiness, max_x, max_tol = is_empty(A, b_mid)
        print(f"[Bisect] iter={iteration}, k={mid:.6g}, maxTol={max_tol:.6g}")
        if emptiness:
            low = mid
        else:
            high = mid
        iteration += 1

    # --- Bước 4: Kết quả ---
    k_min = high
    b_corr = b_correction(b, k_min)
    print(f"[Result] k_min = {k_min:.6g}")
    return k_min, b_corr

def A_correction(A, b):
    max_tol = ip.linear.Tol.maximize(A, b)
    lower_bound = abs(max_tol[1]) / (abs(max_tol[0][0]) + abs(max_tol[0][1]))
    rad_A = ip.rad(A)
    upper_bound = rad_A[0][0]
    for a_i in rad_A:
        for a_ij in a_i:
            if a_ij < upper_bound:
                upper_bound = a_ij
    e = (lower_bound + upper_bound) / 2
    corrected_A = []
    for i in range(len(A)):
        A_i = []
        for j in range(len(A[0])):
            if ip.rad(A[i][j]) == 0:
                A_i.append([A[i][j]._a, A[i][j]._b])
            else:
                A_i.append([A[i][j]._a + e, A[i][j]._b - e])
        corrected_A.append(A_i)
    return ip.Interval(corrected_A)


def Ab_correction(A, b, max_iter=50):
    emptiness, max_x, max_Tol = is_empty(A, b)
    new_A = copy.deepcopy(A)
    new_b = copy.deepcopy(b)
    iteration = 0

    while emptiness and iteration < max_iter:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        new_A = A_correction(new_A, new_b)
        emptiness, max_x, max_Tol = is_empty(new_A, new_b)
        print(f"After A-correction: Tol = {max_Tol:.6f}")
        if not emptiness:
            break
        new_b = b_correction(new_b, k=iteration)
        emptiness, max_x, max_Tol = is_empty(new_A, new_b)
        print(f"After b-correction: Tol = {max_Tol:.6f}")

    if emptiness:
        print("⚠️ Система осталась неразрешимой после комбинированной коррекции.")
    else:
        print("✅ Система стала разрешимой после Ab-коррекции.")

    return new_A, new_b


def plot_tol(axis, A, b):
    max_tol = ip.linear.Tol.maximize(A, b)
    print("Max tol = ", max_tol[0])
    print("Max tol1 = ", max_tol[1])
    grid_min, grid_max = max_tol[0][0] - 2, max_tol[0][0] + 2
    x_1_, x_2_ = np.mgrid[grid_min:grid_max:70j, grid_min:grid_max:70j]
    list_x_1 = np.linspace(grid_min, grid_max, 70)
    list_x_2 = np.linspace(grid_min, grid_max, 70)

    list_tol = np.zeros((70, 70))

    for idx_x1, x1 in enumerate(list_x_1):
        for idx_x2, x2 in enumerate(list_x_2):
            x = [x1, x2]
            tol_values = []
            for i in range(len(b)):
                sum_ = sum(A[i][j] * x[j] for j in range(len(x)))
                rad_b, mid_b = ip.rad(b[i]), ip.mid(b[i])
                tol = rad_b - ip.mag(mid_b - sum_)
                tol_values.append(tol)
            list_tol[idx_x1, idx_x2] = min(tol_values)

    axis.view_init(elev=30, azim=45)
    surf = axis.plot_surface(x_1_, x_2_, list_tol,
                             cmap='plasma', edgecolor='none', alpha=0.9)
    axis.scatter(*max_tol[0], max_tol[1],
                 color='red', s=60, label='Точка максимума')
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')
    axis.set_zlabel('Tol(x)')
    axis.legend()

def plot_tol_functional(axis, A, b):
    max_tol = ip.linear.Tol.maximize(A, b)
    solution = max_tol[0]
    print("Max tol (solution) = ", solution)
    x = np.linspace(float(solution[0]) - 2, float(solution[0]) + 2, 101)
    y = np.linspace(float(solution[1]) - 2, float(solution[1]) + 2, 101)
    xx, yy = np.meshgrid(x, y)
    zz = np.array(
        [[1 if ip.linear.Tol.value(A, b, [x, y]) >= 0 else 0
          for x, y in zip(x_row, y_row)]
         for x_row, y_row in zip(xx, yy)]
    )

    # vàng = vùng Tol>0, xanh dương = Tol<0
    cmap = plt.cm.colors.ListedColormap(["#4472C4", "#FFD966"])
    axis.contourf(xx, yy, zz, levels=1, cmap=cmap, alpha=0.9)
    axis.scatter(solution[0], solution[1],
                 color='black', marker='x', s=60, label='Максимум Tol')
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')
    axis.legend(loc='upper right')

def visualize_tol(A, b, name: str, show=False, save=True):
    os.makedirs("source/Tol", exist_ok=True)
    os.makedirs("source/TolFunctional", exist_ok=True)

    # --- 3D surface ---
    fig3d = plt.figure(figsize=(6, 5))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.set_title(f"Tol surface ({name})")
    plot_tol(ax3d, A, b)
    fig3d.tight_layout()
    if save:
        fig3d.savefig(f"source/Tol/{name}.png")
    if show:
        plt.show()
    plt.close(fig3d)

    # --- 2D functional ---
    fig2d = plt.figure(figsize=(6, 5))
    ax2d = fig2d.add_subplot(111)
    ax2d.set_title(f"Tol functional ({name})")
    plot_tol_functional(ax2d, A, b)
    fig2d.tight_layout()
    if save:
        fig2d.savefig(f"source/TolFunctional/{name}.png")
    if show:
        plt.show()
    plt.close(fig2d)

A1 = ip.Interval([
    [[0.65, 1.25], [0.7, 1.3]],
    [[0.75, 1.35], [0.7, 1.3]]
])
b1 = ip.Interval([[2.75, 3.15],
                 [2.85, 3.25]])


A2 = ip.Interval([
    [[0.65, 1.25], [0.70, 1.3]],
    [[0.75, 1.35], [0.70, 1.3]],
    [[0.8, 1.4], [0.70, 1.3]]
])

b2 = ip.Interval([
    [2.75, 3.15],
    [2.85, 3.25],
    [2.90, 3.3]
    ])

A3 = ip.Interval([
    [[0.65, 1.25], [0.70, 1.3]],
    [[0.75, 1.35], [0.70, 1.3]],
    [[0.8, 1.4], [0.70, 1.3]],
    [[0.3, 0.3], [0.70, 1.3]]
])

b3 = ip.Interval([
    [2.75, 3.15],
    [2.85, 3.25],
    [2.90, 3.3],
    [1.8, 2.2],
    ])

As = [A1, A2, A3]
bs = [b1, b2, b3]

# As = [A1]
# bs = [b1]


def run(correction=None):
    match correction:
        # ==============================================================
        # 1️⃣  Trường hợp Ab-correction
        # ==============================================================
        case "Ab":
            print("____Ab-correction____")
            for i in range(len(As)):
                print(f"\n=== System {i + 1} ===")
                A_ = As[i]
                b_ = bs[i]

                # Hiệu chỉnh
                A_, b_ = Ab_correction(A_, b_)

                emptiness_, maxX, maxTol = is_empty(A_, b_)
                print(f"Tol max = {maxTol:.6f}, argmax = {maxX}")
        
                tolmax, argmax, envs, ccode = tolsolvty(
                    infA=ip.inf(A_), supA=ip.sup(A_),
                    infb=ip.inf(b_).reshape(-1, 1),
                    supb=ip.sup(b_).reshape(-1, 1)
                )
                print(f"[tolsovlty] Tolmax={tolmax:.6f}, argmax={argmax.ravel()}")
                print(A_)
                print(b_)
                visualize_tol(A_, b_, f"Ab-correction_{i + 1}")

        # ==============================================================
        # 2️⃣  Trường hợp chỉ hiệu chỉnh A
        # ==============================================================
        case "A":
            print("____A-correction____")
            for i in range(len(As)):
                print(f"\n=== System {i + 1} ===")
                A_ = As[i]
                b_ = bs[i]
                A_ = A_correction(A_, b_)

                emptiness_, maxX, maxTol = is_empty(A_, b_)
                print(f"After A-correction: Tol = {maxTol:.6f}, argmax = {maxX}")


                tolmax, argmax, envs, ccode = tolsolvty(
                    infA=ip.inf(A_), supA=ip.sup(A_),
                    infb=ip.inf(b_).reshape(-1, 1),
                    supb=ip.sup(b_).reshape(-1, 1)
                )
                print(f"[tolsovlty] Tolmax={tolmax:.6f}, argmax={argmax.ravel()}")
                visualize_tol(A_, b_, f"A-correction_{i + 1}")

        # ==============================================================
        # 3️⃣  Trường hợp chỉ hiệu chỉnh b
        # ==============================================================
        case "b":
            print("____b-correction____")
            for i in range(len(As)):
                print(f"\n=== System {i + 1} ===")
                A_ = As[i]
                b_ = bs[i]

                # Tìm k nhỏ nhất sao cho hệ khả giải
                k_min, b_corr = find_k_min(A_, b_, eps=1e-3, max_iter=100)
                print(f"k_min cho hệ {i} = {k_min:.6f}")
                b_ = b_correction(b_, 1)

                emptiness_, maxX, maxTol = is_empty(A_, b_)
                print(f"After b-correction: Tol = {maxTol:.6f}, argmax = {maxX}")

                tolmax, argmax, envs, ccode = tolsolvty(
                    infA=ip.inf(A_), supA=ip.sup(A_),
                    infb=ip.inf(b_).reshape(-1, 1),
                    supb=ip.sup(b_).reshape(-1, 1)
                )
                print(f"[tolsovlty] Tolmax={tolmax:.6f}, argmax={argmax.ravel()}")
                
                visualize_tol(A_, b_, f"b-correction_{i + 1}")

        # ==============================================================
        # 4️⃣  Không hiệu chỉnh
        # ==============================================================
        case None:
            print("____Without correction____")
            for i in range(len(As)):
                A_ = As[i]
                b_ = bs[i]

                tolmax, argmax, envs, ccode = tolsolvty(
                    infA=ip.inf(A_), supA=ip.sup(A_),
                    infb=ip.inf(b_).reshape(-1, 1),
                    supb=ip.sup(b_).reshape(-1, 1)
                )

                print(f"System {i + 1}: Tolmax={tolmax:.6f}, argmax={argmax.ravel()}")


#run()
#run("A")
#run("b")
run("Ab")