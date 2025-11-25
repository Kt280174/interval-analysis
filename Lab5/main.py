import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import intvalpy as ip


# ============================================================
# 1. Systems and Jacobian
# ============================================================

def f_vec(X, xc=0.0):
    """
    f(X) với X là np.array([x1, x2]), có thể là điểm hoặc interval
    f1 = x1 - x2^2
    f2 = (x1 - xc)^2 + x2^2 - 1
    """
    x1, x2 = X[0], X[1]
    return np.array([
        x1 - x2**2,
        (x1 - xc)**2 + x2**2 - 1
    ])


def J_point(x, xc=0.0):
    """
    Jacobian tại một điểm (x1, x2) – dùng cho C = J^{-1}
    """
    x1, x2 = x[0], x[1]
    return np.array([
        [1.0,       -2.0 * x2],
        [2.0 * (x1 - xc),  2.0 * x2]
    ])


def J_interval(X, xc=0.0):
    """
    Jacobian dạng interval trên toàn hộp X
    X là np.array([Interval, Interval])
    """
    X1, X2 = X[0], X[1]
    return np.array([
        [ip.Interval(1.0, 1.0),    -2.0 * X2],
        [2.0 * (X1 - xc),          2.0 * X2]
    ])


# ============================================================
# 2. Krawczyk
# ============================================================

def Krawczyk_manual(X, xc=0.0):
    """
    Tính K(X) = x_mid - C f(x_mid) + (I - C J(X)) (X - x_mid)
    X: np.array([Interval, Interval])
    Trả về: np.array([Interval, Interval]) – hộp K(X)
    """

    x_mid = np.array([ip.mid(X[0]), ip.mid(X[1])])

    # f(mid)
    f_mid = f_vec(x_mid, xc=xc)

    J_mid = J_point(x_mid, xc=xc)

    # C = J_mid^{-1}
    C = np.linalg.inv(J_mid)
    y = x_mid - C @ f_mid
    y_int = np.array([
        ip.Interval(y[0], y[0]),
        ip.Interval(y[1], y[1])
    ])

    J_int = J_interval(X, xc=xc)
    I = np.eye(2)

    CJ = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            CJ[i, j] = (C[i, 0] * J_int[0, j] +
                        C[i, 1] * J_int[1, j])

    M = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            M[i, j] = I[i, j] - CJ[i, j]

    # X - mid
    dX = np.array([
        X[0] - x_mid[0],
        X[1] - x_mid[1]
    ])

    # M * dX
    corr = np.empty(2, dtype=object)
    corr[0] = M[0, 0] * dX[0] + M[0, 1] * dX[1]
    corr[1] = M[1, 0] * dX[0] + M[1, 1] * dX[1]

    # Krawczyk map
    K = np.array([
        y_int[0] + corr[0],
        y_int[1] + corr[1]
    ])

    return K


def krawczyk_iterate(X0, xc=0.0, iters=3):
    boxes = []
    diameters = []

    X = X0.copy()
    for k in range(iters + 1):
        boxes.append(X.copy())

        d1 = 2 * ip.rad(X[0])
        d2 = 2 * ip.rad(X[1])
        diameters.append(max(float(d1), float(d2)))

        if k == iters:
            break

        K = Krawczyk_manual(X, xc=xc)

        X_new = np.array([
            ip.intersection(X[0], K[0]),
            ip.intersection(X[1], K[1])
        ])

        X = X_new

    return boxes, diameters



# ============================================================
# 3. plot figure
# ============================================================

def plot_geometry_with_boxes(xc, boxes, iters_to_show=10):
    fig, ax = plt.subplots(figsize=(6, 6))

    x2 = np.linspace(-1.5, 1.5, 400)
    x1_par = x2**2

    ax.plot(x1_par, x2, label=r'$f_1(x)=0: x_1 = x_2^2$')

    # Circle: (x1 - xc)^2 + x2^2 = 1
    t = np.linspace(0, 2 * np.pi, 400)
    x1_circ = xc + np.cos(t)
    x2_circ = np.sin(t)
    ax.plot(x1_circ, x2_circ, label=rf'$f_2(x)=0: (x_1 - {xc})^2 + x_2^2 = 1$')

    colors = [
    "red",
    "orange",
    "green",
    "purple",
    "brown",
    "cyan",
    "magenta",
    "olive",
    "navy"
]
    for k, X in enumerate(boxes[:iters_to_show + 1]):
        x1_min = ip.inf(X[0])
        x1_max = ip.sup(X[0])
        x2_min = ip.inf(X[1])
        x2_max = ip.sup(X[1])
        print(x1_min)
        rect = patches.Rectangle(
            (x1_min, x2_min),
            x1_max - x1_min,
            x2_max - x2_min,
            fill=False,
            linestyle='--',
            linewidth=1.5,
            edgecolor=colors[k % len(colors)],
            label=rf'$X^{{({k})}}$'
        )
        ax.add_patch(rect)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(rf'Геометрия и брусы $X^{{(k)}}$ для $x_c = {xc}$')
    ax.grid(True)
    ax.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.show()


def plot_convergence(diameters_dict):
    plt.figure(figsize=(9, 6))  

    for xc, diams in diameters_dict.items():
        ks = list(range(len(diams)))
        plt.plot(
            ks, diams, 
            marker='o', markersize=6, linewidth=2,
            label=rf"$x_c = {xc}$"
        )

    plt.yscale("log")  
    plt.xlabel("Итерация $k$", fontsize=12)
    plt.ylabel("Диаметр бруса (макс. ширина)", fontsize=12)
    plt.title("Скорость сходимости метода Кравчика для разных $x_c$", fontsize=14)

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.minorticks_on()

    plt.legend()
    plt.tight_layout()
    plt.show()


def print_boxes(boxes, xc):
    print(f"\n===== xc = {xc} =====")
    for k, B in enumerate(boxes):
        x1_L = float(ip.inf(B[0]))
        x1_R = float(ip.sup(B[0]))
        x2_L = float(ip.inf(B[1]))
        x2_R = float(ip.sup(B[1]))

        x1_mid = (x1_L + x1_R) / 2
        x2_mid = (x2_L + x2_R) / 2

        print(f"\nX({k}):")
        print(f"  x1 in [{x1_L:.6f}, {x1_R:.6f}]  → mid = {x1_mid:.6f}")
        print(f"  x2 in [{x2_L:.6f}, {x2_R:.6f}]  → mid = {x2_mid:.6f}")


if __name__ == '__main__':

    xc_values = [0.0, 0.5, 1.0, 1.2]

    init_boxes = {
        0.0: ((0.4, 0.9), (0.5, 1.0)),
        0.5: ((0.6, 1.1), (0.6, 1.1)),
        1.0: ((0.8, 1.2), (0.8, 1.2)),
        1.2: ((0.87, 0.98), (0.90, 1.02)), 
    }

    all_diameters = {}

    for xc in xc_values:
        print(f"\n==============================")
        print(f"=== x_c = {xc} ===")
        print(f"==============================")

        (x1_min, x1_max), (x2_min, x2_max) = init_boxes[xc]

        X0 = np.array([
            ip.Interval(x1_min, x1_max),
            ip.Interval(x2_min, x2_max)
        ])

        boxes, diams = krawczyk_iterate(X0, xc=xc, iters=6)
        all_diameters[xc] = diams
        print_boxes(boxes, xc)
        #plot_geometry_with_boxes(xc, boxes, iters_to_show=6)

    plot_convergence(all_diameters)
