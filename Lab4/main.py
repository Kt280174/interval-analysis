import numpy as np
import struct
import intvalpy as ip
import matplotlib.pyplot as plt
import pandas as pd
import copy
ip.precision.extendedPrecisionQ = False
def read_thomson_bin(path):
    with open(path, 'rb') as f:
        header = f.read(256)
        side, mode, frame_count = struct.unpack('<BBH', header[:4])
        print(f"Side={side}, Mode={mode}, Frames={frame_count}")

        frames = []
        point_dtype = np.dtype('<8H')

        for _ in range(frame_count):
            frame_header_data = f.read(16)
            if len(frame_header_data) < 16:
                break
            stop_point, timestamp = struct.unpack('<HL', frame_header_data[:6])
            frame_data = np.frombuffer(f.read(1024*16), dtype=point_dtype)
            frames.append(frame_data)
    frames = np.array(frames)
    volts = frames/16384.0 - 0.5
    return volts

def union_intervals(x, y):
    a = min(float(x.a), float(y.a))
    b = max(float(x.b), float(y.b))
    return ip.Interval([a, b])

def are_intersection(x, y):
    res = ip.intersection(x, y)
    if np.isnan(res.a) and np.isnan(res.b):
        return False
    return True
def interval_mode(x):
    if x is None:
        return None
    edges = []
    for x_i in x:
        edges.append(x_i.a)
        edges.append(x_i.b)
    edges = sorted(edges)
    Z = [ip.Interval(edges[i], edges[i+1]) for i in range(len(edges) - 1)]
    mu = [sum(1 for x_i in x if z_i in x_i) for z_i in Z]
    max_mu = max(mu)
    K = [index for index, element in enumerate(mu) if element == max_mu]
    m = [Z[k] for k in K]
    merged = []
    curr_int = m[0]
    for next_int in m[1:]:
        if are_intersection(curr_int, next_int):
            curr_int = union_intervals(curr_int, next_int)
        else:
            merged.append(curr_int)
            curr_int = next_int
    print("yeah")
    merged.append(curr_int)
    return merged

def kreinovich_median(x):
    lowers = [float(el.a) for el in x]
    uppers = [float(el.b) for el in x]
    med_lower = float(np.median(lowers))
    med_upper = float(np.median(uppers))
    return ip.Interval([med_lower, med_upper])

def prolubnikov_median(x):
    X = sorted(x, key=lambda t: (float(t.a) + float(t.b)) / 2)
    index_med = len(X) // 2
    if len(X) % 2 == 0:
        return (X[index_med - 1] + X[index_med]) / 2

    # print("\n\ntype(x[index_med]): ", type(x[index_med]))
    return X[index_med]

def jaccard_interval(x, y):
    a1, b1 = float(x.a), float(x.b)
    a2, b2 = float(y.a), float(y.b)
    
    numerator = min(b1, b2) - max(a1, a2)
    denominator = max(b1, b2) - min(a1, a2)
    return numerator / denominator

def jaccard_set(X):
    lowers = [float(x.a) for x in X]
    uppers = [float(x.b) for x in X]
    numerator = min(uppers) - max(lowers)
    denominator = max(uppers) - min(lowers)
    return numerator / denominator

def jaccard_sets(X, Y):
    coeffs = [jaccard_interval(x, y) for x, y in zip(X, Y)]
    return np.array(coeffs)

def coefficient_Jaccard(X, Y=None):
    if Y is None:
        return jaccard_set(X)
    
    if isinstance(X, ip.ClassicalArithmetic) and isinstance(Y, ip.ClassicalArithmetic):
        return jaccard_interval(X, Y)
    return jaccard_sets(X, Y)

def golden(f,a , b, eps=1e-4):
    phi = (3 - np.sqrt(5))/2
    x1, x2 = a + phi*(b-a), b - phi*(b-a)
    f1, f2 = f(x1), f(x2)
    while abs(b - a) > eps:
        if f1 < f2:
            a, x1, f1 = x1, x2, f2
            x2 = b - phi*(b-a)
            f2 = f(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = a + phi*(b-a)
            f1 = f(x1)
    return (a+ b)/2

def est_a(a, X, Y):
    print(np.mean(coefficient_Jaccard(X+a, Y)))
    return np.mean(coefficient_Jaccard(X + a, Y))

def est_t(t, X, Y):
    return np.mean(coefficient_Jaccard(X*t, Y))

def est_a_mode(a, X, Y):
    print(np.mean(coefficient_Jaccard(interval_mode(X+a), interval_mode(Y))))
    return np.mean(coefficient_Jaccard(interval_mode(X+a), interval_mode(Y)))
def est_t_mode(t, X, Y):
    return np.mean(coefficient_Jaccard(interval_mode(X*t), interval_mode(Y)))

def est_a_pro_med(a, X, Y):
    return np.mean(coefficient_Jaccard(prolubnikov_median(X+a), prolubnikov_median(Y)))

def est_t_pro_med(t, X, Y):
    return np.mean(coefficient_Jaccard(prolubnikov_median(X*t), prolubnikov_median(Y)))

def est_a_kre_med(a, X, Y):
    return np.mean(coefficient_Jaccard(kreinovich_median(X+a), kreinovich_median(Y)))

def est_t_kre_med(t, X, Y):
    return np.mean(coefficient_Jaccard(kreinovich_median(X*t), kreinovich_median(Y)))

def scalar_to_interval(x, rad):
    return ip.Interval(x-rad, x+rad)
scalar_to_interval_vec = np.vectorize(scalar_to_interval)

def get_avg(data):
    avg = [[0]*8]*1024
    for i in range(len(data)): # 100
        avg = np.add(avg, data[i])
    return np.divide(avg, len(data))

x_data = read_thomson_bin("-0.205_lvl_side_a_fast_data.bin")
y_data = read_thomson_bin("0.225_lvl_side_a_fast_data.bin")

rad = 2**(-14)
x_data = get_avg(x_data)
y_data = get_avg(y_data)

# print("len(x_data): ", len(x_data), len(x_data[0]))
# print("len(y_data): ", len(y_data), len(y_data[0]))

rad = 2 ** (-14)

X = scalar_to_interval_vec(x_data, rad).flatten()
Y = scalar_to_interval_vec(y_data, rad).flatten()

bound_a_l = float(np.min(Y).a) - float(np.max(X).b)
bound_a_r = float(np.max(Y).b) - float(np.min(X).a)

bound_t_l = float(np.min(Y).a) / float(np.max(X).b) 
bound_t_r = float(np.max(Y).b) / float(np.min(X).a)

number = 100
eps_a = (bound_a_r - bound_a_l) / number
eps_t = (bound_t_r - bound_t_l) / number

print(f"\nДиапазон для a: [{bound_a_l:.4f}, {bound_a_r:.4f}], ε = {eps_a:.4f}")
print(f"Диапазон для t: [{bound_t_l:.4f}, {bound_t_r:.4f}], ε = {eps_t:.4f}")


def plot_functional(name, func_a, func_t, X, Y, 
                    bound_a_l, bound_a_r, bound_t_l, bound_t_r, n_points=100, eps=1e-3):
    """
    Визуализирует функционалы F(a) и F(t), отмечая максимум s_max,
    найденный методом золотого сечения.
    """
    # --- Поиск экстремумов ---
    a_opt = golden(lambda a: func_a(a, X, Y), bound_a_l, bound_a_r, eps)
    t_opt = golden(lambda t: func_t(t, X, Y), bound_t_l, bound_t_r, eps)
    val_a = func_a(a_opt, X, Y)
    val_t = func_t(t_opt, X, Y)

    print(f"{name}: a* = {a_opt:.4f} (F={val_a:.4f}), t* = {t_opt:.4f} (F={val_t:.4f})")

    # --- Построение дискретных графиков ---
    a_values = np.linspace(bound_a_l, bound_a_r, n_points)
    t_values = np.linspace(bound_t_l, bound_t_r, n_points)

    # --- Вычисление значений функционалов ---
    Ji_a = [func_a(a, X, Y) for a in a_values]
    Ji_t = [func_t(t, X, Y) for t in t_values]

    # --- Поиск максимумов ---
    idx_a_max = np.argmax(Ji_a)
    idx_t_max = np.argmax(Ji_t)
    a_opt, val_a = a_values[idx_a_max], Ji_a[idx_a_max]
    t_opt, val_t = t_values[idx_t_max], Ji_t[idx_t_max]
    # --- Построение графиков ---
    print(f"{name}: a* = {a_opt:.4f} (F={val_a:.4f}), t* = {t_opt:.4f} (F={val_t:.4f})")
    plt.figure(figsize=(10, 4))

    # F(a)
    plt.subplot(1, 2, 1)
    plt.plot(a_values, Ji_a, label=f"{name}(a)")
    plt.axvline(x=a_opt, color='red', linestyle='--', label=f"a* = {a_opt:.4f}")
    plt.scatter(a_opt, val_a, color='red')
    plt.title(f"{name}(a) — максимум a*")
    plt.xlabel("a"); plt.ylabel("Jaccard")
    plt.legend(); plt.grid(True)

    # F(t)
    plt.subplot(1, 2, 2)
    plt.plot(t_values, Ji_t, label=f"{name}(t)")
    plt.axvline(x=t_opt, color='red', linestyle='--', label=f"t* = {t_opt:.4f}")
    plt.scatter(t_opt, val_t, color='red')
    plt.title(f"{name}(t) — максимум t*")
    plt.xlabel("t"); plt.ylabel("Jaccard")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=300)
    plt.close()
'''plot_functional("F1",
    func_a=lambda a, X, Y: np.mean(coefficient_Jaccard(X + a, Y)),
    func_t=lambda t, X, Y: np.mean(coefficient_Jaccard(X * t, Y)),
    X=X, Y=Y,
    bound_a_l=bound_a_l, bound_a_r=bound_a_r,
    bound_t_l=-1.25, bound_t_r=bound_t_r
)

# 2️⃣ F2 = Ji(mode(X), mode(Y))
plot_functional("F2",
    func_a=lambda a, X, Y: np.mean(coefficient_Jaccard(interval_mode(X + a), interval_mode(Y))),
    func_t=lambda t, X, Y: np.mean(coefficient_Jaccard(interval_mode(X * t), interval_mode(Y))),
    X=X, Y=Y,
    bound_a_l=bound_a_l, bound_a_r=bound_a_r,
    bound_t_l=-1.25, bound_t_r=bound_t_r
)

# 3️⃣ F3 = Ji(medK(X), medK(Y))
plot_functional("F3",
    func_a=lambda a, X, Y: np.mean(coefficient_Jaccard(kreinovich_median(X + a), kreinovich_median(Y))),
    func_t=lambda t, X, Y: np.mean(coefficient_Jaccard(kreinovich_median(X * t), kreinovich_median(Y))),
    X=X, Y=Y,
    bound_a_l=bound_a_l, bound_a_r=bound_a_r,
    bound_t_l=-1.25, bound_t_r=bound_t_r
)

# 4️⃣ F4 = Ji(medP(X), medP(Y))
plot_functional("F4",
    func_a=lambda a, X, Y: np.mean(coefficient_Jaccard(prolubnikov_median(X + a), prolubnikov_median(Y))),
    func_t=lambda t, X, Y: np.mean(coefficient_Jaccard(prolubnikov_median(X * t), prolubnikov_median(Y))),
    X=X, Y=Y,
    bound_a_l=bound_a_l, bound_a_r=bound_a_r,
    bound_t_l=-1.25, bound_t_r=bound_t_r
)'''
a_opt = golden(lambda a: est_a_mode(a, X, Y), bound_a_l, bound_a_r, 1e-3)
t_opt = golden(lambda t: est_t_mode(t, X, Y), bound_t_l, bound_t_r, 1e-3)
val_a = est_a_mode(a_opt, X, Y)
val_t = est_t_mode(t_opt, X, Y)
print(f"a* = {a_opt:.4f} (F={val_a:.4f}), t* = {t_opt:.4f} (F={val_t:.4f})")
