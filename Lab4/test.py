import copy

import struct
import numpy as np
np.float_ = np.float64
import intvalpy as ip
import matplotlib.pyplot as plt
from functools import cmp_to_key

ip.precision.extendedPrecisionQ = False


def union_intervals(x, y):
    return ip.Interval(min(x.a, y.a), max(x.b, y.b))


def mode(X):
    print("MODE1")
    if X is None:
        return None

    # InterSec = X[0]
    # for el in X[1:]:
    #     InterSec = ip.intersection(InterSec, el)
    #
    # if not (np.isnan(InterSec.a) and np.isnan(InterSec.b)):
    #     return InterSec

    Y = []
    for el in X:
        Y.append(el.a)
        Y.append(el.b)

    Y.sort()

    Z = [ip.Interval(Y[i], Y[i + 1]) for i in range(len(Y) - 1)]

    mu = [sum(1 for x_i in X if z_i in x_i) for z_i in Z]

    max_mu = max(mu)
    K = [index for index, element in enumerate(mu) if element == max_mu]

    m = [Z[k] for k in K]
    mode_ = []

    current_interval = m[0]

    for next_interval in m[1:]:
        print(f"MODE2 : for next_interval in m[1:]: {next_interval}")
        res_inter = ip.intersection(current_interval, next_interval)
        if not (np.isnan(res_inter.a) and np.isnan(res_inter.b)):
            current_interval = union_intervals(current_interval, next_interval)
        else:
            mode_.append(current_interval)
            current_interval = next_interval

    mode_.append(current_interval)

    return mode_


def med_K(X):
    c_inf = [ip.inf(el) for el in X]
    c_sup = [ip.sup(el) for el in X]
    # print("\n\nc_inf: ", c_inf)
    # print("\n\nc_sup: ", c_sup)
    # print("\n\nip.Interval(np.median(c_inf), np.median(c_sup)): ", ip.Interval(np.median(c_inf), np.median(c_sup)))
    # print("np.array([np.median(c_inf), np.median(c_sup)]): ", np.array([np.median(c_inf), np.median(c_sup)]))
    return ip.Interval(np.median(c_inf), np.median(c_sup))


def med_P(X):
    x = sorted(X, key=cmp_to_key(lambda x, y: (x.a + x.b) / 2 - (y.a + y.b) / 2))

    index_med = len(x) // 2

    if len(x) % 2 == 0:
        return (x[index_med - 1] + x[index_med]) / 2

    # print("\n\ntype(x[index_med]): ", type(x[index_med]))

    return x[index_med]


def coefficient_Jakkard(X_data, Y_data = None):
    # print("\n\nX_data: ",  type(X_data))
    # print("\n\nY_data: ", type(Y_data))
    if Y_data is None:
        x_inf = [ip.inf(x) for x in X_data]
        x_sup = [ip.sup(x) for x in X_data]
        return (min(x_sup) - max(x_inf)) / (max(x_sup) - min(x_inf))

    if isinstance(X_data, ip.ClassicalArithmetic) and isinstance(Y_data, ip.ClassicalArithmetic):
        return (min(ip.sup(X_data), ip.sup(Y_data)) - max(ip.inf(X_data), ip.inf(Y_data))) / \
            (max(ip.sup(X_data), ip.sup(Y_data)) - min(ip.inf(X_data), ip.inf(Y_data)))

    jakkard_v = []

    for x, y in zip(X_data, Y_data):
        coeff = (min(ip.sup(x), ip.sup(y)) - max(ip.inf(x), ip.inf(y))) / (max(ip.sup(x), ip.sup(y)) - min(ip.inf(x), ip.inf(y)))
        jakkard_v.append(coeff)

    return jakkard_v


def read_bin_file_with_numpy(file_path):
    with open(file_path, 'rb') as f:
        header_data = f.read(256)
        side, mode_, frame_count = struct.unpack('<BBH', header_data[:4])

        frames = []
        point_dtype = np.dtype('<8H')

        for _ in range(frame_count):
            frame_header_data = f.read(16)
            stop_point, timestamp = struct.unpack('<HL', frame_header_data[:6])
            frame_data = np.frombuffer(f.read(1024 * 16), dtype=point_dtype)
            frames.append(frame_data)
        print("Complete load data")
        return np.array(frames)


def get_avg(data):
    avg = [[0]*8]*1024
    for i in range(len(data)): # 100
        avg = np.add(avg, data[i])
    return np.divide(avg, len(data))


def scalar_to_interval(x, rad):
    return ip.Interval(x - rad, x + rad)


def argmaxF(f, a, b, eps):
    lmbd = a + (3 - 5 ** 0.5) * (b - a)/2
    mu = b - (3 - 5 ** 0.5) * (b - a) / 2
    f_lambda = f(lmbd)
    f_mu = f(mu)
    print(f_mu)
    while 1:
        if f_lambda <= f_mu:
            a = lmbd
            if eps > b - a:
                break
            lmbd = mu
            f_lambda = f_mu
            mu = b - (3 - 5 ** 0.5) * (b - a) / 2
            f_mu = f(mu)
        else:
            b = mu
            if eps > b - a:
                break
            mu = lmbd
            f_mu = f_lambda
            lmbd = a + (3 - 5 ** 0.5) * (b - a)/2
            f_lambda = f(lmbd)

        # print(a)
        # print(b)

    return (a+b) / 2


def func_a(a):
    print(np.mean(coefficient_Jakkard(X+a, Y)))
    return np.mean(coefficient_Jakkard(X + a, Y))


def func_t(t):
    return np.mean(coefficient_Jakkard(X * t, Y))


def func_a_mode(a):
    return np.mean(coefficient_Jakkard(mode(X + a), mode(Y)))


def func_t_mode(t):
    return np.mean(coefficient_Jakkard(mode(X * t), mode(Y)))


def func_a_med_p(a):
    return np.mean(coefficient_Jakkard(med_P(X + a), med_P(Y)))


def func_t_med_p(t):
    return np.mean(coefficient_Jakkard(med_P(X * t), med_P(Y)))


def func_a_med_k(a):
    return np.mean(coefficient_Jakkard(med_K(X + a), med_K(Y)))


def func_t_med_k(t):
    return np.mean(coefficient_Jakkard(med_K(X * t), med_K(Y)))


def draw_func(f, a, b, parametr: str, func=""):
    X_linsp = np.linspace(a, b, 100)
    y = np.array([f(x) for x in X_linsp])
    plt.plot(X_linsp, y, )

    plt.xlabel(f"{parametr}")
    plt.ylabel(f"Ji({parametr}, {func}(X), {func}(Y))")
    plt.title("Jaccard Index")
    plt.savefig(f"Jaccadrd-{parametr}-{func}")
    plt.show()


def draw_func_all(i, f, a, b, parametr: str, func=""):
    colors = ["#EF476F", "#F78C6B", "#FFD166", "#83D483", "#06D6A0", "#0CB0A9", "#118AB2", "#073B4C"]
    X_linsp = np.linspace(a, b, 100)
    y = np.array([f(x) for x in X_linsp])
    plt.plot(X_linsp, y, color=colors[i], label=f"Ji({parametr}, {func}(X), {func}(Y))", alpha=0.7)

    # plt.xlabel(f"{parametr}")
    # plt.ylabel(f"Ji({parametr}, {func}(X), {func}(Y))")
    # plt.title("Jaccard Index")
    # plt.show()
    # plt.savefig(f"Jaccadrd-{parametr}-{func}")


scalar_to_interval_vec = np.vectorize(scalar_to_interval)

x_data = read_bin_file_with_numpy('-0.205_lvl_side_a_fast_data.bin')
y_data = read_bin_file_with_numpy('0.225_lvl_side_a_fast_data.bin')

x_data = get_avg(x_data)
y_data = get_avg(y_data)

# print("len(x_data): ", len(x_data), len(x_data[0]))
# print("len(y_data): ", len(y_data), len(y_data[0]))

x_voltage = x_data / 16384.0 - 0.5
y_voltage = y_data / 16384.0 - 0.5

rad = 2 ** (-14)

X = scalar_to_interval_vec(x_voltage, rad).flatten()
Y = scalar_to_interval_vec(y_voltage, rad).flatten()
print("Convert X and Y")
print("X: ", X)
print("X: ", len(X))
print("X: ", type(X))
print("X: ", type(X[0]))

# # Функционал = Ji(const, X, Y)
#draw_func(func_a, 0, 1, "a")
a_f = argmaxF(func_a, 0, 1, 1e-3)
print(a_f, func_a(a_f))
#draw_func(func_t, -4, 0, "t")
t_f = argmaxF(func_t, -4, 0, 1e-3)
print(t_f, func_t(t_f))

# # Функционал = Ji(const,mode(X), mode(Y))
draw_func(func_a_mode, 0, 1, "a")
#a_f_mode = argmaxF(func_a_mode, 0, 1, 1e-3)
#print(a_f_mode, func_a_mode(a_f_mode))
draw_func(func_t_mode, -4, 0, "t")
#t_f_mode = argmaxF(func_t_mode, -4, 0, 1e-3)
#print(t_f_mode, func_t_mode(t_f_mode))
# # Функционал = Ji(const,med_K(X), med_K(Y))
draw_func(func_a_med_k, 0, 1, "a", "med_K")
# a_f_med_k = argmaxF(func_med_k_a, 0, 1, 1e-3)
# print(a_f_med_k, func_med_k_a(a_f_med_k))
draw_func(func_t_med_k, -4, 0, "t", "med_K")
# t_f_med_k = argmaxF(func_med_k_t, -4, 0, 1e-3)
# print(t_f_med_k, func_med_k_t(t_f_med_k))

# # Функционал = Ji(const,med_р(X), med_р(Y))
draw_func(func_a_med_p, 0, 1, "a", "med_p")
# a_f_med_p = argmaxF(func_med_p_a, 0, 1, 1e-3)
# print(a_f_med_p, func_med_p_a(a_f_med_p))
draw_func(func_t_med_p, -4, 0, "t", "med_p")
# t_f_med_p = argmaxF(func_med_p_t, -4, 0, 1e-3)
# print(t_f_med_p, func_med_p_t(t_f_med_p))

# funcs = [func_a, func_t, func_a_med_k, func_t_med_k, func_a_med_p, func_t_med_p]
# funcs_str = ["", "", "med_k", "med_k", "med_p", "med_p"]
# bounds = [[0, 1], [-4, 0], [0, 1], [-4, 0], [0, 1], [-4, 0]]
# params = ["a", "t", "a", "t", "a", "t"]
#
# for i in range(1, len(funcs)+1, 2):
#     draw_func_all(i, funcs[i], bounds[i][0], bounds[i][1], params[i], funcs_str[i])
# plt.xlabel(f"const")
# plt.ylabel(f"Ji(const, func(X), func(Y))")
# plt.title("Jaccard Index")
# plt.legend()
# plt.savefig(f"Jaccadrd-all-in-one-T")
# plt.show()
#
# for i in range(0, len(funcs), 2):
#     draw_func_all(i, funcs[i], bounds[i][0], bounds[i][1], params[i], funcs_str[i])
# plt.xlabel(f"const")
# plt.ylabel(f"Ji(const, func(X), func(Y))")
# plt.title("Jaccard Index")
# plt.legend()
# plt.savefig(f"Jaccadrd-all-in-one-A")
# plt.show()

