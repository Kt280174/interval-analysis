import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# -----------------------------
# 1. Функции и их производные
# -----------------------------
def f1(x): 
    return x**3 - 3*x**2 + 2

def f2(x): 
    return (x + 1)**3 - np.cos(x)

def f1_derivative(x): 
    return 3*x**2 - 6*x

def f2_derivative(x): 
    return 3*(x+1)**2 + np.sin(x)



# -----------------------------
# 2. Точная область значений
# -----------------------------
def exact_range(func, a, b, num_points=10000):
    xs = np.linspace(a, b, num_points)
    ys = func(xs)
    return np.min(ys), np.max(ys)

ran_f1 = exact_range(f1, 0, 3)
ran_f2 = exact_range(f2, -1, 2)

# -----------------------------
# 3. Интервальные операции
# -----------------------------
def interval_add(I1, I2): 
    return [I1[0] + I2[0], I1[1] + I2[1]]

def interval_mul(I1, I2):
    vals = [I1[0]*I2[0], I1[0]*I2[1], I1[1]*I2[0], I1[1]*I2[1]]
    return [min(vals), max(vals)]

def interval_pow(I, n):
    xs = [I[0]**n, I[1]**n]
    if n % 2 == 0 and I[0] < 0 < I[1]:
        return [0, max(xs)]
    return [min(xs), max(xs)]

# -----------------------------
# 4. Методы интервальных оценок
# -----------------------------
def natural_extension_f1(I):
    # f1(x) = x^3 - 3x^2 + 2
    x3 = interval_pow(I, 3)
    x2 = interval_pow(I, 2)
    three_x2 = interval_mul([-3, -3], x2)
    return interval_add(interval_add(x3, three_x2), [2, 2])

def natural_extension_f2(I):
    # f2(x) = (x+1)^3 - cos(x)
    shift = interval_add(I, [1, 1])
    cubic = interval_pow(shift, 3)
    xs = np.linspace(I[0], I[1], 1000)
    cos_vals = np.cos(xs)
    cosI = [np.min(cos_vals), np.max(cos_vals)]
    neg_cos = [-cosI[1], -cosI[0]]
    return interval_add(cubic, neg_cos)

def horner_f1(I):
    # f1(x) = ((x - 3) * x) * x + 2  (вариант Горнера)
    x_minus_3 = interval_add(I, [-3, -3])   # (x - 3)
    t2 = interval_mul(x_minus_3, I)         # (x - 3)*x
    t3 = interval_mul(t2, I)                # ((x - 3)*x)*x
    return interval_add(t3, [2, 2])         # +2

def differential_centered(func, func_deriv, I, c):
    f_c = func(c)
    dI = [I[0] - c, I[1] - c]
    xs = np.linspace(I[0], I[1], 400)
    df = func_deriv(xs)
    deriv_range = [np.min(df), np.max(df)]
    prod = interval_mul(deriv_range, dI)
    return interval_add([f_c, f_c], prod)

def slope_centered(func, I, c):
    f_c = func(c)
    dI = [I[0] - c, I[1] - c]
    xs = np.linspace(I[0], I[1], 400)
    slopes = (func(xs) - f_c) / (xs - c + 1e-12)
    slope_interval = [np.min(slopes), np.max(slopes)]
    prod = interval_mul(slope_interval, dI)
    return interval_add([f_c, f_c], prod)

def interval_mid(I): 
    return (I[0] + I[1]) / 2

def interval_rad(I): 
    return (I[1] - I[0]) / 2

def cut(val, lo=-1, hi=1):
    """Ограничение значения в интервал [lo, hi]"""
    return max(lo, min(hi, val))

def interval_intersection(I1, I2):
    """Пересечение двух интервалов"""
    left = max(I1[0], I2[0])
    right = min(I1[1], I2[1])
    if left <= right:
        return [left, right]
    else:
        return None

def bicentered(func, func_deriv, I):
    # mid и rad интервала X
    mid_x = interval_mid(I)
    rad_x = interval_rad(I)

    # оценка производной на интервале
    xs = np.linspace(I[0], I[1], 1000)
    df_vals = func_deriv(xs)
    mid_df = (np.max(df_vals) + np.min(df_vals)) / 2
    rad_df = (np.max(df_vals) - np.min(df_vals)) / 2

    # коэффициент p по Бауману
    if rad_df == 0:
        p = 0
    else:
        p = cut(mid_df / rad_df)

    # два центра по Бауману
    c_star = mid_x - p * rad_x
    c_star2 = mid_x + p * rad_x

    # считаем интервалы через дифференциальную центрированную форму
    F1 = differential_centered(func, func_deriv, I, c_star)
    F2 = differential_centered(func, func_deriv, I, c_star2)

    # возвращаем пересечение
    return interval_intersection(F1, F2)
# -----------------------------
# 5. Липшиц и сравнение
# -----------------------------
def lipschitz_constant(func_deriv, I):
    xs = np.linspace(I[0], I[1], 1000)
    vals = np.abs(func_deriv(xs))
    return np.max(vals)

def interval_radius(I): return (I[1] - I[0]) / 2

def compare_with_lipschitz(I_est, ran_exact, L, X):
    rad_est = interval_radius(I_est)
    rad_true = interval_radius(ran_exact)
    error = rad_est - rad_true
    bound = L * interval_radius(X)
    return [I_est, rad_est, rad_true, error, bound]

# -----------------------------
# 6. Применение методов
# -----------------------------
F1_nat = natural_extension_f1([0,3])
F1_horner = horner_f1([0, 3])
F1_diff_center = differential_centered(f1, f1_derivative, [0,3], c=0)
F1_slope_center = slope_centered(f1, [0,3], c=0)
F1_bicenter = bicentered(f1, f1_derivative, [0, 3])

F2_nat = natural_extension_f2([-1,2])
F2_diff_center = differential_centered(f2, f2_derivative, [-1,2], c=1.91)
F2_slope_center = slope_centered(f2, [-1,2], c=0)
F2_bicenter = bicentered(f2, f2_derivative, [-1, 2])

L1 = lipschitz_constant(f1_derivative, [0, 3])
L2 = lipschitz_constant(f2_derivative, [-1, 2])

# -----------------------------
# 7. Таблицы результатов
# -----------------------------
data_f1 = []
for name, est in [("Естественное", F1_nat),
                  ("Горнер", F1_horner),
                  ("Дифф. центр", F1_diff_center),
                  ("Наклонная центр", F1_slope_center),
                  ("Бицентрированная", F1_bicenter)]:
    row = compare_with_lipschitz(est, ran_f1, L1, [0, 3])
    data_f1.append([name] + row)

df_f1 = pd.DataFrame(data_f1,
                     columns=["Метод", "Интервал", "rad_est", "rad_true", "Ошибка", "Оценка Липшица"])

data_f2 = []
for name, est in [("Естественное", F2_nat),
                  ("Дифф. центр", F2_diff_center),
                  ("Наклонная центр", F2_slope_center),
                  ("Бицентрированная", F2_bicenter)]:
    row = compare_with_lipschitz(est, ran_f2, L2, [-1, 2])
    data_f2.append([name] + row)

df_f2 = pd.DataFrame(data_f2,
                     columns=["Метод", "Интервал", "rad_est", "rad_true", "Ошибка", "Оценка Липшица"])

print("\n=== f1(x) = x^3 - 3x^2 + 2 ===")
print(df_f1)

print("\n=== f2(x) = (x+1)^3 - cos(x) ===")
print(df_f2)

# -----------------------------
# 8. Графики
# -----------------------------
xs = np.linspace(0, 3, 400)
plt.plot(xs, f1(xs), label="f1(x)")
xs = np.linspace(-1, 2, 400)
plt.plot(xs, f2(xs), label="f2(x)")
plt.legend()
plt.title("Графики функций f1 и f2 ")
plt.grid(True)
plt.show()

