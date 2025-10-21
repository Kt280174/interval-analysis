import numpy as np

def tolsolvty(infA, supA, infb, supb,
              iprn=0, epsf=1e-6, epsx=1e-6, epsg=1e-6, maxitn=2000):
    """
    Реализация функции tolsolvty для проверки разрешимости интервальной
    системы линейных уравнений A x = b с помощью распознающего функционала Tol.
    Переписано с Scilab/MATLAB версии Шарого (S.P. Shary).
    """

    # ---------------- Проверка корректности размеров ----------------
    m_inf, n_inf = infA.shape
    m_sup, n_sup = supA.shape
    if m_inf != m_sup:
        raise ValueError("Количество строк в infA и supA различно")
    if n_inf != n_sup:
        raise ValueError("Количество столбцов в infA и supA различно")
    m = m_inf
    n = n_inf

    k_inf = infb.shape[0]
    k_sup = supb.shape[0]
    if k_inf != k_sup:
        raise ValueError("Количество элементов в infb и supb различно")
    k = k_inf
    if k != m:
        raise ValueError("Размеры матрицы A не соответствуют вектору b")

    # ---------------- Параметры алгоритма ----------------
    nsims = 30      # число одинаковых шагов
    alpha = 2.3     # коэффициент растяжения пространства
    hs = 1.0        # начальный шаг одномерного поиска
    nh = 3          # число одинаковых шагов перед увеличением шага
    q1, q2 = 0.9, 1.1
    w = 1.0 / alpha - 1.0

    # ---------------- Средняя точечная система ----------------
    Ac = 0.5 * (infA + supA)
    Ar = 0.5 * (supA - infA)
    bc = 0.5 * (infb + supb)
    br = 0.5 * (supb - infb)

    sv = np.linalg.svd(Ac, compute_uv=False)
    minsv, maxsv = np.min(sv), np.max(sv)
    if minsv != 0 and maxsv / minsv < 1e12:
        x = np.linalg.lstsq(Ac, bc, rcond=None)[0]
    else:
        x = np.zeros((n, 1))

    # ---------------- Локальная функция вычисления Tol(x) ----------------
    def calcfg(x):
        absx = np.abs(x)
        Ac_x = Ac @ x
        Ar_absx = Ar @ absx
        infs = bc - (Ac_x + Ar_absx)
        sups = bc - (Ac_x - Ar_absx)
        tt = br - np.maximum(np.abs(infs), np.abs(sups))
        f = np.min(tt)
        mc = np.argmin(tt)

        infA_mc = infA[mc, :].reshape(-1, 1)
        supA_mc = supA[mc, :].reshape(-1, 1)
        x_neg = (x < 0).astype(float)
        x_nonneg = (x >= 0).astype(float)
        dl = infA_mc * x_neg + supA_mc * x_nonneg
        ds = supA_mc * x_neg + infA_mc * x_nonneg
        if -infs[mc] <= sups[mc]:
            g = ds
        else:
            g = -dl
        return f, g, tt

    # ---------------- Инициализация ----------------
    B = np.eye(n)
    vf = np.inf * np.ones((nsims, 1))
    lp = iprn

    f, g0, tt = calcfg(x)
    ff, xx = f, x.copy()
    cal, ncals = 1, 1

    if iprn > 0:
        print("\nПротокол максимизации распознающего функционала Tol")
        print("-" * 60)
        print("Шаг\tTol(x)\t\tTol(xx)\tВычФун/шаг\tВычФун")
        print("-" * 60)
        print(f"{0:3d}\t{f:10.6f}\t{ff:10.6f}\t{cal:5d}\t{ncals:5d}")

    # ---------------- Основной цикл ----------------
    for itn in range(1, maxitn + 1):
        pf = ff

        # критерий останова по норме суперградиента
        if np.linalg.norm(g0) < epsg:
            ccode = 2
            break

        g1 = B.T @ g0
        g = B @ g1 / np.linalg.norm(g1)
        normg = np.linalg.norm(g)

        r = 1.0
        cal = 0
        deltax = 0.0
        while r > 0.0 and cal <= 500:
            cal += 1
            x = x + hs * g
            deltax += hs * normg
            f, g1, tt = calcfg(x)
            if f > ff:
                ff = f
                xx = x.copy()
            if cal % nh == 0:
                hs *= q2
            r = float(g.T @ g1)

        if cal > 500:
            ccode = 5
            break
        if cal == 1:
            hs *= q1

        ncals += cal
        if itn == lp and iprn > 0:
            print(f"{itn:3d}\t{f:10.6f}\t{ff:10.6f}\t{cal:5d}\t{ncals:5d}")
            lp += iprn

        if deltax < epsx:
            ccode = 3
            break

        dg = B.T @ (g1 - g0)
        xi = dg / np.linalg.norm(dg)
        B = B + w * (B @ xi) @ xi.T
        g0 = g1

        vf[1:] = vf[:-1]
        vf[0] = abs(ff - pf)
        if abs(ff) > 1:
            deltaf = np.sum(vf) / abs(ff)
        else:
            deltaf = np.sum(vf)
        if deltaf < epsf:
            ccode = 1
            break
        ccode = 4

    # ---------------- Результаты ----------------
    tolmax = ff
    argmax = xx
    tt = np.column_stack((np.arange(1, m + 1), tt.flatten()))
    ind = np.argsort(tt[:, 1])
    envs = tt[ind, :]

    if iprn > 0:
        if itn % iprn != 0:
            print(f"{itn:3d}\t{f:10.6f}\t{ff:10.6f}\t{cal:5d}\t{ncals:5d}")
        print("-" * 60)

    if tolmax >= 0:
        print("\n✅ Интервальная задача о допусках РАЗРЕШИМА")
    else:
        print("\n❌ Интервальная задача о допусках НЕ имеет решений")

    if tolmax < 0 and abs(tolmax / epsf) < 10:
        print("⚠️  Вычисленный максимум находится в пределах точности.")
        print("    Уменьшите epsf/epsx для получения более точной информации.")

    return tolmax, argmax, envs, ccode
if __name__ == "__main__":
    A_2 = np.array([
        [[0.65, 1.25], [0.70, 1.3]],
        [[0.75, 1.35], [0.70, 1.3]]
    ])
    b_2 = np.array([
        [[2.75, 3.15]],
        [[2.85, 3.25]]
    ])

    infA = A_2[:, :, 0]
    supA = A_2[:, :, 1]
    infb = b_2[:, :, 0]
    supb = b_2[:, :, 1]

    tolmax, argmax, envs, ccode = tolsolvty(infA, supA, infb, supb, iprn=1)
    print("\nTolmax =", tolmax)
    print("Argmax =", argmax)
    print("Env =", envs)
    print("Code =", ccode)
