import numpy as np
import itertools    # care


def dist_m(x, y):
    """
    Матрица расстояний между всеми точками матриц x и y
    """
    d = ((x**2).sum(axis=1) * np.ones((x.shape[0],1))).T + \
    ((y**2).sum(axis=1) * np.ones((y.shape[0],1))) - 2 * np.dot(x,y.T)
    return np.sqrt(d)


def chain_dist(x):
    """
    Алгоритм цепных расстояний
    x - матрица данных
    """
    n = np.shape(x)[0]
    d = dist_m(x, x)
    i = np.arange(n - 1)
    cd = d[i, i + 1]
    return cd


def CKO(x):
    """
    Критерий суммы квадратов отклонений
    x -матрица данных; для 2-х кластеров. Только упорядоченная выборка!
    """
    n = np.shape(x)[0]
    return [np.sum((x[:i,] - ((x[:i,].sum(0)) / i))**2) + \
            np.sum((x[i:,] - ((x[i:,].sum(0)) / (n - i)))**2) for i in range(1, n)]


def transformation(x, c):
    """
    Разделение матрицы данных х на 2 матрицы, соответствуюшие одному из 2 классов
    """
    N = x.shape[1]                             # число столбцов (переменных)
    a = x[np.where(c.ravel() == 0), :]         # 3-мерный массив!?
    a.resize(np.squeeze(a).shape)
    b = x[np.where(c.ravel() == 1), :]
    b.resize(np.squeeze(b).shape)
    return a, b


def Fisher(x, c):
    """
    Одномерный критерий Фишера
    x-матрица данных, с-вектор классов
    """
    a, b = transformation(x, c)
    m1, m2 = a.mean(axis=0), b.mean(axis=0)
    s1 = ((a - m1)**2).sum(axis=0)
    s2 = ((b - m2)**2).sum(axis=0)
    return ((m1 - m2)**2)/(s1 + s2)


def Fisher_m(x, c):
    """
    Многомерный критерий Фишера
    x-матрица данных, с-вектор классов
    """
    a, b = transformation(x, c)
    m1, m2 = a.mean(axis=0), b.mean(axis=0)      # средние значения по столбцам матриц а и b
    s1 = np.dot(np.transpose(a - m1), (a - m1))  # сумма квадратов отклонений. Проверить формулу в pdf
    s2 = np.dot(np.transpose(b - m2), (b - m2))
    f = np.dot((m2 - m1),np.linalg.inv(s1 + s2))
    return f, a, b


def Forel(x, r):                                    
    """
    FOREL
    x - матрица данных, r - радиус кластера
    """
    new_c = x[0,:] # !!!
    mm = []
    while len(x) != 0:
        ms0 = new_c
        d = np.sqrt(np.sum((x - ms0)**2, axis=1))
        m0 = x[d <= r]                                # точки i кластера
        new_c = np.sum(m0, axis=0) / np.shape(m0)[0]  # центр кластера
        if (ms0 == new_c).all():                      # and (len(y[d <= r]) != len(y))
            x = np.delete(x, np.where(d <= r), 0)     # проверить форму массива
            mm.append(new_c)
            if len(x) != 0:
                new_c = x[0,:]
    return mm


def k_mean(x, k = 2):
    """
    Метод k-средних
    k - кол-во кластеров
    """
    n = np.shape(x)[0]
    m = np.shape(x)[1]
    im = np.random.permutation(n)[:k]                         # k случайных индексов точек
    ms = x[im].reshape((k,1,m))
    r = np.dot(((x - ms)**2), np.ones(m))                     # расстояния
    ms0 = ms + 1.
    while (ms != ms0).all():
        ms = ms0
        ms0 = []
        for i in range(k):
            m0 = x[np.all(r[i,:] <= r, 0)]                    # точки i кластера
            ms0.append(np.sum(m0, axis=0) / np.shape(m0)[0])  # нужно делить на кол-во точек
        ms0 = np.reshape(ms0, (k,1,m))
        r = np.dot(((x - ms0)**2), np.ones(m))
    return ms0


def classification(x, ms, k = 2):
    """
    Классификация точек по наименьшему расстоянию
    ms - матрица центров кластеров (3 мерная)
    """
    m = np.shape(x)[1]
    dist = np.dot(((x - ms)**2), np.ones(m))
    return [x[np.all(dist[i,:] <= dist, 0)] for i in range(k)]


def cor(x):
    """
    Корелляционная матрица
    """
    y = x - x.mean(axis=0)
    d = np.diag(1 / x.std(axis=0))
    y = np.dot(y, d)
    return np.dot(np.transpose(y), y) / len(y)
 

def cov(x):
    """
    Ковариационная матрица
    """
    y = x - x.mean(axis=0)
    return np.dot(np.transpose(y), y) / (len(y) - 1)


def Selfic(x, f=cov):
    """
    Метод главных компонент
    x - матрица данных
    f - функция вычисления симметричной матрицы (корреляционной или ковариационной)
    Вычисление собственных чисел, векторов и проекций на эти вектора
    """
    num, vec = np.linalg.eig(f(x))                         # собсвтенные ч. и в.
    sort_ind = np.argsort(-num)                            # сортировка по убыванию
    pro = np.dot((x - x.mean(axis=0)), vec)                # центрирование
    return num[sort_ind], vec[:,sort_ind], pro[:,sort_ind]


def dist(x):
    """
    Вычисление расстояний между каждой парой точек. x-матрица данных
    """
    ln = xrange(len(x))    
    comb = np.array(list(itertools.combinations(ln, 2)))
    a = x[comb[:,0]]
    b = x[comb[:,1]]
    d = np.sqrt(((a - b)**2).sum(axis=1))
    return d, comb


def orl1(x):
    """
    Линейное шкалирование Орлочи
    Проецирование на один вектор
    x - матрица данных
    """
    d, cm = dist(x)
    ij = cm[np.where(max(d) <= d)].ravel()
    xv = x - x[ij[0],:]
    return np.dot(xv, xv[ij[1],:])


def orl2(x):
    """
    Проецирование на плоскость
    """
    d, cm = dist(x)
    ij = cm[np.where(max(d) <= d)].ravel()
    xv = x - x[ij[0],:]
    a = ((xv*xv[ij[1],:]).sum(axis=1))/(xv[ij[1],:]**2).sum()
    a = a.reshape((len(a),1))
    pv = ((xv - a*xv[ij[1],:])**2).sum(axis=1)                   # вектора, перпендикулярные первой оси
    amax = a[np.where(max(pv) <= pv)].ravel()
    xv2 = xv - amax*xv[ij[1],:]
    p1 = (xv2*xv2[ij[1],:]).sum(axis=1)
    p2 = (xv2*xv2[np.where(max(pv) <= pv)].ravel()).sum(axis=1)
    return p1, p2


def r2p(x, c):
    """
    Расстояние от двух фиксированных точек 
    x-матрица данных, с-вектор классов
    """
    a, b, N = transformation(x, c)               # матрицы каждого класса. VSPILT() 
    m1, m2 = a.mean(axis=0), b.mean(axis=0)      # средние значения по столбцам матриц а и b
    r1 = np.sqrt(((x - m1)**2).sum(axis=1))      # scipy.std() 
    r2 = np.sqrt(((x - m2)**2).sum(axis=1))
    return r1, r2


def sammon(x, n = 2, dist_func=dist_m, it_max = 500, hl_max = 20, acr = 1e-4):  # добавлен аргумент dist_func который определяет функцию вычисления расстояний
    """
    x - матрица точек; n - число измерений
    """
    N = x.shape[0]
    y = np.random.normal(0.0, 1.0, (N,n))        # чувствительность к начальным координатам
    k = 0.5 / np.sum(dist_func(x, x))
    mx = dist_func(x, x) + np.eye(N)
    mx[mx == 0.] = 1.
    my = dist_func(y, y) + np.eye(N)
    E0 = ((1.0 / mx) * (mx - my)**2).sum()
    nn = 0
    N_ones = np.ones((N,1))
    for i in range(it_max):
        a1 = N_ones * y.ravel(order="F")         #y.ravel(1)
        a0 = N_ones * y.ravel()
        am = a1.T - (a0).reshape((N * n,N), order="F")
        lm = am.reshape((n,N,N))
        d = (1.0 / mx) - (1.0 / my)
        gg = np.dot(lm, d.T)
        gg = gg * np.diag(np.ones(N))
        g = (np.dot(gg, np.ones(N))).ravel(order="F")
        hh = (1.0 / mx) - ((1.0 / my) * (1.0 - (lm**2).reshape((n,N,N)) / my**2)) 
        h = (hh.sum(axis=1)).ravel(order="F")
        s = (g / h).reshape((N,n))
        y0 = y
        for j in range(hl_max): # better 10?
            y = y0 - s
            my = dist_func(y, y) + np.eye(N)
            E = ((1.0 / mx) * (mx - my)**2).sum()
            if E < E0:
                break
            else:
                nn += 1
                s = 0.5 * s
        if abs(E - E0 ) < acr:
            print("Optimization completed")
            break
        else:
            nn += 1
            E0 = E
    print(k * E)
    return y, k * E, nn
