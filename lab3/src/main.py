import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve, root

# Внешний ток от всех нейронов
I = 5


# Параметры динамической системы
#      TS    PS    C     FS
a = [ 0.02, 0.02, 0.02, 0.1]
b = [  0.2, 0.25,  0.2, 0.2]
c = [  -65,  -65,  -50, -65]
d = [    6,    6,    2,   2]


# Список названий режимов
names = ['Tonic spiking', 'Phasic spiking', 'Chattering', 'Fast spiking']


# Костыль. Индекс для обращения к параметрам
index = 0


#  Первая функция системы
def f_1(v, u):
    # if v >= 30:
    #     v = c[index]
    #     u += d[index]
    return 0.04 * v**2 + 5 * v + 140 - u + I


# Вторая функция системы
def f_2(v, u):
    # if v >= 30:
    #     v = c[index]
    #     u += d[index]
    return a[index] * (b[index]*v - u)


# Пока непонятно. Изучать.
f = [f_1, f_2]


# Дополнительное условие. Импульс
def imp_cond(v, u):
    if v >= 30:
        v = c[index]
        u += d[index]
    return v, u


# Метод Эйлера для численного решения задачи Коши
def euler(h, f, t_n, x_0):
    m = int(t_n / h)
    v = np.zeros((m + 1,))
    u = np.zeros((m + 1,))
    t = np.linspace(0, t_n, m+1)
    v[0] = x_0[0]
    u[0] = x_0[1]
    for i in range(m):
        v[i + 1] = v[i] + h * f[0](v[i], u[i])
        u[i + 1] = u[i] + h * f[1](v[i], u[i])
        if v[i+1] >= 30:
            v[i+1] = c[index]
            u[i+1] += d[index]
    return t, v, u


# Метод Рунге-Кутты 4-го порядка численного решения задачи Коши
def runge_kutta(h, f, t_n, x_0):
    m = int(t_n / h)
    y_1 = np.zeros((m + 1,))
    y_2 = np.zeros((m + 1,))
    t = np.linspace(0, t_n, m+1)

    y_1[0] = x_0[0]
    y_2[0] = x_0[1]

    for i in range(m):
        k_1_1 = f[0](y_1[i], y_2[i])
        k_1_2 = f[1](y_1[i], y_2[i])

        k_2_1 = f[0](y_1[i] + h*k_1_1/2, y_2[i] + h/2,)
        k_2_2 = f[1](y_1[i] + h/2, y_2[i] + h*k_1_2/2)

        k_3_1 = f[0](y_1[i] + h*k_2_1/2, y_2[i] + h/2,)
        k_3_2 = f[1](y_1[i] + h/2, y_2[i] + h*k_2_2/2)

        k_4_1 = f[0](y_1[i] + h*k_3_1/2, y_2[i] + h/2,)
        k_4_2 = f[1](y_1[i] + h/2, y_2[i] + h*k_3_2/2)

        y_1[i + 1] = y_1[i] + h*(k_1_1 + 2*k_2_1 + 2*k_3_1 + k_4_1) / 6
        y_2[i + 1] = y_2[i] + h*(k_1_2 + 2*k_2_2 + 2*k_3_2 + k_4_2) / 6

        if y_1[i+1] >= 30:
            y_1[i+1] = c[index]
            y_2[i+1] += d[index]
    return t, y_1, y_2


# Неявный метод Эйлера
def implicit_euler(h, f, t_n, x_0):
    m = int(t_n / h)
    y_1 = np.zeros((m + 1,))
    y_2 = np.zeros((m + 1,))
    t = np.linspace(0, t_n, m+1)
    y_1[0] = x_0[0]
    y_2[0] = x_0[1]

    def phi_1(v1, v0, u0):
        return v1 - h*f[0](v1, u0) - v0

    def phi_2(u1, v0, u0):
        return u1 - h*f[1](v0, u1) - u0

    for i in range(0, m):
        y_1[i + 1] = fsolve(phi_1, y_1[i], args=(y_1[i], y_2[i]))
        y_2[i + 1] = fsolve(phi_2, y_2[i], args=(y_1[i], y_2[i]))

        if y_1[i+1] >= 30:
            y_1[i+1] = c[index]
            y_2[i+1] += d[index]
    return t, y_1, y_2


if __name__ == '__main__':
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    for i in range(0, 4):
        index = i
        x_0 = [c[index], b[index] * c[index]]
        t, y_1, y_2 = euler(0.5, f, 300, x_0)
        ax[i].plot(t, y_1, label=fr' $Эйлер$')
        t, y_1, y_2 = implicit_euler(0.1, f, 300, x_0)
        ax[i].plot(t, y_1, label=fr'$неявный$ $Эйлер$')
        t, y_1, y_2 = runge_kutta(0.5, f, 300, x_0)
        ax[i].plot(t, y_1, label=fr' $Рунге-Кутта$')
        ax[i].set_xlabel(r'$t$')
        ax[i].set_ylabel(r'$v$')
        ax[i].grid()
        ax[i].set_title(names[i])
        ax[i].legend()
    fig.tight_layout()
    plt.show()
