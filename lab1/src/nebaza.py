import numpy
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
import random

x_nodes = np.array([0, 1, 2], dtype=np.float)
y_nodes = np.array([0, 1, 2], dtype=np.float)

i = 0
x = 1.5


def l_i(i, x, x_nodes):
    result = 1
    for y in range(0, x_nodes.size):
        if y != i:
                result = result * (x - x_nodes[y]) / (x_nodes[i] - x_nodes[y])
    return result


def L(x, x_nodes, y_nodes):
    result = 0
    for i in range(0, x_nodes.size):
        result = result + y_nodes[i]*l_i(i, x, x_nodes)
    return result


def t_3_a():
    x_nodes = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float)
    y_nodes = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)
    result = np.empty((1000, 11), dtype=np.float64)

    for i in range(0, 1000):
        for y in range(0, 11):
            result[i][y] = x_nodes[y] + random.gauss(0, 0.01)

    np.set_printoptions(threshold=100000000, suppress=True, linewidth= 200)
    return result




def t_3_b():
    x_nodes = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float)
    y_nodes = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)
    result = t_3_a()

    fig, ax = plt.subplots()  # создаём рисунок fig и 1 график на нём ax, plt.subplots(1,1,1)
    ax.set(xlabel='x', ylabel='h(x)')
    ax.grid()
    y_dots = np.zeros((1000, 1001))
    x_dots = np.arange(0.0, 1.001, 0.001)

    for i in range(0, 1000):
        for x in range(0, 1001):
           y_dots[i][x] = L(x_dots[x], result[i], y_nodes)
        ax.plot(x_dots, y_dots[i], '-', label='L(x)', zorder=1)
    ax.scatter(x_nodes, y_nodes, marker='o', zorder=2)
    plt.show()  # вывод рисунка
    return y_dots


def t_3_c_d(y_dots):
    # 0.1*1000/2 = 50
    x_dots = np.arange(0.0, 1.001, 0.001)
    result = np.sort(y_dots, axis=0)
    print('буду рисовать')
    h_l = result[49]
    h_u = result[949]
    h_mean = np.mean(result, axis=0)

    fig, ax = plt.subplots()  # создаём рисунок fig и 1 график на нём ax, plt.subplots(1,1,1)
    ax.set(xlabel='x', ylabel='h(x)')
    ax.grid()
    ax.plot(x_dots, h_l, '-', label='h_l')
    ax.plot(x_dots, h_u, '-',  label='h_u')
    ax.plot(x_dots, h_mean, '-', label='h_mean')
    ax.legend()
    plt.show()
    print('нарисовав')


def t_4_a():
    x_nodes = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float)
    y_nodes = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)
    result = np.empty((1000, 11), dtype=np.float64)

    for i in range(0, 1000):
        for y in range(0, 11):
            result[i][y] = y_nodes[y] + random.gauss(0, 0.01)
    np.set_printoptions(threshold=100000000, suppress=True, linewidth= 200)

    return result


def t_4_b():
    x_nodes = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float)
    y_nodes = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)
    result = t_4_a()

    fig, ax = plt.subplots()  # создаём рисунок fig и 1 график на нём ax, plt.subplots(1,1,1)
    ax.set(xlabel='x', ylabel='h(x)')
    ax.grid()
    y_dots = np.zeros((1000, 1001))
    x_dots = np.arange(0.0, 1.001, 0.001)
    for i in range(0, 1000):
        for x in range(0, 1001):
           y_dots[i][x] = L(x_dots[x], x_nodes, result[i])
        ax.plot(x_dots, y_dots[i], '-', zorder=1)
    ax.scatter(x_nodes, y_nodes, marker='o', zorder=2)
    plt.show()  # вывод рисунка
    return y_dots


def t_4_c_d(y_dots):
    # 0.1*1000/2 = 50
    x_dots = np.arange(0.0, 1.001, 0.001)
    result = np.sort(y_dots, axis=0)
    print('буду рисовать')

    h_l = result[49]
    h_u = result[949]
    h_mean = np.mean(result, axis=0)

    fig, ax = plt.subplots()  # создаём рисунок fig и 1 график на нём ax, plt.subplots(1,1,1)
    ax.set(xlabel='x', ylabel='h(x)')
    ax.grid()
    ax.plot(x_dots, h_l, '-', label='h_l')
    ax.plot(x_dots, h_u, '-', label='h_u')
    ax.plot(x_dots, h_mean, '-', label='h_mean')
    ax.legend()
    plt.show()
    print('нарисовав')

print(l_i(i, x, x_nodes))
print(L(x, x_nodes, y_nodes))
print(t_3_a())
t_3_c_d(t_3_b())
t_4_c_d(t_4_b())

