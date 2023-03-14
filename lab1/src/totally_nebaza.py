import numpy
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
import random

x_nodes = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float)
y_nodes = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)


def qubic_spline_coeff(x_nodes, y_nodes):
    # план:
    # 1) найти все h и a_{i+1} - a_i
    # 2) заполнить матрицу b
    # 3) заполнить последовательно три диагонали матрицы А: -1, 0, 1
    # 4) создать для каждой диагонали матрицу и сложить полученные матрицы
    # 5) инвертировать А и перемножить на b => c
    # 6) из с найти b и  d. обрезать лишнее и конкатенировать
    n = x_nodes.size  # колличетсво точек
    h_mas = np.zeros(n - 1, dtype=np.float)
    for i in range(0, n - 1):
        h_mas[i] = x_nodes[i + 1] - x_nodes[i]
    # найдены все необходимые h
    a_div_mas = np.zeros(n - 1, dtype=np.float)  # a_{i+1} - a_i
    for i in range(0, n - 1):
        a_div_mas[i] = y_nodes[i + 1] - y_nodes[i]
    # найдена разница каждой пары коэф а
    b_matrix = np.zeros(n, dtype=np.float)  # инициализация b 0
    for i in range(1, n - 1):
        b_matrix[i] = 3 / h_mas[i] * a_div_mas[i] - 3 / h_mas[i - 1] * a_div_mas[i - 1]  # заполненик мат b
    # заполнена матрица b
    diagonal_0 = np.ones(n, dtype=np.float)  # инициализация 0 1
    for i in range(1, n - 1):
        diagonal_0[i] = 2 * (h_mas[i] + h_mas[i - 1])  # 0[i] = 2 *(h_i + h_{i-1})
    # заполнена главная диагональ
    diagonal_1 = np.copy(h_mas)  # инициализация +1 по H
    diagonal_1[0] = 0  # +1[0] = 0
    d1_m = np.diag(diagonal_1, 1)  # формирование мат +1
    # заполнена верхняя неосновная диагональ
    diagonal_2 = np.copy(h_mas)  # инициализация -1 по H
    diagonal_2[h_mas.size - 1] = 0  # -1[n-1] = 0
    d1_m = d1_m + np.diag(diagonal_2, -1)  # сложение -1 +1
    # заполнена нижняя неосновная диагональ
    a_matrix = np.diag(diagonal_0, 0)  # формирование мат с главной диаг
    a_matrix = a_matrix + d1_m  # сложение  матр. с главной диаг и +1 -1
    # заполнена матрица А
    # тогда с = А^(-1)b
    a_matrix_inv = linalg.inv(a_matrix)  # инвертирование A
    c_cof = np.dot(a_matrix_inv, b_matrix)  # перемножение матриц
    # решение матричного уравнения найдено
    d_cof = np.zeros(n - 1, dtype=np.float)  # инициализация d[]
    for i in range(0, n - 1):
        d_cof[i] = (c_cof[i + 1] - c_cof[i]) / (3 * h_mas[i])  # вычисление d_i
    b_cof = np.zeros(n - 1, dtype=np.float)  # инициализация b[]
    for i in range(0, n - 1):
        b_cof[i] = a_div_mas[i] / h_mas[i] - h_mas[i] * (c_cof[i + 1] + 2 * c_cof[i]) / 3  # вычисление b_i+1
    numpy.set_printoptions(precision=5, suppress=True, linewidth=200)  # форматирвание вывода
    c_cof = np.delete(c_cof, n - 1)
    a_cof = np.delete(y_nodes, n - 1)
    result = numpy.r_['1,1,0', [a_cof, b_cof, c_cof, d_cof]]  # конкатенация по строкам
    #   print(result)  # печать всех трёх кф
    return result


def qubic_spline(x, qs_coef, x_nodes: np.ndarray):
    # план : 1) определить индекс промежутка, которому принадлежит x
    #        2) подставить x  в полином с соотвествующим индексом
    index = -1  # индекс промежутка
    if x < x_nodes[0]:
        x_nodes[0] = x
    if x < x_nodes[0]:
        x_nodes[x_nodes.size-1] = x
    for i in range(0, x_nodes.size - 1):
        if x_nodes[i + 1] > x >= x_nodes[i]:
            index = i
    if index == -1:
        index = x_nodes.size - 2  # проверка последнего полинома
    #  print(index)
    #  индекс определён. исключения по промежуткам обработаны
    result = qs_coef[0, index] + qs_coef[1, index] * (x - x_nodes[index]) + qs_coef[2, index] * (
            x - x_nodes[index]) ** 2 + qs_coef[3, index] * (x - x_nodes[index]) ** 3
    #   print(result)
    return result


def t_5_3_a():
    x_nodes = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float)
    y_nodes = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)
    result = np.empty((1000, 11), dtype=np.float64)

    for i in range(0, 1000):
        for y in range(0, 11):
            result[i][y] = x_nodes[y] + random.gauss(0, 0.01)

    np.set_printoptions(threshold=100000000, suppress=True, linewidth=200)
    return result


def t_5_3_b():
    y_nodes = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)
    x_nodes = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float)
    result = t_5_3_a()
    fig, ax = plt.subplots()  # создаём рисунок fig и 1 график на нём ax, plt.subplots(1,1,1)
    ax.set(xlabel='x', ylabel='h(x)')
    ax.grid()
    y_dots = np.zeros((1000, 101))
    x_dots = np.arange(0.0, 1.01, 0.01)
    for i in range(0, 1000):
        for x in range(0, 101):
            y_dots[i][x] = qubic_spline(x_dots[x], qubic_spline_coeff(result[i], y_nodes), result[i])
        ax.plot(x_dots, y_dots[i], '-', zorder=1)
    ax.scatter(x_nodes, y_nodes, marker='o', zorder=2)
    plt.show()  # вывод рисунка
    return y_dots


def t_5_3_c_d(y_dots):
    # 0.1*1000/2 = 50
    x_dots = np.arange(0.0, 1.01, 0.01)
    result = np.sort(y_dots, axis=0)
    print('буду рисовать')
    h_l = result[49]
    h_u = result[949]
    h_mean = np.mean(result, axis=0)
    fig, ax = plt.subplots()  # создаём рисунок fig и 1 график на нём ax, plt.subplots(1,1,1)
    ax.set(xlabel='x', ylabel='h(x)')
    ax.grid()
    ax.plot(x_dots, h_l, '-')
    ax.plot(x_dots, h_u, '-')
    ax.plot(x_dots, h_mean, '-')
    plt.show()
    print('нарисовав')


def t_5_4_a():
    x_nodes = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float)
    y_nodes = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)
    result = np.empty((1000, 11), dtype=np.float64)
    for i in range(0, 1000):
        for y in range(0, 11):
            result[i][y] = y_nodes[y] + random.gauss(0, 0.01)
    np.set_printoptions(threshold=100000000, suppress=True, linewidth=200)
    return result


def t_5_4_b():
    x_nodes = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float)
    y_nodes = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)
    result = t_5_4_a()
    fig, ax = plt.subplots()  # создаём рисунок fig и 1 график на нём ax, plt.subplots(1,1,1)
    ax.set(xlabel='x', ylabel='h(x)')
    ax.grid()
    y_dots = np.zeros((1000, 101))
    x_dots = np.arange(0.0, 1.01, 0.01)
    for i in range(0, 1000):
        for x in range(0, 101):
            y_dots[i][x] = qubic_spline(x_dots[x], qubic_spline_coeff(x_nodes, result[i]), x_nodes)
        ax.plot(x_dots, y_dots[i], '-', zorder=1)
    ax.scatter(x_nodes, y_nodes, marker='o', zorder=2)
    plt.show()  # вывод рисунка
    return y_dots


def t_5_4_c_d(y_dots):
    # 0.1*1000/2 = 50
    x_dots = np.arange(0.0, 1.01, 0.01)
    result = np.sort(y_dots, axis=0)
    print('буду рисовать')
    h_l = result[49]
    h_u = result[949]
    h_mean = np.mean(result, axis=0)
    fig, ax = plt.subplots()  # создаём рисунок fig и 1 график на нём ax, plt.subplots(1,1,1)
    ax.set(xlabel='x', ylabel='h(x)')
    ax.grid()
    ax.plot(x_dots, h_l, '-')
    ax.plot(x_dots, h_u, '-')
    ax.plot(x_dots, h_mean, '-')
    plt.show()
    print('нарисовав')


#t_5_3_c_d(t_5_3_b())
t_5_4_c_d(t_5_4_b())
