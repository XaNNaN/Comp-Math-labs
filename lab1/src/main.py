import numpy
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from matplotlib.pyplot import ylim

x_1 = 0.1  # для G(x)
x_2 = 0.1  # для G'(x)

# Можно вывести график производной, убрав # в tsk_3

# Массивы входных данных для проверки первых двух пунктов отдельно от третьего
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

    n = x_nodes.size  # Колличетсво точек
    h_mas = np.zeros(n - 1, dtype=np.float)
    for i in range(0, n - 1):
        h_mas[i] = x_nodes[i + 1] - x_nodes[i]
    # Найдены все необходимые h

    a_div_mas = np.zeros(n - 1, dtype=np.float)  # a_{i+1} - a_i
    for i in range(0, n - 1):
        a_div_mas[i] = y_nodes[i + 1] - y_nodes[i]
    # Найдена разница каждой пары коэф а

    b_matrix = np.zeros(n, dtype=np.float)  # Инициализация b 0
    for i in range(1, n - 1):
        b_matrix[i] = 3 / h_mas[i] * a_div_mas[i] - 3 / h_mas[i - 1] * a_div_mas[i - 1]
    # Заполнена матрица b

    diagonal_0 = np.ones(n, dtype=np.float)  # Инициализация 0 1
    for i in range(1, n - 1):
        diagonal_0[i] = 2 * (h_mas[i] + h_mas[i - 1])
    # Заполнена главная диагональ

    diagonal_1 = np.copy(h_mas)  # Инициализация +1 по H
    diagonal_1[0] = 0  # +1[0] = 0
    d1_m = np.diag(diagonal_1, 1)  # Формирование матрицы +1
    # Заполнена верхняя неосновная диагональ

    diagonal_2 = np.copy(h_mas)  # Инициализация -1 по H
    diagonal_2[h_mas.size - 1] = 0  # -1[n-1] = 0
    d1_m = d1_m + np.diag(diagonal_2, -1)  # Сложение -1 +1
    # Заполнена нижняя неосновная диагональ

    a_matrix = np.diag(diagonal_0, 0)  # Формирование матрицы с главной диагональю
    a_matrix = a_matrix + d1_m  # Сложение  матр. с главной диаг и +1 -1
    # Заполнена матрица А

    # Тогда с = А^(-1)b
    a_matrix_inv = linalg.inv(a_matrix)  # Инвертирование A
    c_cof = np.dot(a_matrix_inv, b_matrix)  # Перемножение матриц
    # Решение матричного уравнения найдено

    d_cof = np.zeros(n - 1, dtype=np.float)  # Инициализация d[]
    for i in range(0, n - 1):
        d_cof[i] = (c_cof[i + 1] - c_cof[i]) / (3 * h_mas[i])  # Вычисление d_i
    b_cof = np.zeros(n - 1, dtype=np.float)  # Инициализация b[]
    for i in range(0, n - 1):
        b_cof[i] = a_div_mas[i] / h_mas[i] - h_mas[i] * (c_cof[i + 1] + 2 * c_cof[i]) / 3  # Вычисление b_i+1

    numpy.set_printoptions(precision=5, suppress=True, linewidth=200)  # Форматирвание вывода

    c_cof = np.delete(c_cof, n - 1)
    a_cof = np.delete(y_nodes, n - 1)

    result = numpy.r_['1,1,0', [a_cof, b_cof, c_cof, d_cof]]  # Конкатенация по строкам
    return result


def qubic_spline(x, qs_coef, x_nodes: np.ndarray):
    # план : 1) определить индекс промежутка, которому принадлежит x
    #        2) подставить x  в полином с соотвествующим индексом
    section_index = -1  # Индекс промежутка
    if x < x_nodes[0] or x > x_nodes[x_nodes.size - 1]:
        return 'invalid x. Out of range'
    for i in range(0, x_nodes.size - 1):
        if x_nodes[i + 1] > x >= x_nodes[i]:
            section_index = i
    if section_index == -1:
        section_index = x_nodes.size - 2  # Проверка последнего полинома
    #  Индекс определён. исключения по промежуткам обработаны

    result = qs_coef[0, section_index] + qs_coef[1, section_index] * (x - x_nodes[section_index]) \
             + qs_coef[2, section_index] * (x - x_nodes[section_index]) ** 2 + qs_coef[3, section_index] \
             * (x - x_nodes[section_index]) ** 3

    return result


def d_qubic_spline(x, qs_coef, x_nodes: np.ndarray):
    # план : 1) определить индекс промежутка, которому принадлежит x
    #        3) подставить x  в производную с соотвествующим индексом
    section_index = -1  # Индекс промежутка
    if x < x_nodes[0] or x > x_nodes[x_nodes.size - 1]:
        return 'invalid x. Out of range'
    for i in range(0, x_nodes.size - 1):
        if x_nodes[i + 1] > x >= x_nodes[i]:
            section_index = i
    if section_index == -1:
        section_index = x_nodes.size - 2  # Проверка последнего полинома
    #  Индекс определён. исключения по промежуткам обработаны

    # Первая производная = b + с(2*x- 2*x_i) + 3d(x - x_i)^2
    result = qs_coef[1, section_index] + qs_coef[2, section_index] * (2 * x - 2 * x_nodes[section_index]) + 3 \
             * qs_coef[3, section_index] * (x - x_nodes[section_index]) ** 2

    return result


def task_3():
    x_nodes_3 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float)
    y_nodes_3 = np.array([3.37, 3.95, 3.73, 3.59, 3.15, 3.15, 3.05, 3.86, 3.60, 3.70, 3.02], dtype=np.float)
    coef = qubic_spline_coeff(x_nodes_3, y_nodes_3)

    y_dots = np.zeros(1001)
    d_y_dots = np.zeros(1001)
    x_dots = np.arange(0.0, 1.001, 0.001)
    d_x_dots = np.arange(0.0, 1.001, 0.001)

    for i in range(0, 1001):
        y_dots[i] = qubic_spline(i / 1001, coef, x_nodes_3)
        d_y_dots[i] = d_qubic_spline(i / 1001, coef, x_nodes_3)

    fig, ax = plt.subplots()
    ax.set(xlabel='x', ylabel='h(x)')
    ax.grid()
    ax.plot(x_dots, y_dots, '-', label='S(x)')
    # ax.plot(d_x_dots, d_y_dots, 'r-', label='S\'(x)')
    ax.scatter(x_nodes_3, y_nodes_3, marker='o', )
    ax.legend()
    #  ax.set_ylim([2, 4])
    plt.show()  # вывод рисунка


coef = qubic_spline_coeff(x_nodes, y_nodes)
print(coef)
print('G(x) = ', qubic_spline(x_1, coef, x_nodes))
print('G(x\') = ', d_qubic_spline(x_2, coef, x_nodes))
task_3()
