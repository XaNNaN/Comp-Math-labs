import numpy as np
from random import random
from matplotlib import pyplot as plt


def generate_neurons():
    # Генерация возбуждающих нейронов
    neurons = np.zeros((1000, 5))
    for i in range(0, 800):
        neurons[i][0] = 0.02  # a
        neurons[i][1] = 0.2  # b
        neurons[i][2] = -65 + 15 * random() ** 2  # c
        neurons[i][3] = 8 - 6 * random() ** 2  # d
        neurons[i][4] = 5 * random()  # I_0
    # Генерация тормозных нейронов
    for i in range(800, 1000):
        neurons[i][0] = 0.02 + 0.08 * random()  # a
        neurons[i][1] = 0.25 - 0.05 * random()  # b
        neurons[i][2] = -65  # c
        neurons[i][3] = 2  # d
        neurons[i][4] = 2 * random()  # I_0
    return neurons


#  Первая функция системы
def f_1(v, u, neuron, I):
    return 0.04 * v ** 2 + 5 * v + 140 - u + neuron[4] + I


# Вторая функция системы
def f_2(v, u, neuron):
    return neuron[0] * (neuron[1] * v - u)


# Пока непонятно. Изучать.
f = [f_1, f_2]


# Придётся переписать реализацию метода Эйлера
# Новый вариант будет получать ток, параметры нейрона, а также v, u с прошлой итерации
# Возвращать должен статус импульса и v,u на текущей
def euler(h, f, v_i, u_i, neuron, I):
    if v_i >= 30:
        return 1, neuron[2], u_i + neuron[3]
    else:
        v_1 = v_i + h * f[0](v_i, u_i, neuron, I)
        u_1 = u_i + h * f[1](v_i, u_i, neuron)
        return 0, v_1, u_1


# Генерация матрицы смежности
def generate_w_matrix():
    matrix = np.zeros((1000, 1000))
    for i in range(0, 800):
        for y in range(0, 1000):
            if y != i:
                matrix[i][y] = random() / 2
            else:
                matrix[i][y] = 0

    for i in range(800, 1000):
        for y in range(0, 1000):
            if y != i:
                matrix[i][y] = -1 * random()
            else:
                matrix[i][y] = 0
    return matrix


def init_states(neuron_mat):
    state_mat = np.zeros((1000, 2))
    # инициализация возбуждающих нейронов
    for i in range(0, 800):
        state_mat[i][0] = -65  # v
        state_mat[i][1] = neuron_mat[i][1] * (-65)  # u
    # инициализация тормозных нейронов
    for i in range(800, 1000):
        state_mat[i][0] = -65  # v
        state_mat[i][1] = neuron_mat[i][1] * (-65)  # u
    return state_mat


if __name__ == '__main__':
    W = generate_w_matrix()
    neurons = generate_neurons()
    states_mat = init_states(neurons)
    h = 0.5
    iterations = 2000
    imp_iter = np.zeros((1000, iterations))
    time = []
    impulse = []
    for t in range(1, iterations):  # в каждый момент дискретного времени
        print(t)
        for i in range(0, 1000):  # для каждого нейрона
            I = 0
            for y in range(0, 1000):  # собрать внешний ток
                # Связь от  нейрона к нейрону для передачи импульса идёт по строке, приём по столбцу
                if imp_iter[y][t-1] != 0:
                    I += W[y][i]
            # Отправить ток и прошлое состояние(v, u) в метод Эйлера и получить информацию об импульсе в нейроне
            imp, states_mat[i][0], states_mat[i][1] = euler(h, f, states_mat[i][0], states_mat[i][1], neurons[i], I)
            imp_iter[i][t] = imp
        print(t, 'after')
    # Мы получили матрицу импульсов imp_iter, если импульс есть - то единица, нет - 0
    # Нужно для каждого нейрона выделить моменты времени, когда и него был импульс
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    for i in range(0, 1000):
        for y in range(0, iterations):
            if imp_iter[i][y] == 1:
                time.append(y/2)
                impulse.append(i)
        if i < 800:
            ax.scatter(time, impulse, color='green')
        else:
            ax.scatter(time, impulse, color='red')
        time.clear()
        impulse.clear()
    ax.set_xlabel('time')
    ax.set_ylabel('ID')
    plt.show()

