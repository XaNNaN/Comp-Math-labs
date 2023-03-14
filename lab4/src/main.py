import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from sklearn.cluster import DBSCAN


# Область продвинутая

def Kirch_G1():
    A = np.ones((10, 10)) - np.eye(10)  # Матрица смежности простого графа
    D = np.eye(10) * 9  # Матрица степеней вершин на диагонали
    return D - A


def Kirch_G2():
    A = np.zeros((20, 20))
    # Кластер левый
    # A[0][1] = A[0][3] = A[0][4] = 1  # ok
    # A[1][0] = A[1][2] = A[1][4] = A[1][5] = 1  # ok
    # A[2][1] = A[2][3] = A[2][4] = 1  # ok
    # A[3][0] = A[3][2] = A[3][6] = A[3][8] = 1  # ok
    # A[4][0] = A[3][1] = A[3][2] = A[3][5] = A[3][6] = A[3][7] = 1  # ok
    # A[5][1] = A[3][4] = A[3][6] = A[3][7] = A[3][19] = 1  # ok
    # A[6][3] = A[6][4] = A[6][5] = A[6][7] = A[6][11] = 1  # ok
    # A[7][4] = A[7][5] = A[7][6] = A[7][15] = 1  # ok
    # # Кластер правй
    # A[8][10] = A[8][11] = A[8][3] = 1  # ok
    # A[9][10] = A[9][11] = A[9][12] = 1  # ok
    # A[10][8] = A[10][9] = A[10][11] = A[10][12] = 1  # ok
    # A[11][8] = A[11][9] = A[11][10] = A[11][12] = A[11][6] = 1  # ok
    # A[12][9] = A[12][10] = A[12][11] = A[12][14] = 1  # ok
    # # Кластер центральный
    # A[13][15] = A[13][18] = A[13][19] = 1  # ok
    # A[14][12] = A[14][15] = A[14][18] = A[14][19] = 1  # ok
    # A[15][7] = A[15][13] = A[15][14] = A[15][16] = A[15][19] = 1  # ok
    # A[16][15] = A[16][18] = A[16][19] = 1  # ok
    # A[17][18] = A[17][19] = 1  # ok
    # A[18][13] = A[18][14] = A[18][16] = A[18][17] = A[18][19] = 1  # ok
    # A[19][5] = A[19][13] = A[19][14] = A[19][15] = A[19][16] = A[19][17] = A[19][18] = 1  # ok
    #  Запишем связи каждой вершины
    connections = \
        [[1, 2, 5],  # 0
         [0, 3, 7, 8],  # 1
         [0, 3, 5, 6],  # 2
         [2, 1, 5],  # 3
         [5, 6, 7, 15],  # 4
         [0, 2, 3, 4, 6, 7],  # 5
         [2, 4, 5, 7, 19],  # 6
         [4, 5, 6, 11],  # 7

         [1, 10, 11],  # 8
         [10, 11, 12],  # 9
         [8, 9, 11, 12],  # 10
         [7, 8, 9, 10, 12],  # 11
         [9, 10, 11, 13],  # 12

         [15, 18, 19],  # 13
         [15, 18, 19],  # 14
         [4, 13, 14, 16, 19],  # 15
         [15, 18, 19],  # 16
         [18, 19],  # 17
         [13, 14, 16, 17, 19],  # 18
         [6, 13, 14, 15, 16, 17, 18]]  # 19

    for i in range(len(connections)):
        for y in connections[i]:
            A[i][y] = 1

    D = np.eye(20)
    for i in range(20):
        D[i][i] = len(connections[i])
    # plt.matshow(A)
    # plt.show()
    return D - A, A




def Kirch_G3():
    A = pd.read_csv('adjacency_matrix.txt', delimiter=' ', header=None, skipinitialspace=True)
    A = pd.DataFrame.to_numpy(A)
    D = np.eye(1000)
    for i in range(1000):
        D[i][i] = sum(A[i])
    return D - A, A

def sort_eigen_1(eignvalues, eignvectors):
    idx = np.argsort(eignvalues)
    eignvalues1 = eignvalues[idx]
    eignvectors1 = eignvectors[:, idx]
    return eignvalues1, eignvectors1




# Базовая область


def delete_zero(values, vectors):
    i = 0
    val_1 = []
    vec_1 = []
    while i < len(values):
        if values[i] > 10e-16:
            val_1.append(values[i])
            vec_1.append(vectors[i])
        i += 1
    return val_1, vec_1


def centring_matrix(X):
    m = len(X)
    A = (np.eye(m) - 1 / m * np.ones((m, m))) @ X
    return A


def sort_eigen(eignvalues, eignvectors):
    idx = np.argsort(eignvalues)[::-1]
    eignvalues1 = np.copy(eignvalues)
    eignvectors1 = np.copy(eignvectors)
    for i in range(len(idx)):
        eignvalues1[i] = eignvalues[idx[i]]
       # eignvectors1[i] = eignvectors[idx[i]]
        eignvectors1[:, i] = eignvectors[:, idx[i]]
    return eignvalues1, eignvectors1


def plot_val_num_1(std_dev, name):
    _, ax = plt.subplots()
    ax.plot(1. + np.arange(len(std_dev)), std_dev, 'o--')
    ax.set_xlabel('номер компоненты вектора Фидлера')
    ax.set_ylabel('значение i-той компоненты')
    # ax.set_ylabel(rf'{name}')
    ax.grid()
    plt.show()



def pca(A):

    eignvalues, eignvectors = np.linalg.eig(A.T @ A)

    eignvalues, eignvectors = delete_zero(eignvalues, eignvectors)

    eignvectors = np.array(eignvectors)

    eignvalues, eignvectors = sort_eigen(eignvalues, eignvectors)

    cov_coef = 1 / (len(A)-1)

    for i in range(len(eignvalues)):
        eignvalues[i] = np.sqrt(cov_coef*eignvalues[i])

    return eignvalues, eignvectors.T


def plot_data(X, principal_components, color):
    fig, ax = plt.subplots()
    x_bar = [np.mean(X[:, 0]), np.mean(X[:, 1])]
    sigmas_bar = [st.stdev(X[:, 0]), st.stdev(X[:, 1])]
    X_bar = (X - x_bar) / sigmas_bar
    ax.scatter(X_bar[:, 0], X_bar[:, 1], c=color, s=10, alpha=0.6)
    ax.set_xlabel(r'$x_1$', fontsize=16)
    ax.set_ylabel(r'$x_2$', fontsize=16)
    ax.grid()
    plt.show()


def plot_val_num(std_dev):
    _, ax = plt.subplots()
    ax.plot(1. + np.arange(len(std_dev)), std_dev, 'o--')
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$\sqrt{\nu} \sigma_i$')
    ax.grid()
    plt.show()




# Область ...

if __name__ == '__main__':
    # Read the matrix X
    Y = X = pd.read_csv('wdbc.data', delimiter=',', header=None, skipinitialspace=True)
    X = X.drop([0, 1], 1)
    X = pd.DataFrame.to_numpy(X)
    Y = pd.DataFrame.to_numpy(Y)
    # Градация выхохулей по цвету
    color = []
    for i in Y:
        if i[1] == 'B':
            color.append((0, 1, 0))
        else:
            color.append((1, 0, 0))

    # Centring
    X = centring_matrix(X)
    # Метод главных компонент
    std_dev, main_comps = pca(X)

    # Отклонения и номера
    plot_val_num(std_dev)

    # Проекция на две главные компоненты
    main_comps_k = main_comps[:2]
    X_k = X @ main_comps_k.T
    plot_data(X_k, main_comps @ main_comps.T, color)

    # Продвинутая часть
    L_1 = Kirch_G1()
    L_2, A2 = Kirch_G2()
    L_3, A3 = Kirch_G3()
    # Собственные числа и вектора лапласиан или лапласианинов
    L_1_eigenval, L_1_eigenvec = np.linalg.eig(L_1)
    L_2_eigenval, L_2_eigenvec = np.linalg.eig(L_2)
    L_3_eigenval, L_3_eigenvec = np.linalg.eig(L_3)
    #  Осторитруем по величине соб значений
    L_1_eigenval, L_1_eigenvec = sort_eigen_1(L_1_eigenval, L_1_eigenvec)
    L_2_eigenval, L_2_eigenvec = sort_eigen_1(L_2_eigenval, L_2_eigenvec)
    L_3_eigenval, L_3_eigenvec = sort_eigen_1(L_3_eigenval, L_3_eigenvec)
    #  Выведем на экран
    plot_val_num_1(L_1_eigenval, r'\lambda_i^1')
    plot_val_num_1(L_2_eigenval, r'\lambda_i^2')
    plot_val_num_1(L_3_eigenval, r'\lambda_i^3')

    # Рассмотрим вектор, соответствующий второму по величине собст значению.

    plot_val_num_1(L_2_eigenvec[:, 1], 'as')
    plot_val_num_1(L_3_eigenvec[:, 1], 'as')

    plt.matshow(A3)
    plt.show()
    ii = np.argsort(L_3_eigenvec[:, 1].T)
    A3 = A3[np.ix_(ii, ii)]
    plt.matshow(A3)
    plt.show()



