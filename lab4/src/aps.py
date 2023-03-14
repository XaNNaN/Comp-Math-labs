import numpy as np
import matplotlib.pyplot as pl
import statistics as st


def center(A):
    m = len(A)
    A_center = (np.eye(m) - 1./m * np.ones((m, m))) @ A
    return A_center


def remove_zeros(num, vec):
    remove_idx = []
    for i in range(len(num)):
        if abs(num[i]) < 10e-16:
            remove_idx.append(i)

    num = np.delete(num, remove_idx)
    vec = np.delete(vec, remove_idx, 0)

    return num, vec


def pca(A):
    # Квадраты сингулярных чисел соб Грамма и снг вектора
    self_num, self_vec = np.linalg.eig(A.T @ A)
    # Убрать нулевые элкменты
    self_num, self_vec = remove_zeros(self_num, self_vec)
    # Найти индексы в отсоритрованном массива и развернуть
    idx = np.flip(np.argsort(self_num))
    self_num = self_num[idx]
    self_vec = self_vec[:, idx]

    for i in range(len(self_num)):
        self_num[i] = np.sqrt(1./(len(A)-1.)) * np.sqrt(self_num[i])

    return self_vec.T, self_num


# def pca_test(A):
#     _, sigmas, principal_components = np.linalg.svd(A, full_matrices=False)
#     return principal_components, sigmas


def plot_data(A, principal_components=None, scatter=None):
    if scatter is None:
        scatter = []
    if principal_components is None:
        principal_components = []
    _, ax = pl.subplots()
    x_bar = [np.mean(A[:, 0]), np.mean(A[:, 1])]
    sigmas_bar = [st.stdev(A[:, 0]), st.stdev(A[:, 1])]
    A_bar = (A - x_bar) / sigmas_bar
    ax.scatter(A_bar[:, 0], A_bar[:, 1], c=scatter, s=3)
    ax.plot([0], [0], 'ro', markersize=5)
    max_val = np.max(np.abs(A_bar))
    for pc in principal_components:
        ax.plot([0, max_val/1.5 * pc[0]], [0, max_val/1.5 * pc[1]], linewidth=3)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    # ax.set_xlim((-max_val - 1, max_val + 1))
    # ax.set_ylim((-max_val - 1, max_val + 1))
    ax.grid()
    pl.show()


def plot_power(s_d):
    _, ax = pl.subplots()
    ax.plot(1. + np.arange(len(s_d)), s_d, 'o--')
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$\sqrt{\nu} \sigma_i$')
    ax.grid()
    pl.show()


def read_wdbc():
    with open('wdbc.data', 'r') as f:
        full_data = [[num for num in line.split(',')] for line in f]

    is_benign = []
    for row in full_data:
        if row[1] == 'B':
            is_benign.append((0, 1, 0))
        else:
            is_benign.append((0.5, 0, 1))

    matrix = []
    for row in full_data:
        matrix.append([float(row[i]) for i in range(2, len(full_data[0]))])

    return matrix, is_benign


def base():
    matrix, is_benign = read_wdbc()
    matrix = center(matrix)
    # matrix_test = np.array([[matrix[i, 0], matrix[i, 1]] for i in range(len(matrix))])
    # plot_data(matrix_test, [], is_benign)
    main_comp, std_dev = pca(matrix)
    plot_power(std_dev)
    main_comp_2 = main_comp[:2]

    plot_data(matrix @ main_comp_2.T, main_comp_2 @ main_comp_2.T, is_benign)


def L_G1():
    A = np.ones((10, 10)) - np.eye(10)
    D = np.eye(10) * 9

    return D - A


def L_G2():
    connects = [[2, 4], [1, 3, 4, 6, 7], [2, 4, 5], [1, 2, 3, 5, 6, 7, 14], [3, 4, 6, 7, 16],
                [2, 4, 5, 8], [2, 4, 5], [6, 9, 10, 11], [8, 10, 11, 12], [8, 9, 11, 12, 13],
                [8, 9, 10], [9, 10, 19], [10, 14, 16, 17, 19], [4, 13, 16, 17, 18],
                [17, 18, 19], [5, 13, 14, 17], [13, 14, 15, 16, 18, 20], [14, 15, 17, 20],
                [12, 13, 15, 20], [17, 18, 19]]

    a = []
    for i in connects:
        a_row = []
        for j in range(20):
            if i.__contains__(j+1):
                a_row.append(1)
            else:
                a_row.append(0)
        a.append(a_row)

    A = np.array(a)

    d = []
    for i in range(20):
        d_row = []
        for j in range(20):
            if i == j:
                d_row.append(len(connects[i]))
            else:
                d_row.append(0)
        d.append(d_row)

    D = np.array(d)
    pl.matshow(A)
    pl.show()
    return D - A


def L_G3():
    with open('adjacency_matrix.txt', 'r') as f:
        a = [[int(num) for num in line.split(' ')] for line in f]
    A = np.array(a)

    d = []
    for i in range(1000):
        d_row = []
        for j in range(1000):
            if i == j:
                d_row.append(sum(A[i]))
            else:
                d_row.append(0)
        d.append(d_row)

    D = np.array(d)

    return D - A


def advanced():
    L_1 = L_G1()
    L_2 = L_G2()
    L_3 = L_G3()

    L_1_num, L_1_vec = np.linalg.eig(L_1)
    L_2_num, L_2_vec = np.linalg.eig(L_2)
    L_3_num, L_3_vec = np.linalg.eig(L_3)

    # L_1_vec = L_1_vec[:, np.argsort(L_1_num)]
    # # L_1_vec = np.flip(L_1_vec)
    # L_1_num = L_1_num[np.argsort(L_1_num)]
    # # L_1_num = np.flip(L_1_num)
    # L_2_vec = L_2_vec[:, np.argsort(L_2_num)]
    # # L_2_vec = np.flip(L_2_vec)
    # L_2_num = L_2_num[np.argsort(L_2_num)]
    # # L_2_num = np.flip(L_2_num)
    # L_3_vec = L_3_vec[:, np.argsort(L_3_num)]
    # # L_3_vec = np.flip(L_3_vec)
    # L_3_num = L_3_num[np.argsort(L_3_num)]
    # # L_3_num = np.flip(L_3_num)

    plot_power(L_1_num)
    plot_power(L_2_num)
    plot_power(L_3_num)


if __name__ == '__main__':
    base()
    # advanced()

    print(L_G2())
