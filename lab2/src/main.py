import matplotlib.pyplot as plt
import numpy
import numpy as np
import time
from scipy.optimize import *
from math import *

# Constants
t_0 = 0.00000001
C = 1.03439984
T = 1.75418438
g = 9.81
exact_value = T * numpy.sqrt(2 * C / g)
i = 1
#exact_value = np.exp(T) - np.exp(t_0)
#exact_value = 8 / 3
#exact_value = 70 / 3
#exact_value = 226 /15


# Composite Simpson
def composite_simpson(a, b, n: int, f):
    h = (b - a) / (n - 1)
    x = numpy.linspace(a, b, n)
    return h / 3. * (f(x[0]) + 2 * numpy.sum(f(x[2:-1:2]))
                     + 4 * np.sum(f(x[1::2])) + f(x[-1]))


# Composite trapezoid
def composite_trapezoid(a, b, n: int, f):
    h = (b - a) / (n -1)
    x = numpy.linspace(a, b, n)
    return h / 2. * (f(x[0]) + 2 * numpy.sum(f(x[1:-1:])) + f(x[-1]))


def dx_dt(t):
    return C*(1-np.cos(2*t))


def y_t(t):
    return C * (1 / 2 - numpy.cos(2 * t) / 2)


def dy_dx(t):
    return numpy.sin(2 * t) / (1 - numpy.cos(2 * t))


# Function
def f(x):
    return np.exp(x)
    #return x ** 2
    #return 3*x ** 3 + 2*x**2 + x + 2
    #return x ** 4 + x ** 3 + x ** 2 + x


def my_foo(t):
    return np.sqrt((1 + dy_dx(t)**2)/(2*g*y_t(t))) * dx_dt(t)


def x_t(t):
    return C * (t - numpy.sin(2 * t) / 2)

if __name__ == '__main__':
    start_time = time.time()
    simps_h = []
    trap_h = []
    simps_result = []
    trap_result = []
    simps_dif = []
    trap_dif = []

    print(exact_value)

    for n in range(3, 10000):
        #print(n)
        if n % 2 != 0:
            simps_result.append(composite_simpson(t_0, T, n, my_foo))
            simps_dif.append(abs(simps_result[-1] - exact_value))
            simps_h.append((T - t_0) / (n-1))
        trap_result.append(composite_trapezoid(t_0, T, n, my_foo))
        trap_dif.append(abs(trap_result[-1] - exact_value))
        trap_h.append((T - t_0) / (n-1))

    print(simps_result[-1], simps_result[0])
   # print(len(simps_result), len(trap_result))

    h_for_scaling = numpy.logspace(-4, 0, 10000)
    fig, ax = plt.subplots()
    print("--- %s seconds ---" % (time.time() - start_time))
    ax.grid()
    ax.scatter(simps_h, simps_dif, marker='o', label='simpson')
    ax.scatter(trap_h, trap_dif, marker='o', label='trapezoid')
    ax.loglog(h_for_scaling, h_for_scaling**2, 'k-', label='$o(h^2)$')
    ax.loglog(h_for_scaling, 10**(-2) * h_for_scaling ** 4, 'k--', label='$o(h^4)$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("E")
    ax.legend()
    plt.show()  # вывод рисунка