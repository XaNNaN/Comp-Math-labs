import matplotlib.pyplot as plt
import numpy
import numpy as np
import time
from scipy.optimize import *
from math import *

# Constants
t_0 = 0.000001
C = 1.03439984
T = 1.75418438
g = 9.81
exact_value = (T - t_0) * numpy.sqrt(2 * C / g)
i = 1
x = []
y = []
dy = []
#exact_value = np.exp(T) - np.exp(t_0)
#exact_value = 8 / 3
#exact_value = 70 / 3
#exact_value = 226 /15


# Composite Simpson
def composite_simpson(a, b, n: int, f):
    # if n % 2 != 0:
    #     n += 1
    h = (b - a) / (n - 1)
    x = numpy.linspace(a, b, n)
    return h / 3. * (f(x[0]) + 2 * numpy.sum(f(x[2:-1:2])) + 4 * np.sum(f(x[1::2])) + f(x[-1]))


# Composite trapezoid
def composite_trapezoid(a, b, n: int, f):
    h = (b - a) / (n -1)
    x = numpy.linspace(a, b, n)
    return h / 2. * (f(x[0]) + 2 * numpy.sum(f(x[1:-1:])) + f(x[-1]))


def x_t(t):
    return C * (t - numpy.sin(2 * t) / 2)


def y_t(t):
    return C * (1 / 2 - numpy.cos(2 * t) / 2)


def dy_t(t):
    return numpy.sin(2 * t) / (1 - numpy.cos(2 * t))


# Function
def f(x):
    #return np.exp(x)
    #return x ** 2
    #return 3*x ** 3 + 2*x**2 + x + 2
    return x ** 4 + x ** 3 + x ** 2 + x


def yx(x):
    t = find_t(x)
    return C * (1 / 2 - numpy.cos(2 * t) / 2)


def dyx(x, dx=0.001):
    return (yx(x+dx) - yx(x))/dx


def Fy(x):
    return np.sqrt((1+dyx(x)**2)/(2*g*yx(x)))

def my_foo(my_x):
    my_t = find_t(my_x)
    return np.sqrt((1 + dy_t(my_t) ** 2)/(2*g*y_t(my_t)))


def find_t(value):
    mid = len(x)//2
    low = 0
    high = len(x) - 1
    result = []
    if type(value) != numpy.float64 and type(value) != int :
        for i in value:
            while not(i - 0.00001 < x[mid] < i + 0.00001) and low <= high:
                if i > x[mid]:
                    low = mid + 1
                else:
                    high = mid - 1
                mid = (low + high) // 2
            if low > high:
             #   print("Not found", i)
                result.append(mid)
            else:
                result.append(mid)
        return result
    while not (value - step < x[mid] < value + step) and low <= high:
        if value > x[mid]:
            low = mid + 1
        else:
            high = mid - 1
        mid = (low + high) // 2
    if low > high:
       # print("Not found", value)
        return mid
    else:
        return mid



if __name__ == '__main__':
    start_time = time.time()
    simps_h = []
    trap_h = []
    simps_result = []
    trap_result = []
    simps_dif = []
    trap_dif = []
    step = (T - t_0) / 10000

    t = t_0
    while t < T:
        y.append(y_t(t))
        x.append(x_t(t))
        dy.append(dy_t(t))
        t += step
    print(find_t(2), x[find_t(2)])
    fig, ax = plt.subplots()
    ax.plot(np.linspace(t_0, T, 10000), x)


    print(exact_value)

    for n in range(3, 10000):
        print(n)
        if n % 2 != 0:
            simps_result.append(composite_simpson(t_0, 2, n, Fy))
            simps_dif.append(abs(simps_result[-1] - exact_value))
            simps_h.append((T - t_0) / (n-1))
        trap_result.append(composite_trapezoid(t_0, 2, n, Fy))
        trap_dif.append(abs(trap_result[-1] - exact_value))
        trap_h.append((T - t_0) / (n-1))

    print(simps_result[-1], simps_result[0])
    print(len(simps_result), len(trap_result))

    h_for_scaling = numpy.logspace(-2, 0, 10000)
    fig, ax = plt.subplots()
    print("--- %s seconds ---" % (time.time() - start_time))
    ax.grid()
    ax.scatter(simps_h, simps_dif, marker='o', label='simpson')
    ax.scatter(trap_h, trap_dif, marker='o', label='trapezoid')
    ax.loglog(h_for_scaling, h_for_scaling**2, 'k-', label='$o(h^2)$')
    ax.loglog(h_for_scaling, 10**(-2) * h_for_scaling ** 4, 'k--', label='$o(h^4)$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    plt.show()  # вывод рисунка