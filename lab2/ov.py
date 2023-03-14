import numpy as np


def xt(t, x):
    return C * (t - numpy.sin(2 * t) / 2) - x


def tx(x):
    return  fsolve(xt, 0,x)


def yx(x):
    t = tx(x)
    return C * (1 / 2 - numpy.cos(2 * t) / 2)


def dyx(x, dx=0.001):
    return (yx(x+dx) - yx(x))/dx


def Fy(x):
    return np.sqrt((1+dyx(x)**2)/(2*g*yx(x)))