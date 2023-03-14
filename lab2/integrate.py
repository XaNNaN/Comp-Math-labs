
# Composite Simpson
def composite_simpson(a, b, n: int, f):
    # Step
    h = (b - a) / (n - 1)
    y = numpy.zeros(n)
    for i in range(0, n):
        y[i] = f(a + h * i)
    # Here starts integrating calculations
    left_sum = 0
    right_sum = 0
    for i in range(1, int((n-1) / 2)):
        left_sum += y[2*i]
    for i in range(1, int((n-1) / 2 + 1)):
        right_sum += y[2*i - 1]
    # Result = h/3 * Z, Z = A + B + C + D
    z = y[0] + 2 * left_sum + 4 * right_sum + y[-1]
    return h * z / 3


# Composite trapezoid
def composite_trapezoid(a, b, n: int, f):
    h = (b - a) / (n - 1)
    y = numpy.zeros(n)
    for i in range(0, n):
        y[i] = f(a + h * i)
    # Here starts calc
    my_sum = 0
    for i in range(1, n):
        my_sum += y[i]
    # result = h/2 + Z, Z = f[0] + 2 * sum + f[-1]
    z = y[0] / 2 + my_sum + y[-1] / 2
    return h * z


# Composite Simpson with parameters
def prm_composite_simpson(x, y):
    result = 0
    for i in range(2, len(x), 2):
        result += (x[i] - x[i-2]) / 6 * (y[i-2] + 4 * y[i-1] + y[i])
    return result


# Composite Trapezoid with parameters
def prm_composite_trapezoid(x, y):
    result = 0
    for i in range(1, len(x)):
        result += (x[i] - x[i-1]) / 2 * (y[i-1] + y[i])
    return result


# Composite Simpson
def composite_simpson(a, b, n: int, f):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = numpy.linspace(a, b, n+1)
    return h / 3. * (f(x[0]) + 2 * numpy.sum(f(x[2:-1:2])) + 4 * np.sum(f(x[1::2])) + f(x[-1]))

