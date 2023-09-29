import numpy as np
import math
from scipy.interpolate import lagrange

a = 0.0
b = math.pi

x_train = np.random.uniform(a, b, 100)
y_train = np.sin(x_train)
x_test = np.random.uniform(a, b, 10)
y_test = np.sin(x_test)

poly = lagrange(x_train, y_train)
y_pred = poly(x_train)

error_train = np.square(np.subtract(y_train, y_pred)).mean()

y_pred = poly(x_test)

error_test = np.square(np.subtract(y_test, y_pred)).mean()

print("Training Error: ", error_train, " Test error: ", error_test)

stddev_list = [0.1, 1, 10, 100, 1000]

for stddev in stddev_list:
    noise = np.random.normal(0, stddev*0.001, 100)
    y_train = np.sin(x_train)
    poly = lagrange(x_train +  noise, y_train)

    y_pred = poly(x_train)

    error_train = np.square(np.subtract(y_train, y_pred)).mean()

    y_pred = poly(x_test)

    error_test = np.square(np.subtract(y_test, y_pred)).mean()

    print("Training Error: ", error_train, " Test error: ", error_test)

