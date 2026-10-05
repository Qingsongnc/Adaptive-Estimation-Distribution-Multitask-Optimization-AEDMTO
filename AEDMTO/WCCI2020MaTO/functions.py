import numpy as np

"""
# import format
from functions import Griewank, Rastrigin, Ackley, Sphere, Schwefel, Rosenbrock, Weierstrass
"""


# functions
class Task:
    def __init__(self, Low=0, High=1, n=50, coeffi=None, bias=None):
        self.Low = Low
        self.High = High
        self.Dimension = n
        if bias is not None:
            self.center = bias
        else:
            self.center = np.zeros(shape=n)
        if coeffi is not None:
            self.M = coeffi
        else:
            self.M = np.zeros(shape=(n, n))
            for i in range(n):
                self.M[i, i] = 1

    def decode(self, X):
        X1 = self.Low + (self.High - self.Low) * X
        X1 = np.dot(self.M, (X1[0:self.Dimension] - self.center).T)
        return X1.reshape(self.Dimension)


class Sphere(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X, **kwargs):
        temp = self.decode(X)
        return np.sum(temp ** 2)

    def Info(self):
        return 'Sphere ' + str(self.Dimension)


class Ackley(Task):
    def __init__(self, Low=-50, High=50, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X, **kwargs):
        temp = self.decode(X)
        return 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.sum(temp ** 2) / self.Dimension)) - np.exp(
            np.sum(np.cos(2 * np.pi * temp)) / self.Dimension)

    def Info(self):
        return 'Ackley ' + str(self.Dimension)


class Rastrigin(Task):
    def __init__(self, Low=-50, High=50, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X, **kwargs):
        temp = self.decode(X)
        return np.sum(temp ** 2 - 10 * np.cos(2 * np.pi * temp) + 10)

    def Info(self):
        return 'Rastrigin ' + str(self.Dimension)


class Griewank(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X, **kwargs):
        temp = self.decode(X)
        return 1 + np.sum(temp ** 2) / 4000 - np.prod(np.cos(temp / np.sqrt(np.arange(1, self.Dimension + 1))))

    def Info(self):
        return 'Griewank ' + str(self.Dimension)


class Schwefel(Task):
    def __init__(self, Low=-500, High=500, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X, **kwargs):
        temp = self.decode(X)
        return 418.9829 * self.Dimension - np.sum(temp * np.sin(np.sqrt(np.abs(temp))))

    def Info(self):
        return 'Schwefel ' + str(self.Dimension)


class Rosenbrock(Task):
    def __init__(self, Low=-50, High=50, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X, **kwargs):
        temp = self.decode(X)
        return np.sum(100 * ((temp[:self.Dimension - 1] ** 2 - temp[1:]) ** 2) + (temp[:self.Dimension - 1] - 1) ** 2)

    def Info(self):
        return 'Rosenbrock ' + str(self.Dimension)


class Weierstrass(Task):
    def __init__(self, Low=-0.5, High=0.5, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X, **kwargs):
        temp = self.decode(X)
        a = 0.5
        b = 3
        kmax = 20
        sums = 0
        for k in range(0, kmax + 1):
            sums += np.sum(a ** k * np.cos(2 * np.pi * b ** k * (temp + 0.5))) - self.Dimension * a ** k * np.cos(
                np.pi * b ** k)
        return sums

    def Info(self):
        return 'Weierstrass ' + str(self.Dimension)
