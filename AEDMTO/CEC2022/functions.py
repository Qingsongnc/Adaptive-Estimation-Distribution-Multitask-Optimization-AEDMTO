import numpy as np


class Task:
    def __init__(self, Low=0, High=1, n=50, coeffi=None, bias=None, sh_rate=1.0, shuffle=None):
        self.Low = Low
        self.High = High
        self.Dimension = n
        self.sh_rate = sh_rate
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
        if shuffle is None:
            self.shuffle = np.arange(0, self.Dimension, dtype=int)
        else:
            self.shuffle = shuffle

    def decode(self, X):
        X1 = self.Low + (self.High - self.Low) * X
        X1 = np.dot(self.M, (X1 - self.center).T * self.sh_rate)
        return X1[self.shuffle]


class Ellips(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X):
        temp = self.decode(X)
        indexs = np.arange(self.Dimension)
        return np.sum((10 ** (6 * indexs / (self.Dimension - 1))) * temp * temp)

    def Info(self):
        return 'Ellips ' + str(self.Dimension)


class Discus(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X):
        temp = self.decode(X)
        return temp[0] * temp[0] * (10 ** 6) + np.sum(temp[1:] * temp[1:])

    def Info(self):
        return 'Discus ' + str(self.Dimension)


class Rosenbrock(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, sh_rate=2.048 / 100.0):
        super().__init__(Low, High, n, coeffi, bias, sh_rate)

    def function(self, X):
        temp = self.decode(X) + 1
        return np.sum(100 * ((temp[:self.Dimension - 1] ** 2 - temp[1:]) ** 2) + (temp[:self.Dimension - 1] - 1) ** 2)

    def Info(self):
        return 'Rosenbrock ' + str(self.Dimension)


class Ackley(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X):
        temp = self.decode(X)
        return 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.sum(temp ** 2) / self.Dimension)) - np.exp(
            np.sum(np.cos(2 * np.pi * temp)) / self.Dimension) + 500

    def Info(self):
        return 'Ackley ' + str(self.Dimension)


class Weierstrass(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, sh_rate=0.5 / 100):
        super().__init__(Low, High, n, coeffi, bias, sh_rate)

    def function(self, X):
        temp = self.decode(X)
        a = 0.5
        b = 3
        kmax = 20
        sums = 0
        for k in range(0, kmax + 1):
            sums += np.sum((a ** k) * np.cos(2 * np.pi * (b ** k) * (temp + 0.5))) - self.Dimension * (a ** k) * np.cos(
                np.pi * (b ** k))
        return sums + 600

    def Info(self):
        return 'Weierstrass ' + str(self.Dimension)


class Griewank(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, sh_rate=600.0 / 100):
        super().__init__(Low, High, n, coeffi, bias, sh_rate)

    def function(self, X):
        temp = self.decode(X)
        return 1 + np.sum(temp ** 2) / 4000 - np.prod(np.cos(temp / np.sqrt(np.arange(1, self.Dimension + 1)))) + 700

    def Info(self):
        return 'Griewank ' + str(self.Dimension)


class Rastrigin(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, sh_rate=5.12 / 100):
        super().__init__(Low, High, n, coeffi, bias, sh_rate)

    def function(self, X):
        temp = self.decode(X)
        return np.sum(temp ** 2 - 10 * np.cos(2 * np.pi * temp) + 10)

    def Info(self):
        return 'Rastrigin ' + str(self.Dimension)


class Schwefel(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, sh_rate=1000.0 / 100):
        super().__init__(Low, High, n, coeffi, bias, sh_rate)

    def function(self, X):
        temp = self.decode(X) + 420.9687462275036
        temp1 = temp.copy()
        temp1[temp1 < -500] = -500 + np.fmod(np.abs(temp1[temp1 < -500]), 500)
        temp1[temp1 > 500] = 500 - np.fmod(np.abs(temp1[temp1 > 500]), 500)
        return 418.9828872724338 * self.Dimension - np.sum(temp1 * np.sin(np.sqrt(np.abs(temp1)))) + np.sum(
            (temp[temp < -500] + 500) ** 2 / 10000 / self.Dimension) + np.sum(
            (temp[temp > 500] - 500) ** 2 / 10000 / self.Dimension) + 1100

    def Info(self):
        return 'Schwefel ' + str(self.Dimension)


class Katsuura(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, sh_rate=5.0 / 100):
        super().__init__(Low, High, n, coeffi, bias, sh_rate)

    def function(self, X):
        temp = self.decode(X)
        radix = np.zeros(self.Dimension)
        for j in range(1, 33):
            radix += np.abs((2 ** j) * temp - np.floor((2 ** j) * temp + 0.5)) / (2 ** j)
        radix = (1 + (np.arange(self.Dimension) + 1) * radix) ** (10 / (self.Dimension ** 1.2))
        radix = np.prod(radix)
        return 10 / self.Dimension / self.Dimension * (radix - 1)

    def Info(self):
        return 'Katsuura ' + str(self.Dimension)


class GrieRosen(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, sh_rate=5.0 / 100):
        super().__init__(Low, High, n, coeffi, bias, sh_rate)

    def function(self, X):
        temp = self.decode(X) + 1
        temp1 = np.append(temp[1:], temp[0])
        temp1 = 100 * (temp * temp - temp1) * (temp * temp - temp1) + (temp - 1) * (temp - 1)

        return np.sum(temp1 * temp1 / 4000 - np.cos(temp1) + 1) + 1500

    def Info(self):
        return 'Grie_Rosen ' + str(self.Dimension)


class Escaffer6(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None):
        super().__init__(Low, High, n, coeffi, bias)

    def function(self, X):
        temp = self.decode(X)
        temp1 = np.append(temp[1:], temp[0])
        temp1 = temp * temp + temp1 * temp1
        temp2 = np.sin(np.sqrt(temp1))
        return np.sum(0.5 + (temp2 * temp2 - 0.5) / (1 + 0.001 * temp1) / (1 + 0.001 * temp1)) + 1600

    def Info(self):
        return 'Escaffer6 ' + str(self.Dimension)


class HappyCat(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, sh_rate=5.0 / 100):
        super().__init__(Low, High, n, coeffi, bias, sh_rate)

    def function(self, X):
        temp = self.decode(X) - 1
        return (np.abs(np.sum(temp * temp) - self.Dimension) ** (1 / 4)) + (0.5 * np.sum(
            temp * temp) + np.sum(temp)) / self.Dimension + 0.5 + 1300

    def Info(self):
        return 'Happycat ' + str(self.Dimension)


class Hgbat(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, sh_rate=5.0 / 100):
        super().__init__(Low, High, n, coeffi, bias, sh_rate)

    def function(self, X):
        temp = self.decode(X) - 1
        return np.sqrt(np.abs((np.sum(temp * temp) ** 2 - np.sum(temp) ** 2))) + (
                np.sum(temp * temp) / 2 + np.sum(temp)) / self.Dimension + 0.5

    def Info(self):
        return 'Hgbat ' + str(self.Dimension)


class Hf01(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, shuffle=None):
        super().__init__(Low, High, n, coeffi, bias, 1.0, shuffle)
        self.Sch = Schwefel(n=15)
        self.Ras = Rastrigin(n=15)
        self.Elp = Ellips(n=20)

    def function(self, X):
        temp = (self.decode(X) + 100) / 200
        func = 0
        func += self.Sch.function(temp[:15])
        func += self.Ras.function(temp[15:30])
        func += self.Elp.function(temp[30:])
        return func + 600

    def Info(self):
        return 'Hf01 ' + str(self.Dimension)


class Hf04(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, shuffle=None):
        super().__init__(Low, High, n, coeffi, bias, 1.0, shuffle)
        self.Hg = Hgbat(n=10)
        self.Dis = Discus(n=10)
        self.GR = GrieRosen(n=15)
        self.Ras = Rastrigin(n=15)

    def function(self, X):
        temp = (self.decode(X) + 100) / 200
        func = 0
        func += self.Hg.function(temp[:10])
        func += self.Dis.function(temp[10:20])
        func += self.GR.function(temp[20:35])
        func += self.Ras.function(temp[35:])
        return func + 500

    def Info(self):
        return 'Hf04 ' + str(self.Dimension)


class Hf05(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, shuffle=None):
        super().__init__(Low, High, n, coeffi, bias, 1.0, shuffle)
        self.Es = Escaffer6(n=5)
        self.Hg = Hgbat(n=10)
        self.Ros = Rosenbrock(n=10)
        self.Sch = Schwefel(n=10)
        self.Elp = Ellips(n=15)

    def function(self, X):
        temp = (self.decode(X) + 100) / 200
        func = 0
        func += self.Es.function(temp[:5])
        func += self.Hg.function(temp[5:15])
        func += self.Ros.function(temp[15:25])
        func += self.Sch.function(temp[25:35])
        func += self.Elp.function(temp[35:])
        return func - 600

    def Info(self):
        return 'Hf05 ' + str(self.Dimension)


class Hf06(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None, shuffle=None):
        super().__init__(Low, High, n, coeffi, bias, 1.0, shuffle)
        self.Kat = Katsuura(n=5)
        self.HC = HappyCat(n=10)
        self.GR = GrieRosen(n=10)
        self.Sch = Schwefel(n=10)
        self.Ack = Ackley(n=15)

    def function(self, X):
        temp = (self.decode(X) + 100) / 200
        func = 0
        func += self.Kat.function(temp[:5])
        func += self.HC.function(temp[5:15])
        func += self.GR.function(temp[15:25])
        func += self.Sch.function(temp[25:35])
        func += self.Ack.function(temp[35:])
        return func - 2200

    def Info(self):
        return 'Hf06 ' + str(self.Dimension)
