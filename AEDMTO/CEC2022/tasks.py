from CEC2022.functions import Weierstrass, Griewank, HappyCat, GrieRosen, Ackley, Schwefel, Escaffer6, \
    Hf01, Hf04, Hf05, Hf06
import numpy as np
import os

"""
The import format
from CEC2022.tasks import Benchmark1, Benchmark2, Benchmark3, Benchmark4, Benchmark5, Benchmark6, Benchmark7, \
    Benchmark8, Benchmark9, Benchmark10
"""


def GetMatrixs(filepath):
    path = os.path.abspath(os.path.dirname(__file__))
    filepath = path + filepath
    bias1 = np.loadtxt(filepath + 'bias_1')
    bias2 = np.loadtxt(filepath + 'bias_2')
    matrix1 = np.loadtxt(filepath + 'matrix_1')
    matrix2 = np.loadtxt(filepath + 'matrix_2')
    return {'bias1': bias1, 'bias2': bias2, 'ma1': matrix1, 'ma2': matrix2}


def GetShuffledata(index):
    filepath = '/Tasks/shuffle/shuffle_data_' + str(index) + '_D50.txt'
    path = os.path.abspath(os.path.dirname(__file__))
    filepath = path + filepath
    shuffle_data = np.loadtxt(filepath, dtype=int) - 1
    return shuffle_data


def Benchmark1(filename='/Tasks/benchmark_1/'):
    params = GetMatrixs(filename)
    Task1 = Weierstrass(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Weierstrass(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark2(filename='/Tasks/benchmark_2/'):
    params = GetMatrixs(filename)
    Task1 = Griewank(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Griewank(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark3(filename='/Tasks/benchmark_3/'):
    params = GetMatrixs(filename)
    shuffle = GetShuffledata(17)
    Task1 = Hf01(coeffi=params['ma1'], bias=params['bias1'], shuffle=shuffle)
    Task2 = Hf01(coeffi=params['ma2'], bias=params['bias2'], shuffle=shuffle)
    return [Task1, Task2]


def Benchmark4(filename='/Tasks/benchmark_4/'):
    params = GetMatrixs(filename)
    Task1 = HappyCat(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = HappyCat(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark5(filename='/Tasks/benchmark_5/'):
    params = GetMatrixs(filename)
    Task1 = GrieRosen(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = GrieRosen(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark6(filename='/Tasks/benchmark_6/'):
    params = GetMatrixs(filename)
    shuffle = GetShuffledata(21)
    Task1 = Hf05(coeffi=params['ma1'], bias=params['bias1'], shuffle=shuffle)
    Task2 = Hf05(coeffi=params['ma2'], bias=params['bias2'], shuffle=shuffle)
    return [Task1, Task2]


def Benchmark7(filename='/Tasks/benchmark_7/'):
    params = GetMatrixs(filename)
    shuffle = GetShuffledata(22)
    Task1 = Hf06(coeffi=params['ma1'], bias=params['bias1'], shuffle=shuffle)
    Task2 = Hf06(coeffi=params['ma2'], bias=params['bias2'], shuffle=shuffle)
    return [Task1, Task2]


def Benchmark8(filename='/Tasks/benchmark_8/'):
    params = GetMatrixs(filename)
    Task1 = Ackley(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Ackley(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark9(filename='/Tasks/benchmark_9/'):
    params = GetMatrixs(filename)
    Task1 = Schwefel(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Escaffer6(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark10(filename='/Tasks/benchmark_10/'):
    params = GetMatrixs(filename)
    shuffle1 = GetShuffledata(20)
    shuffle2 = GetShuffledata(21)
    Task1 = Hf04(coeffi=params['ma1'], bias=params['bias1'], shuffle=shuffle1)
    Task2 = Hf05(coeffi=params['ma2'], bias=params['bias2'], shuffle=shuffle2)
    return [Task1, Task2]
