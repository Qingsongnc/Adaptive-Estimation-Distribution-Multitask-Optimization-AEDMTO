import numpy as np
import scipy.io as sio
import os
from CEC2017.functions import Griewank, Rastrigin, Ackley, Sphere, Schwefel, Rosenbrock, Weierstrass

"""
The import format
from CEC2017.tasks import CI_HS, CI_MS, CI_LS, PI_HS, PI_MS, PI_LS, NI_HS, NI_MS, NI_LS
"""


# all nine tasks
def mat2python(filename, flags):
    path = os.path.abspath(os.path.dirname(__file__))
    file = path + filename
    data = sio.loadmat(file)
    names = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    parameters = []
    for i, flag in enumerate(flags):
        if flag is not None:
            name = names[i]
            parameters.append(data[name])
        else:
            parameters.append(None)
    return parameters


def CI_HS(filename='/Tasks/CI_H.mat'):
    #  Complete Intersection and High Similarity (CI+HS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Griewank(coeffi=params[2], bias=params[0])
    Task2 = Rastrigin(coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def CI_MS(filename='/Tasks/CI_M.mat'):
    # Complete Intersection and Medium Similarity (CI+MS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Ackley(coeffi=params[2], bias=params[0])
    Task2 = Rastrigin(coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def CI_LS(filename='/Tasks/CI_L.mat'):
    #  Complete Intersection and Low Similarity
    flags = ['GO_Task1', None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Ackley(coeffi=params[2], bias=params[0])
    Task2 = Schwefel()
    return [Task1, Task2]


def PI_HS(filename='/Tasks/PI_H.mat'):
    #  Partial Intersection and High Similarity (PI+HS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Rastrigin(coeffi=params[2], bias=params[0])
    Task2 = Sphere(bias=params[1])
    return [Task1, Task2]


def PI_MS(filename='/Tasks/PI_M.mat'):
    # Partial Intersection and Medium Similarity (PI+MS)
    flags = ['GO_Task1', None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Ackley(coeffi=params[2], bias=params[0])
    Task2 = Rosenbrock()
    return [Task1, Task2]


def PI_LS(filename='/Tasks/PI_L.mat'):
    # Partial Intersection and Low Similarity (PI+LS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Ackley(coeffi=params[2], bias=params[0])
    Task2 = Weierstrass(coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def NI_HS(filename='/Tasks/NI_H.mat'):
    # No Intersection and High Similarity
    flags = [None, 'GO_Task2', None, 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Rosenbrock()
    Task2 = Rastrigin(coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def NI_MS(filename='/Tasks/NI_M.mat'):
    # No Intersection and Medium Similarity (NI+MS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Griewank(coeffi=params[2], bias=params[0])
    Task2 = Weierstrass(n=50, coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def NI_LS(filename='/Tasks/NI_L.mat'):
    # No Intersection and Low Similarity (NI+LS)
    flags = ['GO_Task1', None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Rastrigin(coeffi=params[2], bias=params[0])
    Task2 = Schwefel()
    return [Task1, Task2]
