import numpy as np
import scipy.io as sio
import os
import scipy.linalg
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
            if flag == 'GO_Task1' or flag == 'GO_Task2':
                data[name] = np.repeat(data[name], 20, axis=None)
            else:
                data[name] = np.random.choice(data[name].flatten(),
                                              size=[data[name].shape[0] * 20, data[name].shape[0] * 20], replace=True) / 20
            parameters.append(data[name])
        else:
            parameters.append(None)
    return parameters


def CI_HS(filename='/Tasks/CI_H.mat'):
    #  Complete Intersection and High Similarity (CI+HS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Griewank(n=1000, coeffi=params[2], bias=params[0])
    Task2 = Rastrigin(n=1000, coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def CI_MS(filename='/Tasks/CI_M.mat'):
    # Complete Intersection and Medium Similarity (CI+MS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Ackley(n=1000, coeffi=params[2], bias=params[0])
    Task2 = Rastrigin(n=1000, coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def CI_LS(filename='/Tasks/CI_L.mat'):
    #  Complete Intersection and Low Similarity
    flags = ['GO_Task1', None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Ackley(n=1000, coeffi=params[2], bias=params[0])
    Task2 = Schwefel(n=1000)
    return [Task1, Task2]


def PI_HS(filename='/Tasks/PI_H.mat'):
    #  Partial Intersection and High Similarity (PI+HS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Rastrigin(n=1000, coeffi=params[2], bias=params[0])
    Task2 = Sphere(n=1000, bias=params[1])
    return [Task1, Task2]


def PI_MS(filename='/Tasks/PI_M.mat'):
    # Partial Intersection and Medium Similarity (PI+MS)
    flags = ['GO_Task1', None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Ackley(n=1000, coeffi=params[2], bias=params[0])
    Task2 = Rosenbrock(n=1000)
    return [Task1, Task2]


def PI_LS(filename='/Tasks/PI_L.mat'):
    # Partial Intersection and Low Similarity (PI+LS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Ackley(n=1000, coeffi=params[2], bias=params[0])
    Task2 = Weierstrass(n=500, coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def NI_HS(filename='/Tasks/NI_H.mat'):
    # No Intersection and High Similarity
    flags = [None, 'GO_Task2', None, 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Rosenbrock(n=1000)
    Task2 = Rastrigin(n=1000, coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def NI_MS(filename='/Tasks/NI_M.mat'):
    # No Intersection and Medium Similarity (NI+MS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Griewank(n=1000, coeffi=params[2], bias=params[0])
    Task2 = Weierstrass(n=1000, coeffi=params[3], bias=params[1])
    return [Task1, Task2]


def NI_LS(filename='/Tasks/NI_L.mat'):
    # No Intersection and Low Similarity (NI+LS)
    flags = ['GO_Task1', None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Rastrigin(n=1000, coeffi=params[2], bias=params[0])
    Task2 = Schwefel(n=1000)
    return [Task1, Task2]
