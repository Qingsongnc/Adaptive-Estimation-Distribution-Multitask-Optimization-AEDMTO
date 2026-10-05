import numpy as np
import scipy.io as sio
import os
from CEC2019MaTO.functions import Griewank, Rastrigin, Ackley, Schwefel, Rosenbrock, Weierstrass

"""
# The import format
from CEC2019MaTO.tasks import MaTO19
"""
"""
# usage
Tasks = [MaTO19(i) for i in range(6)]
"""

Functions = [Rosenbrock, Ackley, Rastrigin, Griewank, Weierstrass, Schwefel]
def mat2python(filename):
    path = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(path, 'Tasks', filename + '.mat')
    data = sio.loadmat(file)
    return data[filename]

def MaTO19(ProbNum):
    CurFunc = Functions[ProbNum]
    Tasks = []
    RotateMat = 'RotationTask' + str(ProbNum + 1)
    BiasVec = 'GoTask' + str(ProbNum + 1)
    RotateMat = mat2python(RotateMat)[0]
    BiasVec = mat2python(BiasVec)
    for i in range(50):
        Task = CurFunc(coeffi=RotateMat[i], bias=BiasVec[i])
        Tasks.append(Task)
    return Tasks
