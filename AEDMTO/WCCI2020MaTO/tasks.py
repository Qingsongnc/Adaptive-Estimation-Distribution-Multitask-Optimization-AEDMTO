import numpy as np
import os
from WCCI2020MaTO.functions import Griewank, Rastrigin, Ackley, Sphere, Schwefel, Rosenbrock, Weierstrass

"""
# The import format
from WCCI2020MaTO.tasks import WCCI2020MaTO
"""
"""
# usage
Tasks = [WCCI2020MaTO for i in range(10)]
"""

functions = [Sphere, Rosenbrock, Ackley, Rastrigin, Griewank, Weierstrass, Schwefel]
FuncNums = [
    [0], [1], [3],
    [0, 1, 2], [3, 4, 5], [1, 4, 6], [2, 3, 5],
    [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6]
]
def loadfile(ProbNum, FuncNum):
    path = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(path, 'Tasks', 'benchmark_' + str(ProbNum))
    matrix = np.loadtxt(os.path.join(file, 'matrix_' + str(FuncNum)), dtype=float, delimiter=None)
    bias = np.loadtxt(os.path.join(file, 'bias_' + str(FuncNum)), dtype=float, delimiter=None)
    return matrix, bias

def WCCI2020MaTO(ProbNum):
    Tasks = []
    FuncList = FuncNums[ProbNum]
    for i in range(50):
        matrix, bias = loadfile(ProbNum + 1, i + 1)
        function = functions[FuncList[i % len(FuncList)]]
        Tasks.append(function(coeffi=matrix, bias=bias))
    return Tasks
