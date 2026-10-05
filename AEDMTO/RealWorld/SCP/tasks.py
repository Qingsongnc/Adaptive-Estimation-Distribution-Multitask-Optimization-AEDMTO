import numpy as np
import scipy.io as sio
import os
from sklearn.metrics import pairwise_distances

"""
# The import format
from RealWorld.SCP.tasks import SCP
"""

class Sensors:
    def __init__(self, A, n, Low=-1, High=1):
        self.A = A
        self.Dimension = n
        self.Low = np.array([-1 if (m % 3 != 2) else 0.1 for m in range(self.Dimension)])
        self.High = np.array([1 if (m % 3 != 2) else 0.25 for m in range(self.Dimension)])

    def decode(self, X):
        X1 = self.Low + (self.High - self.Low) * X[:self.Dimension]
        return X1.reshape(int(self.Dimension / 3 + 0.1), 3)

    def function(self, X):
        """
        function [Objs, Cons] = SCP_func(var, A, dim)
        Objs = [];
        for i = 1:size(var, 1)
            x = var(i, :);
            a = 1000; b = 10; c0 = 1;
            x = x(1:dim);
            k = dim / 3;
            x = reshape(x, 3, k)';
            d = pdist2(A, x(:, 1:2));
            isconverage = (d <= repmat(x(:, 3)', size(A, 1), 1));
            maxisconverage = max(isconverage, [], 2);
            convarage_ratio = sum(maxisconverage) / (size(A, 1));
            f = a * (1 - convarage_ratio) + c0 * k + sum(b * x(:, 3).^2);
            Objs(i, :) = f;
        end
        Cons = zeros(size(var, 1), 1);
        end
        """
        temp = self.decode(X)
        a = 1000
        b = 10
        c0 = 1
        d = np.array(pairwise_distances(self.A, temp[:, :2]))
        iscoverage = (d <= np.tile(temp[:, 2].T, (self.A.shape[0], 1)))
        maxiscoverage = np.max(iscoverage, axis=1)
        covarage_ratio = sum(maxiscoverage) / self.A.shape[0]
        f = a * (1 - covarage_ratio) + c0 * self.Dimension / 3 + sum(b * (temp[:, 2] ** 2))
        return f

    def Info(self):
        return 'SCP ' + str(self.Dimension)

def mat2python(filename):
    path = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(path, filename)
    data = sio.loadmat(file)
    return data['A']


def SCP(Nmin=25, Nmax=35):
    filename = 'SCP_Adata.mat'
    A = mat2python(filename)
    Tasks = []
    for i in range(Nmax - Nmin + 1):
        Task = Sensors(A, (Nmin + i) * 3)
        Tasks.append(Task)
    return Tasks
