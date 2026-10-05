import os
import numpy as np


# The Individual in Multiple Population
class MPIndividual:
    def __init__(self, Task, X=None, **kwargs):
        """
        :param Task: The Task for this individual.
        :param X: Default is None, under this condition, self.X will Generate randomly.
                  if X is specifically set, X will be assigned this parameter.
        After set Task and X, the initialzation function will calculate the function value for paramter Task.
        This class is a base class, which can be inherited to append its own properties
            for different algorithms demands.
        """
        self.Task = Task
        self.Dimension = self.Task.Dimension
        if X is None:
            self.X = np.random.uniform(0., 1., size=self.Dimension)
        else:
            X[X < 0] = 0
            X[X > 1] = 1
            self.X = X[:self.Dimension]
        self.function = self.Task.function(self.X, **kwargs)

    def evaluate(self, **kwargs):
        self.function = self.Task.function(self.X, **kwargs)


# The Population for Multiple Population for a single task in Multi-task
class MPPopulation:
    def __init__(self, Task, IndividualClass=MPIndividual, Num=50, Dimension=None, **kwargs):
        """
        :param Task: The task for this population
        :param IndividualClass: The class of inherited Individual class. It's used to create individuals.
                So it needs the concrete class.
        :param Num: The Population size for this Population
        It's an abstract class.
        The Evolve method should be realized with concrete operation in the inherited class.
        """
        self.Task = Task
        self.Dimension = self.Task.Dimension
        self.Num = Num
        self.Population = [IndividualClass(self.Task, **kwargs) for _ in range(self.Num)]
        self.best = min([Indi.function for Indi in self.Population])
        self.gbest = self.Population[np.argmin([Indi.function for Indi in self.Population])].X.copy()

    def Evolve(self, MPop, **kwargs):
        """
        The Optimization operation for a single Population in a single iteration.
        :param MPop: The Population of MultiPos.
            During evolution process, any evolve operations should call extra MPop.FEIn() followed.
        :return: The FE through this Self Evolve Process
        """
        pass


class MultiPops:
    def __init__(self, Tasks, PopulationClass, iteration=1000, Num=50,
                 UseFE=True, MaxFEs=1e+5, CurveNode=None, **kwargs):
        """
        :param Tasks: The Tasks for the algorithm
        :param PopulationClass: The class of inherited Population class.
                    It's used to create populations for different tasks.
        :param iteration: The number of iterations for this Population
        :param Num: The size of one single Population.
        :param UseFE: Whether to use FE or not.
        :param MaxFEs: The maximum number of FEs allowed for this Population.
        :param CurveNode: Related with the convergence curve. If output, this parameter is number. Otherwise, None.
        It's an abstract class, which contains all the populations for each task.
        The Optimize method should be realized with concrete operation in the inherited class.
        """
        self.Tasks = Tasks
        self.T = len(Tasks)
        self.Ps = [PopulationClass(T, Num, **kwargs) for T in self.Tasks]
        self.FE = 0
        self.UseFE = UseFE
        if self.UseFE:
            self.MaxFEs = MaxFEs
        else:
            self.MaxFEs = Num * self.T * iteration
        self.CurveNode = CurveNode
        if self.CurveNode is not None:
            self.curve = np.zeros((self.T, self.CurveNode + 1))
            self.curve[:, 0] = np.array([pop.best for pop in self.Ps])

    def Optimize(self, **kwargs):
        """
        The Optimization operation for a single iteration. The counting of FE is necessary with function self.FEIn().
        """
        pass

    def FEIn(self):
        """
        Make the function evaluation number +1 after any functions (tasks) called.
        This function is set to realize some extra functions like convergence curve (process) output.
        """
        self.FE += 1
        if self.CurveNode is not None:
            if self.FE % int(self.MaxFEs / self.CurveNode + 0.1) == 0:
                for t in range(self.T):
                    if int(self.FE / (self.MaxFEs / self.CurveNode) + 0.1) > self.CurveNode:
                        break
                    self.curve[t, int(self.FE / (self.MaxFEs / self.CurveNode) + 0.1)] = self.Ps[t].best


    def GetBest(self, scout=False, fout=None):
        """
        Output the best values for all self.T tasks.
        :param scout: Whether to output the value to screen. Default False.
        :param fout: Whether to output the value to a file. Default None, which means no output.
                    fout should be an opened file with write or append mode.
                    example: fout=open(‘filename.txt’, 'w')
        :return: The optimization results for all tasks. An array with self.T values.
        """
        outputs = np.zeros(self.T)
        for i in range(self.T):
            if scout:
                print(i, self.Tasks[i].Info(), ': ', self.Ps[i].best, end=' ')
            outputs[i] = self.Ps[i].best
            if fout is not None:
                fout.write('function ' + str(i) + ' ' + self.Tasks[i].Info() + ': ')
                fout.write(str(self.Ps[i].best) + '\n')
        if scout:
            print()
        return outputs


"""
# The code to realize the algorithm
import numpy as np
from Multi_Population import MPIndividual, MPPopulation, MultiPops, main
from Evolve_Operator import *


class Individual(MPIndividual):
    def __init__(self, Task, X=None, **kwargs):
        super().__init__(Task, X, **kwargs)


# noinspection PyAttributeOutsideInit
class Population(MPPopulation):
    def __init__(self, Task, Num, **kwargs):
        super().__init__(Task, Individual, Num, **kwargs)

    def Evolve(self, MPop, **kwargs):
        # Evolve process for a single population, 
        # MPop is used to increase the function evaluation number with function MPop.FEIn().
        pass


class Algorithm(MultiPops):
    def __init__(self, Tasks, iteration=1000, Num=50, UseFE=True, MaxFEs=1e+5, CurveNode=None, **kwargs):
        super().__init__(Tasks, Population, iteration, Num, UseFE, MaxFEs, CurveNode, **kwargs)

    def Optimize(self, **kwargs):
        # Codes for concrete optimization steps.
        for t in range(self.T):
            self.Ps[t].Evolve(self)


if __name__ == '__main__':
    # Algorithm should be replaced by the Algorithm class name.
    main(iteration=1000, Num=50, time=30, Pop=Algorithm, filename='output', Problems='17', 
         UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50)
    
"""

def TasksIntro(Problems):
    Ps = []
    ProbList = Problems.split('+')
    for Prob in ProbList:
        if Prob == '17':
            from CEC2017.tasks import CI_HS, CI_MS, CI_LS, PI_HS, PI_MS, PI_LS, NI_HS, NI_MS, NI_LS
            np.random.seed(0)
            Ps.extend(
                [
                    CI_HS(),
                    CI_MS(),
                    CI_LS(),
                    PI_HS(),
                    PI_MS(),
                    PI_LS(),
                    NI_HS(),
                    NI_MS(),
                    NI_LS()
                ]
            )
            np.random.seed()
        if Prob == '22':
            from CEC2022.tasks import Benchmark1, Benchmark2, Benchmark3, Benchmark4, Benchmark5, \
                                       Benchmark6, Benchmark7, Benchmark8, Benchmark9, Benchmark10
            Ps.extend(
                [
                    Benchmark1(),
                    Benchmark2(),
                    Benchmark3(),
                    Benchmark4(),
                    Benchmark5(),
                    Benchmark6(),
                    Benchmark7(),
                    Benchmark8(),
                    Benchmark9(),
                    Benchmark10(),
                ]
            )
        if Prob == 'Ma19':
            from CEC2019MaTO.tasks import MaTO19
            for i in range(6):
                Ps.append(MaTO19(i))
        if Prob == 'WCCI20':
            from WCCI2020MaTO.tasks import WCCI2020MaTO
            for i in range(10):
                Ps.append(WCCI2020MaTO(i))
        if Prob == 'PKACP':
            from RealWorld.PKACP.tasks import PKACP
            dims = [20, 25, 30, 35, 40, 50, 100]
            for dim in dims:
                Ps.append(PKACP(dim))
        if Prob == 'SCP':
            from RealWorld.SCP.tasks import SCP
            Ps.append(SCP(Nmin=25, Nmax=35))
    return Ps

def main(iteration=1000, Num=50, time=30, Pop=MultiPops, filename='output', Problems='17',
         UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, **kwargs):
    """
    The main process of an Evolutionary Multitask Algorithm.
    :param iteration: MaxGeneration. Default value is 1000.
    :param Num: The size of one single Population.
    :param time: The repetition time. Default value is 30.
    :param Pop: The Algorithm class Name (In The Enherited class) contains several populations.
    :param filename: The File name to output to.
    :param Problems: The Problem set, Default '17'.
            The values are from ['17', '22', 'PKACP'，‘SCP’, 'Ma19'] and their combination connected with '+',
            For example, Problems='17+22' means tasks contains both 17 and 22.
            The combination could be infinite like '17+22+17+17'
                (But it's not recommended to repeat the same problem sets).
    :param UseFE: Whether using Function Evaluation number for the termination condition or not. Default True.
    :param MaxFEs: If Using FE, The Max Function Evaluation number. Default 1e+5.
    :param OutputCurve: Whether to output the convergence curve data to screen. Default False.
    :param OutputNum: If output curves, how many nodes for each task are to output.
    :param kwargs: The extra parameters set by the inherited class.
                   The parameters should appear in the parameters in the classes.
    :return: None
    """
    f = open(filename + '.txt', 'w', encoding='UTF-8')
    f.close()
    fcsv = open(filename + '.csv', 'w', encoding='UTF-8')
    fcsv.close()
    # Get test suites
    Ps = TasksIntro(Problems)
    # Output arrays initialization
    outputfits = []
    Curves = []
    for index, tasks in enumerate(Ps):
        f = open(filename + '.txt', 'a', encoding='UTF-8')
        f.write('Population ' + str(index + 1) + ':\n')
        f.close()
        Ntasks = len(tasks)
        if UseFE:
            iteration = int(MaxFEs / Num / Ntasks)
        if OutputCurve:
            Curves = np.zeros((Ntasks, OutputNum + 1))
        outputs = np.zeros(Ntasks)
        outputfitt = [np.zeros(time) for _ in range(Ntasks)]
        for t in range(time):
            print('Time ' + str(t + 1) + ' for population ' + str(index + 1) + ':')
            P = Pop(tasks, iteration=iteration, Num=Num, UseFE=UseFE, MaxFEs=MaxFEs,
                    CurveNode=OutputNum if OutputCurve else None, **kwargs)
            for Nt, task in enumerate(P.Tasks):
                print('Task ' + str(Nt), task.Info(), end=' ')
            print()
            itrt = 0
            while (UseFE and P.FE < P.MaxFEs) or (not UseFE and itrt < iteration):
                itrt += 1
                P.Optimize()
                if iteration >= 20 and itrt % int(iteration / 20) == 0:
                    # Print results for approximate 20 times through all iterations or FEs
                    print('Iteration', str(itrt), ':', end=' ')
                    P.GetBest(scout=True)
            f = open(filename + '.txt', 'a', encoding='UTF-8')
            f.write('Time' + str(t + 1) + ':\n')
            ret = P.GetBest(scout=True, fout=f)
            f.close()
            for Nt in range(Ntasks): outputfitt[Nt][t] = ret[Nt]
            outputs += ret
            if OutputCurve:
                Curves += P.curve
        # Output result for tasks
        outputs /= time
        fcsv = open(filename + '.csv', 'a', encoding='UTF-8')
        for i in range(len(outputs)):
            fcsv.write(str(outputs[i]) + ',')
        fcsv.write('\n')
        fcsv.close()
        outputfits.extend(outputfitt)
        # Output curve data
        if OutputCurve:
            Curves /= time
            for ti in range(Ntasks):
                curvefilename = 'Curve' + filename
                os.makedirs(curvefilename, exist_ok=True)
                curvefilename = os.path.join(curvefilename, 'T' + str(index) + 't' + str(ti) + '.csv')
                np.savetxt(curvefilename, Curves[ti].reshape(1, OutputNum + 1), fmt='%.15g', delimiter=',')
        f = open(filename + '.txt', 'a', encoding='UTF-8')
        f.write('\n')
        f.close()
        # Output results
        np.savetxt(filename + '_fit.csv', np.array(outputfits).T, fmt='%.15g', delimiter=',')
