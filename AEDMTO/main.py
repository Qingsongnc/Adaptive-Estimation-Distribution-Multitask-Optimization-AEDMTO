import numpy as np
import scipy.stats
import os
os.environ["OMP_NUM_THREADS"] = '1'
from Multi_Population import MPIndividual, MPPopulation, MultiPops, main
from Evolve_Operator import binomial_crossover, Roulette_Wheel_Selection, Elitist_Selection, GauMutate
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class Individual(MPIndividual):
    def __init__(self, Task, X=None, **kwargs):
        super().__init__(Task, X, **kwargs)


class Population(MPPopulation):
    def __init__(self, Task, Num, F=0.5, CR=0.9, nt=50, ptsf_p=None, **kwargs):
        super().__init__(Task, Individual, Num, **kwargs)
        self.F = F
        self.CR = CR
        self.nt = nt

        # self.cntr = None
        # self.std = None
        # # uniform
        # self.lb = None
        # self.hb = None
        # # skewed
        # self.SK = None
        # # mixture
        # self.mixK = 5
        # self.cntrK = np.zeros((self.mixK, self.Dimension))
        # self.stdK = np.zeros((self.mixK, self.Dimension))
        # self.nK = np.zeros((self.mixK, self.Dimension))

        # self.clusters = None
        # self.B = 3

        self.ptsf = ptsf_p
        self.shuffle = None
        self.Distrib()

    def Evolve(self, MPop, **kwargs):
        # The Concrete intraSE Process. FE number should be concerned.
        Num = len(self.Population)
        for i in range(Num):
            rands = np.random.choice(range(0, Num - 1), 3, replace=False)
            rands[rands >= i] += 1
            V = self.Population[rands[0]].X + self.F * (self.Population[rands[1]].X - self.Population[rands[2]].X)
            # Binomial crossover
            U = binomial_crossover(V, self.Population[i].X, self.CR)
            offspring = Individual(self.Task, U)
            self.Population.append(offspring)
            # Update best
            if offspring.function < self.best:
                self.best = offspring.function
                self.gbest = offspring.X.copy()
            MPop.FEIn()
        # Eltist Selection
        self.Population = Elitist_Selection(self.Population, self.Num)

    def Distrib(self):
        Xs = np.array([Indi.X for Indi in self.Population])
        # Gaussian
        self.cntr = np.mean(Xs, axis=0)
        self.std = np.std(Xs, axis=0)
        # # Uniform
        # self.lb = np.min(Xs, axis=0)
        # self.hb = np.max(Xs, axis=0)
        # # Skewed
        # self.SK = np.mean(((Xs - self.cntr) ** 3) / (self.std ** 3 + 1e+6), axis=0)
        # # Mixture
        # self.nK = np.zeros((self.mixK, self.Dimension))
        # pca = PCA(n_components=3)
        # Xs1 = pca.fit_transform(Xs)
        # kmeans = KMeans(n_clusters=self.mixK, random_state=0)
        # lbs = kmeans.fit_predict(Xs1)
        # for i in range(self.mixK):
        #     Xsk = [Xs[j] for j in range(self.Num) if lbs[j] == i]
        #     self.nK[i] = len(Xsk)
        #     self.cntrK[i, :] = np.mean(Xsk, axis=0)
        #     self.stdK[i, :] = np.std(Xsk, axis=0)
        # Xs = Xs.T
        # Xs.sort(axis=1)
        # Xs = Xs.reshape([self.Dimension, self.mixK, int(self.Num / self.mixK + 0.1)])
        # self.cntrK = Xs.mean(axis=2).T
        # self.stdK = Xs.std(axis=2).T

    def ComMemFunc(self, source):
        # Shuffle Dimension
        self.Distrib()
        shuffle = np.arange(source.Dimension)

        if self.Dimension <= source.Dimension:
            shuffle = np.random.choice(shuffle, self.Dimension, replace=False)
        else:
            while shuffle.shape[0] < self.Dimension:
                shf_add = np.random.choice(shuffle, source.Dimension, replace=False)
                shuffle = np.append(shuffle, shf_add)
            shuffle = shuffle[:self.Dimension]
            np.random.shuffle(shuffle)

        # # Repeatedly select dimensions
        # shuffle = np.random.choice(shuffle, self.Dimension, replace=True)

        # # Dynamic based on distribution
        # # for i in range(self.Dimension):
        # #     probs = np.log(source.std / (self.std[i] + 1e-6)) - 1.0 / 2 + ((self.std[i] * self.std[i]) + ((
        # #         source.cntr - self.cntr[i]) * (source.cntr - self.cntr[i]))) / (2 * source.std * source.std + 1e-6)
        # #     probs = 1 / (probs + 1e-6)
        # #     shuffle[i], _  = Roulette_Wheel_Selection(probs)
        # for i in range(self.Dimension):
        #     probs = np.exp(-(source.gbest - self.cntr[i]) * (source.gbest - self.cntr[i]) / (
        #         2 * self.std[i] * self.std[i] + 1e-6))
        #     shuffle[i], _  = Roulette_Wheel_Selection(probs + 1e-6)

        # Index-align
        # if self.shuffle is None:
        #     self.shuffle = shuffle
        # else:
        #     shuffle = self.shuffle
        # shuffle = np.arange(source.Dimension)[:self.Dimension]
        # if len(shuffle) != self.Dimension:
        #     shuffle = np.append(shuffle, shuffle[:self.Dimension - len(shuffle)])

        # New Scheme
        # GetMembership
        # Gaussian
        memFunc = np.array([np.mean(np.exp(-(Indi.X[shuffle] - self.cntr) * (Indi.X[shuffle] - self.cntr) / (
                2 * self.std * self.std + 1e-6))) for Indi in source.Population])
        # # Uniform
        # memFunc = np.array([np.mean(np.array((self.lb < Indi.X[shuffle]) * (Indi.X[shuffle] < self.hb)))
        #                     for Indi in source.Population])
        # # Skewed
        # memFunc = np.array([np.mean(self.std * self.std * np.sqrt(np.pi / 2) * np.mean(scipy.stats.skewnorm.pdf(
        #     Indi.X[shuffle], self.SK, loc=self.cntr, scale=self.std))) for Indi in source.Population])
        # memFunc = np.array([np.mean(2 / self.std * np.mean(scipy.stats.skewnorm.pdf(
        #     Indi.X[shuffle], self.SK, loc=self.cntr, scale=self.std))) for Indi in source.Population])
        # # Mixture
        # memFunc = np.array([np.mean(self.nK * self.mixK / self.Num * np.exp(-(Indi.X[shuffle] - self.cntrK) * (Indi.X[shuffle] - self.cntrK) / (
        #         2 * self.stdK * self.stdK + 1e-6))) for Indi in source.Population])
        # # Cauchy
        # memFunc = np.array([np.mean(1 / (1 + (((Indi.X[shuffle] - self.cntr) / (
        #         self.std + 1e-6)) ** 2))) for Indi in source.Population])
        
        # Transfer Probability
        memFunc1 = memFunc.copy()
        memFunc1.sort()
        memFunc1 = memFunc1[::-1]

        # Exponential Weight
        ptsf = np.sum((0.5 ** np.arange(1, source.Num + 1)) * memFunc1)
        # # Linear Weight
        # ptsf = np.sum((2 * np.arange(1, source.Num + 1) / (self.Num * (self.Num - 1))) * memFunc1)
        # # Log Weight
        # ptsf = np.sum((np.log(np.arange(1, source.Num + 1)) / (np.sum(np.log(np.arange(1, source.Num + 1))))) * memFunc1)

        # if self.shuffle is None:
        #     self.shuffle = shuffle
        # # Primary Scheme
        # # GetMembership
        # memFunc2 = np.array([np.mean(np.exp(-(Indi.X[self.shuffle] - self.cntr) * (Indi.X[self.shuffle] - self.cntr) / (
        #         2 * self.std * self.std + 1e-6))) for Indi in source.Population])
        # # Transfer Probability
        # memFunc3 = memFunc2.copy()
        # memFunc2.sort()
        # ptsf1 = np.sum((0.5 ** np.arange(1, source.Num + 1)) * memFunc2)
        #
        # # True Dynamic
        # # if ptsf > ptsf1:
        # if np.random.random() < ptsf / (ptsf + ptsf1):
        #     self.shuffle = shuffle
        # else:
        #     ptsf = ptsf1
        #     shuffle = self.shuffle
        #     memFunc = memFunc2

        return memFunc, ptsf, shuffle

    def Transfer(self, MPop, source, cur=None):
        memFunc, ptsf, shuffle = self.ComMemFunc(source)
        # MPop.CurPtsf[cur] = ptsf
        # Fixed ptsf
        if self.ptsf is not None:
            ptsf = self.ptsf
        # # Randomly select individuals
        # memFunc = np.zeros(self.Dimension) + 1

        # Knowledge Transfer Process
        if np.random.random() < ptsf:
            Trsindexs, _ = Roulette_Wheel_Selection(memFunc, Num=self.nt)
            indexs = np.random.choice(range(0, self.Num), self.nt, replace=False)
            for i in range(0, self.nt):
                U = binomial_crossover(source.Population[Trsindexs[i]].X[shuffle],
                                       self.Population[indexs[i]].X)
                offspring = Individual(self.Task, U)
                if offspring.function <= self.best:
                    self.best = offspring.function
                    self.gbest = offspring.X.copy()
                self.Population.append(offspring)
                MPop.FEIn()
            self.Population = Elitist_Selection(self.Population, self.Num)

    def ELS(self, MPop, sigma=None):
        ElsIn = GauMutate(self.Population[0], sigma)
        ELS = Individual(self.Task, ElsIn)
        if ELS.function < self.Population[0].function:
            self.Population[0] = ELS
            if ELS.function < self.best:
                self.best = ELS.function
                self.gbest = ELS.X.copy()
        else:
            self.Population[-1] = ELS
        MPop.FEIn()


class AEDMTO(MultiPops):
    def __init__(self, Tasks, iteration=1000, Num=50, UseFE=True, MaxFEs=1e+5, CurveNode=None, F=0.5, CR=0.9, **kwargs):
        super().__init__(Tasks, Population, iteration, Num, UseFE, MaxFEs, CurveNode, **kwargs)
        self.F = F
        self.CR = CR
        self.sigmamin = 0.05
        self.sigmamax = 0.5
        # # For f_KT Curves
        # self.CurPtsf = [self.Ps[0].ComMemFunc(self.Ps[1])[1], self.Ps[1].ComMemFunc(self.Ps[0])[1]]
        # if self.CurveNode is not None:
        #     self.curve[:, 0] = self.CurPtsf

    def FEIn(self):
        self.FE += 1
        if self.CurveNode is not None:
            if self.FE % int(self.MaxFEs / self.CurveNode + 0.1) == 0:
                for t in range(self.T):
                    if int(self.FE / (self.MaxFEs / self.CurveNode) + 0.1) > self.CurveNode:
                        break
                    self.curve[t, int(self.FE / (self.MaxFEs / self.CurveNode) + 0.1)] = self.Ps[t].best
                    # # For f_KT Curves
                    # self.curve[t, int(self.FE / (self.MaxFEs / self.CurveNode) + 0.1)] = self.CurPtsf[t]

    def Optimize(self, **kwargs):
        for t in range(self.T):
            self.Ps[t].Evolve(self)
            # The Knowledge Transfer Process. The FE number should be concerned
            j = np.random.randint(0, self.T - 1)
            if j >= t:
                j += 1
            # For f_KT Curves
            # self.CurPtsf[t] = self.Ps[t].Transfer(self, self.Ps[j])
            self.Ps[t].Transfer(self, self.Ps[j], t)
            # ELS Process
            sigma = self.sigmamax - (self.sigmamax - self.sigmamin) * self.FE / self.MaxFEs
            sigma = max(sigma, self.sigmamin)
            self.Ps[t].ELS(self, sigma)


if __name__ == '__main__':
    # Algorithm should be replaced by the Algorithm class name.
    # main(iteration=1000, Num=50, time=30, Pop=AEDMTO, filename='outputMaTO19', Problems='Ma19',
    #      UseFE=True, MaxFEs=2.5e+6, OutputCurve=False, OutputNum=50)
    # main(iteration=1000, Num=50, time=30, Pop=AEDMTO, filename='outputSCPT11', Problems='SCP',
    #      UseFE=True, MaxFEs=5.5e+5, OutputCurve=False, OutputNum=50)
    main(iteration=1000, Num=50, time=30, Pop=AEDMTO, filename='output', Problems='17+22',
         UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50)
    # for nt in range(0, 30, 10):
    #     # Algorithm should be replaced by the Algorithm class name.
    #     main(iteration=1000, Num=50, time=30, Pop=AEDMTO, filename='Differentnt/nt'+str(nt), Problems='17',
    #          UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, nt=nt)

    # parser = argparse.ArgumentParser(description='parameters')
    # parser.add_argument("--ptsf", type=float, default=None, help="KT_frequency")
    # args = parser.parse_args()
    # ptsf_p = args.ptsf
    # # Algorithm should be replaced by the Algorithm class name.
    # main(iteration=1000, Num=50, time=30, Pop=AEDMTO, filename='DifferentPtsf/ptsf{}'.format(ptsf_p), Problems='17',
    #      UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, ptsf_p=ptsf_p)
    # # for CR in [0.3,0.5,0.7,0.9]:
    #     # Algorithm should be replaced by the Algorithm class name.
    #     main(iteration=1000, Num=50, time=30, Pop=AEDMTO, filename='DifferentCR/CR'+str(CR), Problems='17',
    #          UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, CR=CR)
    # for F in [0.5,0.7,0.9]:
    #     # Algorithm should be replaced by the Algorithm class name.
    #     main(iteration=1000, Num=50, time=30, Pop=AEDMTO, filename='DifferentF/F'+str(F), Problems='17',
    #          UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, F=F)
