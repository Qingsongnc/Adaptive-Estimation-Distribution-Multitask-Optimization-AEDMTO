import numpy as np


# Crossover
def SBX_Crossover(P1, P2, ita=2.):
    """
    Simulated Binary Crossover Operator
    Generate a random number sequence r[Dimension] in interval [0, 1],
    Then calculate the coefficiency beta,
        beta = (2r)^[1/(ita+1)]     , if r <=0.5
        beta = (2-2r)^[-1/(ita+1)], if r > 0.5
    Finally, C1 = (1 + beta)P1 + (1 - beta)P2
             C2 = (1 + beta)P2 + (1 - beta)P1
    :param P1: The first parent.
    :param P2: The second parent.
    :param ita: The parameter used in SBX Crossover
    :return: Two Generated offspring
    """
    temp1, temp2 = P1.copy(), P2.copy()
    Dimension = max(len(temp1), len(temp2))
    r = np.random.random(Dimension)
    beta = np.zeros(Dimension)
    beta[r <= 0.5] = (2 * r[r <= 0.5]) ** (1 / (1 + ita))
    beta[r > 0.5] = (1 / (2 - 2 * r[r > 0.5])) ** (1 / (1 + ita))
    temp1 = ((1 + beta) * temp1 + (1 - beta) * temp2) / 2
    temp2 = ((1 - beta) * temp1 + (1 + beta) * temp2) / 2
    return temp1, temp2


def binomial_crossover(v, x, CR=None):
    """
    :param v: Differential vector
    :param x: The previous vector
    :param CR: Cross Probability
    :return: The vector after crossover operation.
    """
    if CR is None:
        CR = np.random.uniform(0.1, 0.9)
    jr = np.random.randint(x.size)
    rands = np.random.random(x.size)
    rands[rands > CR] = 1
    rands[rands <= CR] = 0
    U = rands * x + (1 - rands) * v
    U[jr] = v[jr]
    return U


def binary_crossover(P1, P2):
    """
    :param P1: The first parent.
    :param P2: The second parent.
    :return: The vector after crossover operation.
    """
    rands = np.random.random(P1.size)
    rands[rands > 0.5] = 1
    rands[rands <= 0.5] = 0
    U = rands * P1 + (1 - rands) * P2
    return U


# Mutation
def PolyMutate(P, itam=5.):
    """
    Polynomial Mutation Operator
    Generate a random number sequence r[Dimension] in interval[0, 1],
    Then calculate the coefficiency beta,
        beta = (2r)^[1 / (itam + 1)] - 1      , r <= 0.5
        beta = 1 - (2 - 2r)^[1 / (itam + 1)]  , r >  0.5
    Finally
        P = P + beta * P       , r <= 0.5
        P = P + beta * (1 - P) , r > 0.5
    Another Version, the mutate operation was like:
        P = P + beta
    :param P: A Individual
    :param itam: The parameter used in polymutation, default 5.
    :return: The individual mutated
    """
    temp = P.copy()
    for i in range(len(temp)):
        if np.random.random() < (1 / len(temp)):
            r = np.random.random()
            # One version
            if r <= 0.5:
                temp[i] += ((2 * r) ** (1 / (1 + itam)) - 1) * temp[i]
            else:
                temp[i] += (1 - (2 * (1 - r)) ** (1 / (1 + itam))) * (1 - temp[i])
            # Another version
            # if r <= 0.5:
            #     temp[i] += ((2 * r) ** (1 / (1 + itam)) - 1)
            # else:
            #     temp[i] += (1 - (2 * (1 - r)) ** (1 / (1 + itam)))
    return temp


def GauMutate(P, deviation, index=None):
    """
    Gaussian Mutation Operator
    :param P: A Individual
    :param deviation: The standard deviation for Guassian Mutation Operator
    :param index: A preset index to be mutated
    :return: The individual mutated
    """
    temp = P.X.copy()
    if index is None:
        index = np.random.randint(len(temp))
    temp[index] += np.random.normal(scale=deviation)
    return temp


# Selection
def Roulette_Wheel_Selection(P, key=lambda x: x, Num=1):
    """
    Simulate the roulette wheel to select an item. Each item in the wheel has its own area,
    which in algorithms usually stands for the probability.
    :param P: The items or probabilities.
    :param key: A lambda expression, which will be extracted and normalized first to use as selection probabilities p.
                After this procedure, the array p satisties p.sum() == 1.
                The key usually stands for a value related to function fitness in evolutionary process.
    :param Num: The Number to be selected.
    :return: The indexes and items selected in this function. If any selection occurs an error,
             this function will return two empty lists. If Num is 1, return a number and an item instead of two lists.
    """
    p = np.array([key(indi) for indi in P])
    p = p / p.sum()
    indexs = []
    for n in range(Num):
        r = np.random.random()
        m = 0
        for i in range(p.size):
            m += p[i]
            if r <= m:
                indexs.append(i)
                break
        if len(indexs) != n + 1:
            indexs.append(n - 1)
    if Num == 1:
        return indexs[0], P[indexs[0]]
    return indexs, P[indexs]


def Elitist_Selection(P, Num, key=lambda x: x.function, DESC=False):
    """
    Sort all the item by the key, and then select the top Num items.
    :param P: The item.
    :param Num: The number of items to be kept.
    :param key: The key used to sort items. Commonly set as lambda x: x.function
    :param DESC: The method to sort, the default set is ASC(False), from low to high.
    :return: The items kept through this procedure.
    """
    P.sort(key=key, reverse=DESC)
    if Num > len(P):
        Num = len(P)
    return P[0: Num]


"""
# Follows are the codes for some modules which can be directly copied to different algorithms

# Differential Evolution. Directly put it into class.
# futsu no DE Strategy
def Differential_Evolve(MPop):
    Num = len(self.Population)
    for i in range(Num):
        rands = np.random.choice(range(0, Num - 1), 3, replace=False)
        rands[rands >= i] += 1
        V = self.Population[rands[0]].X + self.F * (self.Population[rands[1]].X - self.Population[rands[2]].X)
        # Binomial crossover
        U = binomial_crossover(V, self.Population[i].X, self.CR)
        offspring = Individual(self.Task, U)
        # Selection
        if offspring.function < self.Population[i].function:
            self.Population[i] = offspring
        # Update best
        if offspring.function < self.best:
            self.best = offspring.function
            self.gbest = offspring.X.copy()
        MPop.FEIn()

# DE with Elitist Selection Strategy
def Differential_Evolve(MPop):
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
    self.Population = Elitist_Selection(self.Population, lambda x: 1 / x.function, self.Num)

# GA algorithm
def GA(MPop):
    np.random.shuffle(self.Population)
        for i in range(int(self.Num / 2)):
            c1, c2 = SBX_Crossover(self.Population[i].X, self.Population[int(self.Num / 2) + i].X)
            c1 = PolyMutate(c1)
            c2 = PolyMutate(c2)
            self.Population.append(Individual(self.Task, c1))
            MPop.FEIn()
            self.Population.append(Individual(self.Task, c2))
            MPop.FEIn()
        self.Population = Elitist_Selection(self.Population, lambda x: x.function, self.Num, False)
        if self.Population[0].function < self.best:
            self.best = self.Population[0].function
            self.gbest = self.Population[0].X.copy()

# PSO
def PSO(MPop):
    # Remember to append velocity, pbest, pbestval to Individual class
    #    self.velocity = np.random.uniform(-0.2, 0.2, self.Dimension)
    #    self.pbest = self.X.copy()
    #    self.pbestval = self.function
    # Remember to append w, c1, c2 to Population class
    #    self.w = 0.8
    #    self.c1 = 2.0
    #    self.c2 = 2.0
    for i in range(self.Num):
        v = (
                self.w * self.Population[i].velocity +
                self.c1 * np.random.random() * (self.Population[i].pbest - self.Population[i].X) +
                self.c2 * np.random.random() * (self.gbest - self.Population[i].X)
            )
        v[v > 0.2] = 0.2
        v[v < -0.2] = -0.2
        self.Population[i].velocity = v
        self.Population[i].X = self.Population[i].X + v
        self.Population[i].X[self.Population[i].X > 1] = 1
        self.Population[i].X[self.Population[i].X < 0] = 0
        func = self.Task.function(self.Population[i].X)
        self.Population[i].function = func
        if func < self.best:
            self.best = func
            self.gbest = self.Population[i].X.copy()
        if func < self.Population[i].pbestval:
            self.Population[i].pbest = self.Population[i].X.copy()
            self.Population[i].pbestval = func
        MPop.FEIn()

# Elitist Learning Strategy
# Mutate the best Individual with a zero mean sigma std Gaussian Mutate.
# Then compare it with the best. If better, replace the best one. Otherwise, replace the worst one.
def Elitist_Learning_Strategy(self, MPop, sigma):
    ElsIn = GauMutate(self.Population[0], sigma)
    ELS = Individual(self.Task, ElsIn)
    MPop.FEIn()
    if ELS.function < self.Population[0].function:
        self.Population[0] = ELS
    else:
        self.Population[-1] = ELS



# The main function for CEC2017+CEC2022 (Old Version Abandoned)
if __name__ == '__main__':
    filename = 'output'
    f = open(filenam + '.txt', 'w', encoding='UTF-8')
    iteration = 1000
    time = 20
    # For Single Population, Pop should be replaced by the Population class name
    # For Multi Population, Pop should be replaced by the inherited MultiPop class
    Ps = [
        CI_HS(),
        CI_MS(),
        CI_LS(),
        PI_HS(),
        PI_MS(),
        PI_LS(),
        NI_HS(),
        NI_MS(),
        NI_LS(),
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
    outputs = np.zeros((len(Ps), 2))
    for t in range(1, time + 1):
        f = open(filename + '.txt', 'a', encoding='UTF-8')
        f.write('time' + str(t) + ':\n')
        print('time ' + str(t) + ':')
        for index, tasks in enumerate(Ps):
            # For Single Population, Pop should be replaced by the Population class name
            # For Multi Population, Pop should be replaced by the inherited MultiPop class
            P = Pop(tasks)
            print('Population ' + str(index + 1))
            for ti, task in enumerate(P.Tasks):
                print('Task ' + str(ti), task.Info(), end=' ')
            print()
            for iter in range(1, iteration + 1):
                P.Optimize()  # for Multi Population
                # P.Evolve()    # for Single Population
                if iter % int(iteration / 20) == 0:
                    print('Iteration', str(iter), ':', end=' ')
                    P.GetBest()
            outputs[index] += P.GetBest(f)
        f.close()
    outputs /= time
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            print(outputs[i, j], end=',')
        print()
    fcsv = open(filename + '.csv', 'w', encoding='UTF-8')
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            fcsv.write(str(outputs[i, j]) + ',')
        fcsv.write('\n')
    fcsv.close()
"""