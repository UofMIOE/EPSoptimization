from math import gamma, pi, sin
import numpy as np
from random import normalvariate, randint, random
from base_algorithm import BaseAlgorithm

class CuckooSearch(BaseAlgorithm):
    """
    Cuckoo Search Optimization
    """

    def __init__(self, function_wrapper, number_of_variables = 1, objective = "maximization"):
        super().__init__(function_wrapper,number_of_variables,objective)

    def search(self, n=100, iteration=100, pa=0.25,
                 nest=100):
        """
        :param n: number of agents
        :param iteration: number of iterations
        :param pa: probability of cuckoo's egg detection (default value is 0.25)
        :param nest: number of nests (default value is 100)

        CONSUMED BY FUNCTION WRAPPER
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes        
        """

        dimension = self.number_of_variables + 1
        super(CuckooSearch, self).__init__(self.function_wrapper, self.number_of_variables, self.objective)

        self.__Positions = []
        self.__Gbest = []
        self.__Nests = []

        #Calculate step size
        beta = 3 / 2
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (
            gamma((1 + beta) / 2) * beta *
            2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.array([normalvariate(0, 1) for k in range(dimension)]) * sigma
        v = np.array([normalvariate(0, 1) for k in range(dimension)])
        step = u / abs(v) ** (1 / beta)


#        self.__agents = list()
#        for j in range(n):
#            oneAgent= list()
#            for i in range(dimension):
#                oneAgent.append(np.random.uniform(function_wrapper.minimum_decision_variable_values()[i], function_wrapper.maximum_decision_variable_values()[i])
#            self.__agents.append(oneAgent)
#        self.__agents = np.array(self.__agents)
#
#        for j in range(nest):
#            oneNest= list()
#            for i in range(dimension):
#                oneNest.append(np.random.uniform(function_wrapper.minimum_decision_variable_values()[i], function_wrapper.maximum_decision_variable_values()[i])
#            self.__nests.append(oneNest)
#        self.__nests = np.array(self.__nests)

        #Initialize nest locations
        self.__agents = np.random.uniform(self.function_wrapper.minimum_decision_variable_values(), self.function_wrapper.maximum_decision_variable_values(), (n, dimension))
        self.__nests = np.random.uniform(self.function_wrapper.minimum_decision_variable_values(), self.function_wrapper.maximum_decision_variable_values(), (nest, dimension))
        Pbest = self.__nests[np.array([self.function_wrapper.objective_function_value(x)
                                       for x in self.__nests]).argmax()]
        Gbest = Pbest
        self._points(self.__agents)

        for t in range(iteration):

            for i in self.__agents:
                val = randint(0, nest - 1)
                if self.function_wrapper.objective_function_value(i) > self.function_wrapper.objective_function_value(self.__nests[val]):
                    self.__nests[val] = i

            fnests = [(self.function_wrapper.objective_function_value(self.__nests[i]), i) for i in range(nest)]
            fnests.sort()
            fcuckoos = [(self.function_wrapper.objective_function_value(self.__agents[i]), i) for i in range(n)]
            fcuckoos.sort(reverse=True)

            nworst = nest // 2
            worst_nests = [fnests[-i - 1][1] for i in range(nworst)]

            for i in worst_nests:
                if random() < pa:
                    self.__nests[i] = np.random.uniform(self.function_wrapper.minimum_decision_variable_values(), self.function_wrapper.maximum_decision_variable_values(), (1, dimension))
#                    oneNest = list()
#                    for j in range(dimension):
#                        oneNest.append(np.random.uniform(function_wrapper.minimum_decision_variable_values()[j], function_wrapper.maximum_decision_variable_values()[j])
#                    oneNest = np.array(oneNest)
#                    self.__nests[i] = oneNest

            if nest < n:
                mworst = n
            else:
                mworst = nest

            for i in range(mworst):

                if fnests[i][0] > fcuckoos[i][0]:
                    self.__agents[fcuckoos[i][1]] = self.__nests[fnests[i][1]]


#            for j in range(nest):
#                for i in range(dimension):
#                    if self.__nests[i][j] < function_wrapper.maximum_decision_variable_values()[i] and self.__nests[i][j] > function_wrapper.minimum_decision_variable_values()[i]
#                        continue
#                    else if self.__nests[i][j] < function_wrapper.minimum_decision_variable_values()[i]
#                        self.__nests[i][j] = function_wrapper.minimum_decision_variable_values()[i]
#                    else 
#                        self.__nests[i][j] = function_wrapper.maximum_decision_variable_values()[i]
                    
            self.__nests = np.clip(self.__nests, self.function_wrapper.minimum_decision_variable_values(), self.function_wrapper.maximum_decision_variable_values())
            self.__Levyfly(step, Pbest, n, dimension)

#            for j in range(n):
#                for i in range(dimension):
#                    if self.__agents[i][j] < function_wrapper.maximum_decision_variable_values()[i] and self.__agents[i][j] > function_wrapper.minimum_decision_variable_values()[i]
#                        continue
#                    else if self.__agents[i][j] < function_wrapper.minimum_decision_variable_values()[i]
#                        self.__agents[i][j] = function_wrapper.minimum_decision_variable_values()[i]
#                    else 
#                        self.__agents[i][j] = function_wrapper.maximum_decision_variable_values()[i]
            self.__agents = np.clip(self.__agents, self.function_wrapper.minimum_decision_variable_values(), self.function_wrapper.maximum_decision_variable_values())
            self._points(self.__agents)
            self.__nest()

            Pbest = self.__nests[np.array([self.function_wrapper.objective_function_value(x)
                                        for x in self.__nests]).argmax()]

            if self.function_wrapper.objective_function_value(Pbest) > self.function_wrapper.objective_function_value(Gbest):
                Gbest = Pbest

        self._set_Gbest(Gbest)

        return {"best_decision_variable_values": self.get_Gbest(), "best_objective_function_value":self.function_wrapper.objective_function_value(self.get_Gbest())}

    def __nest(self):
        self.__Nests.append([list(i) for i in self.__nests])

    def __Levyfly(self, step, Pbest, n, dimension):

        for i in range(n):
            stepsize = 0.2 * step * (self.__agents[i] - Pbest)
            self.__agents[i] += stepsize * np.array([normalvariate(0, 1)
                                                    for k in range(dimension)])

    # def get_nests(self):
    #     """Return a history of cuckoos nests (return type: list)"""

    #     return self.__Nests

    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    def _points(self, agents):
        self.__Positions.append([list(i) for i in agents])

    # def get_agents(self):
    #     """Returns a history of all agents of the algorithm (return type:
    #     list)"""

    #     return self.__Positions

    def get_Gbest(self):
        """Return the best position of algorithm (return type: list)"""

        return list(self.__Gbest)
