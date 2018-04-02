from base_algorithm import BaseAlgorithm
from random import random
from math import exp, pi, sqrt, fabs
import numpy as np

class AntColony(BaseAlgorithm):

	class Ant(object):
		def __init__(self, function_wrapper, location_coordinates, objective):
			self.function_wrapper = function_wrapper
			self.location_coordinates = location_coordinates
			self.objective = objective
			self.probability = 0
			self.weight = 0
			self.update_fitness_value()

		def update_fitness_value(self):
			self.fitness = self.function_wrapper.objective_function_value(self.location_coordinates)
	

	def __init__(self, function_wrapper, number_of_variables = 1, objective = "maximization"):
		super().__init__(function_wrapper, number_of_variables, objective)


	def search(self, archive_size = 10, number_of_new_solution = 5, evaporation_rate = 0.9, number_of_iteration_before_stop = 100):
		self.archive_size = archive_size;
		self.number_of_new_solution = number_of_new_solution
		self.evaporation_rate = evaporation_rate
		self.number_of_iteration_before_stop = number_of_iteration_before_stop

		self.__initialize_ants()

		self.__sort_ant_by_fitness(self.__ants)
		self.__update_probability(self.__ants)

		iteration = 1
		while iteration < number_of_iteration_before_stop:

			new_ants = list()

			for l in range(number_of_new_solution):

				decision_variable_values = [super(AntColony, self).get_decision_variable_value_by_randomization(variable_index) for variable_index in range(self.number_of_variables)]
				new_ant = AntColony.Ant(self.function_wrapper, decision_variable_values, self.objective)

				for i in range(self.number_of_variables):
					ant_S_j = self.__select_ant_based_on_probability() 
					mu_i = self.__ants[ant_S_j].location_coordinates[i]
					sigma_i = self.__return_sigma(i, ant_S_j)
					new_coordinate_s = np.random.normal(mu_i, sigma_i)
					new_coordinate_s = self.__constrain_within_range(new_coordinate_s, i)
					new_ant.location_coordinates[i] = new_coordinate_s

				new_ant.update_fitness_value()
				new_ants.append(new_ant)

			new_ants_collect = self.__ants + new_ants

			self.__sort_ant_by_fitness(new_ants_collect)

			for l in range(number_of_new_solution):
				new_ants_collect.pop()

			self.__ants = new_ants_collect
			self.__update_probability(self.__ants)
			
			iteration += 1

		solution_ant = self.__select_best_ant_by_fitness(self.__ants)
		return { "best_decision_variable_values": solution_ant.location_coordinates, "best_objective_function_value": solution_ant.fitness }


	def __initialize_ants(self):
		self.__ants = list()

		for i in range(self.archive_size):
			decision_variable_values = [super(AntColony, self).get_decision_variable_value_by_randomization(variable_index) for variable_index in range(self.number_of_variables)]
			ant = AntColony.Ant(self.function_wrapper, decision_variable_values, self.objective)
			self.__ants.append(ant)

	def __sort_ant_by_fitness(self, ants):
		if self.objective == "maximization":
			ants.sort(key = lambda ant: ant.fitness, reverse = True)
		elif self.objective == "minimization":
			ants.sort(key = lambda ant: ant.fitness, reverse = False)

	def __update_probability(self, ants):
		q = 5
		k = len(ants)
		c1 = 1.0 / (q*k*sqrt(2*pi))
		c2 = -2.0*q*q*k*k

		position = 1
		summa = 0.0
		for ant in ants:
			ant.weight = c1 * exp(c2*(position - 1)**2)
			summa += ant.weight
			position += 1

		for ant in ants:
			ant.probability = ant.weight /  summa

	def __select_ant_based_on_probability(self):
		# return a index for selected ant
		self.__ants.sort(key = lambda ant: ant.probability, reverse = False) # from small probability to big
		ran = random()
		for ant_index in range(len(self.__ants)):
			if ran < self.__ants[ant_index].probability:
				return ant_index

	def __return_sigma(self, index_of_dimension, index_of_ant):
		summa = 0.0
		ant_S_j_i = self.__ants[index_of_ant].location_coordinates[index_of_dimension]
		for ant in self.__ants:
			summa += fabs(ant.location_coordinates[index_of_dimension] - ant_S_j_i)
		return summa / (len(self.__ants)-1) * self.evaporation_rate
		
	def __select_best_ant_by_fitness(self, ants):
		if self.objective == "maximization":
			best_ant = max(ants, key = lambda ant: ant.fitness)
		elif self.objective == "minimization":
			best_ant = min(ants, key = lambda ant: ant.fitness)
		return best_ant

	def __constrain_within_range(self, location_coordinate, variable_index):
		if location_coordinate < self.function_wrapper.minimum_decision_variable_values()[variable_index]:
			return self.function_wrapper.minimum_decision_variable_values()[variable_index]
		elif location_coordinate > self.function_wrapper.maximum_decision_variable_values()[variable_index]:
			return self.function_wrapper.maximum_decision_variable_values()[variable_index]
		else:
			return location_coordinate