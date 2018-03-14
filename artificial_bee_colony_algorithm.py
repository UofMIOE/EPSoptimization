from base_algorithm import BaseAlgorithm
import random
import copy

class ArtificialBeeColony(BaseAlgorithm):
	
	class Bee(object):
		
		def __init__(self,function_wrapper, location_coordinates, objective):
			self.function_wrapper = function_wrapper
			self.location_coordinates = location_coordinates
			self.trial = 0
			self.possibility = 0
			self.objective = objective
			self.update_nectar_amount()
			self.update_fitness()
			
		def update_nectar_amount(self):
			self.nectar_amount = self.function_wrapper.objective_function_value(self.location_coordinates)

		def update_fitness(self):
			if self.objective == "maximization":
				# self.fitness = self.nectar_amount
				if self.nectar_amount >= 0:
					self.fitness = 1.0 + self.nectar_amount
				else:
					self.fitness = 1.0 / (1.0 + abs(self.nectar_amount))
			elif self.objective == "minimization":
				if self.nectar_amount >= 0:
					self.fitness = 1.0 / (1.0 + self.nectar_amount)
				else:
					self.fitness = 1.0 + abs(self.nectar_amount)

		def fuction_wrapper_preserving_clone(self):
			clone_object = copy.copy(self)
			clone_object.location_coordinates = copy.deepcopy(self.location_coordinates)
			clone_object.nectar_amount = copy.deepcopy(self.nectar_amount)
			clone_object.trial = copy.deepcopy(self.trial)
			clone_object.possibility = copy.deepcopy(self.possibility)
			clone_object.fitness = copy.deepcopy(self.fitness)
			clone_object.objective = copy.deepcopy(self.objective)
			return clone_object
	

	def __init__(self, function_wrapper, number_of_variables = 1, objective = "maximization"):
		super().__init__(function_wrapper,number_of_variables,objective)
	

	def search(self, number_of_bees = 10, number_of_trail_before_abandon = 10, number_of_fitness_evaluation = 50):

		self.number_of_bees = number_of_bees
		self.number_of_trail_before_abandon = number_of_trail_before_abandon
		self.number_of_fitness_evaluation = number_of_fitness_evaluation

		self.__initialize_bees(number_of_bees)
		

		num_eval = number_of_bees

		while num_eval < number_of_fitness_evaluation:
			
			# Employed Bees Phase
			for bee_i in range(number_of_bees):
				
				self.__mutate_location(bee_i)

				num_eval += 1

				if num_eval >= self.number_of_fitness_evaluation:
					solution_bee = self.__select_best_bee_by_nectar_amount(self.__bees)
					return { "best_decision_variable_values": solution_bee.location_coordinates, "best_objective_function_value": solution_bee.nectar_amount }

			self.__update__possibility()

			# Onlooker Bees Phase
			bee_s = 1
			bee_s_index = bee_s - 1
			bee_t = 1

			while bee_t <= (self.number_of_bees - 1):
				r = random.random()

				if r < self.__bees[bee_s_index].possibility:
					bee_t += 1
					self.__mutate_location(bee_s_index)
					num_eval += 1

					if num_eval >= self.number_of_fitness_evaluation:
						solution_bee = self.__select_best_bee_by_nectar_amount(self.__bees)
						return { "best_decision_variable_values": solution_bee.location_coordinates, "best_objective_function_value": solution_bee.nectar_amount }

				bee_s = bee_s % self.number_of_bees + 1
				bee_s_index = bee_s - 1


			# Scout Bees Phase
			
			mi = self.__maximum_trial()
			if self.__bees[mi].trial >= self.number_of_trail_before_abandon:
				decision_variable_values = [super(ArtificialBeeColony, self).get_decision_variable_value_by_randomization(variable_index) for variable_index in range(self.number_of_variables)]
				bee = ArtificialBeeColony.Bee(self.function_wrapper, decision_variable_values, self.objective)
				self.__bees[mi] = bee.fuction_wrapper_preserving_clone()

				num_eval += 1

				if num_eval >= self.number_of_fitness_evaluation:
						solution_bee = self.__select_best_bee_by_nectar_amount(self.__bees)
						return { "best_decision_variable_values": solution_bee.location_coordinates, "best_objective_function_value": solution_bee.nectar_amount }


	def __initialize_bees(self, number_of_bees):
		self.__bees = list()

		for i in range(number_of_bees):
			decision_variable_values = [super(ArtificialBeeColony, self).get_decision_variable_value_by_randomization(variable_index) for variable_index in range(self.number_of_variables)]
			bee = ArtificialBeeColony.Bee(self.function_wrapper, decision_variable_values, self.objective)
			self.__bees.append(bee)

	def __constrain_within_range(self, location_coordinate, variable_index):
		if location_coordinate < self.function_wrapper.minimum_decision_variable_values()[variable_index]:
			return self.function_wrapper.minimum_decision_variable_values()[variable_index]
		elif location_coordinate > self.function_wrapper.maximum_decision_variable_values()[variable_index]:
			return self.function_wrapper.maximum_decision_variable_values()[variable_index]
		else:
			return location_coordinate


	def __select_best_bee_by_nectar_amount(self, bees):
		if self.objective == "maximization":
			best_bee = max(bees, key = lambda bee: bee.nectar_amount)
		elif self.objective == "minimization":
			best_bee = min(bees, key = lambda bee: bee.nectar_amount)
		return best_bee

	def __mutate_location(self,bee_i):
		bee_i_copy = self.__bees[bee_i].fuction_wrapper_preserving_clone()

		# draws a dimension to be crossed-over and mutated
		j = random.randint(0, self.number_of_variables-1)

		# selects another bee
		bee_k = bee_i;
		while (bee_k == bee_i): bee_k = random.randint(0, self.number_of_bees-1)

		new_location_coordinate = self.__bees[bee_i].location_coordinates[j] + 2.0*(random.random() - 0.5) * \
		(self.__bees[bee_i].location_coordinates[j] - self.__bees[bee_k].location_coordinates[j])

		new_location_coordinate = self.__constrain_within_range(new_location_coordinate, j)

		bee_i_copy.location_coordinates[j] = new_location_coordinate

		bee_i_copy.update_nectar_amount()
		bee_i_copy.update_fitness()

		if bee_i_copy.nectar_amount < self.__bees[bee_i].nectar_amount and self.objective == "minimization":
			self.__bees[bee_i] = bee_i_copy.fuction_wrapper_preserving_clone()
			self.__bees[bee_i].trial = 0
		elif bee_i_copy.nectar_amount > self.__bees[bee_i].nectar_amount and self.objective == "maximization":
			self.__bees[bee_i] = bee_i_copy.fuction_wrapper_preserving_clone()
			self.__bees[bee_i].trial = 0
		else:
			self.__bees[bee_i].trial += 1

	def __update__possibility(self):
		sum = 0.0
		for bee_i in range(self.number_of_bees):
			sum += self.__bees[bee_i].fitness
		for bee_i in range(self.number_of_bees):
			self.__bees[bee_i].possibility = self.__bees[bee_i].fitness / sum

	def __maximum_trial(self):
		index = 0
		trial = 0
		for i in range(self.number_of_bees):
			if self.__bees[i].trial > trial:
				trial = self.__bees[i].trial
				index = i
		return index