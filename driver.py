# Example Solution Script for U of M IOE

# from metaheuristic_algorithms.function_wrappers.abstract_wrapper import AbstractWrapper
from abstract_wrapper import AbstractWrapper
# from metaheuristic_algorithms.harmony_search import HarmonySearch
from harmony_search import HarmonySearch
from direct_optimization import Direct
from ant_colony_optimization import AntColony
from artificial_bee_colony_algorithm import ArtificialBeeColony

# Declare HLMM Object, methods of Simple Equality Constraints, and Simple Objective Function
class HLMM(AbstractWrapper):
    def maximum_decision_variable_values(self):
        return[250]
    def minimum_decision_variable_values(self):
        return[0]
    def objective_function_value(self, decision_variable_values): # Decision Variable is Spend Level
        #return(0.09828 + 0.01035*decision_variable_values[0] + 2.0566*X[0]**2+ m[0] + m[1])
        return(-(0.01035*(decision_variable_values[0]-210)**2 -10))
    def initial_decision_variable_value_estimates(self):
        return[10]
 


number_of_variables = 1 # Number Decision Variables
objective = "maximization"
#objective = "minimization"


# Declare Harmony Search Hyperparameters
maximum_attempt = 2500
pitch_adjusting_range = 100
harmony_search_size = 2000
harmony_memory_acceping_rate = 0.95
pitch_adjusting_rate = 0.3

# Declare Artificial Bee Colony Hyperparameters
number_of_bees = 20
number_of_trail_before_abandon = 50
number_of_fitness_evaluation = 2000

# Declare Ant Colony Hyperparameters
archive_size = 20
number_of_new_solution = 5 
evaporation_rate = 0.9 
number_of_iteration_before_stop = 1000

# Declare Direct Hyperparameters
iterations = 30

# Recurse Harmony Search over GLMM Structure Combindations    
# for m,n in zip(combination,combinationlabel):
#     print(m)
HLMMwrapper = HLMM()


# 1. Harmony Search
harmony_search = HarmonySearch(HLMMwrapper, number_of_variables, objective)
result1 = harmony_search.search(maximum_attempt = maximum_attempt, 
                               pitch_adjusting_range = pitch_adjusting_range, 
                               harmony_search_size = harmony_search_size, 
                               harmony_memory_acceping_rate = harmony_memory_acceping_rate, 
                               pitch_adjusting_rate = pitch_adjusting_rate)

print("Harmony Search X1:",result1["best_decision_variable_values"][0]) 
print("Harmony Search extrema(R):",result1["best_objective_function_value"])

print("----------------------// end of Harmony Search")


# 2. Artificial Bee Colony
artificial = ArtificialBeeColony(HLMMwrapper, number_of_variables, objective)
result2 = artificial.search(number_of_bees = number_of_bees, 
                            number_of_trail_before_abandon = number_of_trail_before_abandon,  
                            number_of_fitness_evaluation = number_of_fitness_evaluation)

print("Artificial Bee Colony Search X1:",result2["best_decision_variable_values"][0]) 
print("Artificial Bee Colony Search extrema(R):",result2["best_objective_function_value"])


print("----------------------// end of Artificial Bee Colony Search")

# 3. Ant Colony
ant_colony = AntColony(HLMMwrapper, number_of_variables, objective)
result3 = ant_colony.search(archive_size = archive_size,
                            number_of_new_solution = number_of_new_solution,
                            evaporation_rate = evaporation_rate,
                            number_of_iteration_before_stop = number_of_iteration_before_stop)

print("Ant Colony Search X1:",result3["best_decision_variable_values"][0]) 
print("Ant Search extrema(R):",result3["best_objective_function_value"])

print("----------------------// end of Ant Colony Search")


# 4. Direct 
direct = Direct(HLMMwrapper, number_of_variables, objective)
result4 = direct.search(iterations = iterations)

print("Direct Search X1:",result4["best_decision_variable_values"][0]) 
print("Direct Search extrema(R):",result4["best_objective_function_value"])

print("----------------------// end of Direct Search")


print("==============================================")