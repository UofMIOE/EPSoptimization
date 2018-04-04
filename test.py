# Example Solution Script for U of M IOE
import numpy as np
import time
# from metaheuristic_algorithms.function_wrappers.abstract_wrapper import AbstractWrapper
from abstract_wrapper import AbstractWrapper
# from metaheuristic_algorithms.harmony_search import HarmonySearch
from harmony_search import HarmonySearch
from artificial_bee_colony_algorithm import ArtificialBeeColony
from ant_colony_optimization import AntColony
from direct_optimization import Direct
from Cuckoo_Search_Algorithm import CuckooSearch
from simulated_annealing import SimulatedAnnealing
from simplified_particle_swarm_optimization import SimplifiedParticleSwarmOptimization
from firefly_algorithm import FireflyAlgorithm
import itertools
import csv

# Declare HLMM Object, methods of Simple Equality Constraints, and Simple Objective Function
class HLMM(AbstractWrapper):
    def maximum_decision_variable_values(self):
        return[250]
    def minimum_decision_variable_values(self):
        return[0]
    def objective_function_value(self, decision_variable_values): # Decision Variable is Spend Level
        #return(0.09828 + 0.01035*decision_variable_values[0] + 2.0566*X[0]**2+ m[0] + m[1])
        return(-(0.09828 + 0.01035*(decision_variable_values[0]-210)**2 -10))
    def initial_decision_variable_value_estimates(self):
        return[10]
 
## Declare Random Effects       
#randomintercept1 = [41.00978,1.1874,-0.1972]
#randominterceptlabel1 = ["T1 Ratio of H:","T1 Ratio of M:","T1 Ratio of L:"] 
#randomintercept2 = [3.8006,4.5607,-1.0345,0.8921,0.1739,-2.6423]
#randominterceptlabel2 = ["Seasonal Effect of Jan:","Seasonal Effect of May:"
#                         ,"Seasonal Effect of Jul:","Seasonal Effect of Sep:"
#                         ,"Seasonal Effect of Nov:","Seasonal Effec of Dec:"]
## Declare Fixed Effects    
#X = np.array([0.6305345867, 0.7833810757, 0.6440576699, 0.8037949529,
#              0.7903449391, 0.7537774028, 0.6950504602, 0.8466272186,
#              0.6437516156, 0.6724549363, 0.4211727918, 0.4942251099,
#              0.367134762, 0.4014854039, 0.5375975169, 0.3925483311,
#              0.7621946848, 0.7264823503, 0.7726027173, 0.7386701467,
#              0.8714490891, 0.669595422, 0.6998836716, 0.7239443721,
#              0.8262506103, 0.8044416493, 0.8163021048, 0.8690081363,
#              0.908831589, 0.8205715489, 0.8654079287, 0.8275721834,
#              0.8659668951, 0.8628531392, 0.8636805982, 0.8279350805,
#              0.7636361372, 0.7409485405, 0.8863498437, 0.9325482106,
#              0.8940163041, 0.7043902685, 0.7884980803, 0.7107404379,
#              0.8304674924, 0.8338695519, 0.7854999826, 0.8005034871,
#              0.7720266928, 0.7582271188, 0.6405862582, 0.8632760029,
#              0.7188567889, 0.5833872249, 0.7095633792, 0.8632616842,
#              0.3447492612, 0.4405042791, 0.6181982207, 0.7407103167])
#
#Xlabel = ['% HH > 35K USD/Annum (Zip:62002):','% HH > 35K USD/Annum (Zip:62010):',
#		 '% HH > 35K USD/Annum (Zip:62024):','% HH > 35K USD/Annum (Zip:62025):',
#		 '% HH > 35K USD/Annum (Zip:62034):','% HH > 35K USD/Annum (Zip:62035):',
#		 '% HH > 35K USD/Annum (Zip:62040):','% HH > 35K USD/Annum (Zip:62062):',
#		 '% HH > 35K USD/Annum (Zip:62084):','% HH > 35K USD/Annum (Zip:62095):',
#		 '% HH > 35K USD/Annum (Zip:62201):','% HH > 35K USD/Annum (Zip:62203):',
#		 '% HH > 35K USD/Annum (Zip:62204):','% HH > 35K USD/Annum (Zip:62205):',
#		 '% HH > 35K USD/Annum (Zip:62206):','% HH > 35K USD/Annum (Zip:62207):',
#		 '% HH > 35K USD/Annum (Zip:62208):','% HH > 35K USD/Annum (Zip:62220):',
#		 '% HH > 35K USD/Annum (Zip:62221):','% HH > 35K USD/Annum (Zip:62223):',
#		 '% HH > 35K USD/Annum (Zip:62225):','% HH > 35K USD/Annum (Zip:62226):',
#		 '% HH > 35K USD/Annum (Zip:62232):','% HH > 35K USD/Annum (Zip:62234):',
#		 '% HH > 35K USD/Annum (Zip:62243):','% HH > 35K USD/Annum (Zip:62249):',
#		 '% HH > 35K USD/Annum (Zip:62269):','% HH > 35K USD/Annum (Zip:62294):',
#		 '% HH > 35K USD/Annum (Zip:63005):','% HH > 35K USD/Annum (Zip:63010):',
#		 '% HH > 35K USD/Annum (Zip:63011):','% HH > 35K USD/Annum (Zip:63012):',
#		 '% HH > 35K USD/Annum (Zip:63017):','% HH > 35K USD/Annum (Zip:63021):',
#		 '% HH > 35K USD/Annum (Zip:63025):','% HH > 35K USD/Annum (Zip:63026):',
#		 '% HH > 35K USD/Annum (Zip:63031):','% HH > 35K USD/Annum (Zip:63033):',
#		 '% HH > 35K USD/Annum (Zip:63034):','% HH > 35K USD/Annum (Zip:63038):',
#		 '% HH > 35K USD/Annum (Zip:63040):','% HH > 35K USD/Annum (Zip:63042):',
#		 '% HH > 35K USD/Annum (Zip:63043):','% HH > 35K USD/Annum (Zip:63044):',
#		 '% HH > 35K USD/Annum (Zip:63049):','% HH > 35K USD/Annum (Zip:63050):',
#		 '% HH > 35K USD/Annum (Zip:63051):','% HH > 35K USD/Annum (Zip:63052):',
#		 '% HH > 35K USD/Annum (Zip:63069):','% HH > 35K USD/Annum (Zip:63070):',
#		 '% HH > 35K USD/Annum (Zip:63074):','% HH > 35K USD/Annum (Zip:63088):',
#		 '% HH > 35K USD/Annum (Zip:63102):','% HH > 35K USD/Annum (Zip:63103):',
#		 '% HH > 35K USD/Annum (Zip:63104):','% HH > 35K USD/Annum (Zip:63105):',
#		 '% HH > 35K USD/Annum (Zip:63106):','% HH > 35K USD/Annum (Zip:63107):',
#		 '% HH > 35K USD/Annum (Zip:63108):','% HH > 35K USD/Annum (Zip:63109):',
#		 '% HH > 35K USD/Annum (Zip:63110):','% HH > 35K USD/Annum (Zip:63111):']
#
## Generate GLMM Structure Combinations for Recursion
#combination = [randomintercept1,randomintercept2,X]
#combinationlabel = [randominterceptlabel1,randominterceptlabel2,Xlabel]
#combination = list(itertools.product(*combination))
#combinationlabel = list(itertools.product(*combinationlabel))

# Declare Harmony Search Hyperparameters
number_of_variables = 1 # Number Decision Variables
objective = "maximization"
#objective = "minimization"
maximum_attempt = 2500
pitch_adjusting_range = 100
harmony_search_size = 2000
harmony_memory_acceping_rate = 0.95
pitch_adjusting_rate = 0.3

# Declare Artificial Bee Colony Hyperparameters
number_of_bees = 100
number_of_trail_before_abandon = 30
number_of_fitness_evaluation = 2000

# Declare Ant Colony Hyperparameters
archive_size = 10
number_of_new_solution = 5
evaporation_rate = 0.9
number_of_iteration_before_stop = 100

# Declare Direct Hyperparameters
iterations = 30

#Declare Cuckoo Search Hyperparameters
number_agents = 100
number_iterations = 100
probabilty_found = 0.25
number_nests = 100

#Declare Simulate annealing Hyperparameters
temperature = 1000
minimal_temperature = 1
bolzmann_constant = 1.38065e-23
energy_norm = 10
maximum_number_of_rejections = 2500
maximum_number_of_runs = 500
maximum_number_of_acceptances = 15
cooling_factor = 0.95
standard_diviation_for_estimation = 6
ratio_of_energy_delta_over_evaluation_delta = 10

#Declare SPSO Hyperparameters
number_of_particiles = 20
number_of_iterations = 15
social_coefficient = 0.5
random_variable_coefficient = 0.2

#Declare Firefly Hyperparameters
number_of_fireflies = 10
maximun_generation = 10
randomization_parameter_alpha = 0.2
absorption_coefficient_gamma = 1.0

# Recurse Harmony Search over GLMM Structure Combindations    
# for m,n in zip(combination,combinationlabel):
# print(m)
t = 1
result_hs = []
result_artificial = []
result_direct = []
result_ant = []
result_cuckoo = []
result_pso = []
result_ff = []
time_hs = []
time_artificial = []
time_ant = []
time_direct = []
time_cuckoo = []
time_pso = []
time_ff = []


while t <= 1000:
    
    HLMMwrapper = HLMM()

    # Harmony Search
    harmony_search = HarmonySearch(HLMMwrapper, number_of_variables, objective)
    time_hs_start = time.time()
    result1 = harmony_search.search(maximum_attempt = maximum_attempt, 
                                   pitch_adjusting_range = pitch_adjusting_range, 
                                   harmony_search_size = harmony_search_size, 
                                   harmony_memory_acceping_rate = harmony_memory_acceping_rate, 
                                   pitch_adjusting_rate = pitch_adjusting_rate)
    time_hs_elapsed = time.time() - time_hs_start
    time_hs.append(time_hs_elapsed)
    
#    print("Harmony Search X1:",result1["best_decision_variable_values"][0]) 
#    print("Random Intercept for",n[0],m[0])
#    print("Random Intercept for",n[1],m[1])
#    print("Fixed Effect for", n[2],m[2])
#   print("Harmony Search extrema(R):",result1["best_objective_function_value"])
    
#   print("----------------------")
    

    artificial = ArtificialBeeColony(HLMMwrapper, number_of_variables, objective)
    time_artificial_start = time.time()
    result2 = artificial.search(number_of_bees = number_of_bees, 
                                number_of_trail_before_abandon = number_of_trail_before_abandon,  
                                number_of_fitness_evaluation = number_of_fitness_evaluation)
    time_artificial_elapsed = time.time() - time_artificial_start
    time_artificial.append(time_artificial_elapsed)
    
#   print("Artificial Bee Colony Search X1:",result2["best_decision_variable_values"][0]) 
#    print("Random Intercept for",n[0],m[0])
#    print("Random Intercept for",n[1],m[1])
#    print("Fixed Effect for", n[2],m[2])
#   print("Artificial Bee Colony Search extrema(R):",result2["best_objective_function_value"])
    
    
#    print("----------------------")
    
    ant_colony = AntColony(HLMMwrapper, number_of_variables, objective)
    time_ant_start = time.time()
    result3 = ant_colony.search(archive_size = archive_size,
                                number_of_new_solution = number_of_new_solution,
                                evaporation_rate = evaporation_rate,
                                number_of_iteration_before_stop = number_of_iteration_before_stop)
    time_ant_elapsed = time.time() - time_ant_start
    time_ant.append(time_ant_elapsed)
#    print("Ant Colony Search X1:",result3["best_decision_variable_values"][0]) 
#    print("Random Intercept for",n[0],m[0])
#    print("Random Intercept for",n[1],m[1])
#    print("Fixed Effect for", n[2],m[2])
#   print("Ant Colony Search extrema(R):",result3["best_objective_function_value"])
    
    
#    print("----------------------")
    
      
    direct = Direct(HLMMwrapper, number_of_variables, objective)
    time_direct_start = time.time()
    
    result4 = direct.search(iterations = iterations)
    time_direct_elapsed = time.time() - time_direct_start
    time_direct.append(time_direct_elapsed)
    
#    print("Direct Search X1:",result4["best_decision_variable_values"][0]) 
#    print("Random Intercept for",n[0],m[0])
#    print("Random Intercept for",n[1],m[1])
#    print("Fixed Effect for", n[2],m[2])
#    print("Direct Search extrema(R):",result4["best_objective_function_value"])
    
    
#    print("----------------------")


    cuckoo_search = CuckooSearch(HLMMwrapper, number_of_variables, objective)
    time_cuckoo_start = time.time()
    
    result5 = cuckoo_search.search(n = number_agents,
                                   iteration = number_iterations,
                                   pa = probabilty_found,
                                   nest = number_nests)
    time_cuckoo_elapsed = time.time() - time_cuckoo_start
    time_cuckoo.append(time_cuckoo_elapsed)
    
#    print("Cuckoo Search X1:",result5["best_decision_variable_values"][0]) 
#    print("Random Intercept for",n[0],m[0])
#    print("Random Intercept for",n[1],m[1])
#    print("Fixed Effect for", n[2],m[2])
#    print("Cuckoo Search extrema(R):",result5["best_objective_function_value"])
    
#    print("----------------------")
    
    # simulated_annealing

    simulatted_annealing = SimulatedAnnealing(HLMMwrapper, number_of_variables, objective)
    time_hs_start = time.time()
    #result6 = simulatted_annealing.search(temperature = temperature,
    """                            minimal_temperature = minimal_temperature,
                                   bolzmann_constant = bolzmann_constant,
                                   energy_norm = energy_norm,
                                   maximum_number_of_rejections = maximum_number_of_rejections, 
                                   maximum_number_of_runs = maximum_number_of_runs, 
                                   maximum_number_of_acceptances = maximum_number_of_acceptances, 
                                   cooling_factor = cooling_factor, 
                                   standard_diviation_for_estimation = standard_diviation_for_estimation,
                                   ratio_of_energy_delta_over_evaluation_delta = ratio_of_energy_delta_over_evaluation_delta)

    time_hs_elapsed = time.time() - time_hs_start
    time_hs.append(time_hs_elapsed)
    
    print("Simulated Annealing X1:",result6["best_decision_variable_values"][0]) 
#    print("Random Intercept for",n[0],m[0])
#    print("Random Intercept for",n[1],m[1])
#    print("Fixed Effect for", n[2],m[2])
    print("Simulated Annealing extrema(R):",result6["best_objective_function_value"])
    
    print("----------------------")
   
    """  
    
    simplified_particle_swarm_optimization = SimplifiedParticleSwarmOptimization(HLMMwrapper, number_of_variables, objective)
    time_pso_start = time.time()
    
    result7 = simplified_particle_swarm_optimization.search(number_of_particiles = number_of_particiles,
                                   number_of_iterations = number_of_iterations,
                                   social_coefficient = social_coefficient,
                                   random_variable_coefficient = random_variable_coefficient)
    time_pso_elapsed = time.time() - time_pso_start
    time_pso.append(time_pso_elapsed)
    
#    print("Particle Swarm Optimization X1:",result7["best_decision_variable_values"][0]) 
#   print("Particle Swarm Optimization extrema(R):",result7["best_objective_function_value"])
    
#   print("----------------------")
    
    firefly_algorithm = FireflyAlgorithm(HLMMwrapper, number_of_variables, objective)
    
    time_ff_start = time.time()
    
    result8 = firefly_algorithm.search(number_of_fireflies = number_of_fireflies,
                                       maximun_generation = maximun_generation,
                                       randomization_parameter_alpha = randomization_parameter_alpha,
                                       absorption_coefficient_gamma = absorption_coefficient_gamma)
    
    time_ff_elapsed = time.time() - time_pso_start
    time_ff.append(time_ff_elapsed)
    
    
    
#   print("Firefly Algorithm X1:",result8["best_decision_variable_values"][0]) 
#   print("Firefly Algorithm extrema(R):",result8["best_objective_function_value"])
    
    
    #print("==============================================")
    
    result_hs.append(result1["best_objective_function_value"])
    result_artificial.append(result2["best_objective_function_value"])
    result_ant.append(result3["best_objective_function_value"])
    result_direct.append(result4["best_objective_function_value"])
    result_cuckoo.append(result5["best_objective_function_value"])
    result_pso.append(result7["best_objective_function_value"])
    result_ff.append(result8["best_objective_function_value"])
    
    t = t + 1
    

