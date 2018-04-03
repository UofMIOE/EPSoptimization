# Example Solution Script for U of M IOE
import numpy as np
# from metaheuristic_algorithms.function_wrappers.abstract_wrapper import AbstractWrapper
from abstract_wrapper import AbstractWrapper
# from metaheuristic_algorithms.harmony_search import HarmonySearch
from harmony_search import HarmonySearch
from artificial_bee_colony_algorithm import ArtificialBeeColony
from Cuckoo_Search_Algorithm import CuckooSearch
import itertools

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
 
# Declare Random Effects       
randomintercept1 = [41.00978,1.1874,-0.1972]
randominterceptlabel1 = ["T1 Ratio of H:","T1 Ratio of M:","T1 Ratio of L:"] 
randomintercept2 = [3.8006,4.5607,-1.0345,0.8921,0.1739,-2.6423]
randominterceptlabel2 = ["Seasonal Effect of Jan:","Seasonal Effect of May:"
                         ,"Seasonal Effect of Jul:","Seasonal Effect of Sep:"
                         ,"Seasonal Effect of Nov:","Seasonal Effec of Dec:"]
# Declare Fixed Effects    
X = np.array([0.6305345867, 0.7833810757, 0.6440576699, 0.8037949529,
              0.7903449391, 0.7537774028, 0.6950504602, 0.8466272186,
              0.6437516156, 0.6724549363, 0.4211727918, 0.4942251099,
              0.367134762, 0.4014854039, 0.5375975169, 0.3925483311,
              0.7621946848, 0.7264823503, 0.7726027173, 0.7386701467,
              0.8714490891, 0.669595422, 0.6998836716, 0.7239443721,
              0.8262506103, 0.8044416493, 0.8163021048, 0.8690081363,
              0.908831589, 0.8205715489, 0.8654079287, 0.8275721834,
              0.8659668951, 0.8628531392, 0.8636805982, 0.8279350805,
              0.7636361372, 0.7409485405, 0.8863498437, 0.9325482106,
              0.8940163041, 0.7043902685, 0.7884980803, 0.7107404379,
              0.8304674924, 0.8338695519, 0.7854999826, 0.8005034871,
              0.7720266928, 0.7582271188, 0.6405862582, 0.8632760029,
              0.7188567889, 0.5833872249, 0.7095633792, 0.8632616842,
              0.3447492612, 0.4405042791, 0.6181982207, 0.7407103167])

Xlabel = ['% HH > 35K USD/Annum (Zip:62002):','% HH > 35K USD/Annum (Zip:62010):',
		 '% HH > 35K USD/Annum (Zip:62024):','% HH > 35K USD/Annum (Zip:62025):',
		 '% HH > 35K USD/Annum (Zip:62034):','% HH > 35K USD/Annum (Zip:62035):',
		 '% HH > 35K USD/Annum (Zip:62040):','% HH > 35K USD/Annum (Zip:62062):',
		 '% HH > 35K USD/Annum (Zip:62084):','% HH > 35K USD/Annum (Zip:62095):',
		 '% HH > 35K USD/Annum (Zip:62201):','% HH > 35K USD/Annum (Zip:62203):',
		 '% HH > 35K USD/Annum (Zip:62204):','% HH > 35K USD/Annum (Zip:62205):',
		 '% HH > 35K USD/Annum (Zip:62206):','% HH > 35K USD/Annum (Zip:62207):',
		 '% HH > 35K USD/Annum (Zip:62208):','% HH > 35K USD/Annum (Zip:62220):',
		 '% HH > 35K USD/Annum (Zip:62221):','% HH > 35K USD/Annum (Zip:62223):',
		 '% HH > 35K USD/Annum (Zip:62225):','% HH > 35K USD/Annum (Zip:62226):',
		 '% HH > 35K USD/Annum (Zip:62232):','% HH > 35K USD/Annum (Zip:62234):',
		 '% HH > 35K USD/Annum (Zip:62243):','% HH > 35K USD/Annum (Zip:62249):',
		 '% HH > 35K USD/Annum (Zip:62269):','% HH > 35K USD/Annum (Zip:62294):',
		 '% HH > 35K USD/Annum (Zip:63005):','% HH > 35K USD/Annum (Zip:63010):',
		 '% HH > 35K USD/Annum (Zip:63011):','% HH > 35K USD/Annum (Zip:63012):',
		 '% HH > 35K USD/Annum (Zip:63017):','% HH > 35K USD/Annum (Zip:63021):',
		 '% HH > 35K USD/Annum (Zip:63025):','% HH > 35K USD/Annum (Zip:63026):',
		 '% HH > 35K USD/Annum (Zip:63031):','% HH > 35K USD/Annum (Zip:63033):',
		 '% HH > 35K USD/Annum (Zip:63034):','% HH > 35K USD/Annum (Zip:63038):',
		 '% HH > 35K USD/Annum (Zip:63040):','% HH > 35K USD/Annum (Zip:63042):',
		 '% HH > 35K USD/Annum (Zip:63043):','% HH > 35K USD/Annum (Zip:63044):',
		 '% HH > 35K USD/Annum (Zip:63049):','% HH > 35K USD/Annum (Zip:63050):',
		 '% HH > 35K USD/Annum (Zip:63051):','% HH > 35K USD/Annum (Zip:63052):',
		 '% HH > 35K USD/Annum (Zip:63069):','% HH > 35K USD/Annum (Zip:63070):',
		 '% HH > 35K USD/Annum (Zip:63074):','% HH > 35K USD/Annum (Zip:63088):',
		 '% HH > 35K USD/Annum (Zip:63102):','% HH > 35K USD/Annum (Zip:63103):',
		 '% HH > 35K USD/Annum (Zip:63104):','% HH > 35K USD/Annum (Zip:63105):',
		 '% HH > 35K USD/Annum (Zip:63106):','% HH > 35K USD/Annum (Zip:63107):',
		 '% HH > 35K USD/Annum (Zip:63108):','% HH > 35K USD/Annum (Zip:63109):',
		 '% HH > 35K USD/Annum (Zip:63110):','% HH > 35K USD/Annum (Zip:63111):']

# Generate GLMM Structure Combinations for Recursion
combination = [randomintercept1,randomintercept2,X]
combinationlabel = [randominterceptlabel1,randominterceptlabel2,Xlabel]
combination = list(itertools.product(*combination))
combinationlabel = list(itertools.product(*combinationlabel))

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

#Declare Cuckoo Search Hyperparameters
number_agents = 100
number_iterations = 100
probabilty_found = 0.25
number_nests = 100

# Recurse Harmony Search over GLMM Structure Combindations    
for m,n in zip(combination,combinationlabel):
    print(m)
    HLMMwrapper = HLMM()

    # Harmony Search
    harmony_search = HarmonySearch(HLMMwrapper, number_of_variables, objective)
    result1 = harmony_search.search(maximum_attempt = maximum_attempt, 
                                   pitch_adjusting_range = pitch_adjusting_range, 
                                   harmony_search_size = harmony_search_size, 
                                   harmony_memory_acceping_rate = harmony_memory_acceping_rate, 
                                   pitch_adjusting_rate = pitch_adjusting_rate)
    
    print("Harmony Search X1:",result1["best_decision_variable_values"][0]) 
    print("Random Intercept for",n[0],m[0])
    print("Random Intercept for",n[1],m[1])
    print("Fixed Effect for", n[2],m[2])
    print("Harmony Search extrema(R):",result1["best_objective_function_value"])
    
    print("----------------------")


    artificial = ArtificialBeeColony(HLMMwrapper, number_of_variables, objective)
    result2 = artificial.search(number_of_bees = number_of_bees, 
                                number_of_trail_before_abandon = number_of_trail_before_abandon,  
                                number_of_fitness_evaluation = number_of_fitness_evaluation)

    print("Artificial Bee Colony Search X1:",result2["best_decision_variable_values"][0]) 
    print("Random Intercept for",n[0],m[0])
    print("Random Intercept for",n[1],m[1])
    print("Fixed Effect for", n[2],m[2])
    print("Artificial Bee Colony Search extrema(R):",result2["best_objective_function_value"])
    
    
    print("----------------------")

    cuckoo_search = CuckooSearch(HLMMwrapper, number_of_variables, objective)
    result3 = cuckoo_search.search(n = number_agents,
                                   iteration = number_iterations,
                                   pa = probabilty_found,
                                   nest = number_nests)

    print("Cuckoo Search X1:",result3["best_decision_variable_values"][0]) 
    print("Random Intercept for",n[0],m[0])
    print("Random Intercept for",n[1],m[1])
    print("Fixed Effect for", n[2],m[2])
    print("Cuckoo Search extrema(R):",result3["best_objective_function_value"])
    
    
    print("==============================================")