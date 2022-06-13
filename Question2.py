import math
import numpy as np
from GeneticProgram import GeneticProgram
from GeneticProgram_Set import function_set, terminal_set

# Data
# Generate data
D = 2
X = np.linspace(-1, 1, 21)
Y = np.linspace(-1, 1, 21)
xs = []
ys = []
sphere = []
rastrigin = []
for x in X:
    for y in Y:
        xs.append(x)
        ys.append(y)
        sphere.append((x**2) + (y**2))
        rastrigin.append((10 * D) + ((x**2) - (10 * math.cos(2 * math.pi * x))) + ((y**2) - (10 * math.cos(2 * math.pi * y))))
input = { 'x': xs, 'y': ys }
data_1 = { 'input': input, 'output': sphere }
data_2 = { 'input': input, 'output': rastrigin }

parameters_1 = {
    "time": 360,
    "max_depth": 5,
    "population_size": 250,
    "sample_size": 50,
    "loss_type": "manhattan",
    "loss_threshold": 100,
    "perfect_match": 0.02,
    "crossover_rate": 0.9,
    "mutation_rate": 0.02,
    "shrink_prob": 0.25,
    "hoist_prob": 0.25,
    "grow_prob": 0.1,
    "hillclimb_rate": 0.02,
    "multiple_hillclimb": True,
    "selection_method": "tournament_selection",
    "percent_elite": 0.1,
    "mutate_elite": False,
    "keep_best": 25
}

# Create genetic program

# Sphere
# genetic_program_1 = GeneticProgram(
#     data=data_1,
#     function_set=function_set,
#     terminal_set=terminal_set,
#     parameters=parameters_1,
# )
# genetic_program_1.run()

# Rastrigin
genetic_program_2 = GeneticProgram(
    data=data_2,
    function_set=function_set,
    terminal_set=terminal_set,
    parameters=parameters_1,
)
genetic_program_2.run()
