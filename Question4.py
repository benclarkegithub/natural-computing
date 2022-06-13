from GeneticProgram import GeneticProgram
from GeneticProgram_Set import function_set, terminal_set

# Data
X = []
Y = []
Z = []

with open("datafile.txt") as file:
    for line in file:
        x, y, z = line.split()
        X.append(float(x))
        Y.append(float(y))
        Z.append(float(z))

input = { 'x': X, 'y': Y }
data = { 'input': input, 'output': Z }

# For depth 2 or 3
parameters_1 = {
    "time": 360,
    "max_depth": 3,
    "population_size": 500,
    "sample_size": 25,
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

genetic_program_1 = GeneticProgram(
    data=data,
    function_set=function_set,
    terminal_set=terminal_set,
    parameters=parameters_1,
)

genetic_program_1.run()

# For depth >= 3
# parameters_2 = {
#     "time": 360,
#     "max_depth": 5,
#     "population_size": 250,
#     "sample_size": 100,
#     "loss_type": "manhattan",
#     "loss_threshold": 100,
#     "perfect_match": 0.02,
#     "crossover_rate": 0.9,
#     "mutation_rate": 0.02,
#     "shrink_prob": 0.25,
#     "hoist_prob": 0.25,
#     "grow_prob": 0.1,
#     "hillclimb_rate": 0.02,
#     "multiple_hillclimb": True,
#     "selection_method": "tournament_selection",
#     "percent_elite": 0.1,
#     "mutate_elite": False,
#     "keep_best": 25
# }
#
# genetic_program_2 = GeneticProgram(
#     data=data,
#     function_set=function_set,
#     terminal_set=terminal_set,
#     parameters=parameters_2,
# )
#
# genetic_program_2.run()
