import copy
import math
import random
import timeit
import numpy as np


class Tree:
    def __init__(self, root):
        self.root = root

    def get_root(self):
        return self.root

    def set_root(self, root):
        self.root = root


class Program(Tree):
    def __init__(self, root):
        super().__init__(root)

    def calculate(self, input):
        """ Input is a dictionary of variables (e.g. 'x') and a corresponding value """
        return self.root.calculate(input)

    def to_str(self):
        return self.root.make_str()


class Node:
    children_total = 0
    max_depth = 0

    def __init__(self, parent, children):
        self.parent = parent
        self.children = children
        self.calculate_children_total()
        self.calculate_max_depth()

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_children_total(self):
        return self.children_total

    def get_max_depth(self):
        return self.max_depth

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)
        self.add_child_to_children_total(child)
        self.calculate_max_depth()

    def add_child_at_index(self, child, index):
        self.children.insert(index, child)
        self.add_child_to_children_total(child)
        self.calculate_max_depth()

    def remove_child(self, child):
        self.remove_child_from_children_total(child)
        self.children.remove(child)
        self.calculate_max_depth()

    def remove_child_at_index(self, index):
        self.remove_child_from_children_total(self.children[index])
        del self.children[index]
        self.calculate_max_depth()

    def calculate_children_total(self):
        self.children_total = 0
        if self.children is not None:
            for child in self.children:
                self.add_child_to_children_total(child)
        if self.parent is not None:
            self.parent.calculate_children_total()

    def add_child_to_children_total(self, child):
        self.children_total += 1 + child.children_total

    def remove_child_from_children_total(self, child):
        self.children_total -= 1 + child.children_total

    def calculate_max_depth(self):
        if (self.children is None) or (len(self.children) == 0):
            self.max_depth = 0
        else:
            max_depth = self.children[0].get_max_depth()
            for child in self.children:
                if child.get_max_depth() > max_depth:
                    max_depth = child.get_max_depth()
            self.max_depth = 1 + max_depth
        if self.parent is not None:
            self.parent.calculate_max_depth()


class NonTerminal(Node):
    def __init__(self, parent, function):
        super().__init__(parent, [])
        self.function = function

    def get_function(self):
        return self.function

    def calculate(self, input):
        values = [child.calculate(input) for child in self.get_children()]

        # Check values to ensure they are all valid
        for value in values:
            if value is None:
                # The value is None
                return None

        if (not self.function["function"].are_allowed_values(values)) or self.function["function"].are_disallowed_values(values):
            return None

        return self.function["function"].calculate(values)

    def make_str(self):
        start_str = self.function["function"].get_start_str()
        middle_str = self.function["function"].get_middle_str()
        end_str = self.function["function"].get_end_str()
        middle_str_2 = middle_str.join([child.make_str() for child in self.children])
        return start_str + middle_str_2 + end_str


class Terminal(Node):
    def __init__(self, parent, terminal):
        super().__init__(parent, None)
        self.terminal = terminal

    def calculate(self, input):
        if type(self.terminal["value"]) is not str:
            return self.terminal["value"]
        else:
            return input[self.terminal["value"]]

    def set_value(self, value):
        self.terminal["value"] = value

    def make_str(self):
        if type(self.terminal["value"]) is not str:
            if not math.isclose(self.terminal["value"], math.pi):
                return str(self.terminal["value"])
            else:
                return 'π'
        else:
            return self.terminal["value"]


class Function:
    def __init__(self, function, arity, allowed_values, disallowed_values, start_str, middle_str, end_str):
        self.function = function
        self.arity = arity
        self.allowed_values = allowed_values
        self.disallowed_values = disallowed_values
        self.start_str = start_str
        self.middle_str = middle_str
        self.end_str = end_str

    def calculate(self, x):
        try:
            return self.function(x)
        except OverflowError:
            return None

    def get_arity(self):
        return self.arity

    def are_allowed_values(self, values):
        if self.allowed_values is None:
            return True
        else:
            return self.allowed_values(values)

    def are_disallowed_values(self, values):
        if self.disallowed_values is None:
            return False
        else:
            return self.disallowed_values(values)

    def get_start_str(self):
        return self.start_str

    def get_middle_str(self):
        return self.middle_str

    def get_end_str(self):
        return self.end_str


# Helper function
def item_from_pdf(items, pdf):
    pdf_a = np.array(pdf)
    if not math.isclose(pdf_a.sum(), 1):
        raise ValueError("PDF does not sum to 1.")

    random_number = random.random()
    i = 0
    cumulative_density = pdf[i]
    while cumulative_density < random_number:
        i += 1
        cumulative_density += pdf[i]

    return items[i]


class GeneticProgram:
    """
    The genetic program class.
    """

    programs = []
    sample = []
    sample_size = 100
    losses = np.array([])
    loss = np.array([])
    loss_threshold = 100
    fitness = np.array([])
    best_programs = []
    best_programs_loss = np.array([])

    def __init__(self, data, function_set, terminal_set, parameters, seed=None):
        self.input = data["input"]
        self.output = np.array(data["output"])
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.time = parameters["time"]
        self.max_depth = parameters["max_depth"]
        self.population_size = parameters["population_size"]
        self.sample_size = parameters["sample_size"]
        self.loss_type = parameters["loss_type"]
        self.loss_threshold = parameters["loss_threshold"]
        self.perfect_match = parameters["perfect_match"]
        self.crossover_rate = parameters["crossover_rate"]
        self.mutation_rate = parameters["mutation_rate"]
        self.shrink_prob = parameters["shrink_prob"]
        self.hoist_prob = parameters["hoist_prob"]
        self.grow_prob = parameters["grow_prob"]
        self.hillclimb_rate = parameters["hillclimb_rate"]
        self.multiple_hillclimb = parameters["multiple_hillclimb"]
        self.selection_method = parameters["selection_method"]
        self.percent_elite = parameters["percent_elite"]
        self.mutate_elite = parameters["mutate_elite"]
        self.keep_best = parameters["keep_best"]

        self.init_population(seed)

    def get_programs(self):
        return self.programs

    def get_loss(self):
        return self.loss

    def get_min_loss(self):
        return self.loss.min()

    def get_avg_loss(self):
        return self.loss.mean()

    def get_fitness(self):
        return self.fitness

    def get_best_programs(self):
        return self.best_programs

    def get_best_programs_loss(self):
        return self.best_programs_loss

    def init_population(self, seed):
        """
        Based on "A Field Guide to Genetic Programming" 2.2 Initialising
        the Population (pages 11-14).
        Initialises the population using the Ramped half-and-half method.
        """
        if seed is not None:
            self.programs = seed

        pdf_proportional = []
        for f in self.function_set:
            pdf_proportional.append(f["probability"])
        pdf_proportional = np.array(pdf_proportional)
        pdf = pdf_proportional / pdf_proportional.sum()

        while len(self.programs) < self.population_size:
            # Pick the root node function from the function set based
            # on the functions' probabilities.
            function = item_from_pdf(self.function_set, pdf)
            root = NonTerminal(None, function)
            program = Program(root)
            self.programs.append(program)
            full = (len(self.programs) % 2) == 0
            self.init_node(program.get_root(), full, 0)

    def init_node(self, node, full, depth):
        for _ in range(node.get_function()["function"].get_arity()):
            if full:
                if (depth + 1) < self.max_depth:
                    # Initialise node as non-terminal
                    function = self.get_function_terminal(node.get_function(), True, False)
                    child = NonTerminal(node, function)
                    self.init_node(child, True, depth + 1)
                else:
                    # Initialise node as terminal
                    terminal = self.get_function_terminal(node.get_function(), False, True)
                    child = Terminal(node, terminal)
            else:
                if (depth + 1) < self.max_depth:
                    function_terminal = self.get_function_terminal(node.get_function(), True, True)
                    if "function" in function_terminal:
                        # Initialise node as non-terminal
                        child = NonTerminal(node, function_terminal)
                        self.init_node(child, False, depth + 1)
                    else:
                        # Initialise node as terminal
                        child = Terminal(node, function_terminal)
                else:
                    # Initialise node as terminal
                    terminal = self.get_function_terminal(node.get_function(), False, True)
                    child = Terminal(node, terminal)

            node.add_child(child)

    def get_function_terminal(self, parent_function, function, terminal, arity=None):
        functions = self.get_possible_functions(parent_function, arity) if function else []
        terminals = self.get_possible_terminals(parent_function) if terminal else []
        functions_terminals = functions + terminals

        # Proportional PDF
        pdf_proportional = []
        for f_t in (functions + terminals):
            pdf_proportional.append(f_t["probability"])
        pdf_proportional = np.array(pdf_proportional)
        pdf = pdf_proportional / pdf_proportional.sum()

        return item_from_pdf(functions_terminals, pdf)

    def get_possible_functions(self, parent_function, arity=None):
        # Prioritise allowed functions over disallowed functions (whitelist takes priority over
        # blacklist).
        if parent_function is not None:
            if "allowed_functions" in parent_function:
                return [f for f in self.function_set if (f["function"] in parent_function["allowed_functions"]) \
                                                        and ((arity is None) or (f["function"].get_arity() == arity))]
            elif "disallowed_functions" in parent_function:
                return [f for f in self.function_set if (f["function"] not in parent_function["disallowed_functions"]) \
                                                        and ((arity is None) or (f["function"].get_arity() == arity))]

        return [f for f in self.function_set if (arity is None) or (f["function"].get_arity() == arity)]

    def get_possible_terminals(self, parent_function):
        if parent_function is not None:
            if "allowed_terminals" in parent_function:
                return [t for t in self.terminal_set if t["value"] in parent_function["allowed_terminals"]]
            elif "disallowed_terminals" in parent_function:
                return [t for t in self.terminal_set if t["value"] not in parent_function["disallowed_terminals"]]

        return self.terminal_set

    def run(self):
        # Sample indexes at the beginning of the run to keep
        # them consistent throughout iterations.
        input_length = len(next(iter(self.input.values())))
        if input_length > self.sample_size:
            self.sample = random.sample(range(input_length), self.sample_size)
        else:
            self.sample = range(input_length)

        # For the first 4/5ths of the time alternate between a
        # new population and a seeded one. Gradually increase
        # the number of iterations. Keep the best individuals
        # at all times.
        print(f"Starting Genetic Program.")
        start = timeit.default_timer()
        i = 0
        perfect_match_found = False

        while (timeit.default_timer() - start) < self.time:
            print(f"Iteration: {i}")
            self.programs = []
            self.losses = np.array([])
            self.loss = np.array([])
            self.fitness = np.array([])

            # For the first 6 major iterations, alternate between new and
            # seeded populations. Then continue with seeded only.
            if (i < 6) and ((i % 2) == 0):
                # New population
                print("Using new population.")
                self.init_population(None)
            else:
                # Seeded population
                print("Using seeded population.")
                self.init_population(self.best_programs)

            for j in range(20 + (2 * i)):
                self.iterate()
                # Check to see if there is a perfect match, and
                # if there is stop the program and print it out.
                if np.any((self.loss / self.sample_size) < self.perfect_match):
                    print("Found a perfect match!")
                    perfect_match_found = True
                    break
                # Otherwise print analysis
                analysis = self.analyse()
                print(f"Iteration: {i} ({j}). "
                      f"\tBest loss: {self.get_min_loss() / self.sample_size:.2f}"
                      f"\tAvg. loss: {self.get_avg_loss() / self.sample_size:.2f}"
                      f"\tAvg. children: {analysis['avg_children']:.2f}"
                      f"\tAvg. max depth: {analysis['avg_max_depth']:.2f}"
                      f"\tBest program: {self.programs[self.loss.argmin()].to_str()}")

            if perfect_match_found:
                break

            i += 1

        self.print_best_programs()

    def print_best_programs(self):
        best_programs_str = []
        for p_i, p in enumerate(self.best_programs):
            avg_loss = self.best_programs_loss[p_i] / self.sample_size
            best_programs_str.append(f"{avg_loss:.2f}\t\t{p.to_str()}")
        print("Best Programs:")
        print("Loss\t\tProgram")
        for best_program_str in best_programs_str:
            print(best_program_str)

    def get_terminals_from_program(self, program):
        terminals = []
        stack = [program.get_root()]

        while len(stack) > 0:
            node = stack.pop()

            if (type(node) is Terminal) and (type(node["terminal"]["value"]) is not str):
                terminals.append(node)

            if node.get_children() is not None:
                for child in node.get_children():
                    stack.append(child)

        return terminals

    def iterate(self):
        print_time = False

        start = timeit.default_timer()
        self.calculate_loss()
        if print_time:
            print(f"calculate_loss() {timeit.default_timer() - start}")

        start = timeit.default_timer()
        self.calculate_fitness()
        if print_time:
            print(f"calculate_fitness() {timeit.default_timer() - start}")

        start = timeit.default_timer()
        self.selection_and_recombination()
        if print_time:
            print(f"selection_and_recombination() {timeit.default_timer() - start}")

        start = timeit.default_timer()
        self.mutation()
        if print_time:
            print(f"mutation() {timeit.default_timer() - start}")

        start = timeit.default_timer()
        self.hillclimb()
        if print_time:
            print(f"hillclimb() {timeit.default_timer() - start}")

        start = timeit.default_timer()
        self.check_max_depth()
        if print_time:
            print(f"check_max_depth() {timeit.default_timer() - start}")

    def analyse(self):
        total_children = 0
        total_max_depth = 0

        for program in self.programs:
            total_children += program.get_root().get_children_total()
            total_max_depth += program.get_root().get_max_depth()

        avg_children = total_children / len(self.programs)
        avg_max_depth = total_max_depth / len(self.programs)
        # unique_programs = len(np.unique(self.losses, axis=0))

        return {
            "avg_children": avg_children,
            "avg_max_depth": avg_max_depth,
            # "unique_programs": unique_programs
        }

    def calculate_loss(self):
        programs = []
        program_outputs = []

        for program in self.programs:
            output, invalid_output = self.calculate_program_output(program)

            if not invalid_output:
                programs.append(program)
                program_outputs.append(output)

        self.programs = programs
        program_outputs = np.array(program_outputs)

        self.losses = self.calculate_losses_from_program_outputs(program_outputs)
        # Calculate the loss by summing over the losses for each
        # input combination.
        self.loss = self.losses.sum(axis=1)

        # Filter out losses above threshold
        if self.loss_type == "squared":
            loss_filtered = np.nonzero((self.loss / self.sample_size) <= (self.loss_threshold ** 2))[0]
        elif self.loss_type == "manhattan":
            loss_filtered = np.nonzero((self.loss / self.sample_size) <= self.loss_threshold)[0]
        programs = []
        for p_i in loss_filtered:
            programs.append(self.programs[p_i])
        self.programs = programs
        self.losses = self.losses[loss_filtered]
        self.loss = self.loss[loss_filtered]

        # REALLY MESSY CODE BELOW!
        # Find the keep_best best
        unique_loss, unique_loss_i = np.unique(self.loss, return_index=True)
        best_unique_loss_i = np.argsort(unique_loss)[:self.keep_best]
        for i in range(len(best_unique_loss_i)):
            loss_i = unique_loss_i[best_unique_loss_i[i]]
            if (len(self.best_programs_loss) < self.keep_best) or (self.loss[loss_i] < self.best_programs_loss.max()):
                self.best_programs_loss = np.append(self.best_programs_loss, self.loss[loss_i])
                self.best_programs.append(copy.deepcopy(self.programs[loss_i]))

        best_programs = []
        best_programs_loss = []
        unique_best_programs_loss, unique_best_programs_loss_i = np.unique(self.best_programs_loss, return_index=True)
        best_unique_best_programs_loss_i = np.argsort(unique_best_programs_loss)[:self.keep_best]
        for i in range(len(best_unique_best_programs_loss_i)):
            best_loss_i = unique_best_programs_loss_i[best_unique_best_programs_loss_i[i]]
            best_programs.append(copy.deepcopy(self.best_programs[best_loss_i]))
            best_programs_loss.append(self.best_programs_loss[best_loss_i])
        # Set the new best programs and their loss
        self.best_programs = best_programs
        self.best_programs_loss = np.array(best_programs_loss)

    def calculate_program_output(self, program):
        outputs = []
        invalid_output = False

        for i in self.sample:
            input = {}
            for input_symbol in self.input.keys():
                input[input_symbol] = self.input[input_symbol][i]

            output = program.calculate(input)
            if output is None:
                invalid_output = True
                break
            else:
                # Convert to float in order to be able to use
                # numpy's unique (otherwise "The axis argument
                # to unique is not supported for dtype object").
                outputs.append(float(output))

        return outputs, invalid_output

    def calculate_losses_from_program_outputs(self, program_outputs):
        if self.loss_type == "squared":
            return (self.output[self.sample] - program_outputs) ** 2
        elif self.loss_type == "manhattan":
            return np.abs(self.output[self.sample] - program_outputs)

    def calculate_fitness(self):
        children_total = np.array([p.get_root().get_children_total() for p in self.programs])
        max_depth = np.array([p.get_root().get_max_depth() for p in self.programs])
        self.fitness = 1 / (1 + (self.loss / self.sample_size) + (0.25 * max_depth) + (0.05 * children_total))

    def selection_and_recombination(self):
        """
        Based on "A Field Guide to Genetic Programming" 2.3 Selection (pages 14-15).
        Selection by tournament selection.
        """
        programs = []

        if self.percent_elite > 0:
            # Get only unique programs
            unique_losses, unique_losses_i = np.unique(self.losses, return_index=True, axis=0)
            # Sort the unique programs by fitness
            # elite_i corresponds to an index in unique_losses_i
            elite_i = np.argsort(self.fitness[unique_losses_i])[::-1]

            for i in range(math.floor(self.percent_elite * self.population_size)):
                # Get the program index by getting the original index from
                # unique_losses_i.
                program_i = unique_losses_i[elite_i[i]]
                programs.append(copy.deepcopy(self.programs[program_i]))

        while len(programs) < self.population_size:
            if self.selection_method == "fitness_proportionate":
                # Select two programs proportional to their fitness
                random_programs = []
                for p in range(2):
                    program_prob = self.fitness / self.fitness.sum()
                    cumulative_density = program_prob[0]
                    random_number = random.random()
                    i = 0
                    while cumulative_density < random_number:
                        i += 1
                        cumulative_density += program_prob[i]
                    random_programs.append(copy.deepcopy(self.programs[i]))
                program = self.recombination(random_programs[0], random_programs[1])
            elif self.selection_method == "tournament_selection":
                # Select four programs at random
                random_programs = []
                for p in range(4):
                    random_programs.append(math.floor(random.random() * len(self.programs)))
                # Apply tournament selection
                if self.loss[random_programs[0]] < self.loss[random_programs[1]]:
                    program_1 = copy.deepcopy(self.programs[random_programs[0]])
                else:
                    program_1 = copy.deepcopy(self.programs[random_programs[1]])
                if self.loss[random_programs[2]] < self.loss[random_programs[3]]:
                    program_2 = copy.deepcopy(self.programs[random_programs[2]])
                else:
                    program_2 = copy.deepcopy(self.programs[random_programs[3]])
                program = self.recombination(program_1, program_2)
            programs.append(program)

        self.programs = programs

    def recombination(self, program_1, program_2):
        """
        Based on "A Field Guide to Genetic Programming" 2.4 Recombination
        and Mutation (pages 15-16).
        """
        if random.random() < self.crossover_rate:
            # Find two crossover points
            # The number of possible crossover points is the number of
            # total number of nodes the program has (minus 1 for program
            # 1 because the the root cannot be used as a crossover point).
            # How many depths are there?
            program_1_max_depth = program_1.get_root().get_max_depth()
            program_2_max_depth = program_2.get_root().get_max_depth()
            # Choose a depth
            # Create a proportional probability distribution
            program_1_depth_bias = 2
            program_2_depth_bias = 1
            program_1_depth_prob_prop = np.array([1 / (program_1_depth_bias**d) for d in range(program_1_max_depth)])
            program_2_depth_prob_prop = np.array([1 / (program_2_depth_bias**d) for d in range(program_2_max_depth + 1)])
            # Create a probability distribution
            program_1_depth_prob = program_1_depth_prob_prop / program_1_depth_prob_prop.sum()
            program_2_depth_prob = program_2_depth_prob_prop / program_2_depth_prob_prop.sum()
            program_depth_prob = [program_1_depth_prob, program_2_depth_prob]
            # Generate random numbers
            random_numbers = [random.random(), random.random()]
            # Use random numbers to choose depth
            program_depth = [1, 0]
            for i in range(2):
                cumulative_density = 0
                for probability in program_depth_prob[i].tolist():
                    cumulative_density += probability
                    if cumulative_density < random_numbers[i]:
                        program_depth[i] += 1
                    else:
                        break
            # Find crossover points
            program_crossover = [program_1.get_root(), program_2.get_root()]
            for i in range(2):
                for d in range(program_depth[i]):
                    potential_children = [child for child in program_crossover[i].get_children() if (child.get_max_depth() + 1) >= (program_depth[i] - d)]
                    child_index = math.floor(random.random() * len(potential_children))
                    program_crossover[i] = potential_children[child_index]
            # Apply crossover
            parent = program_crossover[0].get_parent()
            crossover_index = parent.get_children().index(program_crossover[0])
            parent.remove_child_at_index(crossover_index)
            parent.add_child_at_index(program_crossover[1], crossover_index)
            program_crossover[1].set_parent(parent)

            return program_1
        else:
            # Pick one of the programs at random
            if random.random() < 0.5:
                return program_1
            else:
                return program_2

    def mutation(self):
        """
        Based on "A Field Guide to Genetic Programming" 2.4 Recombination
        and Mutation (pages 16-17).
        """
        for p_i, program in enumerate(self.programs):
            if (p_i >= math.floor(self.percent_elite * self.population_size)) or self.mutate_elite:
                stack = [program.get_root()]
                # Depth-first search
                while len(stack) > 0:
                    node = stack.pop()
                    new_node = None

                    if random.random() < self.mutation_rate:
                        # Apply mutation
                        random_number = random.random()
                        parent = node.get_parent()

                        if random_number < self.shrink_prob:
                            # Shrink mutation
                            new_node, did_shrink = self.shrink_mutation(node, parent)
                        elif random_number < (self.shrink_prob + self.hoist_prob):
                            # Hoist mutation
                            did_hoist = self.hoist_mutation(program, node, parent)
                            if did_hoist:
                                # Clear stack for the DFS algorithm
                                stack = []
                        elif random_number < (self.shrink_prob + self.hoist_prob + self.grow_prob):
                            # Grow mutation
                            new_node = self.grow_mutation(program, node, parent)
                            if parent is None:
                                # The whole program has been replaced by a random tree,
                                # skip to the next program.
                                break
                        else:
                            # Point mutation
                            new_node = self.point_mutation(program, node, parent)

                    if new_node is not None:
                        # Set node to new_node for the DFS algorithm
                        node = new_node

                    if node.get_children() is not None:
                        for child in node.get_children():
                            stack.append(child)

    def shrink_mutation(self, node, parent):
        new_node = None

        # Only apply shrink if node is a non-terminal, and is not the root
        if (type(node) is NonTerminal) and (parent is not None):
            terminal = self.get_function_terminal(parent.get_function(), False, True)
            new_node = Terminal(parent, terminal)
            node_index = parent.get_children().index(node)
            parent.remove_child_at_index(node_index)
            parent.add_child_at_index(new_node, node_index)
            return new_node, True
        return new_node, False

    def hoist_mutation(self, program, node, parent):
        # Only apply hoist if node is a non-terminal, and is not the root
        if (type(node) is NonTerminal) and (parent is not None):
            program.set_root(node)
            node.set_parent(None)
            return True
        return False

    def grow_mutation(self, program, node, parent):
        # Generate random subtree
        parent_function = parent.get_function() if (parent is not None) else None
        function = self.get_function_terminal(parent_function, True, False)
        new_node = NonTerminal(parent, function)
        self.init_node(new_node, False, 0)
        # Remove old tree and add new tree
        if parent is None:
            program.set_root(new_node)
        else:
            node_index = parent.get_children().index(node)
            parent.remove_child_at_index(node_index)
            parent.add_child_at_index(new_node, node_index)

        return new_node

    def point_mutation(self, program, node, parent):
        if type(node) is NonTerminal:
            # Select a non terminal of the same arity at random
            arity = node.get_function()["function"].get_arity()
            if parent is not None:
                function = self.get_function_terminal(parent.get_function(), True, False, arity)
            else:
                function = self.get_function_terminal(None, True, False, arity)
            new_node = self.point_mutation_function(program, node, parent, function)
        else:
            # Select a terminal at random
            terminal = self.get_function_terminal(parent.get_function(), False, True)
            new_node = self.point_mutation_terminal(node, parent, terminal)

        return new_node

    def point_mutation_function(self, program, node, parent, function):
        new_node = NonTerminal(parent, function)
        if parent is None:
            program.set_root(new_node)
        else:
            node_index = parent.get_children().index(node)
            parent.remove_child_at_index(node_index)
            parent.add_child_at_index(new_node, node_index)
        for child in node.get_children():
            new_node.add_child(child)
            child.set_parent(new_node)

        return new_node

    def point_mutation_terminal(self, node, parent, terminal):
        new_node = Terminal(parent, terminal)
        node_index = parent.get_children().index(node)
        parent.remove_child_at_index(node_index)
        parent.add_child_at_index(new_node, node_index)

        return new_node

    def hillclimb(self):
        for p_i in range(len(self.programs)):
            stack = [self.programs[p_i].get_root()]
            node_number = 0
            # Depth-first search
            while len(stack) > 0:
                node = stack.pop()
                node_number += 1

                if random.random() < self.hillclimb_rate:
                    # Apply hillclimb
                    hillclimb_programs = []
                    parent = node.get_parent()

                    # Try random shrink
                    program_shrink = copy.deepcopy(self.programs[p_i])
                    node_shrink = self.get_node_from_node_number(program_shrink.get_root(), node_number)
                    new_node_shrink, did_shrink = self.shrink_mutation(node_shrink, node_shrink.get_parent())
                    if did_shrink:
                        hillclimb_programs.append(program_shrink)

                    # Try hoist
                    program_hoist = copy.deepcopy(self.programs[p_i])
                    node_hoist = self.get_node_from_node_number(program_hoist.get_root(), node_number)
                    did_hoist = self.hoist_mutation(program_hoist, node_hoist, node_hoist.get_parent())
                    if did_hoist:
                        hillclimb_programs.append(program_hoist)

                    # Try random grow
                    program_grow = copy.deepcopy(self.programs[p_i])
                    node_grow = self.get_node_from_node_number(program_grow.get_root(), node_number)
                    self.grow_mutation(program_grow, node_grow, node_grow.get_parent())
                    hillclimb_programs.append(program_grow)

                    # Systematically try every possible point mutation
                    if type(node) is NonTerminal:
                        # Get all functions of the same arity
                        arity = node.get_function()["function"].get_arity()
                        if parent is not None:
                            functions = self.get_possible_functions(parent.get_function(), arity)
                        else:
                            functions = self.get_possible_functions(None, arity)

                        for function in functions:
                            program_point = copy.deepcopy(self.programs[p_i])
                            node_point = self.get_node_from_node_number(program_point.get_root(), node_number)
                            self.point_mutation_function(program_point, node_point, node_point.get_parent(), function)
                            hillclimb_programs.append(program_point)
                    else:
                        terminals = self.get_possible_terminals(parent.get_function())

                        for terminal in terminals:
                            program_point = copy.deepcopy(self.programs[p_i])
                            node_point = self.get_node_from_node_number(program_point.get_root(), node_number)
                            self.point_mutation_terminal(node_point, node_point.get_parent(), terminal)
                            hillclimb_programs.append(program_point)

                    # Calculate the loss of each mutation
                    hillclimb_programs_2 = []
                    program_outputs = []

                    for program in hillclimb_programs:
                        output, invalid_output = self.calculate_program_output(program)

                        if not invalid_output:
                            hillclimb_programs_2.append(program)
                            program_outputs.append(output)

                    hillclimb_programs = hillclimb_programs_2
                    program_outputs = np.array(program_outputs)

                    if len(program_outputs) > 0:
                        losses = self.calculate_losses_from_program_outputs(program_outputs)
                        # Calculate the loss by summing over the losses for each
                        # input combination.
                        loss = losses.sum(axis=1)

                        # Only keep the programs that are better
                        # Calculate the loss of the program
                        output, invalid_output = self.calculate_program_output(self.programs[p_i])

                        if not invalid_output:
                            program_losses = self.calculate_losses_from_program_outputs(np.array(output))
                            program_loss = program_losses.sum()
                        else:
                            program_loss = loss.max()

                        program_hillclimb_program_difference = program_loss - loss
                        hillclimb_better_i = np.nonzero(program_hillclimb_program_difference > 0)[0]
                        hillclimb_programs = [h_p for h_p_i, h_p in enumerate(hillclimb_programs) if h_p_i in hillclimb_better_i]

                        if len(hillclimb_better_i) > 0:
                            # Create a pdf
                            loss_better = program_hillclimb_program_difference[hillclimb_better_i]
                            loss_better_pdf = loss_better / loss_better.sum()
                            new_program = item_from_pdf(hillclimb_programs, loss_better_pdf)
                            self.programs[p_i] = new_program

                            if self.multiple_hillclimb:
                                # Reset stack and node_number for DFS algorithm
                                stack = [new_program.get_root()]
                            else:
                                stack = []
                            node_number = 0
                            continue

                if node.get_children() is not None:
                    for child in node.get_children():
                        stack.append(child)

    def get_node_from_node_number(self, root, node_number):
        """ Given a root and node number, assuming the depth-first search
            algorithm, return the node at the node number. """
        stack_2 = [root]
        node_2 = None
        node_number_2 = 0
        while node_number > node_number_2:
            node_2 = stack_2.pop()
            node_number_2 += 1
            if node_2.get_children() is not None:
                for child in node_2.get_children():
                    stack_2.append(child)

        return node_2

    def check_max_depth(self):
        for program in self.programs:
            root = program.get_root()
            if root.get_max_depth() > self.max_depth:
                # Prune the tree
                self.prune_tree(root, self.max_depth)

    def prune_tree(self, node, max_depth):
        if max_depth == 0:
            # Replace the non-terminal with a terminal (same as shrink mutation)
            parent = node.get_parent()
            terminal = self.get_function_terminal(parent.get_function(), False, True)
            new_node = Terminal(parent, terminal)
            node_index = parent.get_children().index(node)
            parent.remove_child_at_index(node_index)
            parent.add_child_at_index(new_node, node_index)
        else:
            for child in node.get_children():
                if child.get_max_depth() > max_depth - 1:
                    self.prune_tree(child, max_depth - 1)
