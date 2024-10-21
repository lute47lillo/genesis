import numpy as np
import copy
import zss
import matplotlib.pyplot as plt


class Node:
    def __init__(self, value, children=None):
        self.value = value  # Function or terminal
        self.children = children if children is not None else []

    def is_terminal(self):
        return len(self.children) == 0

    def __str__(self):
        if self.is_terminal():
            return str(self.value)
        else:
            return f"({self.value} {' '.join(str(child) for child in self.children)})"


class Individual:
    def __init__(self, tree=None, id=None, ancestors=None, generation=0):
        self.tree = tree if tree is not None else self.random_tree(depth=3)
        self.fitness = None
        self.id = id if id is not None else np.random.randint(1e9)
        self.ancestors = ancestors if ancestors is not None else set()
        self.generation = generation  # Track the generation of the individual

    def random_tree(self, depth):
        if depth == 0:
            # Return a terminal node
            terminal = np.random.choice(['x', '1', '2', '3'])  # Example terminals
            return Node(terminal)
        else:
            # Return a function node with children
            # function = np.random.choice(['+', '-', '*', '/'])  # Example functions
            function = np.random.choice(['+', '-', '*', '/', 'cos', 'sin', 'exp', 'sqrt']) # Ackley and rastrigin
            children = [self.random_tree(depth - 1) for _ in range(2)]  # Binary functions
            return Node(function, children)
    
    def __str__(self):
        return str(self.tree)

class GeneticAlgorithmGP:
    def __init__(self, pop_size, generations, mutation_rate, allowed_distance=None, max_depth=5):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.allowed_distance = allowed_distance
        self.max_depth = max_depth
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []
        self.lineage_data = []
    
    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            individual = Individual()
            self.population.append(individual)
    
    def calculate_fitness(self, fitness_function):
        for individual in self.population:
            individual.fitness = fitness_function(individual.tree)
    
    def tournament_selection(self, k=3):
        selected = []
        for _ in range(self.pop_size):
            participants = np.random.choice(self.population, k)
            winner = max(participants, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected
    
    def genetic_distance(self, ind1, ind2):
        return self.tree_edit_distance(ind1.tree, ind2.tree)
    
    def select_random_node(self, tree):
        nodes = self.get_all_nodes(tree)
        return np.random.choice(nodes)

    def get_all_nodes(self, tree):
        nodes = [tree]
        for child in tree.children:
            nodes.extend(self.get_all_nodes(child))
        return nodes
    
    def can_mate(self, ind1, ind2, allowed_distance):
        distance = self.tree_edit_distance(ind1.tree, ind2.tree)
        return distance >= allowed_distance
    
    def tree_depth(self, node):
        if node is None or node.is_terminal():
            return 1
        else:
            return 1 + max(self.tree_depth(child) for child in node.children)
    
    def tree_edit_distance(self, node1, node2):
        if node1 is None and node2 is None:
            return 0
        if node1 is None or node2 is None:
            return 1
        if node1.value != node2.value:
            cost = 1
        else:
            cost = 0
        # Calculate distance for children
        child_distances = 0
        for child1, child2 in zip(node1.children, node2.children):
            child_distances += self.tree_edit_distance(child1, child2)
            
        # Add distances for unmatched children
        child_distances += abs(len(node1.children) - len(node2.children))
        return cost + child_distances
    
    def tree_edit_distance_zss(self, node1, node2):
        def get_children(node):
            return node.children
        return zss.simple_distance(node1, node2, get_children)
    
    # ------------- Croosover --------------------- #
    
    def crossover(self, parent1, parent2, max_depth=10):
        if self.allowed_distance is not None:
            if not self.can_mate(parent1, parent2, self.allowed_distance):
                return None, None
        
        # Clone parents to avoid modifying originals
        child1 = copy.deepcopy(parent1.tree)
        child2 = copy.deepcopy(parent2.tree)

        # Select random crossover points
        node1 = self.select_random_node(child1)
        node2 = self.select_random_node(child2)

        # Swap subtrees
        node1.value, node2.value = node2.value, node1.value
        node1.children, node2.children = node2.children, node1.children
        
         # Check for depth constraints
        if self.tree_depth(child1) > max_depth or self.tree_depth(child2) > max_depth:
            return None, None  # Discard offspring exceeding max depth

        # Create new individuals
        offspring1 = Individual(tree=child1, ancestors=parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id}), generation=parent1.generation + 1)
        offspring2 = Individual(tree=child2, ancestors=parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id}), generation=parent1.generation + 1)

        return offspring1, offspring2

    # ------------- Croosover --------------------- #
    
    def mutate(self, individual, max_depth=10):

        # Clone individual to avoid modifying original
        mutated_tree = copy.deepcopy(individual.tree)

        # Select a random node to mutate
        node_to_mutate = self.select_random_node(mutated_tree)

        # Replace the subtree with a new random subtree
        new_subtree = individual.random_tree(depth=2)  # Adjust depth as needed
        node_to_mutate.value = new_subtree.value
        node_to_mutate.children = new_subtree.children
        
        # Ensure the mutated tree does not exceed max depth
        if self.tree_depth(mutated_tree) > max_depth:
            return  # Discard mutation or handle accordingly

        # Update individual
        individual.tree = mutated_tree

    def measure_diversity(self):
        # Calculate diversity based on tree structures
        # Example: Average pairwise tree edit distance
        total_distance = 0
        count = 0
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self.tree_edit_distance(self.population[i].tree, self.population[j].tree)
                total_distance += distance
                count += 1
        if count == 0:
            return 0
        diversity = total_distance / count
        return diversity
    
    def run(self, fitness_function):
        self.initialize_population()
    
        for gen in range(self.generations):
            
            # Calculate fitness
            self.calculate_fitness(fitness_function)
    
            # Record best fitness
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_fitness_list.append(best_individual.fitness)
    
            # Measure diversity
            diversity = self.measure_diversity()
            self.diversity_list.append(diversity)
    
            # Collect lineage data
            for ind in self.population:
                self.lineage_data.append({
                    'id': ind.id,
                    'ancestors': ind.ancestors,
                    'generation': gen,
                    'fitness': ind.fitness
                })
    
            # Selection
            selected = self.tournament_selection()
    
            # Crossover and Mutation
            next_population = []
            i = 0
            while len(next_population) < self.pop_size:
                parent1 = selected[i % len(selected)]
                parent2 = selected[(i + 1) % len(selected)]
                offspring = self.crossover(parent1, parent2)
    
                if offspring[0] is not None and offspring[1] is not None:
                    self.mutate(offspring[0])
                    self.mutate(offspring[1])
                    next_population.extend(offspring)
                else:
                    # Introduce new random individuals to maintain population size
                    new_individual = Individual()
                    next_population.append(new_individual)
                    if len(next_population) < self.pop_size:
                        new_individual = Individual()
                        next_population.append(new_individual)
                i += 2
    
            self.population = next_population[:self.pop_size]
    
            # Optional: Print progress
            if (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}: Best Fitness = {best_individual.fitness:.4f}, Diversity = {diversity:.4f}")
    
        return self.best_fitness_list, self.diversity_list
    

def ackley_function(x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return term1 + term2 + a + np.exp(1)

def rastrigin_function(x, A=10):
    d = len(x)
    return A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Define target functions
def ackley_target(x):
    return ackley_function(x)

def rastrigin_target(x):
    return rastrigin_function(x)

# Define input vectors (sampled within the search space)
def generate_input_vectors(d=2, num_samples=100):
    # For Ackley, typically x_i ∈ [-5, 5]
    # For Rastrigin, typically x_i ∈ [-5.12, 5.12]
    return [np.random.uniform(-5, 5, d) for _ in range(num_samples)]

# Initialize input vectors for Ackley
input_vectors_ackley = generate_input_vectors(d=2, num_samples=100)

# Initialize input vectors for Rastrigin
input_vectors_rastrigin = generate_input_vectors(d=2, num_samples=100)

# Define fitness functions
def fitness_ackley(genome):
    return complex_fitness_function(genome, ackley_target, input_vectors_ackley)

def fitness_rastrigin(genome):
    return complex_fitness_function(genome, rastrigin_target, input_vectors_rastrigin)

def count_nodes(node):
    """
        Count the number of nodes (functions and terminals) in a program tree.
    """
    if node is None:
        return 0
    count = 1  # Count the current node
    for child in node.children:
        count += count_nodes(child)
    return count

def evaluate_tree(node, x):
    """
    Evaluate the program tree with input vector x.

    Parameters:
    - node (Node): Current node in the program tree.
    - x (numpy.ndarray): Input vector.

    Returns:
    - result (float): Result of the program's evaluation.
    """
    if node.is_terminal():
        if node.value == 'x':
            # Assume 'x' corresponds to a specific dimension, e.g., x1
            return x[0]  # Modify as needed for multi-dimensional x
        else:
            return float(node.value)
    else:
        # Define function implementations
        func = node.value
        args = [evaluate_tree(child, x) for child in node.children]
        try:
            if func == '+':
                return args[0] + args[1]
            elif func == '-':
                return args[0] - args[1]
            elif func == '*':
                return args[0] * args[1]
            elif func == '/':
                return args[0] / args[1] if args[1] != 0 else 1.0  # Protected division
            elif func == 'cos':
                return np.cos(args[0])
            elif func == 'sin':
                return np.sin(args[0])
            elif func == 'exp':
                return np.exp(args[0])
            elif func == 'sqrt':
                return np.sqrt(args[0]) if args[0] >= 0 else 0.0  # Protected sqrt
            else:
                # Undefined function
                raise ValueError(f"Undefined function: {func}")
        except:
            # Handle any unexpected errors
            return 0.0

def complex_fitness_function(genome, target_function, input_vectors, lambda_complexity=0.1):
    mse_total = 0.0
    complexity = count_nodes(genome)
    
    for x in input_vectors:
        try:
            output = evaluate_tree(genome, x)
            target = target_function(x)
            mse = (output - target) ** 2
            mse_total += mse
        except Exception as e:
            mse_total += 1e6  # Large penalty for errors
    
    mse_average = mse_total / len(input_vectors)
    fitness = (1 / (mse_average + 1e-6)) * np.exp(-lambda_complexity * complexity)
    return fitness

# Initialize GP-based GA for Ackley
ga_gp_ackley = GeneticAlgorithmGP(
    pop_size=200,
    generations=100,
    mutation_rate=0.005,
    allowed_distance=10,  # Adjust based on inbreeding prevention
    max_depth=10
)
import os
# Run GP-based GA for Ackley
best_fitness_ackley, diversity_ackley = ga_gp_ackley.run(fitness_ackley)

# Plotting Results for Ackley
plt.figure(figsize=(14, 6))

# Best Fitness Over Generations
plt.subplot(1, 2, 1)
plt.plot(best_fitness_ackley, label='Best Fitness')
plt.title('Best Fitness Over Generations (Ackley GP)')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()

# Genetic Diversity Over Generations
plt.subplot(1, 2, 2)
plt.plot(diversity_ackley, label='Genetic Diversity', color='orange')
plt.title('Genetic Diversity Over Generations (Ackley GP)')
plt.xlabel('Generation')
plt.ylabel('Diversity')
plt.legend()

plt.tight_layout()
plt.savefig(f'{os.getcwd()}/NO_Inbreeding_test_ackley.png')
plt.close()

# Similarly, you can run and plot for Rastrigin
ga_gp_rastrigin = GeneticAlgorithmGP(
    pop_size=200,
    generations=100,
    mutation_rate=0.005,
    allowed_distance=10,
    max_depth=10
)

# Run GP-based GA for Rastrigin
best_fitness_rastrigin, diversity_rastrigin = ga_gp_rastrigin.run(fitness_rastrigin)

# Plotting Results for Rastrigin
plt.figure(figsize=(14, 6))

# Best Fitness Over Generations
plt.subplot(1, 2, 1)
plt.plot(best_fitness_rastrigin, label='Best Fitness')
plt.title('Best Fitness Over Generations (Rastrigin GP)')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()

# Genetic Diversity Over Generations
plt.subplot(1, 2, 2)
plt.plot(diversity_rastrigin, label='Genetic Diversity', color='orange')
plt.title('Genetic Diversity Over Generations (Rastrigin GP)')
plt.xlabel('Generation')
plt.ylabel('Diversity')
plt.legend()

plt.tight_layout()
plt.savefig(f'{os.getcwd()}/NO_Inbreeding_ratrigin.png')
plt.close()

# ------------- Symbolic

# def symbolic_regression_fitness(genome, target_function, input_values):
#     """
#     Evaluate the fitness of a genome (function tree) based on how well it approximates the target function.

#     Parameters:
#     - genome (Node): The root node of the function tree.
#     - target_function (callable): The target function to approximate.
#     - input_values (list or numpy.ndarray): Input values for evaluation.

#     Returns:
#     - fitness (float): The inverse of the mean squared error.
#     """
#     def evaluate_tree(node, x):
#         if node.is_terminal():
#             if node.value == 'x':
#                 return x
#             else:
#                 return float(node.value)
#         else:
#             func = node.value
#             left = evaluate_tree(node.children[0], x)
#             right = evaluate_tree(node.children[1], x)
#             if func == '+':
#                 return left + right
#             elif func == '-':
#                 return left - right
#             elif func == '*':
#                 return left * right
#             elif func == '/':
#                 return left / right if right != 0 else 1.0  # Handle division by zero
#             else:
#                 raise ValueError(f"Unknown function: {func}")

#     errors = []
#     for x in input_values:
#         try:
#             output = evaluate_tree(genome, x)
#             target = target_function(x)
#             error = (output - target) ** 2
#             errors.append(error)
#         except Exception as e:
#             errors.append(float('inf'))  # Penalize invalid programs

#     mse = np.mean(errors)
#     fitness = 1 / (mse + 1e-6)  # Avoid division by zero
#     return fitness

# def target_function(x):
#     return x ** 2 + 2 * x + 1  # Example: quadratic function

# def symbolic_regression_fitness_function(genome):
#     input_values = np.linspace(-10, 10, 50)
#     return symbolic_regression_fitness(genome, target_function, input_values)

# import matplotlib.pyplot as plt

# # GA Parameters
# pop_size = 200
# generations = 100
# mutation_rate = 0.05
# allowed_distance = 10  # Adjust based on your inbreeding prevention strategy

# # Initialize GP GA
# ga_gp = GeneticAlgorithmGP(
#     pop_size=pop_size,
#     generations=generations,
#     mutation_rate=mutation_rate,
#     allowed_distance=allowed_distance,
#     max_depth=5
# )

# import os
# # Run GP GA
# best_fitness_list, diversity_list = ga_gp.run(symbolic_regression_fitness_function)

# # Plotting Results
# plt.figure(figsize=(14, 6))

# # Best Fitness Over Generations
# plt.subplot(1, 2, 1)
# plt.plot(best_fitness_list, label='Best Fitness')
# plt.title('Best Fitness Over Generations (Symbolic Regression GP)')
# plt.xlabel('Generation')
# plt.ylabel('Fitness')
# plt.legend()

# # Genetic Diversity Over Generations
# plt.subplot(1, 2, 2)
# plt.plot(diversity_list, label='Genetic Diversity', color='orange')
# plt.title('Genetic Diversity Over Generations (Symbolic Regression GP)')
# plt.xlabel('Generation')
# plt.ylabel('Diversity')
# plt.legend()

# plt.tight_layout()
# plt.savefig(f"{os.getcwd()}/test_div_fitGP_noInbreeding.png")
# plt.close()


# """Nuanced version of tree edit distance"""
# # import zss

# # def tree_edit_distance_zss(node1, node2):
# #     def get_children(node):
# #         return node.children
    
# #     return zss.simple_distance(node1, node2, get_children)