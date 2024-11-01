import numpy as np
import copy
import os
import zss
import random
import matplotlib.pyplot as plt
import benchmark_factory as bf
import util
import experiments as exp
import plotting as plot
import gp_math
from argparse import Namespace
# import tests as tests
import unittest

"""
    TODO LIST
    
        - Fix issues such as:
            /gpfs1/home/e/l/elillopo/diversity/src/genetic_programming.py:337: RuntimeWarning: overflow encountered in exp
            return np.exp(args[0])
            /gpfs1/home/e/l/elillopo/diversity/src/genetic_programming.py:327: RuntimeWarning: overflow encountered in divide
            result = np.true_divide(args[0], denominator)
            /gpfs1/home/e/l/elillopo/diversity/src/genetic_programming.py:335: RuntimeWarning: invalid value encountered in sin
            return np.sin(args[0])
            /gpfs1/home/e/l/elillopo/diversity/src/genetic_programming.py:333: RuntimeWarning: invalid value encountered in cos
            return np.cos(args[0])
            /gpfs1/home/e/l/elillopo/diversity/src/genetic_programming.py:321: RuntimeWarning: overflow encountered in multiply
            return args[0] * args[1]
            /gpfs1/home/e/l/elillopo/diversity/src/genetic_programming.py:321: RuntimeWarning: invalid value encountered in multiply
            return args[0] * args[1]
            /users/e/l/elillopo/.conda/envs/neurobotics/lib/python3.11/site-packages/numpy/core/fromnumeric.py:88: RuntimeWarning: invalid value encountered in reduce
            return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
            /gpfs1/home/e/l/elillopo/diversity/src/genetic_programming.py:321: RuntimeWarning: invalid value encountered in scalar multiply
            return args[0] * args[1]
            
            that are in evaluate TREE function
            
        - Add the NGUYEN benchmark functions from "Effective Adaptive Mutation Rates for Program Synthesis" Paper
        and from "Better GP benchmarks: community survey results and proposals" paper.

"""


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
    def __init__(self, args, tree=None, id=None, ancestors=None, generation=0):
        self.args = args
        self.bounds = self.args.bounds
        self.max_depth = self.args.max_depth # TODO
        self.initial_depth = self.args.initial_depth
        self.tree = tree if tree is not None else self.random_tree(depth=self.initial_depth) # Initial depth of 6 as in paper
        self.fitness = None
        self.id = id if id is not None else np.random.randint(1e9)
        self.ancestors = ancestors if ancestors is not None else set()
        self.generation = generation  # Track the generation of the individual
        
    def get_function_arity(self, function):
        arity_dict = {
            '+': 2,
            '-': 2,
            '*': 2,
            '/': 2,
            'sin': 1,
            'cos': 1,
            'log': 1
        }
        return arity_dict.get(function, 0)
    
    def random_tree(self, depth):
    
        if depth == 0:
            # Return a terminal node
            terminal = np.random.choice(['x', '1.0'])
            if terminal == '1.0':
                return Node(1.0)
            else:
                return Node('x')
        else:
            # Return a function node with appropriate arity
            function = np.random.choice(['+', '-', '*', '/', 'sin', 'cos', 'log'])
            arity = self.get_function_arity(function)
            children = [self.random_tree(depth - 1) for _ in range(arity)]
            return Node(function, children)
    
    def __str__(self):
        return str(self.tree)

class GeneticAlgorithmGP:
    def __init__(self, args, inbred_threshold=None):
        self.args = args
        self.pop_size = args.pop_size
        self.generations = args.generations
        self.mutation_rate = args.mutation_rate
        self.inbred_threshold = inbred_threshold
        self.max_depth = args.max_depth
        self.initial_depth = args.initial_depth
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []
    
    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            individual = Individual(self.args)
            self.population.append(individual)
    
    def calculate_fitness(self, fitness_function, curr_gen):
        for individual in self.population:
            individual.fitness, success = fitness_function(individual.tree)
            if success:
                print(f"Successful individual found in generation {curr_gen}")
                print(f"Function: {individual.tree}")
            
    def tournament_selection(self, k=3):
        selected = []
        for _ in range(self.pop_size):
            participants = np.random.choice(self.population, k)
            winner = max(participants, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected
    
    def select_random_node(self, tree):
        nodes = self.get_all_nodes(tree)
        return np.random.choice(nodes)

    def get_all_nodes(self, tree):
        nodes = [tree]
        for child in tree.children:
            nodes.extend(self.get_all_nodes(child))
        return nodes
    
    def select_random_node_with_parent(self, tree):
        """
        Selects a random node along with its parent.
        Returns a tuple (parent_node, selected_node).
        If the selected node is the root, parent_node is None.
        """
        all_nodes = self.get_all_nodes_with_parent(tree)
        if not all_nodes:
            return None, None
        return random.choice(all_nodes)

    def get_all_nodes_with_parent(self, node, parent=None):
        """
        Recursively collects all nodes in the tree along with their parent.
        """
        nodes = [(parent, node)]
        for child in node.children:
            nodes.extend(self.get_all_nodes_with_parent(child, node))
        return nodes
    
    def can_mate(self, ind1, ind2, inbred_threshold):
        distance = self.tree_edit_distance(ind1.tree, ind2.tree)
        return distance >= inbred_threshold
    
    def tree_depth(self, node):
        if node is None:
            return 0
        if node.is_terminal():
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
    
    def crossover(self, parent1, parent2):
        print(f"\nCurrent nº individuals in population: {len(self.population)}")
        if self.inbred_threshold is not None:
            if not self.can_mate(parent1, parent2, self.inbred_threshold):
                return None, None

        # Clone parents to avoid modifying originals
        child1 = copy.deepcopy(parent1.tree)
        child2 = copy.deepcopy(parent2.tree)

        max_attempts = 10
        for attempt in range(max_attempts+1):
            # Select random nodes with their parents
            parent_node1, node1 = self.select_random_node_with_parent(child1)
            parent_node2, node2 = self.select_random_node_with_parent(child2)

            if node1 is None or node2 is None:
                print(f"Attempt {attempt+1}: One of the selected nodes is None. Retrying...")
                continue  # Try again

            # Check if both nodes have the same arity
            arity1 = parent1.get_function_arity(node1.value)
            arity2 = parent2.get_function_arity(node2.value)
            if arity1 != arity2:
                print(f"Attempt {attempt+1}: Arities do not match (arity1={arity1}, arity2={arity2}). Retrying...")
                continue  # Arities don't match, select another pair

            # Swap entire subtrees
            if parent_node1 is None:
                # node1 is root of child1
                child1 = copy.deepcopy(node2)
            else:
                # Find the index of node1 in its parent's children and replace it
                try:
                    index = parent_node1.children.index(node1)
                    parent_node1.children[index] = copy.deepcopy(node2)
                except ValueError:
                    print(f"Attempt {attempt}: node1 not found in parent_node1's children. Retrying...")
                    continue  # node1 not found, try again

            if parent_node2 is None:
                # node2 is root of child2
                child2 = copy.deepcopy(node1)
            else:
                # Find the index of node2 in its parent's children and replace it
                try:
                    index = parent_node2.children.index(node2)
                    parent_node2.children[index] = copy.deepcopy(node1)
                except ValueError:
                    print(f"Attempt {attempt}: node2 not found in parent_node2's children. Retrying...")
                    continue  # node2 not found, try again

            # Check for depth constraints
            depth_child1 = self.tree_depth(child1)
            depth_child2 = self.tree_depth(child2)
            if depth_child1 > self.max_depth or depth_child2 > self.max_depth:
                print(f"Attempt {attempt}: Offspring exceed max depth ({self.max_depth}). Retrying...")
                # Revert the swap by reinitializing the trees
                child1 = copy.deepcopy(parent1.tree)
                child2 = copy.deepcopy(parent2.tree)
                continue  # Try again

            # Successful crossover
            print(f"Attempt {attempt}: Crossover successful.")
            break
        else:
            # Failed to perform a valid crossover within max_attempts
            print("Crossover failed after maximum attempts.")
            return None, None

        # Create new individuals
        offspring1 = Individual(
            self.args,
            tree=child1,
            ancestors=parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id}),
            generation=parent1.generation + 1
        )
        offspring2 = Individual(
            self.args,
            tree=child2,
            ancestors=parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id}),
            generation=parent1.generation + 1
        )

        return offspring1, offspring2


    # ------------- Croosover --------------------- #
    
    def mutate(self, individual):
        
        # Clone individual to avoid modifying original
        mutated_tree = copy.deepcopy(individual.tree)

        # Select a random node to mutate
        node_to_mutate = self.select_random_node(mutated_tree)
        if node_to_mutate is None:
            print("Warning: No node selected for mutation")
            return  # Cannot mutate without a node

        # Replace the subtree with a new random subtree
        new_subtree = individual.random_tree(self.initial_depth) 
        
        # Ensure that the new_subtree has the correct arity
        required_children = individual.get_function_arity(new_subtree.value)
        if len(new_subtree.children) != required_children:
            print(f"Warning: New subtree has incorrect arity for function '{new_subtree.value}'")
            return  # Discard mutation
        
        node_to_mutate.value = new_subtree.value
        node_to_mutate.children = new_subtree.children
        
        # Ensure the mutated tree does not exceed max depth
        if self.tree_depth(mutated_tree) > self.max_depth:
            print("Warning: Mutated tree exceeds max depth")
            return  # Discard mutation

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
            
            print(f"Current generation: {gen+1}")
            print(f"The population:")
            for indiv in self.population:
                print(f"Tree: {indiv.tree}, depth: {self.tree_depth(indiv.tree)}")
            print()
            
            # Calculate fitness
            self.calculate_fitness(fitness_function, gen)
                
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_fitness_list.append(best_individual.fitness)
    
            # Measure diversity
            diversity = self.measure_diversity()
            self.diversity_list.append(diversity)
    
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
                    new_individual = Individual(self.args)
                    next_population.append(new_individual)
                    if len(next_population) < self.pop_size:
                        new_individual = Individual(self.args)
                        next_population.append(new_individual)
                i += 2
    
            self.population = next_population[:self.pop_size]
    
            # Optional: Print progress
            if (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}: Best Fitness = {best_individual.fitness:.4f}, Diversity = {diversity:.4f}")
    
        return self.best_fitness_list, self.diversity_list
    
class GPLandscape:
    
    def __init__(self, args, bounds):
        self.args = args
        self.bounds = bounds
        self.lambda_complexity = 0.01
        self.data = self.generate_data() # Generate all data points
        self.total_exc = 0
    
    def count_nodes(self, node):
        """
            Count the number of nodes (functions and terminals) in a program tree.
        """
        if node is None:
            return 0
        count = 1  # Count the current node
        for child in node.children:
            count += self.count_nodes(child)
        return count
    
    def evaluate_tree(self, node, x):
        
        if node is None:
            print(f"Warning: evaluate_tree called with node={node}")
            return 0.0

        if node.is_terminal():
            if node.value == 'x':
                # print(f"Evaluating terminal node 'x': {x}")
                return x  # x is a scalar
            else:
                try:
                    val = float(node.value)
                    # print(f"Evaluating terminal node '{node.value}': {val}")
                    return val
                except ValueError as e:
                    # print(f"Error converting node value to float: {e}")
                    return 0.0
        else:
            # TODO: Problem IDENTIFIED. Some func have arity 1 not arity 2. But, there is always two children added.
            func = node.value
            args_tree = [self.evaluate_tree(child, x) for child in node.children]
            # print(args_tree[0], func, args_tree[1])
            # actual_arity = len(node.children)
            # expected_arity = util.get_function_arity(func)
            # print(f"Error: Function '{func}' expects {expected_arity} children, but got {actual_arity}. Returning 0.0")
  
            try:
                if func == '+':
                    result = gp_math.protected_sum(args_tree[0], args_tree[1])
                elif func == '-':
                    result = gp_math.protected_subtract(args_tree[0], args_tree[1])
                elif func == '*':
                    result = gp_math.protected_mult(args_tree[0], args_tree[1])
                elif func == '/':
                    result = gp_math.protected_divide(args_tree[0], args_tree[1])
                elif func == 'sin':
                    result = gp_math.protected_sin(args_tree[0])
                elif func == 'cos':
                    result = gp_math.protected_cos(args_tree[0])
                elif func == 'log':
                    result = gp_math.protected_log(args_tree[0])
                else:
                    raise ValueError(f"Undefined function: {func}")

                # Clamp the result to the interval [-1e6, 1e6]
                # Extracted of paper: Effective Adaptive Mutation Rates for Program Synthesis by Ni, Andrew and Spector, Lee 2024
                result = np.clip(result, -1e6, 1e6)

                return result
            except Exception as e:
                # print(f"Error in evaluate_tree: {e}")
                self.total_exc += 1
                
                # if self.total_exc > 10:
                #     exit()
   
                # Handle any unexpected errors
                return 0.0  # Return 0.0 for any error
            
    def target_function(self, x):
        
        # Define target functions
        if self.args.benchmark == 'ackley':
            return bf.ackley_function(x)
        
        if self.args.benchmark == "nguyen1":
            return bf.nguyen1(x)
        
    # Define input vectors (sampled within the search space)
    def generate_data(self):
        x_values = np.arange(self.bounds[0], self.bounds[1] + 0.1, 0.1)  # Include 4.0. TODO Include the step size as hyper parameters
        y_values = self.target_function(x_values)
        data = list(zip(x_values, y_values))
        return data

    def symbolic_fitness_function(self, genome):
        total_error = 0.0
        success = True  # Assume success initially
        epsilon = 1e-3  # Small threshold for success

        for x, target in self.data:

            try:
                output = self.evaluate_tree(genome, x)
                # print(f"output value: {output}, target value: {target}")
                error = output - target
                
                # proxy_error = np.sign(error) * np.log(0.01 + abs(error))
                # total_error += abs(proxy_error)
                
                total_error += abs(error)

                if abs(error) > epsilon:
                    success = False  # Error exceeds acceptable threshold
                    
            except Exception as e:
                # print(f"Error evaluating fitness: {e}")
                total_error += 1e6  # Penalize invalid outputs
                success = False
                
        fitness = 1 / (total_error + 1e-6)  # Fitness increases as total error decreases
        return fitness, success
    
class TestGeneticProgramming(unittest.TestCase):
            
    def setUp(self):

        # Define args as an argparse.Namespace with necessary attributes
        self.args = Namespace(
            benchmark='nguyen1',
            initial_depth=2,
            max_depth=6,
            pop_size=50,
            mutation_rate=0.005,
            generations=50,
            tournament_size=3,
            exp_num_runs=3,
            inbred_threshold=5, 
            bounds=(-4.0, 4.0),
            config_plot="tests"
            # Add other necessary arguments here
        )
     
        # Initialize necessary objects
        self.gp = GeneticAlgorithmGP(self.args)
        self.gp_landscape = GPLandscape(self.args, bounds=(-4, 4))

    def test_get_function_arity(self):
        self.assertEqual(util.get_function_arity('+'), 2)
        self.assertEqual(util.get_function_arity('sin'), 1)
        self.assertEqual(util.get_function_arity('x'), 0)
        self.assertEqual(util.get_function_arity('unknown'), 0)

    def test_crossover_same_arity(self):
        # Create two parent trees with matching arities
        parent1 = Individual(self.args, tree=Node('+', [Node('x'), Node('1.0')]))
        parent2 = Individual(self.args, tree=Node('+', [Node('*', [Node('x'), Node('1.0')]), Node('x')]))
        offspring1, offspring2 = self.gp.crossover(parent1, parent2)
        self.assertIsNotNone(offspring1)
        self.assertIsNotNone(offspring2)
        # Further assertions can be added to verify subtree swapping

    def test_crossover_different_arity(self):
        # Create two parent trees with different arities
        parent1 = Individual(self.args, tree=Node('+', [Node('x'), Node('1.0')]))
        parent2 = Individual(self.args, tree=Node('sin', [Node('x')]))
        offspring1, offspring2 = self.gp.crossover(parent1, parent2)
        self.assertIsNone(offspring1)
        self.assertIsNone(offspring2)

    def test_evaluate_tree(self):
        # Create a simple tree and evaluate
        tree = Node('+', [Node('x'), Node('1.0')])
        result = self.gp_landscape.evaluate_tree(tree, 2.0)
        self.assertEqual(result, 3.0)
        
    def test_evaluate_tree_complex(self):
        # Test a more complex tree
        tree = Node('-', [
            Node('/', [Node('1.0'), Node('x')]),
            Node('-', [Node('x'), Node('x')])
        ])
        # Expected: (1.0 / x) - (x - x) = (1.0 / 2.0) - (2.0 - 2.0) = 0.5 - 0.0 = 0.5
        result = self.gp_landscape.evaluate_tree(tree, 2.0)
        self.assertEqual(result, 0.5)
        
    def test_mutate_correct_arity(self):
        # Create an individual and mutate
        individual = Individual(
            self.args, 
            tree=Node('+', [Node('x'), Node('1.0')])
        )
        original_arity = individual.get_function_arity(individual.tree.value)
        self.gp.mutate(individual)
        # After mutation, check that the arity remains the same
        new_arity = individual.get_function_arity(individual.tree.value)
        self.assertEqual(original_arity, new_arity)
    
    def test_mutate_correct_arity_2(self):
        # Create an individual and mutate
        individual = Individual(self.args, tree=Node('+', [Node('x'), Node('1.0')]))
        self.gp.mutate(individual)
        # After mutation, check that the arity remains 2
        self.assertEqual(len(individual.tree.children), 2)
    
    def test_mutate_incorrect_arity(self):
        # Attempt to mutate and ensure arity is preserved
        individual = Individual(
            self.args, 
            tree=Node('sin', [Node('x')])
        )
        original_arity = individual.get_function_arity(individual.tree.value)
        self.gp.mutate(individual)
        
        # After mutation, check that the arity remains the same
        new_arity = individual.get_function_arity(individual.tree.value)
        self.assertEqual(original_arity, new_arity)

    

    
    
if __name__ == "__main__":
    
    # Get args
    args = util.set_args()
    
    # Set file plotting name
    args.config_plot = f"genetic_programming/{args.benchmark}/PopSize:{args.pop_size}_InThres:{args.inbred_threshold}_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}" 
    
    args.bounds = util.get_function_bounds(args.benchmark)
    
    # unittest.main()
    # # tests = TestGeneticProgramming(args)
    # # tests.test_get_function_arity()
    # # tests.test_crossover_same_arity()
    # # tests.test_crossover_different_arity()
    # # tests.test_evaluate_tree()
    # # tests.test_mutate_correct_arity()
    # exit()

    
    # Create Landscape
    gp_landscape = GPLandscape(args, args.bounds)

    # Run experiments
    print("Running GA with NO Inbreeding Mating...")
    results_no_inbreeding = exp.multiple_runs_function_gp(args, gp_landscape, args.inbred_threshold)
    
    print("Running GA with Inbreeding Mating...")
    results_inbreeding = exp.multiple_runs_function_gp(args, gp_landscape, None)
    
    gs_list, fit_list, div_list, label_list = plot.collect_bootstrapping_data(args, results_no_inbreeding, results_inbreeding)
    plot.plot_multiple_runs_GP_functions(args, gs_list, fit_list, div_list, label_list)

# ------------- An example of how to change that --------- #


# def target_function(x):
#     return x ** 2 + 2 * x + 1  # Example: quadratic function

# def symbolic_regression_fitness_function(genome):
#     input_values = np.linspace(-10, 10, 50)
#     return symbolic_regression_fitness(genome, target_function, input_values)

# TODO: THIS FOR ACKLEY
    # def evaluate_tree(self, node, x):
    #     """
    #         Evaluate the program tree with input vector x.

    #         Parameters:
    #         - node (Node): Current node in the program tree.
    #         - x (numpy.ndarray): Input vector.

    #         Returns:
    #         - result: Result of the program's evaluation. Can be a scalar or vector.
    #     """
    #     if node.is_terminal():
    #         if node.value == 'x':
    #             return x  # Return the entire input vector
    #         else:
    #             return float(node.value)
    #     else:
    #         func = node.value
    #         args = [self.evaluate_tree(child, x) for child in node.children]
    #         try:
    #             if func == '+':
    #                 return args[0] + args[1]
    #             elif func == '-':
    #                 return args[0] - args[1]
    #             elif func == '*':
    #                 return args[0] * args[1]
    #             elif func == '/':
    #                 # Protected division
    #                 denominator = args[1]
    #                 if isinstance(denominator, (int, float, np.ndarray)):
    #                     with np.errstate(divide='ignore', invalid='ignore'):
    #                         result = np.true_divide(args[0], denominator)
    #                         result[~np.isfinite(result)] = 1.0  # Replace inf, -inf, NaN with 1.0
    #                         return result
    #                 else:
    #                     return 1.0
    #             elif func == 'cos':
    #                 return np.cos(args[0])
    #             elif func == 'sin':
    #                 return np.sin(args[0])
    #             elif func == 'exp':
    #                 return np.exp(args[0])
    #             elif func == 'sqrt':
    #                 return np.sqrt(np.abs(args[0]))  # Protected sqrt
    #             elif func == 'sum':
    #                 return np.sum(args[0])
    #             elif func == 'norm':
    #                 return np.linalg.norm(args[0])
    #             else:
    #                 # Undefined function
    #                 raise ValueError(f"Undefined function: {func}")
                
    #             # After computing result, check if it's an array
    #             if isinstance(result, np.ndarray):
    #                 # Reduce array to scalar
    #                 result = np.mean(result)
    #             return result
            
    #         except Exception as e:
    #             # Handle any unexpected errors
    #             return 0.0
    
    # def complex_fitness_function(self, genome, input_vectors, lambda_complexity=0.1):
    #     """
    #         Definition
    #         -----------
    #             Calculation of fitness value based off Symbolic  Regression
    #             input_vectors = self.generate_input_vectors(d=2, num_samples=100)
    #             to generate samples
    #     """
    #     mse_total = 0.0
    #     complexity = self.count_nodes(genome)
        
    #     for x in input_vectors:
    #         try:
    #             output = self.evaluate_tree(genome, x)
    #             target = self.target_function(x)
    #             # Ensure output and target are scalars
    #             output = float(output)
    #             target = float(target)
    #             mse = (output - target) ** 2
    #             mse_total += mse
    #         except Exception as e:
    #             mse_total += 1e6  # Large penalty for errors
        
    #     mse_average = mse_total / len(input_vectors)
    #     fitness = (1 / (mse_average + 1e-6)) * np.exp(-lambda_complexity * complexity)
    #     return fitness
    
    # def fitness_function(self, genome):
    # input_vectors = self.generate_input_vectors(d=2, num_samples=100)
    #     return self.complex_fitness_function(genome)
    
    # def get_function_arity(self, function):
    #     # Define arity for each function
    #     arity_dict = {
    #         '+': 2,
    #         '-': 2,
    #         '*': 2,
    #         '/': 2,
    #         'cos': 1,
    #         'sin': 1,
    #         'exp': 1,
    #         'sqrt': 1,
    #         'sum': 1,
    #         'norm': 1
    #     }
    #     return arity_dict.get(function, 0)  # Default to 0 if function not found
    
    # ACKLEY
    # def random_tree(self, depth):
    #     if depth == self.max_depth:
    #         # At the root, use a reduction function
    #         function = np.random.choice(['sum', 'norm'])
    #     elif depth == 0:
    #         # Return a terminal node
    #         terminal = np.random.choice(['x', 'ERC'])  # Terminals are 'x' or random constants
    #         if terminal == 'ERC':
    #             value = np.random.uniform(self.bounds[0], self.bounds[1])  # Bounds for specific functions. TODO: only working with functions like ackley, rastrigin, etc
    #             return Node(value)
    #         else:
    #             return Node('x')
    #     else:
    #         # Return a function node with children
    #         function = np.random.choice(['+', '-', '*', '/', 'cos', 'sin', 'exp', 'sqrt', 'sum', 'norm'])  # Include vector functions
    #         arity = self.get_function_arity(function)
    #         children = [self.random_tree(depth - 1) for _ in range(arity)]
    #         return Node(function, children)

