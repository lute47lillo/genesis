"""
    Author: Lute Lillo
    
    Definition
    ------------
        Genetic Programming Landscape class. 
            - Evaluate Tree semantics.
            - Calculate introns.
            - Generate Input-Output Samples.
"""

import util
import random
import numpy as np
import gp_math
import time
import experiments as exp
import multiprocessing
from typing import List, Tuple, Dict

# Global variable to hold the GPIntronAnalyzer instance
global_analyzer = None

def init_worker(args):
    """
    Initializer function for each worker process.
    Instantiates a GPIntronAnalyzer and assigns it to the global variable.
    
    Parameters:
    -----------
    args : Namespace or custom object
        The arguments required to initialize GPIntronAnalyzer.
    """
    global global_analyzer
    global_analyzer = GPIntronAnalyzer(args)

def process_individual_helper(individual):
    """
    Helper function to process a single individual for intron detection.
    This function uses the global GPIntronAnalyzer instance initialized in the worker.
    
    Parameters:
    -----------
    individual : Individual
        The individual to process.
    
    Returns:
    --------
    tuple:
        (total_intron_nodes, total_nodes, intron_ratio)
    """
    global global_analyzer
    if global_analyzer is None:
        raise ValueError("Global analyzer not initialized.")

    introns = global_analyzer.detect_pattern_introns(individual.tree)
    total_intron_nodes = sum(intron_size for _, intron_size in introns)
    total_nodes = global_analyzer.count_nodes(individual.tree)
    intron_ratio = total_intron_nodes / total_nodes if total_nodes > 0 else 0
    return total_intron_nodes, total_nodes, intron_ratio

# Define Node and Individual classes as provided
class Node:
    def __init__(self, value, tree_id, children=None):
        self.value = value  # Function or terminal
        self.children = children if children is not None else []
        self.tree_id = tree_id if tree_id is not None else None

    def is_terminal(self):
        return len(self.children) == 0

    def __str__(self):
        if self.is_terminal():
            return str(self.value)
        else:
            return f"({self.value} {' '.join(str(child) for child in self.children)})"

class Individual:
    def __init__(self, args, fitness_function=None, tree=None, id=None, ancestors=None, parents=None, generation=0):
        self.args = args
        self.bounds = self.args.bounds
        self.max_depth = self.args.max_depth  # TODO
        self.initial_depth = self.args.initial_depth
        self.id = id if id is not None else np.random.randint(1e9)
        self.parents = parents if parents is not None else []
        self.tree = tree if tree is not None else self.random_tree(depth=self.initial_depth)  # Initial depth of 6 as in paper
        self.ancestors = ancestors if ancestors is not None else set()
        self.generation = generation  # Track the generation of the individual
        self.succ_kinship = None

        # Init fitness for individual in creation and self.success
        self.fitness, self.success = fitness_function(self.tree)

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
                return Node(1.0, tree_id=self.id)
            else:
                return Node('x', tree_id=self.id)
        else:
            # Return a function node with appropriate arity
            function = np.random.choice(['+', '-', '*', '/', 'sin', 'cos', 'log'])
            arity = self.get_function_arity(function)
            children = [self.random_tree(depth - 1) for _ in range(arity)]
            return Node(function, self.id, children)

    def __str__(self):
        return str(self.tree)

# Define the GPIntronAnalyzer class
class GPIntronAnalyzer:
    def __init__(self, args):
        self.args = args
        self.target_function = util.select_gp_benchmark(args)
        self.bounds = args.bounds
        self.data = self.generate_data()  # Generate all data points

    def generate_data(self):
        """
        Define input vectors (sampled within the search space).
        """
        x_values = np.arange(self.bounds[0], self.bounds[1] + 0.1, 0.1)  # TODO: Make step size a hyperparameter
        y_values = self.target_function(x_values)
        data = list(zip(x_values, y_values))
        return data

    # ---------------------- Introns Detection Methods ---------------------- #
    
    def sample_input_values(self, num_samples=10):
        """
        Generates a set of sample input values for x.
        """
        lower_bound, upper_bound = self.bounds
        return [random.uniform(lower_bound, upper_bound) for _ in range(num_samples)]

    def subtree_evaluates_to_zero(self, node):
        """
        Determines if a subtree always evaluates to zero for all input values.
        """
        x_values = self.sample_input_values()
        outputs = [self.evaluate_tree(node, x) for x in x_values]
        epsilon = 1e-6
        return all(abs(output) < epsilon for output in outputs)

    def detect_pattern_introns(self, node) -> List[Tuple]:
        """
        Detects introns based on specific patterns where adding or subtracting zero
        does not affect the program output.
        """
        introns = []

        if node is None or node.is_terminal():
            return introns

        # Pattern: Node + 0 or 0 + Node
        if node.value == '+':
            if len(node.children) < 2:
                # Handle cases where '+' has less than two children
                return introns
            left_child = node.children[0]
            right_child = node.children[1]

            if self.subtree_evaluates_to_zero(left_child):
                # Pattern: 0 + Node
                intron_node = left_child
                intron_size = self.count_nodes(intron_node)
                introns.append((intron_node, intron_size))
            elif self.subtree_evaluates_to_zero(right_child):
                # Pattern: Node + 0
                intron_node = right_child
                intron_size = self.count_nodes(intron_node)
                introns.append((intron_node, intron_size))

        # Pattern: Node - 0
        elif node.value == '-':
            if len(node.children) < 2:
                # Handle cases where '-' has less than two children
                return introns
            right_child = node.children[1]

            if self.subtree_evaluates_to_zero(right_child):
                intron_node = right_child
                intron_size = self.count_nodes(intron_node)
                introns.append((intron_node, intron_size))

        # Recursively check child nodes
        for child in node.children:
            introns.extend(self.detect_pattern_introns(child))

        return introns

    def measure_introns(self, population: List[Individual]) -> Dict:
        """
        Measures intron statistics in the population using parallel processing.

        Parameters:
        -----------
        population (List[Individual]): The list of Individual objects in the population.

        Returns:
        --------
        intron_info (dict): A dictionary containing total and average intron metrics.
        """
        # Start timing
        # start_time = time.time()

        population_total_intron_nodes = 0
        population_total_nodes = 0
        individual_intron_ratios = []

        # Use multiprocessing.Pool with initializer
        with multiprocessing.Pool(initializer=init_worker, initargs=(self.args, )) as pool:
            # Map the helper function across the population
            results = pool.map(process_individual_helper, population)

        # Process the results
        for idx, result in enumerate(results):
            try:
                total_intron_nodes, total_nodes, intron_ratio = result

                # Update population totals
                population_total_intron_nodes += total_intron_nodes
                population_total_nodes += total_nodes
                individual_intron_ratios.append(intron_ratio)

                # print(f"Processed individual {idx + 1}/{len(population)}: "
                #       f"Intron Nodes={total_intron_nodes}, Total Nodes={total_nodes}, "
                #       f"Ratio={intron_ratio:.4f}")
            except Exception as e:
                print(f"Error processing individual {idx + 1}: {e}")

        # Population-level metrics
        population_intron_ratio = (population_total_intron_nodes / population_total_nodes
                                   if population_total_nodes > 0 else 0)
        average_intron_ratio = (sum(individual_intron_ratios) / len(individual_intron_ratios)
                                if individual_intron_ratios else 0)

        intron_info = {
            'population_total_intron_nodes': population_total_intron_nodes,
            'population_total_nodes': population_total_nodes,
            'population_intron_ratio': population_intron_ratio,
            'average_intron_ratio': average_intron_ratio
        }

        print(f"\nThe total nodes: {population_total_nodes}.")
        print(f"\nThe total intron nodes: {population_total_intron_nodes}.")
        print(f"\nThe intron ratio: {population_intron_ratio}.")
        
        # # End timing
        # end_time = time.time()

        # elapsed_time = end_time - start_time
        # print(f"\nTime taken to measure introns: {elapsed_time:.4f} seconds")

        return intron_info

    # ------------------ Evaluate Tree Functions ------------------ #

    def count_nodes(self, node) -> int:
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
        """
        Evaluates the program tree for a given input x.
        """
        # Base case checking for error
        if node is None:
            return 0.0

        # Check if node is terminal (leaf) or not
        if node.is_terminal():
            if node.value == 'x':
                return x  # x is a scalar
            else:
                try:
                    val = float(node.value)
                    return val
                except ValueError as e:
                    return 0.0
        else:
            func = node.value
            args_tree = [self.evaluate_tree(child, x) for child in node.children]

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
                result = np.clip(result, -1e6, 1e6)

                return result
            except Exception as e:
                return 0.0  # Return 0.0 for any error

    # ------------------ Fitness Function ------------------ #

    def symbolic_fitness_function(self, genome):
        """
        Calculate the fitness of the individual after evaluating the tree.
        """
        total_error = 0.0
        success = True  # Assume success initially
        epsilon = 1e-4  # Small threshold for success

        for x, target in self.data:
            try:
                output = self.evaluate_tree(genome, x)
                error = output - target

                # TODO: As used in original Paper
                total_error += abs(error)

                if abs(error) > epsilon:
                    success = False  # Error exceeds acceptable threshold

            except Exception as e:
                total_error += 1e6  # Penalize invalid outputs
                success = False

        fitness = 1 / (total_error + 1e-6)  # Fitness increases as total error decreases
        return fitness, success

# Example Usage
if __name__ == "__main__":
    
    # Get args
    args = util.set_args()
    
    # Create Landscape
    landscape = GPIntronAnalyzer(args)

    # -------------------------------- Experiment --------------------------- #
    
    term1 = f"genetic_programming/{args.benchmark}/"
    term2 = "bloat/"
    term3 = f"Parallel_PopSize:{args.pop_size}_InThres:{args.inbred_threshold}_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_MaxD:{args.max_depth}_InitD:{args.initial_depth}" 
    args.config_plot = term1 + term2 + term3
    
    # print("Running GA with NO Inbreeding Mating...")
    # results_no_inbreeding = exp.test_multiple_runs_function_bloat(args, landscape, args.inbred_threshold)
    # util.save_accuracy(results_no_inbreeding, f"{args.config_plot}_no_inbreeding.npy")
    
    print("Running GA with Inbreeding Mating...")
    results_inbreeding = exp.test_multiple_runs_function_bloat(args, landscape, None)
    util.save_accuracy(results_inbreeding, f"{args.config_plot}_inbreeding.npy")

    # ------- TEST ----------- #
    # Create a sample population
    # population = [Individual(tree) for tree in your_tree_list]  # Replace `your_tree_list` with actual trees
    
    # analyzer = GPIntronAnalyzer()
    # intron_stats = analyzer.measure_introns(population)
    
    # print("Intron Statistics:")
    # print(intron_stats)
