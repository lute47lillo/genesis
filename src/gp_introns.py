"""
    Author: Lute Lillo
    
    Definition
    ------------
        Genetic Programming Landscape for intron and bloat computation. 
            - Is used by gp_bloat.py main GA algorithm.
            - Implements multiprocessing.
"""

import multiprocessing
from typing import List, Tuple, Dict
import time
import util
import gp_math
import random
import numpy as np
import experiments as exp
from gp_node import Individual

# Global variable to hold the GPIntronAnalyzer instance in worker processes
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
    global_analyzer = GPIntronAnalyzer(args, initialize_pool=False)  # Avoid nested pools

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

def serialize_node(node):
    """
        converts a node (and its subtree) into a nested tuple, which is hashable and can be used as a dictionary key
    """
    if node.is_terminal():
        return (node.value,)
    else:
        return (node.value, tuple(serialize_node(child) for child in node.children))
        
class GPIntronAnalyzer:
    def __init__(self, args, initialize_pool=True):
        self.args = args
        self.target_function = util.select_gp_benchmark(args)
        self.bounds = args.bounds
        self.data = self.generate_data()  # Generate all data points
        self.node_id_map = {} 

        if initialize_pool:
            # Initialize the multiprocessing pool with the initializer
            self.pool = multiprocessing.Pool(initializer=init_worker, initargs=(self.args, ))
        else:
            # No pool initialization to prevent nested pools
            self.pool = None

    def generate_data(self):
        """
        Define input vectors (sampled within the search space).
        """
        x_values = np.arange(self.bounds[0], self.bounds[1] + 0.1, 0.1)  # Make step size a hyperparameter
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

    def memoize(fn):
        cache = {}
        def memoized_fn(self, node):
            key = serialize_node(node)
            if key in cache:
                return cache[key]
            else:
                result = fn(self, node)
                cache[key] = result
                return result
        return memoized_fn
    
    def memoize_two_args(fn):
        cache = {}
        def memoized_fn(self, node1, node2):
            key = (serialize_node(node1), serialize_node(node2))
            if key in cache:
                return cache[key]
            else:
                result = fn(self, node1, node2)
                cache[key] = result
                return result
        return memoized_fn

    @memoize
    def subtree_evaluates_to_zero(self, node):
        x_values = self.sample_input_values()
        outputs = [self.evaluate_tree(node, x) for x in x_values]
        epsilon = 1e-6
        return all(abs(output) < epsilon for output in outputs)

    @memoize
    def subtree_evaluates_to_one(self, node):
        x_values = self.sample_input_values()
        outputs = [self.evaluate_tree(node, x) for x in x_values]
        epsilon = 1e-6
        return all(abs(output - 1.0) < epsilon for output in outputs)
    
    def mark_subtree_as_visited(self, node, visited_nodes):
        serialized_node = serialize_node(node)
        if serialized_node in visited_nodes:
            return
        visited_nodes.add(serialized_node)
        for child in node.children:
            self.mark_subtree_as_visited(child, visited_nodes)
    
    def detect_addition_subtraction_zero_introns(self, node, visited_nodes):
        introns = []
        if node.value in ['+', '-']:
            if len(node.children) < 2:
                return introns
            left_child, right_child = node.children[:2]

            # Check left child
            serialized_left_child = serialize_node(left_child)
            if serialized_left_child not in visited_nodes and self.subtree_evaluates_to_zero(left_child):
                intron_size = self.count_nodes(left_child)
                introns.append((left_child, intron_size))
                self.mark_subtree_as_visited(left_child, visited_nodes)

            # Check right child
            serialized_right_child = serialize_node(right_child)
            if serialized_right_child not in visited_nodes and self.subtree_evaluates_to_zero(right_child):
                intron_size = self.count_nodes(right_child)
                introns.append((right_child, intron_size))
                self.mark_subtree_as_visited(right_child, visited_nodes)
        return introns
    

    def detect_multiplication_by_one_introns(self, node, visited_nodes):
        introns = []
        if node.value == '*':
            if len(node.children) < 2:
                return introns
            left_child, right_child = node.children[:2]

            # Check left child
            serialized_left_child = serialize_node(left_child)
            if serialized_left_child not in visited_nodes and self.subtree_evaluates_to_one(left_child):
                intron_size = self.count_nodes(left_child)
                introns.append((left_child, intron_size))
                self.mark_subtree_as_visited(left_child, visited_nodes)

            # Check right child
            serialized_right_child = serialize_node(right_child)
            if serialized_right_child not in visited_nodes and self.subtree_evaluates_to_one(right_child):
                intron_size = self.count_nodes(right_child)
                introns.append((right_child, intron_size))
                self.mark_subtree_as_visited(right_child, visited_nodes)
        return introns

    def detect_division_by_one_introns(self, node, visited_nodes):
        introns = []
        if node.value == '/':
            if len(node.children) < 2:
                return introns
            right_child = node.children[1]
            if self.subtree_evaluates_to_one(right_child):
                serialized_node = serialize_node(node)
                if serialized_node not in visited_nodes:
                    # The entire division node is redundant (dividing by one)
                    intron_size = self.count_nodes(node)
                    introns.append((node, intron_size))
                    self.mark_subtree_as_visited(node, visited_nodes)
        return introns


    def detect_sin_zero_introns(self, node, visited_nodes):
        introns = []
        if node.value in ['+', '-']:
            for child in node.children:
                if child.value == 'sin' and len(child.children) > 0:
                    sin_child = child.children[0]
                    if self.subtree_evaluates_to_zero(sin_child):
                        serialized_child_node = serialize_node(child)
                        if serialized_child_node not in visited_nodes:
                            # The sin(0) node evaluates to zero, redundant in addition/subtraction
                            intron_size = self.count_nodes(child)
                            introns.append((child, intron_size))
                            self.mark_subtree_as_visited(child, visited_nodes)
        return introns

    def detect_log_one_introns(self, node, visited_nodes):
        introns = []
        if node.value in ['+', '-']:
            for child in node.children:
                if child.value == 'log' and len(child.children) > 0:
                    log_child = child.children[0]
                    if self.subtree_evaluates_to_one(log_child):
                        serialized_child_node = serialize_node(child)
                        if serialized_child_node not in visited_nodes:
                            # The log(1) node evaluates to zero, redundant in addition/subtraction
                            intron_size = self.count_nodes(child)
                            introns.append((child, intron_size))
                            self.mark_subtree_as_visited(child, visited_nodes)
        return introns

    def detect_cos_zero_introns(self, node, visited_nodes):
        introns = []
        if node.value in ['*', '/']:
            for child in node.children:
                if child.value == 'cos' and len(child.children) > 0:
                    cos_child = child.children[0]
                    if self.subtree_evaluates_to_zero(cos_child):
                        serialized_child_node = serialize_node(child)
                        if serialized_child_node not in visited_nodes:
                            # The cos(0) node evaluates to one, which is redundant in multiplication/division
                            intron_size = self.count_nodes(child)
                            introns.append((child, intron_size))
                            self.mark_subtree_as_visited(child, visited_nodes)
        return introns

    def detect_subtraction_of_identical_introns(self, node, visited_nodes):
        introns = []
        if node.value == '-':
            if len(node.children) < 2:
                return introns
            left_child, right_child = node.children[:2]

            serialized_node = serialize_node(node)
            if serialized_node not in visited_nodes and self.subtrees_are_identical(left_child, right_child):
                # The entire subtraction node is an intron
                intron_size = self.count_nodes(node)
                introns.append((node, intron_size))
                self.mark_subtree_as_visited(node, visited_nodes)
        return introns

    @memoize_two_args
    def subtrees_are_identical(self, node1, node2):
        """
        Checks if two subtrees are structurally identical.
        """
        # If they are not equal, the subtrees cannot be equal
        if node1.value != node2.value or len(node1.children) != len(node2.children):
            return False
        for c1, c2 in zip(node1.children, node2.children):
            if not self.subtrees_are_identical(c1, c2):
                return False
        return True
    
    def detect_division_of_identical_introns(self, node, visited_nodes):
        introns = []
        if node.value == '/':
            if len(node.children) < 2:
                return introns
            left_child, right_child = node.children[:2]

            serialized_node = serialize_node(node)
            if (serialized_node not in visited_nodes and
                self.subtrees_are_identical(left_child, right_child) and
                not self.subtree_evaluates_to_zero(left_child)):

                # The entire division node is an intron
                intron_size = self.count_nodes(node)
                introns.append((node, intron_size))
                self.mark_subtree_as_visited(node, visited_nodes)
        return introns


    def detect_pattern_introns(self, node) -> List[Tuple]:
        introns = []
        visited_nodes = set()
        stack = [node]

        while stack:
            current_node = stack.pop()
            serialized_current_node = serialize_node(current_node)
            if current_node is None or serialized_current_node in visited_nodes:
                continue

            # Mark current node as visited
            visited_nodes.add(serialized_current_node)

            # Apply intron detection patterns to the current node
            introns.extend(self.detect_addition_subtraction_zero_introns(current_node, visited_nodes))
            introns.extend(self.detect_multiplication_by_one_introns(current_node, visited_nodes))
            introns.extend(self.detect_division_by_one_introns(current_node, visited_nodes))
            introns.extend(self.detect_sin_zero_introns(current_node, visited_nodes))
            introns.extend(self.detect_log_one_introns(current_node, visited_nodes))
            introns.extend(self.detect_cos_zero_introns(current_node, visited_nodes))
            introns.extend(self.detect_subtraction_of_identical_introns(current_node, visited_nodes))
            introns.extend(self.detect_division_of_identical_introns(current_node, visited_nodes))

            # Add children to the stack for further traversal
            if not current_node.is_terminal():
                stack.extend(current_node.children)

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
        if self.pool is None:
            raise ValueError("Multiprocessing pool not initialized.")

        # Start timing
        start_time = time.time()
        print(f"\nLength of population: {len(population)}")

        population_total_intron_nodes = 0
        population_total_nodes = 0
        individual_intron_ratios = []

        # Map the helper function across the population using the existing pool
        results = self.pool.map(process_individual_helper, population)

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
        print(f"Avg pop nodes: {population_total_nodes/len(population):.3f}.")
        print(f"The total intron nodes: {population_total_intron_nodes}.")
        print(f"The intron ratio: {population_intron_ratio}.")

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTime taken to measure introns: {elapsed_time:.4f} seconds")

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

    
if __name__ == "__main__":
    
    # Get args
    args = util.set_args()
    
    # Create Landscape
    landscape = GPIntronAnalyzer(args)

    try:
        term1 = f"genetic_programming/{args.benchmark}/"
        term2 = "bloat/"
        
        # introns_ referes to classic intron run. introns_mutation_ are doing mutation to introduce specific introns.
        if args.inbred_threshold == 1:
            term3 = f"random_plus_PopSize:{args.pop_size}_InThres:None_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_MaxD:{args.max_depth}_InitD:{args.initial_depth}" 
        else:
            term3 = f"random_plus_PopSize:{args.pop_size}_InThres:{args.inbred_threshold}_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_MaxD:{args.max_depth}_InitD:{args.initial_depth}" 
        
        # Text to save files and plot.
        args.config_plot = term1 + term2 + term3
        
        if args.inbred_threshold == 1:
            print("Running GA with Inbreeding Mating...")
            results_inbreeding = exp.test_multiple_runs_function_bloat(args, landscape, None)
            util.save_accuracy(results_inbreeding, f"{args.config_plot}_inbreeding.npy")
        else:
            print("Running GA with NO Inbreeding Mating...")
            results_no_inbreeding = exp.test_multiple_runs_function_bloat(args, landscape, args.inbred_threshold)
            util.save_accuracy(results_no_inbreeding, f"{args.config_plot}_no_inbreeding.npy")
        
    finally:
        # Ensure that the pool is properly closed
        if landscape.pool is not None:
            landscape.pool.close()
            landscape.pool.join()