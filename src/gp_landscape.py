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


class GPLandscape:
    
    def __init__(self, args):
        
        self.args = args
        self.target_function = util.select_gp_benchmark(args)
        self.bounds = args.bounds
        self.data = self.generate_data() # Generate all data points
        
    def generate_data(self):
        """
            Definition
            -----------
                Define input vectors (sampled within the search space).
        """
        x_values = np.arange(self.bounds[0], self.bounds[1] + 0.1, 0.1)  # TODO Include the step size (0.1) as hyper parameters if adding more benchmarks
        y_values = self.target_function(x_values)
        data = list(zip(x_values, y_values))
        return data
        
    # ---------------------- Introns detection. EXPERIMENTAL ------------- #
    
    def sample_input_values(self, num_samples=10):
        """
        Generates a set of sample input values for x.
        
        Parameters:
        -----------
        num_samples : int
            The number of sample input values to generate.
        
        Returns:
        --------
        list of floats
            The list of sample x values.
        """
        lower_bound, upper_bound = self.bounds
        return [random.uniform(lower_bound, upper_bound) for _ in range(num_samples)]
    
    def subtree_evaluates_to_zero(self, node):
        """
        Determines if a subtree always evaluates to zero for all input values.
        
        Parameters:
        -----------
        node : Node
            The root node of the subtree to evaluate.
        
        Returns:
        --------
        bool
            True if the subtree evaluates to zero for all x in x_values, False otherwise.
        """
        x_values = self.sample_input_values()
        outputs = [self.evaluate_tree(node, x) for x in x_values]
        epsilon = 1e-6
        return all(abs(output) < epsilon for output in outputs)

    def detect_pattern_introns(self, node):
        """
        Detects introns based on specific patterns where adding or subtracting zero
        does not affect the program output.
        
        Parameters:
        -----------
        node : Node
            The root node of the subtree to check.
        
        Returns:
        --------
        list of tuples
            Each tuple contains (intron_node, intron_size).
        """
        introns = []
        
        if node is None or node.is_terminal():
            return introns
        
        # Pattern: Node + 0 or 0 + Node
        if node.value == '+':
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
            right_child = node.children[1]
            
            if self.subtree_evaluates_to_zero(right_child):
                intron_node = right_child
                intron_size = self.count_nodes(intron_node)
                introns.append((intron_node, intron_size))
        
        # Recursively check child nodes
        for child in node.children:
            introns.extend(self.detect_pattern_introns(child))
        
        return introns
    
    def measure_introns(self, population):
        """
            Definition
            -----------
                Measures intron statistics in the population.
            
            Parameters
            -----------
                - population (List.list): The list of Individual objects in the population.
            
            Returns
            -----------
                - intron_info (dict): A dictionary containing total and average intron metrics.
        """
        population_total_intron_nodes = 0
        population_total_nodes = 0
        individual_intron_ratios = []
        
        for individual in population:
            # Detect introns and their sizes
            introns = self.detect_pattern_introns(individual.tree)
            total_intron_nodes = sum(intron_size for _, intron_size in introns)
            
            # Total nodes in the individual's tree
            total_nodes = self.count_nodes(individual.tree)
            
            # Intron ratio for the individual
            intron_ratio = total_intron_nodes / total_nodes if total_nodes > 0 else 0
            
            # Update population totals
            population_total_intron_nodes += total_intron_nodes
            population_total_nodes += total_nodes
            individual_intron_ratios.append(intron_ratio)
        
        # Population-level metrics
        population_intron_ratio = population_total_intron_nodes / population_total_nodes if population_total_nodes > 0 else 0
        average_intron_ratio = sum(individual_intron_ratios) / len(individual_intron_ratios) if individual_intron_ratios else 0
        
        intron_info = {
            'population_total_intron_nodes': population_total_intron_nodes,
            'population_total_nodes': population_total_nodes,
            'population_intron_ratio': population_intron_ratio,
            'average_intron_ratio': average_intron_ratio
        }
        
        return intron_info
    
    # ------------- Evaluate Tree Functions ------------- #
    
    def count_nodes(self, node):
        """
            Definition
            -----------
                Count the number of nodes (functions and terminals) in a program tree.
        """
        if node is None:
            return 0
        count = 1  # Count the current node
        for child in node.children:
            count += self.count_nodes(child)
        return count
    
    def evaluate_tree(self, node, x):
        
        # Base case checking for error
        if node is None:
            return 0.0

        # Check if node is terminal (leave) or not
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
                # Extracted of paper: Effective Adaptive Mutation Rates for Program Synthesis by Ni, Andrew and Spector, Lee 2024
                result = np.clip(result, -1e6, 1e6)

                return result
            except Exception as e:

                return 0.0  # Return 0.0 for any error

    # ------------------ Fitness Function ----------------- #

    def symbolic_fitness_function(self, genome):
        """
            Definition
            -----------
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
                
        fitness = 1 / (total_error + 1e-6)  # Fitness increases as total error decreases or could return just total error
        return fitness, success