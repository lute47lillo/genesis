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
        self.poulation_success = False
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []
    
    # ----------------------- Tree ~ Node functions ------------------ #
    
    def select_random_node(self, tree):
        """
            Definition
            -----------
                Returns a single random node of a given tree (not Individual object).
                Example:
                    - tree1 = Node('+', [Node('x'), Node('1')])
                      could return Node(+) or Node(x) or Node(1)
        """
        nodes = self.get_all_nodes(tree)
        return np.random.choice(nodes)

    def get_all_nodes(self, tree):
        """
            Definition
            -----------
                Returns all nodes of a given tree (not Individual object).
                Example:
                    - tree1 = Node('+', [Node('x'), Node('1')])
                      returns [Node(+), Node(x), Node(1)]
        """
        nodes = [tree]
        for child in tree.children:
            nodes.extend(self.get_all_nodes(child))
        return nodes
    
    def select_random_node_with_parent(self, tree):
        """
            Definition
            -----------
                Selects a random node along with its parent.
                
                Example:
                    - tree1 = Node('+', [Node('x'), Node('1')])
                      could return (Node(+), Node(x)) or (Node(+), Node(1)) as [parent, child] pairs.
                
            Parameters
            -----------
                - tree (Node): children node of a given individual tree
                
            Returns
            -----------  
                - tuple (parent_node, selected_node).
                If the selected node is the root, parent_node is None.
        """
        all_nodes = self.get_all_nodes_with_parent(tree)
        if not all_nodes:
            return None, None
        return random.choice(all_nodes)

    def get_all_nodes_with_parent(self, node, parent=None):
        """
            Definition
            -----------
                Recursively collects all nodes in the tree along with their parent.
                
                Example:
                    - tree1 = Node('+', [Node('x'), Node('1')])
                      returns [(None, Node(+)), (Node(+), Node(x)), (Node(+), Node(1))] as [parent, child] pairs.
                              
        """
        nodes = [(parent, node)]
        for child in node.children:
            nodes.extend(self.get_all_nodes_with_parent(child, node))
        return nodes
    
    def can_mate(self, ind1, ind2, inbred_threshold):
        """
            Definition
            -----------
                Inbreeding prevention mechanism based of average pairwise distance between trees.
                
                Example:  # Tree 1: (x + 1) -> tree1 = Node('-', [Node('x'), Node('1')])
                          # Tree 2: (x + 2) -> tree2 = Node('+', [Node('x'), Node('2')])
                          have distance of 2. 
                          If inbred_threshold = None, 1 or 2 they could generate offspring.
                          If inbred_threshold = 3 or larger they could NOT generate offspring.
        """
        distance = self.compute_trees_distance(ind1.tree, ind2.tree)
        return distance >= inbred_threshold
    
    def tree_depth(self, node):
        """
            Definition
            -----------
                Returns the depth of a given individual node.
                Example:
                    - tree1 = Node('+', [Node('x'), Node('1')]) for Node(+) will return 2 -> 1 depth of children + 1 for itself.
        """
        if node is None:
            return 0
        if node.is_terminal():
            return 1
        else:
            return 1 + max(self.tree_depth(child) for child in node.children)
    
    def compute_trees_distance(self, node1, node2):
        """
            Definition
            -----------
                Computes the distance between 2 different trees through recursion.
                Example:
                          # Tree 1: (x + 1) -> tree1 = Node('-', [Node('x'), Node('1')])
                          # Tree 2: (x + 2) -> tree2 = Node('+', [Node('x'), Node('2')])
                          have distance of 2. 
        """
        # Base cases
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
            child_distances += self.compute_trees_distance(child1, child2)
            
        # Add distances for unmatched children
        child_distances += abs(len(node1.children) - len(node2.children))
        return cost + child_distances

    # ----------------- General GP Functions ------------------------- #
    
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
                self.poulation_success = True
            
    def tournament_selection(self, k=3):
        selected = []
        for _ in range(self.pop_size):
            participants = np.random.choice(self.population, k)
            winner = max(participants, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected
    
    def crossover(self, parent1, parent2):
      
        # Check if there is inbreeding prevention mechanism. (None means inbreeding is allowed)
        if self.inbred_threshold is not None:
            if not self.can_mate(parent1, parent2, self.inbred_threshold): # If distance(p1, p2) >= inbred_thres then skip bc [not False ==  True]
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
                # print(f"Attempt {attempt+1}: One of the selected nodes is None. Retrying...")
                continue  # Try again

            # Check if both nodes have the same arity
            arity1 = parent1.get_function_arity(node1.value)
            arity2 = parent2.get_function_arity(node2.value)
            if arity1 != arity2:
                # print(f"Attempt {attempt+1}: Arities do not match (arity1={arity1}, arity2={arity2}). Retrying...")
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
                    # print(f"Attempt {attempt}: node2 not found in parent_node2's children. Retrying...")
                    continue  # node2 not found, try again

            # Check for depth constraints
            depth_child1 = self.tree_depth(child1)
            depth_child2 = self.tree_depth(child2)
            if depth_child1 > self.max_depth or depth_child2 > self.max_depth:
                # print(f"Attempt {attempt}: Offspring exceed max depth ({self.max_depth}). Retrying...")
                # Revert the swap by reinitializing the trees
                child1 = copy.deepcopy(parent1.tree)
                child2 = copy.deepcopy(parent2.tree)
                continue  # Try again

            # Successful crossover
            # print(f"Attempt {attempt}: Crossover successful.")
            break
        else:
            # Failed to perform a valid crossover within max_attempts
            # print("Crossover failed after maximum attempts.")
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
    
    def mutate(self, individual):
        
        # Clone individual to avoid modifying original
        mutated_tree = copy.deepcopy(individual.tree)

        # Select a random node to mutate
        node_to_mutate = self.select_random_node(mutated_tree)
        if node_to_mutate is None:
            # print("Warning: No node selected for mutation")
            return  # Cannot mutate without a node

        # Replace the subtree with a new random subtree
        new_subtree = individual.random_tree(self.initial_depth) 
        
        # Ensure that the new_subtree has the correct arity
        required_children = individual.get_function_arity(new_subtree.value)
        if len(new_subtree.children) != required_children:
            # print(f"Warning: New subtree has incorrect arity for function '{new_subtree.value}'")
            return  # Discard mutation
        
        node_to_mutate.value = new_subtree.value
        node_to_mutate.children = new_subtree.children
        
        # Ensure the mutated tree does not exceed max depth
        if self.tree_depth(mutated_tree) > self.max_depth:
            # print("Warning: Mutated tree exceeds max depth")
            return  # Discard mutation

        # Update individual
        individual.tree = mutated_tree

    def measure_diversity(self):
        """
            Definition
            -----------
                Calculate diversity based on tree structures as the given average pairwise tree edit distance
                
                Example:  # Tree 1: (x + 1) -> tree1 = Node('+', [Node('x'), Node('1')])
                          # Tree 2: (x + 2) -> tree2 = Node('+', [Node('x'), Node('2')])
                          have distance of 1. 
                          
        """
        total_distance = 0
        count = 0
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self.compute_trees_distance(self.population[i].tree, self.population[j].tree)
                total_distance += distance
                count += 1
        if count == 0:
            return 0
        diversity = total_distance / count
        return diversity
    
    # ----------------- Main execution loop ------------------------- #
    
    def run(self, fitness_function):
        self.initialize_population()
    
        for gen in range(self.generations):
            
            # print(f"Current generation: {gen+1}")
            # print(f"The population:")
            # for indiv in self.population:
            #     print(f"Tree: {indiv.tree}, depth: {self.tree_depth(indiv.tree)}")
            # print()
            
            # Calculate fitness
            self.calculate_fitness(fitness_function, gen)
            
            # Update best fitness list
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_fitness_list.append(best_individual.fitness)
    
            # Measure diversity
            diversity = self.measure_diversity()
            self.diversity_list.append(diversity)
            
            # Early Stopping condition if successful individual has been found
            # TODO: Return generation as well, in order to compare methods
            if self.poulation_success == True:
                return self.best_fitness_list, self.diversity_list, gen+1
    
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
    
        return self.best_fitness_list, self.diversity_list, gen+1
    
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
            # print(f"Warning: evaluate_tree called with node={node}")
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
        # TODO: Refactor this to be in utils.py
        
        # Define target functions
        if self.args.benchmark == 'ackley':
            return bf.ackley_function(x)
        
        if self.args.benchmark == "nguyen1":
            return bf.nguyen1(x)
        
        if self.args.benchmark == "nguyen2":
            return bf.nguyen2(x)
        
        if self.args.benchmark == "nguyen3":
            return bf.nguyen3(x)
        
        if self.args.benchmark == "nguyen4":
            return bf.nguyen4(x)
        
    # Define input vectors (sampled within the search space)
    def generate_data(self):
        x_values = np.arange(self.bounds[0], self.bounds[1] + 0.1, 0.1)  # Include 4.0. TODO Include the step size as hyper parameters
        y_values = self.target_function(x_values)
        data = list(zip(x_values, y_values))
        return data

    def symbolic_fitness_function(self, genome):
        total_error = 0.0
        success = True  # Assume success initially
        epsilon = 1e-4  # Small threshold for success

        for x, target in self.data:

            try:
                output = self.evaluate_tree(genome, x)
                # print(f"output value: {output}, target value: {target}")
                error = output - target
                
                # TODO: As used in Paper Effective Adaptive MR
                # proxy_error = np.sign(error) * np.log(0.01 + abs(error))
                # total_error += abs(proxy_error)
                
                # TODO: As used in original Paper
                total_error += abs(error)

                if abs(error) > epsilon:
                    success = False  # Error exceeds acceptable threshold
                    
            except Exception as e:
                # print(f"Error evaluating fitness: {e}")
                total_error += 1e6  # Penalize invalid outputs
                success = False
                
        fitness = 1 / (total_error + 1e-6)  # Fitness increases as total error decreases or could return just total error
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
    args.config_plot = f"genetic_programming/{args.benchmark}/PopSize:{args.pop_size}_InThres:{args.inbred_threshold}_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_MaxD:{args.max_depth}_InitD:{args.initial_depth}" 
    
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

    # # Run experiments
    print("Running GA with NO Inbreeding Mating...")
    results_no_inbreeding = exp.multiple_runs_function_gp(args, gp_landscape, args.inbred_threshold)
    util.save_accuracy(results_no_inbreeding, f"{args.config_plot}_no_inbreeding.npy")
    
    # print("Running GA with Inbreeding Mating...")
    # results_inbreeding = exp.multiple_runs_function_gp(args, gp_landscape, None)
    # util.save_accuracy(results_inbreeding, f"{args.config_plot}_inbreeding.npy")
    
    # Plot the generation of successful runs
    # plot.plot_gen_vs_run(args, results_no_inbreeding, results_inbreeding)
    
    # Plot with bootstraping only if all runs are same length of generations
    # gs_list, fit_list, div_list, label_list = plot.collect_bootstrapping_data(args, results_no_inbreeding, results_inbreeding)
    # plot.plot_multiple_runs_GP_functions(args, gs_list, fit_list, div_list, label_list)