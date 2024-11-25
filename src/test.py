import unittest
import random
import numpy as np
import copy
import benchmark_factory as bf
import gp_math
    
# ----------------- Genetic Programming -------------- #

class Node:
    def __init__(self, value, children=None):
        self.value = value  # Operator or operand
        self.children = children if children is not None else []
        
    def is_terminal(self):
        return len(self.children) == 0
    
    def __str__(self):
        if self.is_terminal():
            return str(self.value)
        else:
            return f"({self.value} {' '.join(str(child) for child in self.children)})"

class Individual:
    def __init__(self, tree, initial_depth=3):
        self.tree = tree if tree is not None else self.random_tree(depth=initial_depth) # Initial depth of 6 as in paper
    
    def __repr__(self):
        return f"Individual({self.tree})"
    
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

class GeneticProgrammingSystem:
    def __init__(self, population=None):
        self.population = population  # List of Individual objects
        
    def initialize_population(self, pop_size, initial_depth=3):
        self.population = []
        for _ in range(pop_size):
            individual = Individual(None, initial_depth=initial_depth)
            self.population.append(individual)
    
        return self.population
    
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
    
    def select_random_node_with_parent(self, tree):
        """
            Definition
            -----------
                Selects a random node along with its parent.
                
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
        distance = self.tree_edit_distance(ind1.tree, ind2.tree)
        return distance >= inbred_threshold
    
    def select_random_node(self, tree):
        nodes = self.get_all_nodes(tree)
        return np.random.choice(nodes)
    
    def get_all_nodes(self, tree):
        nodes = [tree]
        for child in tree.children:
            nodes.extend(self.get_all_nodes(child))
        return nodes
    
    def tree_depth(self, node):
        """
            Definition
            -----------
                Returns the height of a given individual tree.
                Example:
                    - tree1 = Node('+', [Node('x'), Node('1')]) for Node(+) will return 2 -> 1 depth of children + 1 for itself.
        """
        if node is None:
            return 0
        if node.is_terminal():
            return 1
        else:
            return 1 + max(self.tree_depth(child) for child in node.children)
        
    def crossover(self, parent1, parent2, inbred_threshold, max_depth):
      
        # Check if there is inbreeding prevention mechanism. (None means inbreeding is allowed)
        if inbred_threshold is not None:
            if not self.can_mate(parent1, parent2, inbred_threshold): # If distance(p1, p2) >= inbred_thres then skip bc [not False ==  True]
                return None, None

        # Clone parents to avoid modifying originals
        child1 = copy.deepcopy(parent1.tree)
        child2 = copy.deepcopy(parent2.tree)

        # Attempt crossover
        max_attempts = 10
        for attempt in range(max_attempts+1):
            
            # Select random nodes with their parents
            parent_node1, node1 = self.select_random_node_with_parent(child1)
            parent_node2, node2 = self.select_random_node_with_parent(child2)
            
            print(f"\nChosen Parent_node1: {parent_node1} and node1: {node1}")
            print(f"Chosen Parent_node2: {parent_node2} and node2: {node2}")

            if node1 is None or node2 is None:
                continue  # Try again

            # Check if both nodes have the same arity
            arity1 = parent1.get_function_arity(node1.value)
            arity2 = parent2.get_function_arity(node2.value)
            
            if arity1 != arity2:
                continue  # Arities don't match, select another pair
            
            print(f"\nThe arity of node1: {arity1}")
            print(f"The arity of node2: {arity2}")
            print("PD: At this point they need to match.")

            # Swap entire subtrees
            if parent_node1 is None:
                # node1 is root of child1
                child1 = copy.deepcopy(node2)
                print(f"\nCase1. Parent_node1 is NONE. Child 1 is copy of child 2: {child1}")
            else:
                # Find the index of node1 in its parent's children and replace it
                try:
                    index = parent_node1.children.index(node1)
                    print(f"\nCase2. Index: {index} of parent_node1 {parent_node1} is where node1 {node1} happens.")
                    
                    parent_node1.children[index] = copy.deepcopy(node2)
                    print(f"NEW Children of parent_node1 {parent_node1} at index {index} where node1 {node1} happened is now node2 {node2}")
                    
                except ValueError:
                    continue  # node1 not found, try again

            if parent_node2 is None:
                # node2 is root of child2
                child2 = copy.deepcopy(node1)
                print(f"Case3. Parent_node2 is NONE. Child 2 is copy of child 1: {child2}")
            else:
                # Find the index of node2 in its parent's children and replace it
                try:
                    index = parent_node2.children.index(node2)
                    print(f"\nCase4. Index: {index} of parent_node2 {parent_node2} where node1 {node2} happens.")
                    
                    parent_node2.children[index] = copy.deepcopy(node1)
                    print(f"NEW Children of parent_node2 {parent_node1} where node2 {node1} happens is node1 {node2}")
                except ValueError:
                    continue  # node2 not found, try again

            # Check for depth constraints
            depth_child1 = self.tree_depth(child1)
            depth_child2 = self.tree_depth(child2)
            
            print(f"\nThe depth of child1: {depth_child1}")
            print(f"The depth of child2: {depth_child2}")
   
            if depth_child1 > max_depth or depth_child2 > max_depth:
                # Revert the swap by reinitializing the trees
                child1 = copy.deepcopy(parent1.tree)
                child2 = copy.deepcopy(parent2.tree)
                continue  # Try again

            # Successful crossover
            break
        else:
            # Failed to perform a valid crossover within max_attempts
            return None, None

        # Create new individuals
        offspring1 = Individual(tree=child1)
        offspring2 = Individual(tree=child2)

        return offspring1, offspring2
    
    def mutate(self, individual):
        
        # Clone individual to avoid modifying original
        mutated_tree = copy.deepcopy(individual.tree)
        
        print(f"Tree to mutate: {mutated_tree}") # Tree to mutate: (+ x 1.0)

        # Select a random node to mutate
        node_to_mutate = self.select_random_node(mutated_tree)
        print(f"Node to mutate: {node_to_mutate}") # Could be either x or 1
        if node_to_mutate is None:
            # print("Warning: No node selected for mutation")
            return  # Cannot mutate without a node

        # Replace the subtree with a new random subtree
        new_subtree = individual.random_tree(3) 
        print(f"Mutation by replacement. New subtree to replace node: {new_subtree}")
        
        # Ensure that the new_subtree has the correct arity
        required_children = individual.get_function_arity(new_subtree.value)
        print(f"Arity required by new_subtree: {required_children}")
        if len(new_subtree.children) != required_children:
            print(f"Warning: New subtree has incorrect arity for function '{new_subtree.value}'")
            return  # Discard mutation
        
        
        print(f"Node to mutate value: {node_to_mutate.value} changed with {new_subtree.value}")
        print(f"Node to mutate children: {node_to_mutate.children} changed with {new_subtree.children[0]}")
        node_to_mutate.value = new_subtree.value
        node_to_mutate.children = new_subtree.children
        
        # Ensure the mutated tree does not exceed max depth
        if self.tree_depth(mutated_tree) > 8:
            print("Warning: Mutated tree exceeds max depth")
            return  # Discard mutation

        # Update individual
        individual.tree = mutated_tree
        
class GPLandscapeTest:
    
    def __init__(self):
        
        self.target_function = bf.nguyen1
        self.data = self.generate_data() # Generate all data points
        
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
        lower_bound, upper_bound = -4.0, 4.0
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
        Measures intron statistics in the population.
        
        Parameters:
        -----------
        population : list
            The list of Individual objects in the population.
        
        Returns:
        --------
        dict
            A dictionary containing total and average intron metrics.
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
        
        return {
            'population_total_intron_nodes': population_total_intron_nodes,
            'population_total_nodes': population_total_nodes,
            'population_intron_ratio': population_intron_ratio,
            'average_intron_ratio': average_intron_ratio
        }


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
            
    def generate_data(self):
        """
            Definition
            -----------
                Define input vectors (sampled within the search space).
        """
        x_values = np.arange(-4.0, 4.0 + 0.1, 0.1)  # TODO Include the step size (0.1) as hyper parameters if adding more benchmarks
        y_values = self.target_function(x_values)
        data = list(zip(x_values, y_values))
        return data

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
    
import unittest

# Assume the necessary classes and methods have been imported
# from your GP implementation
def create_tree_with_zero_introns():
    # Subtree (x - x) evaluates to zero
    zero_subtree = Node('-', [Node('x'), Node('x')])
    
    # Full tree: x + (x - x)
    root = Node('+', [Node('x'), zero_subtree])
    return root



class TestPatternIntrons(unittest.TestCase):
    
    def test_detect_pattern_introns(self):
        
        # Create the tree with a known intron
        tree = create_tree_with_zero_introns()
        landscape = GPLandscapeTest()
        
        # Detect introns in the tree
        introns = landscape.detect_pattern_introns(tree)
 
        tree = introns[0][0]
        size = introns[0][1]
        print(tree, size)
        
        # Expected intron size is 3 (nodes: '-', 'x', 'x')
        expected_intron_size = 3
        
        # Verify that introns were detected
        self.assertTrue(len(introns) > 0, "No introns detected when there should be.")
        
        # Extract the sizes of detected introns
        detected_intron_sizes = [intron_size for _, intron_size in introns]
        
        # Check that the expected intron size is in the detected intron sizes
        self.assertIn(expected_intron_size, detected_intron_sizes, "Intron of expected size not detected.")

    def test_detect_introns_pop(self):
        
        # Create a population
        gp = GeneticProgrammingSystem(None)
        pop = gp.initialize_population(20, 3)
        
        land = GPLandscapeTest()
        out = land.measure_introns(pop)
        
        print(out)


# class TestMeasureDiversity(unittest.TestCase):
    
    # def test_init_pop(self):
        # gp_system = GeneticProgrammingSystem(None)
        # population = gp_system.initialize_population(10, 3)
        
        # for i, indiv in enumerate(population):
        #     depth_i = gp_system.tree_depth(indiv.tree)
        #     print(f"\n({i}) Depth: {depth_i}, tree: {indiv.tree}")
            
    
    # def test_measure_diversity(self):
    #     # Create sample trees for testing
        
    #     # Tree 1: (x + 1)
    #     tree1 = Node('-', [Node('2'), Node('1')])
        
    #     # Tree 2: (x + 2)
    #     tree2 = Node('+', [Node('x'), Node('2')])
        
    #     # Tree 3: (x * x)
    #     tree3 = Node('-', [Node('x'), Node('1')])
        
        # Tree 4: (3 - x)
        # tree4 = Node('/', [Node('3'), Node('x')])
        
        # # Tree 5: 
        # tree5 = Node('+', [Node('*', [Node('-', [Node('x'), Node('1.0')])]), Node('+', [Node('x'), Node('1.0')])])
        
        # tree6 = Node('+', [Node('*', [Node('x'), Node('1.0')]), Node('x')])
        
        # # Create individuals
        # individual1 = Individual(tree1)
        # individual2 = Individual(tree2)
        # individual3 = Individual(tree3)
        # individual4 = Individual(tree4)
        
        # # Create population
        # population = [individual1, individual2, individual3, individual4]
        
        # # Initialize GP system
        # gp_system = GeneticProgrammingSystem(population)
        
        # # Calculate diversity
        # diversity = gp_system.measure_diversity()
        
        # nodes = gp_system.get_all_nodes(tree1)
        # print(nodes)
        
        # depth = gp_system.tree_depth(tree4)
        # print(f"The depth: {depth}")
        
        # random_node = gp_system.select_random_node(tree1)
        # print(random_node)
        
        # # Manually compute expected diversity
        # # Pairwise distances:
        # # Distance between tree1 and tree2
        # dist1_2 = gp_system.tree_edit_distance(tree1, tree2)  
        # inbred_thres = 2      
        # can = gp_system.can_mate(individual1, individual2, inbred_thres)
        # if not can:
        #     print(f"Cant reproduce with distance: {dist1_2} and threshold: {inbred_thres}")
        # else:
        #     print(f"Can reproduce with distance: {dist1_2} and threshold: {inbred_thres}")
            
        # # Distance between tree1 and tree3
        # dist1_3 = gp_system.tree_edit_distance(tree1, tree3)
        # inbred_thres = 2      
        # can = gp_system.can_mate(individual1, individual3, inbred_thres)
        # if not can:
        #     print(f"Cant reproduce with distance: {dist1_3} and threshold: {inbred_thres}")
        # else:
        #     print(f"Can reproduce with distance: {dist1_3} and threshold: {inbred_thres}")
            
        # # Distance between tree1 and tree4
        # dist1_4 = gp_system.tree_edit_distance(tree1, tree4)
        # # Distance between tree2 and tree3
        # dist2_3 = gp_system.tree_edit_distance(tree2, tree3)
        # # Distance between tree2 and tree4
        # dist2_4 = gp_system.tree_edit_distance(tree2, tree4)
        # # Distance between tree3 and tree4
        # dist3_4 = gp_system.tree_edit_distance(tree3, tree4)
        
        # total_distance = dist1_2 + dist1_3 + dist1_4 + dist2_3 + dist2_4 + dist3_4
        # count = 6  # Total number of pairs
        
        # expected_diversity = total_distance / count
        
        # # Assert that the calculated diversity matches the expected diversity
        # self.assertEqual(diversity, expected_diversity)
        
        # # Optional: Print the diversity
        # print(f"Calculated Diversity: {diversity}")
        # print(f"Expected Diversity: {expected_diversity}")
        
        # # Additionally, assert that the diversity is greater than zero
        # self.assertGreater(diversity, 0)
        
    # def test_redundant_code(self, ):
        
    #     # Create Landscape
    #     landscape = GPLandscapeTest()
        
    #     # Create individual
    #     tree5 = Node('+', [Node('*', [Node('-', [Node('x'), Node('1.0')])]), Node('+', [Node('x'), Node('1.0')])])
    #     # tree5 = Node('-', [Node('x'), Node('1')])
    #     individual = Individual(tree5)
        
    #     # Create Data
    #     self.data = landscape.generate_data()
    #     x_values = [x for x, _ in self.data]  # Use existing data inputs
    #     redundant_nodes = landscape.detect_redundant_nodes(individual.tree, x_values)
    #     num_redundant_nodes = len(redundant_nodes)
        
    #     print(num_redundant_nodes)
        
    # def test_crossover_same_arity(self):
        
    #     tree1 = Node('+', [Node('x'), Node('x')]) # x + x = 2x
    #     tree2 = Node('+', [Node('*', [Node('x'), Node('1.0')]), Node('x')]) # x + x = 2x
        
    #     # Semantically equivalent, but syntactic distance of 3. 
    #     # The childs they produce are:
    #     # Individual((* x 1.0))  -> x
    #     # Individual((+ (+ x x) x)) -> x + 2x = 3x
        
    #     # Create two parent trees with matching arities
    #     individual1 = Individual(tree1)
    #     individual2 = Individual(tree2)
        
    #     # Create population
    #     population = [individual1, individual2]
        
    #     # Initialize GP system
    #     gp_system = GeneticProgrammingSystem(population)
        
    #     dist1_2 = gp_system.tree_edit_distance(tree1, tree2)
    #     print(f"Distance: {dist1_2}")
        
    #     offspring1, offspring2 = gp_system.crossover(individual1, individual2, 2, 10)
        
    #     print(f"{individual1} + {individual2} = {offspring1}")
    #     print(f"{individual1} + {individual2} = {offspring2}")
        
        
    #     self.assertIsNotNone(offspring1)
    #     self.assertIsNotNone(offspring2)
        
    # def test_mutate_correct_arity(self):
        
    #     # Create an individual and mutate
    #     tree1 = Node('+', [Node('x'), Node('1.0')])
    #     tree2 = Node('+', [Node('*', [Node('x'), Node('1.0')]), Node('x')])
        
    #     # Create two parent trees with matching arities
    #     individual1 = Individual(tree1)
    #     individual2 = Individual(tree2)
        
    #     # Create population
    #     population = [individual1, individual2]
        
    #     # Initialize GP system
    #     gp_system = GeneticProgrammingSystem(population)
    #     original_arity = individual1.get_function_arity(individual1.tree.value)
    #     gp_system.mutate(individual1)
        
    #     # After mutation, check that the arity remains the same
    #     new_arity = individual1.get_function_arity(individual1.tree.value)
    #     self.assertEqual(original_arity, new_arity)

    
    # def test_mutate_incorrect_arity(self):
    #     # Attempt to mutate and ensure arity is preserved
    #     individual = Individual(
    #         self.args, 
    #         tree=Node('sin', [Node('x')])
    #     )
    #     original_arity = individual.get_function_arity(individual.tree.value)
    #     self.gp.mutate(individual)
        
    #     # After mutation, check that the arity remains the same
    #     new_arity = individual.get_function_arity(individual.tree.value)
    #     self.assertEqual(original_arity, new_arity)

if __name__ == '__main__':
    # testing()
    unittest.main()
    
# TODO: Re-insert as real unit-tests if needed
# class TestGeneticProgramming(unittest.TestCase):
            
#     def setUp(self):

#         # Define args as an argparse.Namespace with necessary attributes
#         self.args = Namespace(
#             benchmark='nguyen1',
#             initial_depth=2,
#             max_depth=6,
#             pop_size=50,
#             mutation_rate=0.005,
#             generations=50,
#             tournament_size=3,
#             exp_num_runs=3,
#             inbred_threshold=5, 
#             bounds=(-4.0, 4.0),
#             config_plot="tests"
#             # Add other necessary arguments here
#         )
     
#         # Initialize necessary objects
#         self.gp = GeneticAlgorithmGP(self.args)
#         self.gp_landscape = GPLandscape(self.args, bounds=(-4, 4))

#     def test_get_function_arity(self):
#         self.assertEqual(util.get_function_arity('+'), 2)
#         self.assertEqual(util.get_function_arity('sin'), 1)
#         self.assertEqual(util.get_function_arity('x'), 0)
#         self.assertEqual(util.get_function_arity('unknown'), 0)

#     def test_crossover_same_arity(self):
#         # Create two parent trees with matching arities
#         parent1 = Individual(self.args, tree=Node('+', [Node('x'), Node('1.0')]))
#         parent2 = Individual(self.args, tree=Node('+', [Node('*', [Node('x'), Node('1.0')]), Node('x')]))
#         offspring1, offspring2 = self.gp.crossover(parent1, parent2)
#         self.assertIsNotNone(offspring1)
#         self.assertIsNotNone(offspring2)
#         # Further assertions can be added to verify subtree swapping

#     def test_crossover_different_arity(self):
#         # Create two parent trees with different arities
#         parent1 = Individual(self.args, tree=Node('+', [Node('x'), Node('1.0')]))
#         parent2 = Individual(self.args, tree=Node('sin', [Node('x')]))
#         offspring1, offspring2 = self.gp.crossover(parent1, parent2)
#         self.assertIsNone(offspring1)
#         self.assertIsNone(offspring2)

#     def test_evaluate_tree(self):
#         # Create a simple tree and evaluate
#         tree = Node('+', [Node('x'), Node('1.0')])
#         result = self.gp_landscape.evaluate_tree(tree, 2.0)
#         self.assertEqual(result, 3.0)
        
#     def test_evaluate_tree_complex(self):
#         # Test a more complex tree
#         tree = Node('-', [
#             Node('/', [Node('1.0'), Node('x')]),
#             Node('-', [Node('x'), Node('x')])
#         ])
#         # Expected: (1.0 / x) - (x - x) = (1.0 / 2.0) - (2.0 - 2.0) = 0.5 - 0.0 = 0.5
#         result = self.gp_landscape.evaluate_tree(tree, 2.0)
#         self.assertEqual(result, 0.5)
        
#     def test_mutate_correct_arity(self):
#         # Create an individual and mutate
#         individual = Individual(
#             self.args, 
#             tree=Node('+', [Node('x'), Node('1.0')])
#         )
#         original_arity = individual.get_function_arity(individual.tree.value)
#         self.gp.mutate(individual)
#         # After mutation, check that the arity remains the same
#         new_arity = individual.get_function_arity(individual.tree.value)
#         self.assertEqual(original_arity, new_arity)
    
#     def test_mutate_correct_arity_2(self):
#         # Create an individual and mutate
#         individual = Individual(self.args, tree=Node('+', [Node('x'), Node('1.0')]))
#         self.gp.mutate(individual)
#         # After mutation, check that the arity remains 2
#         self.assertEqual(len(individual.tree.children), 2)
    
#     def test_mutate_incorrect_arity(self):
#         # Attempt to mutate and ensure arity is preserved
#         individual = Individual(
#             self.args, 
#             tree=Node('sin', [Node('x')])
#         )
#         original_arity = individual.get_function_arity(individual.tree.value)
#         self.gp.mutate(individual)
        
#         # After mutation, check that the arity remains the same
#         new_arity = individual.get_function_arity(individual.tree.value)
#         self.assertEqual(original_arity, new_arity)
