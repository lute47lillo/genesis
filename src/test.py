import unittest
import random
import numpy as np
import copy

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
    def __init__(self, tree):
        self.tree = tree  # The GP tree representing the individual's program
    
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

class GeneticProgrammingSystem:
    def __init__(self, population):
        self.population = population  # List of Individual objects
    
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
                    print(f"\nCase2. Index: {index} of parent_node1 {parent_node1} where node1 {node1} happens.")
                    
                    parent_node1.children[index] = copy.deepcopy(node2)
                    print(f"NEW Children of parent_node1 {parent_node1} where node1 {node1} happens is node2 {node2}")
                    
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

class TestMeasureDiversity(unittest.TestCase):
    def test_measure_diversity(self):
        # Create sample trees for testing
        
        # Tree 1: (x + 1)
        tree1 = Node('-', [Node('2'), Node('1')])
        
        # Tree 2: (x + 2)
        tree2 = Node('+', [Node('x'), Node('2')])
        
        # Tree 3: (x * x)
        tree3 = Node('-', [Node('x'), Node('1')])
        
        # Tree 4: (3 - x)
        tree4 = Node('/', [Node('3'), Node('x')])
        
        # Tree 5: 
        tree5 = Node('+', [Node('*', [Node('-', [Node('x'), Node('1.0')])]), Node('+', [Node('x'), Node('1.0')])])
        
        tree6 = Node('+', [Node('*', [Node('x'), Node('1.0')]), Node('x')])
        
        # Tree 4: (3 - x)
        tree7 = Node('/', [Node('3'), Node('x')], Node('*', [Node('x'), Node('1.0')]))
        
        # Create individuals
        individual1 = Individual(tree1)
        individual2 = Individual(tree2)
        individual3 = Individual(tree3)
        individual4 = Individual(tree4)
        
        # Create population
        population = [individual1, individual2, individual3, individual4]
        
        # Initialize GP system
        gp_system = GeneticProgrammingSystem(population)
        
        # Calculate diversity
        diversity = gp_system.measure_diversity()
        
        nodes = gp_system.get_all_nodes(tree1)
        print(nodes)
        
        depth = gp_system.tree_depth(tree7)
        print(f"The depth: {depth}")
        
        random_node = gp_system.select_random_node(tree1)
        print(random_node)
        
        # Manually compute expected diversity
        # Pairwise distances:
        # Distance between tree1 and tree2
        dist1_2 = gp_system.tree_edit_distance(tree1, tree2)  
        inbred_thres = 2      
        can = gp_system.can_mate(individual1, individual2, inbred_thres)
        if not can:
            print(f"Cant reproduce with distance: {dist1_2} and threshold: {inbred_thres}")
        else:
            print(f"Can reproduce with distance: {dist1_2} and threshold: {inbred_thres}")
            
        # Distance between tree1 and tree3
        dist1_3 = gp_system.tree_edit_distance(tree1, tree3)
        inbred_thres = 2      
        can = gp_system.can_mate(individual1, individual3, inbred_thres)
        if not can:
            print(f"Cant reproduce with distance: {dist1_3} and threshold: {inbred_thres}")
        else:
            print(f"Can reproduce with distance: {dist1_3} and threshold: {inbred_thres}")
            
        # Distance between tree1 and tree4
        dist1_4 = gp_system.tree_edit_distance(tree1, tree4)
        # Distance between tree2 and tree3
        dist2_3 = gp_system.tree_edit_distance(tree2, tree3)
        # Distance between tree2 and tree4
        dist2_4 = gp_system.tree_edit_distance(tree2, tree4)
        # Distance between tree3 and tree4
        dist3_4 = gp_system.tree_edit_distance(tree3, tree4)
        
        total_distance = dist1_2 + dist1_3 + dist1_4 + dist2_3 + dist2_4 + dist3_4
        count = 6  # Total number of pairs
        
        expected_diversity = total_distance / count
        
        # Assert that the calculated diversity matches the expected diversity
        self.assertEqual(diversity, expected_diversity)
        
        # Optional: Print the diversity
        print(f"Calculated Diversity: {diversity}")
        print(f"Expected Diversity: {expected_diversity}")
        
        # Additionally, assert that the diversity is greater than zero
        self.assertGreater(diversity, 0)
        
    def test_crossover_same_arity(self):
        
        tree1 = Node('+', [Node('x'), Node('1.0')])
        tree2 = Node('+', [Node('*', [Node('x'), Node('1.0')]), Node('x')])
        
        # Create two parent trees with matching arities
        individual1 = Individual(tree1)
        individual2 = Individual(tree2)
        
        # Create population
        population = [individual1, individual2]
        
        # Initialize GP system
        gp_system = GeneticProgrammingSystem(population)
        
        dist1_2 = gp_system.tree_edit_distance(tree1, tree2)
        print(f"Distance: {dist1_2}")
        
        offspring1, offspring2 = gp_system.crossover(individual1, individual2, 2, 10)
        
        print(f"{individual1} + {individual2} = {offspring1}")
        print(f"{individual1} + {individual2} = {offspring2}")
        
        
        self.assertIsNotNone(offspring1)
        self.assertIsNotNone(offspring2)

if __name__ == '__main__':
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
