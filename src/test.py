import unittest
import random
import numpy as np

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
        
        depth = gp_system.tree_depth(tree1.children[0])
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

if __name__ == '__main__':
    unittest.main()
