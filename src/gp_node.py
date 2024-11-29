"""
    Author: Lute Lillo
    
    Definition
    ------------
        Genetic Programming Node and Individual classes. 
            - Generation of Random Trees (Individuals) based off Node class.
"""

import numpy as np

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
        self.max_depth = self.args.max_depth # TODO
        self.initial_depth = self.args.initial_depth
        self.id = id if id is not None else np.random.randint(1e9)
        self.parents = parents if parents is not None else []
        self.tree = tree if tree is not None else self.random_tree(depth=self.initial_depth) # Initial depth of 6 as in paper
        self.ancestors = ancestors if ancestors is not None else set()
        self.generation = generation  # Track the generation of the individual
        self.succ_kinship = None
        
        # Init fitness for individual in creation and self.success
        self.fitness, self.success = fitness_function(self.tree)
        
        # # Initial generation individual
        # if not self.parents:
        #     self.ancestors.add(self.id)
            
        # # Update ancestors with depth limitation
        # ancestry_max_depth = 10  # Set the desired ancestry depth. Edit for computational issues
        # for parent in self.parents:
        #     if parent.generation >= self.generation - ancestry_max_depth:
        #         self.ancestors.update(parent.ancestors)
        #         self.ancestors.add(parent.id)
        
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