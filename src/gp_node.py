"""
    Author: Lute Lillo
    
    Definition
    ------------
        Genetic Programming Node and Individual classes. 
            - Generation of Random Trees (Individuals) based off Node class.
"""

import numpy as np

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
    def __init__(self, args, fitness_function=None, tree=None, id=None, init_method="full"):
        self.args = args
        self.bounds = self.args.bounds
        self.max_depth = self.args.max_depth
        self.init_method = init_method
        self.id = id if id is not None else np.random.randint(1e9)
        
        if tree is not None:
            self.tree = tree
        elif self.init_method == "grow":
            self.tree = self.grow_tree(depth=self.args.initial_depth)
        elif self.init_method == "full":
            self.tree = self.full_tree(depth=self.args.initial_depth)
        
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
    
    def full_tree(self, depth):
        """
            Definition
            -----------
                In the "full" method, all the nodes at each level (except the leaves at the maximum depth) are function (non-terminal) nodes
                and all leaves at the maximum depth are terminal nodes. This creates "bushy" and more balanced trees.
                
                    1. At every level of the tree (except the deepest level), we place a function (internal) node.
                    2. Only at the maximum depth do we place terminal nodes (leaves).
                    
                    Therefore, the number of nodes in a "full" tree depends heavily on the arity (the number of children) of the chosen function nodes.
                    Theoretical maximum nº of nodes is given by 2^(max_depth+1) - 1. Ex: Max Depth = 3. Then 15 is maximum.
        """
        if depth == 0:
            # Return a terminal node
            terminal = np.random.choice(['x', '1.0'])
            return Node(1.0 if terminal == '1.0' else 'x')
        else:
            # Choose a function node
            function = np.random.choice(['+', '-', '*', '/', 'sin', 'cos', 'log'])
            arity = self.get_function_arity(function)
            
            # Ensure all nodes at this level are expanded to full depth (non-terminal if possible)
            children = [self.full_tree(depth - 1) for _ in range(arity)]
            return Node(function, children)
        
    def grow_tree(self, depth):
        """
            Definition
            -----------
                At each node (except at maximum depth), randomly decide whether it will be a function node or a terminal node.
                If picks a terminal before reaching the maximum depth, that branch stops growing, resulting in irregularly shaped trees.
                Over many generated individuals, this leads to a variety of shapes and sizes rather than only "full" shapes.
        """
        
        # If we are at max depth, we must choose a terminal
        if depth == 0:
            terminal = np.random.choice(['x', '1.0'])
            return Node(1.0 if terminal == '1.0' else 'x')

        # Otherwise, randomly decide whether to select a function or terminal
        # Let's say there's a 50% chance of choosing a function node and a 50% chance of choosing a terminal node
        # You can adjust this probability as needed
        if np.random.rand() < 0.5:
            # Choose a terminal (end branch early)
            terminal = np.random.choice(['x', '1.0'])
            return Node(1.0 if terminal == '1.0' else 'x')
        else:
            # Choose a function node and recurse for children
            function = np.random.choice(['+', '-', '*', '/', 'sin', 'cos', 'log'])
            arity = self.get_function_arity(function)
            children = [self.grow_tree(depth - 1) for _ in range(arity)]
            return Node(function, children)
    
    def __str__(self):
        return str(self.tree)