"""            
    Definition
    -----------
    
        Contains stable version to compute 'basic' mu+lambda runs with HBC or without.
            - No Bloat or intron computation.
            - Used to compute performance runs.
"""

import numpy as np
import copy
import random
import util
import experiments as exp
import gp_math

import multiprocessing
import time

# Global variable to hold the GPLandscape instance in worker processes
gp_landscape = None

def init_worker(args):
    """
    Initializer function for each worker process.
    Instantiates a GPLandscape and assigns it to a global variable.
    
    Parameters:
    -----------
    args : Namespace or custom object
        The arguments required to initialize GPLandscape.
    """
    global gp_landscape
    gp_landscape = GPLandscape(args, initialize_pool=False)
    
# Top-level helper function for multiprocessing
def evaluate_fitness(tree):
    """
    Evaluates the fitness of a given tree using the global GPLandscape instance.

    Parameters:
    -----------
    tree : Node
        The root node of the individual's program tree.

    Returns:
    --------
    tuple
        A tuple containing the fitness value and a success flag.
    """
    return gp_landscape.symbolic_fitness_function(tree)

# TODO: Testing has its own self-contained classes. So, the iterations can be quick.
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
        self.fitness, self.success = fitness_function(self.tree) # Computes only the Absolute Error fitness
        
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

class GeneticAlgorithmGPPerformance:
    
    def __init__(self, args, mut_rate, inbred_threshold=None):
        
        # Variables
        self.args = args
        self.pop_size = args.pop_size
        self.generations = args.generations
        self.mutation_rate = mut_rate
        self.inbred_threshold = inbred_threshold
        self.max_depth = args.max_depth
        self.initial_depth = args.initial_depth
        self.poulation_success = False
        
        # Trackers
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []
        
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
            
    def compute_population_size_depth(self):
        
        # Get Tree sizes for entire population (nº nodes)
        tree_sizes = [self.count_nodes(ind.tree) for ind in self.population]
        average_size = sum(tree_sizes) / len(tree_sizes)
        self.average_size_list.append(average_size)
        
        # Get Tree depths for entire population
        tree_depths = [self.tree_depth(ind.tree) for ind in self.population]
        average_depth = sum(tree_depths) / len(tree_depths)
        self.average_depth_list.append(average_depth)  
    
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
        
    def compute_trees_distance(self, node1, node2):
        """
            Definition
            -----------
                Computes the distance between 2 different trees through recursion.
                Example:
                          # Tree 1: (x - 1) -> tree1 = Node('-', [Node('x'), Node('1')])
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
    
    def initialize_population_half_and_half(self):
        print(f"\nInitializing population.")
        self.population = []
        can_mate_full = 0
        can_mate_grow = 0

        # Custom half-n-half -> 75/25 
        # half = self.pop_size // 2
        # half = int(self.pop_size * 0.80)
        half = self.pop_size # TODO: This is the gp_lambda performance run
        
        # The first half is full initialization
        for i in range(half):
            individual = Individual(self.args, fitness_function=self.fitness_function, init_method="full") # Usually full
            indiv_total_nodes = self.count_nodes(individual.tree)
            if self.inbred_threshold is not None and indiv_total_nodes >= self.inbred_threshold:
                can_mate_full += 1
            self.population.append(individual)
            
            # print(f"Full ({i}) - individual total nodes: {indiv_total_nodes}")

        # The second half is grow (random) initialization
        for j in range(self.pop_size - half):
            individual = Individual(self.args, fitness_function=self.fitness_function, init_method="grow")
            indiv_total_nodes = self.count_nodes(individual.tree)
            if self.inbred_threshold is not None and indiv_total_nodes >= self.inbred_threshold:
                can_mate_grow += 1
            self.population.append(individual)
            
            # print(f"Grow ({j}) - individual total nodes: {indiv_total_nodes}")
            
        print(f"\nStarting with MaxDepth: {self.max_depth} and initDepth: {self.initial_depth}. Out of {self.pop_size}.")
        print(f"Grow: {can_mate_grow} Individuals can mate. Full: {can_mate_full} Individuals can mate.")
        print(f"Total: {(can_mate_grow + can_mate_full)} ({(can_mate_grow + can_mate_full) / self.pop_size * 100:.3f}%).\n")
        
    def calculate_fitness(self, curr_gen):
        
        # Determine the number of worker processes
        num_processes = multiprocessing.cpu_count()

        # Initialize the multiprocessing pool with the initializer
        with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(self.args, )) as pool:
            # Extract all trees from the population
            trees = [individual.tree for individual in self.population]

            # Map the evaluate_fitness function to all trees in parallel
            results = pool.map(evaluate_fitness, trees)
            
        # Process the results
        for individual, (fitness, success) in zip(self.population, results):
            
            individual.fitness = fitness
            individual.success = success
            
            if individual.success:
                print(f"Successful individual found in generation {curr_gen}")
                print(f"Function: {individual.tree}")
                self.poulation_success = True
                
    def check_succcess_new_pop(self, curr_gen, next_population):
        for individual in next_population:
            if individual.success:
                print(f"Successful individual found in new population in generation {curr_gen}")
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
        """
            Definition
            -----------
                Crossover of 2 parents in the population that produces 2 different offspring.
                Given tree1 = Node('+', [Node('x'), Node('1.0')])
                        tree2 = Node('+', [Node('*', [Node('x'), Node('1.0')]), Node('x')])
                Example 1:
                
                    Chosen Parent_node1: (+ x 1.0) and Node1:   'x' from tree1
                    Chosen Parent_node2: (* x 1.0) and Node2: '1.0' from tree2
                    
                    Arity of node1 and node2 are equal = 0. Node 'x' happens at Index 0 of parent_node1 (+ x 1.0).
                    New children where Node 'x' happens at Index 0 in parent_node1 is Node2 '1.0'. Therefore, New offspring is (+ 1.0 1.0).
                    
                Example 2:
                    Chosen Parent_node1: None and Node1:       (+ x 1.0) from tree1
                    Chosen Parent_node2: None and Node2: (+ (* x 1.0) x) from tree2
                    
                    Arity of node1 and node2 are equal = 2. Parent_node1 is NONE. New offspring is copy of child 2: (+ (* x 1.0) x)        
        """
      
        # Check if there is inbreeding prevention mechanism. (None means inbreeding is allowed)
        if self.inbred_threshold is not None:
            if not self.can_mate(parent1, parent2, self.inbred_threshold): # If distance(p1, p2) >= inbred_thres then skip bc [not False ==  True]
                return None, None

        self.allowed_mate_by_dist += 1
        
        # Clone parents to avoid modifying originals
        child1 = copy.deepcopy(parent1.tree)
        child2 = copy.deepcopy(parent2.tree)

        # Attempt crossover
        max_attempts = 10
        for attempt in range(max_attempts+1):
            
            # Select random nodes with their parents
            parent_node1, node1 = self.select_random_node_with_parent(child1)
            parent_node2, node2 = self.select_random_node_with_parent(child2)

            if node1 is None or node2 is None:
                continue  # Try again

            # Check if both nodes have the same arity
            arity1 = parent1.get_function_arity(node1.value)
            arity2 = parent2.get_function_arity(node2.value)
            if arity1 != arity2:
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
                    continue  # node2 not found, try again

            # Check for depth constraints
            depth_child1 = self.tree_depth(child1)
            depth_child2 = self.tree_depth(child2)
            if depth_child1 > self.max_depth or depth_child2 > self.max_depth:
                
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
        offspring1 = Individual(
            self.args,
            fitness_function=self.fitness_function,
            tree=child1,
            init_method="full"
        )
        
        offspring2 = Individual(
            self.args,
            fitness_function=self.fitness_function,
            tree=child2, 
            init_method="full"
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
        new_subtree = individual.full_tree(self.initial_depth) 
        
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

    def measure_diversity(self, population):
        """
            Definition
            -----------
                Calculate diversity based on tree structures.
                
                Example:  # Tree 1: (x + 1) -> tree1 = Node('+', [Node('x'), Node('1')])
                          # Tree 2: (x + 2) -> tree2 = Node('+', [Node('x'), Node('2')])
                          have distance of 1. 
                          
        """
        total_distance = 0
        count = 0
        
        # Iterate pairwise for all individuals in the population.
        for i in range(len(population)):
            for j in range(len(population)):
                if population[i].id != population[j].id:
                    distance = self.compute_trees_distance(population[i].tree, population[j].tree)
                    total_distance += distance
                    count += 1
            
        if count == 0:
            return 0
        diversity = total_distance / count
        return diversity
    
    # ----------------- Main execution loop ------------------------- #
    
    def run(self, fitness_function):
        # Assign fitness function to in class variable
        self.fitness_function = fitness_function
        
        # Init population
        self.initialize_population_half_and_half() # ramped half-n-half initializaiton
        
        # Initialize lists to store bloat metrics
        self.average_size_list = []
        self.average_depth_list = []
        
        for gen in range(self.generations):

            # Start timing
            start_time = time.time()
            
            # Calculate fitness
            self.calculate_fitness(gen) # Performance runs
            
            # Update best fitness list
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_fitness_list.append(best_individual.fitness)
    
            # Measure diversity
            diversity = self.measure_diversity(self.population)
            self.diversity_list.append(diversity)
            
            # Early Stopping condition if successful individual has been found
            if self.poulation_success == True:
                return self.best_fitness_list, self.diversity_list, gen + 1
    
            # Tournament Selection
            selected = self.tournament_selection()
    
            # Crossover and Mutation
            next_population = []
            i = 0
            
            # Lambda (parent+children)
            lambda_pop = self.pop_size * 2
            
            # Start metrics for debugging
            self.allowed_mate_by_dist = 0 # Reset to 0 on every generation
            offspring_count = 0
            none_count = 0
            
            # Lambda (parent+children)
            lambda_pop = self.pop_size * 2
            
            while i < lambda_pop: 
                parent1 = selected[i % len(selected)]
                parent2 = selected[(i + 1) % len(selected)]
                offspring = self.crossover(parent1, parent2)
    
                if offspring[0] is not None and offspring[1] is not None:
                    self.mutate(offspring[0])
                    self.mutate(offspring[1])
                    next_population.extend(offspring)
                    offspring_count += 1
                else:
                    none_count += 1
                    # Append parents if Inbreeding is allowed
                    if self.inbred_threshold is None: 
                        next_population.append(copy.deepcopy(parent1))
                        next_population.append(copy.deepcopy(parent2))
                    else:
                        # Introduce new random individuals to maintain population size if inbreeding is not allowed
                        new_individual = Individual(self.args, fitness_function=self.fitness_function, init_method="full")
                        next_population.append(new_individual)
                                                
                        if len(next_population) < self.pop_size:
                            new_individual = Individual(self.args, fitness_function=self.fitness_function, init_method="full")
                            next_population.append(new_individual)
        
                i += 2
                
            # print(f"Generation {gen + 1}: Checking by distance. Only {self.allowed_mate_by_dist} Individuals can actually  mate -> {self.allowed_mate_by_dist/len(selected) * 100:.3f}\n")
            # print(f"New individuals added from real offspring: {offspring_count} vs added from random (they could'nt mate): {none_count}")

            # Check if individual of next population is already successful. No need to recombination as it will always have largest fitness
            self.check_succcess_new_pop(gen+1, next_population)
            
            if self.poulation_success == True:
                
                # Update best fitness list
                best_individual = max(next_population, key=lambda ind: ind.fitness)
                self.best_fitness_list.append(best_individual.fitness)

                # Measure diversity
                diversity = self.measure_diversity(next_population)
                self.diversity_list.append(diversity)
                
                # Returns 2 + gens because technically we are just shortcutting the crossover of this current generation. So, +1 for 0th-indexed offset, and +1 for skipping some steps.
                # This added values will have been returned in the next gen loop iteration.
                return self.best_fitness_list, self.diversity_list, gen + 2
            
            # Combine the population (mu+lambda)
            combined_population = next_population[:lambda_pop] + self.population     
            combined_population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Update the population
            self.population = combined_population[:self.pop_size]
            self.pop_size = len(self.population)
            
            # Print progress
            if (gen + 1) % 10 == 0:
                # Measure Size, Depth statistics
                self.compute_population_size_depth()                
           
                print(f"\nInbreeding threshold set to: {self.inbred_threshold}.")
                print(f"Generation {gen + 1}: Best Fitness = {best_individual.fitness:.3f}\n"
                      f"Diversity = {self.diversity_list[gen]:.3f}\n"
                      f"Avg Size = {self.average_size_list[-1]:.3f}\n"
                      f"Avg Depth = {self.average_depth_list[-1]:.3f}\n")
                
                # End timing
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"\nTime taken to run 10 gen: {elapsed_time:.4f} seconds")
    
        return self.best_fitness_list, self.diversity_list, gen+1

# ---------- Landscape --------- # 

# Define this at the top level of your module
FUNC_MAP = {
    '+': gp_math.protected_sum,
    '-': gp_math.protected_subtract,
    '*': gp_math.protected_mult,
    '/': gp_math.protected_divide,
    'sin': gp_math.protected_sin,
    'cos': gp_math.protected_cos,
    'log': gp_math.protected_log
}

class GPLandscape:
    
    def __init__(self, args, initialize_pool=True):
        
        self.args = args
        self.target_function = util.select_gp_benchmark(args)
        self.bounds = args.bounds
        self.generate_data() # Generate all data points
        
        if initialize_pool:
            # Initialize the multiprocessing pool with the initializer
            self.pool = multiprocessing.Pool(initializer=init_worker, initargs=(self.args, ))
        else:
            # No pool initialization to prevent nested pools
            self.pool = None
    
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
    
    def generate_data(self):
        """
            Define input vectors (sampled within the search space).
        """
        self.x_values = np.arange(self.bounds[0], self.bounds[1] + 0.1, 0.1)  # Include the step size (0.1) as hyperparameters if adding more benchmarks
        self.y_values = self.target_function(self.x_values)
        self.data = list(zip(self.x_values, self.y_values))
    
    def evaluate_tree_vectorized(self, node, x_array):
        if node.is_terminal():
            return x_array if node.value == 'x' else np.full_like(x_array, float(node.value))
        else:
            func = FUNC_MAP.get(node.value, lambda a, b: np.zeros_like(a))
            args = [self.evaluate_tree_vectorized(child, x_array) for child in node.children]
            return func(*args)
    
    def symbolic_fitness_function(self, genome):
        try:
            outputs = self.evaluate_tree_vectorized(genome, self.x_values)  # Pass entire array        
            errors = outputs - self.y_values
            total_error = np.sum(np.abs(errors))
            success = np.all(np.abs(errors) <= 1e-4)
        except Exception as e:
            total_error = 1e6
            success = False
        fitness = 1 / (total_error + 1e-6)
        return fitness, success
        
if __name__ == "__main__":
    
    # Get args
    args = util.set_args()
    
    # Create Landscape
    gp_landscape = GPLandscape(args)

    # -------------------------------- Experiment: Multiple Runs w/ fixed population and fixed mutation rate --------------------------- #
    try:
        term1 = f"genetic_programming/{args.benchmark}/"
        term2 = "gp_lambda/"
        
        if args.inbred_threshold == 1:
            term3 = f"PopSize:{args.pop_size}_InThres:None_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_MaxD:{args.max_depth}_InitD:{args.initial_depth}" 
        else:
            term3 = f"PopSize:{args.pop_size}_InThres:{args.inbred_threshold}_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_MaxD:{args.max_depth}_InitD:{args.initial_depth}" 
              
        # Text to save files and plot.
        args.config_plot = term1 + term2 + term3
        
        if args.inbred_threshold == 1:
            print("Running GA with Inbreeding Mating...")
            results_inbreeding = exp.test_multiple_runs_function_gp_based(args, gp_landscape, None)
            util.save_accuracy(results_inbreeding, f"{args.config_plot}_inbreeding.npy")
        else:
            print("Running GA with NO Inbreeding Mating...")
            results_no_inbreeding = exp.test_multiple_runs_function_gp_based(args, gp_landscape, args.inbred_threshold)
            util.save_accuracy(results_no_inbreeding, f"{args.config_plot}_no_inbreeding.npy")
        
    finally:
        # Ensure that the pool is properly closed
        if gp_landscape.pool is not None:
            gp_landscape.pool.close()
            gp_landscape.pool.join()
