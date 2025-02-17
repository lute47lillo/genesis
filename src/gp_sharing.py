"""            
    Definition
    -----------
    
        Contains stable version to compute 'basic' mu+lambda runs with inbreeding threshold.
            - Experimental treatments of diversity + fitness guided selection.
            - No Bloat or intron computation.
"""

import numpy as np
import copy
import random
import util
import experiments as exp
import plotting as plot
import gp_math
import multiprocessing

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
    global_analyzer = GPLandscape(args, initialize_pool=False)  # Avoid nested pools


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
    def __init__(self, args, fitness_function=None, tree=None, id=None, ancestors=None, generation=0):
        self.args = args
        self.bounds = self.args.bounds
        self.max_depth = self.args.max_depth 
        self.initial_depth = self.args.initial_depth
        self.tree = tree if tree is not None else self.random_tree(depth=self.initial_depth) # Initial depth of 6 as in paper
        self.diversity = 0
        self.id = id if id is not None else np.random.randint(1e9)
        self.ancestors = ancestors if ancestors is not None else set()
        self.generation = generation  # Track the generation of the individual
        
        # Init fitness for individual in creation and self.success
        self.fitness, self.success = fitness_function(self.tree) # Computes only the Absolute Error fitness
        
        self.sharing_factor = 0
        
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

class GeneticAlgorithmGPSharing:
    
    def __init__(self, args, mut_rate, inbred_threshold=None):
        self.args = args
        self.pop_size = args.pop_size
        self.generations = args.generations
        self.mutation_rate = mut_rate
        self.inbred_threshold = inbred_threshold
        self.max_depth = args.max_depth
        self.initial_depth = args.initial_depth
        self.poulation_success = False
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []
        
        # TODO: For using diversity in the loss function
        self.min_fitness = 0
        self.max_fitness = - np.inf
        
        self.min_div = 0
        self.max_div = - np.inf
        
        # TODO: Experimental move around fitness and diversity importance
        self.diversity_weight = args.diversity_weight
        self.fitness_weight = args.fitness_weight
        
        # TODO: Experimental for fitness sharing
        self.sigma_share = args.sigma_share
        self.sigma_share_weight = args.sigma_share_weight
        
    # ------------------------ Fitness sharing ------------------------ #
    
    def sharing_function(self, d):
        """
            Definition
            -----------
                Quantifies how much fitness should be shared between two individuals based on their distance.
            
            External Parameters
            --------------------    
                - sigma_share (float). Is threshold for how close two individuals must be for their fitness values to influence each other.
                - d (int). Is a distance between two individuals. Computed from our custom distance metric. Calculated in the measure diversity function.
        """
        # print(d, self.sigma_share)
        if d < self.sigma_share:
            # print(f"Sharing distance becase d: {d} and sigma share: {self.sigma_share}")
            return 1 - (d / self.sigma_share)
        else:
            return 0

    def calculate_fitness_with_sharing(self, curr_gen):
        
        # Compute Absolute Error fitness
        for individual in self.population:
            fitness, individual.success = self.fitness_function(individual.tree)
            
            # Scale up fitness.
            fitness = util.scale_fitness_values(fitness, self.max_fitness, self.min_fitness)
            
            # Compute final fitness based off sharing
            if individual.sharing_factor > 0:
                individual.fitness = fitness / individual.sharing_factor
            else:
                individual.fitness = fitness
            
            if individual.success:
                print(f"Successful individual found in generation {curr_gen}")
                print(f"Function: {individual.tree}")
                self.poulation_success = True

        # Adjust fitness
        # for individual in self.population:
        #     if individual.sharing_factor > 0:
        #         individual.fitness = individual.fitness / individual.sharing_factor
        #     else:
        #         individual.fitness = individual.fitness
        
    # ------------------------ Fitness sharing ------------------------ #
    
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
    
    def compute_individual_fit(self, individual):
        """
            Definition
            ------------
                Helper function that is only used after offspring is created, or when new random individuals are added to the population.
        """
          
        # Scale-up
        fitness = util.scale_fitness_values(individual.fitness, self.max_fitness, self.min_fitness)
       
        # Adjust fitness by sharing factor
        # TODO: Might need to adjust fitness like in the div+fit
        for individual in self.population:
            if individual.sharing_factor > 0:
                try:
                    # print(individual.fitness, individual.sharing_factor)
                    individual.fitness = fitness / individual.sharing_factor
                except RuntimeWarning as e:
                    print(f"RuntimeWarning: {e}")
            else:
                individual.fitness = fitness
        
    def compute_individual_sharing_factor(self, population, individual):
     
        S_i = 0
        for i in range(len(population)):
            if population[i].id != individual.id:
                # Compute fitness sharing dist for later calculation
                distance = self.compute_trees_distance(population[i].tree, individual.tree)                
                sh = self.sharing_function(distance)
                S_i += sh
        
        # Assign diversity to specific individual.
        individual.sharing_factor = S_i

    # ----------------- General GP Functions ------------------------- #
    
    def initialize_population(self):
        print(f"\nInitializing population with fitness sharing and inital sigma share of {self.sigma_share}, weighted per gen as {self.sigma_share_weight}.")
        self.population = []
        for _ in range(self.pop_size):
            individual = Individual(self.args, fitness_function=self.fitness_function)
            self.population.append(individual)
        
        # Measure initial diversity and get min~max range 
        self.measure_diversity(self.population, 0)
        self.max_fitness, self.min_fitness = util.compute_min_max_fit(self.population, self.max_fitness, self.min_fitness)
        
        # Adjust fitness by sharing factor
        for individual in self.population:
            fitness = util.scale_fitness_values(individual.fitness, self.max_fitness, self.min_fitness)

            if individual.sharing_factor > 0: # If 0 then Individual is sufficiently different from all other individuals in the population
                individual.fitness = fitness / individual.sharing_factor
            else:
                individual.fitness = fitness
            
           
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

            # TODO: Currenlty enforcnig that the parents need to have the same arity. 
            # TODO: In the future we could deal with different arities by removing, adding nodes rather than swapping entire subtrees
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
            ancestors=parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id}),
            generation=parent1.generation + 1
        )
        self.compute_individual_sharing_factor(self.population, offspring1)
        self.compute_individual_fit(offspring1)
        
        offspring2 = Individual(
            self.args,
            fitness_function=self.fitness_function,
            tree=child2,
            ancestors=parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id}),
            generation=parent1.generation + 1
        )
        self.compute_individual_sharing_factor(self.population, offspring2)
        self.compute_individual_fit(offspring2)
        
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

    def measure_diversity(self, population, curr_gen):
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
            individual_diversity = 0
            S_i = 0
            for j in range(len(population)):
                if population[i].id != population[j].id:
                    
                    # Compute custom distance metric for diversity
                    distance = self.compute_trees_distance(population[i].tree, population[j].tree)
                    individual_diversity += distance
                    total_distance += distance
                    count += 1
                    
                    # Compute fitness sharing dist for later calculation
                    sh = self.sharing_function(distance)
                    S_i += sh
            
            # Assign sharing factor to individual.
            # print(f"({i}) - Sharing factor: {S_i}")
            population[i].sharing_factor = S_i
            population[i].diversity = individual_diversity
            
        # How many comparisons, it goes i- > j and j->i so is double of C(300 over 2)
        if count == 0:
            return 0
        diversity = total_distance / count
        
        # Set sigma share for next iteration to be:
        self.sigma_share = self.sigma_share_weight * diversity
        return diversity
    
    # ----------------- Main execution loop ------------------------- #
    
    def run(self, fitness_function):
        # Assign fitness function to in class variable
        self.fitness_function = fitness_function
        
        # Init population
        self.initialize_population()
        
        # Initialize lists to store bloat metrics
        self.average_size_list = []
        self.sigma_share_list = []
        
        for gen in range(self.generations):

            # Calculate fitness
            self.calculate_fitness_with_sharing(gen)
            
            # Update best fitness list
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_fitness_list.append(best_individual.fitness)
    
            # Measure diversity and progress of sigma share
            diversity = self.measure_diversity(self.population, gen+1)
            self.diversity_list.append(diversity)
            self.sigma_share_list.append(self.sigma_share)
            
            # Early Stopping condition if successful individual has been found
            if self.poulation_success == True:
               
                return self.best_fitness_list, self.diversity_list, self.sigma_share_list, gen + 1
    
            # Selection
            selected = self.tournament_selection()
    
            # Crossover and Mutation
            next_population = []
            i = 0
            
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
                else:
                    # Append parents if Inbreeding is allowed
                    if self.inbred_threshold is None: 
                        next_population.append(copy.deepcopy(parent1))
                        next_population.append(copy.deepcopy(parent2))
                    else:
                        # Introduce new random individuals to maintain population size if inbreeding is not allowed
                        new_individual = Individual(self.args, fitness_function=self.fitness_function)
                        self.compute_individual_sharing_factor(self.population, new_individual)
                        self.compute_individual_fit(new_individual)
                        next_population.append(new_individual)
                                                
                        if len(next_population) < self.pop_size:
                            new_individual = Individual(self.args, fitness_function=self.fitness_function)
                            self.compute_individual_sharing_factor(self.population, new_individual) 
                            self.compute_individual_fit(new_individual)
                            next_population.append(new_individual)
        
                i += 2

            # Check if individual of next population is already successful. No need to recombination as it will always have largest fitness
            self.check_succcess_new_pop(gen+1, next_population)
            
            if self.poulation_success == True:
                
                # Update best fitness list
                best_individual = max(next_population, key=lambda ind: ind.fitness)
                self.best_fitness_list.append(best_individual.fitness)

                # Measure diversity
                diversity = self.measure_diversity(next_population, gen+1)
                self.diversity_list.append(diversity)
                self.sigma_share_list.append(self.sigma_share)
                
                # Returns 2 + gens because technically we are just shortcutting the crossover of this current generation. So, +1 for 0th-indexed offset, and +1 for skipping some steps.
                # This added values will have been returned in the next gen loop iteration.
                return self.best_fitness_list, self.diversity_list, self.sigma_share_list, gen + 2
            
            # Combine the population (mu+lambda)
            combined_population = next_population[:lambda_pop] + self.population     
            combined_population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Update the population
            self.population = combined_population[:self.pop_size]
            self.pop_size = len(self.population)
            
            # Re-compute min - max fitness for normalization
            self.max_fitness, self.min_fitness = util.compute_min_max_fit(self.population, self.max_fitness, self.min_fitness)
            
            # print(f"Generation {gen + 1}: Sigma Share: {self.sigma_share}. Diversity: {diversity}. Best Individual Fitness = {best_individual.fitness:.3f}.\n")
        
            # Print progress
            if (gen + 1) % 10 == 0:
                
                # Measure Size, Depth statistics
                self.compute_population_size_depth()
           
                print(f"\nInbreeding threshold set to: {self.inbred_threshold}.")
                print(f"Generation {gen + 1}: Best Fitness = {best_individual.fitness:.3f}\n"
                      f"Diversity = {self.diversity_list[gen]:.3f}\n"
                      f"Sigma Share: {self.sigma_share_list[gen]:.3f}\n"
                      f"Avg Size = {self.average_size_list[-1]:.3f}\n")
    
        return self.best_fitness_list, self.diversity_list, self.sigma_share_list, gen+1
    
class GPLandscape:
    
    def __init__(self, args, initialize_pool=True):
        
        self.args = args
        self.target_function = util.select_gp_benchmark(args)
        self.bounds = args.bounds
        self.data = self.generate_data() # Generate all data points

    
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
        x_values = np.arange(self.bounds[0], self.bounds[1] + 0.1, 0.1)  # TODO Include the step size (0.1) as hyper parameters if adding more benchmarks
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
        
if __name__ == "__main__":
    
    # Get args
    args = util.set_args()
    
    # Create Landscape
    gp_landscape = GPLandscape(args)

    # -------------------------------- Experiment: Multiple Runs w/ fixed population and fixed mutation rate --------------------------- #
    
    term1 = f"genetic_programming/{args.benchmark}/"
    term2 = "sharing/"

    if args.inbred_threshold == 1:
        term3 = f"SigmaShare:{args.sigma_share_weight}_PopSize:{args.pop_size}_InThres:None_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_MaxD:{args.max_depth}_InitD:{args.initial_depth}" 
    else:
        term3 = f"SigmaShare:{args.sigma_share_weight}_PopSize:{args.pop_size}_InThres:{args.inbred_threshold}_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_MaxD:{args.max_depth}_InitD:{args.initial_depth}" 
        
    # Text to save files and plot.
    args.config_plot = term1 + term2 + term3
        
    if args.inbred_threshold == 1:
        print("Running GA with Inbreeding Mating...")
        results_inbreeding = exp.test_multiple_runs_function_sharing(args, gp_landscape, None)
        util.save_accuracy(results_inbreeding, f"{args.config_plot}_inbreeding.npy")
    else:
        print("Running GA with NO Inbreeding Mating...")
        results_no_inbreeding = exp.test_multiple_runs_function_sharing(args, gp_landscape, args.inbred_threshold)
        util.save_accuracy(results_no_inbreeding, f"{args.config_plot}_no_inbreeding.npy")