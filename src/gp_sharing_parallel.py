"""            
    Definition
    -----------
    
        Contains stable version to compute HBC mu+lambda runs with fitness sharing.
            - No Bloat or intron computation.
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
    def __init__(self, args, fitness_function=None, tree=None, id=None, generation=0):
        self.args = args
        self.bounds = self.args.bounds
        self.max_depth = self.args.max_depth 
        self.initial_depth = self.args.initial_depth
        self.tree = tree if tree is not None else self.full_tree(depth=self.initial_depth) # Initial depth of 6 as in paper
        self.diversity = 0
        self.id = id if id is not None else np.random.randint(1e9)
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
    
    def __str__(self):
        return str(self.tree)

class GeneticAlgorithmGPSharingParallel:
    
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
        
        # For using diversity in the loss function
        self.min_fitness = 0
        self.max_fitness = - np.inf
        
        self.min_div = 0
        self.max_div = - np.inf
        
        # Experimental move around fitness and diversity importance
        self.diversity_weight = args.diversity_weight
        self.fitness_weight = args.fitness_weight
        
        # Experimental for fitness sharing
        self.sigma_share = args.sigma_share
        self.sigma_share_weight = args.sigma_share_weight
        
    # ------------------------ Fitness sharing ------------------------ #
    
    def sharing_function(self, d):
        """
            Quantifies how much fitness should be shared between two individuals based on their distance.
        
            External Parameters
            --------------------    
                - sigma_share (float). Is threshold for how close two individuals must be for their fitness values to influence each other.
                - d (int). Is a distance between two individuals. Computed from our custom distance metric. Calculated in the measure diversity function.
        """
        if d < self.sigma_share:
            return 1 - (d / self.sigma_share)
        else:
            return 0

    def calculate_fitness_with_sharing(self, curr_gen):
        """
            Calculates fitness for all individuals in the population using multiprocessing.
        """
        
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
            # Scale up fitness
            fitness = util.scale_fitness_values(fitness, self.max_fitness, self.min_fitness)

            # Compute final fitness based on sharing factor
            if individual.sharing_factor > 0:
                individual.fitness = fitness / individual.sharing_factor
            else:
                individual.fitness = fitness

            # Check for success
            if success:
                print(f"Successful individual found in generation {curr_gen}")
                print(f"Function: {individual.tree}")
                self.poulation_success = True

    # ------------------------ Fitness sharing ------------------------ #
    
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
            
    def compute_population_size_depth(self):
        """
            Compute and store the average tree size of the population.
        """
        # Get Tree sizes for entire population (nº nodes)
        tree_sizes = [self.count_nodes(ind.tree) for ind in self.population]
        average_size = sum(tree_sizes) / len(tree_sizes)
        self.average_size_list.append(average_size) 
    
    # ----------------------- Tree ~ Node functions ------------------ #
    
    def select_random_node(self, tree):
        """
            Returns a single random node of a given tree (not Individual object).
            Example:
                - tree1 = Node('+', [Node('x'), Node('1')])
                  could return Node(+) or Node(x) or Node(1)
        """
        nodes = self.get_all_nodes(tree)
        return np.random.choice(nodes)
    
    def get_all_nodes(self, tree):
        """
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
            Selects a random node along with its parent.
        """
        all_nodes = self.get_all_nodes_with_parent(tree)
        if not all_nodes:
            return None, None
        return random.choice(all_nodes)
    
    def get_all_nodes_with_parent(self, node, parent=None):
        """
            Recursively collects all nodes in the tree along with their parent.
        """
        nodes = [(parent, node)]
        for child in node.children:
            nodes.extend(self.get_all_nodes_with_parent(child, node))
        return nodes
    
    def can_mate(self, ind1, ind2, inbred_threshold):
        """
            Inbreeding prevention mechanism based on pairwise distance between trees.
        """
        distance = self.compute_trees_distance(ind1.tree, ind2.tree)
        return distance >= inbred_threshold
    
    def tree_depth(self, node):
        """
            Returns the height of a given individual tree.
        """
        if node is None:
            return 0
        if node.is_terminal():
            return 1
        else:
            return 1 + max(self.tree_depth(child) for child in node.children)
        
    def compute_trees_distance(self, node1, node2):
        """
            Computes the distance between 2 different trees through recursion.
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
            Helper function to compute and adjust fitness based on sharing factor.
        """
        # Scale-up
        fitness = util.scale_fitness_values(individual.fitness, self.max_fitness, self.min_fitness)

        # Adjust fitness by sharing factor
        if individual.sharing_factor > 0:
            try:
                individual.fitness = fitness / individual.sharing_factor
            except RuntimeWarning as e:
                print(f"RuntimeWarning: {e}")
        else:
            individual.fitness = fitness
    
    def compute_individual_sharing_factor(self, population, individual):
        """
            Compute and assign the sharing factor for a given individual.
        """
        S_i = 0
        for other in population:
            if other.id != individual.id:
                distance = self.compute_trees_distance(other.tree, individual.tree)
                sh = self.sharing_function(distance)
                S_i += sh
        
        # Assign sharing factor to individual.
        individual.sharing_factor = S_i

    # ----------------- General GP Functions ------------------------- #
    
    def initialize_population(self):
        """
            Initializes the population with random individuals.
        """
        print(f"\nInitializing population with fitness sharing and initial sigma share of {self.sigma_share}, weighted per gen as {self.sigma_share_weight}.")
        self.population = []
        for _ in range(self.pop_size):
            individual = Individual(self.args, fitness_function=self.fitness_function)
            self.population.append(individual)
        
        # Measure initial diversity and get min~max range 
        self.measure_diversity(self.population)
        self.max_fitness, self.min_fitness = util.compute_min_max_fit(self.population, self.max_fitness, self.min_fitness)
        
        # Adjust fitness by sharing factor
        for individual in self.population:
            fitness = util.scale_fitness_values(individual.fitness, self.max_fitness, self.min_fitness)

            if individual.sharing_factor > 0: # If 0 then Individual is sufficiently different from all other individuals in the population
                individual.fitness = fitness / individual.sharing_factor
            else:
                individual.fitness = fitness
            
           
    def check_success_new_pop(self, curr_gen, next_population):
        """
            Checks if any individual in the new population is successful.
        """
        for individual in next_population:
            if individual.success:
                print(f"Successful individual found in new population in generation {curr_gen}")
                print(f"Function: {individual.tree}")
                self.poulation_success = True
            
    def tournament_selection(self, k=3):
        """
            Performs tournament selection.
        """
        selected = []
        for _ in range(self.pop_size):
            participants = random.sample(self.population, k)
            winner = max(participants, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected
    
    def crossover(self, parent1, parent2):
        """
            Performs crossover between two parents to produce two offspring.
        """
        # Check if there is inbreeding prevention mechanism. (None means inbreeding is allowed)
        if self.inbred_threshold is not None:
            if not self.can_mate(parent1, parent2, self.inbred_threshold):
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
            generation=parent1.generation + 1
        )
        self.compute_individual_sharing_factor(self.population, offspring1)
        self.compute_individual_fit(individual=offspring1)
        
        offspring2 = Individual(
            self.args,
            fitness_function=self.fitness_function,
            tree=child2,
            generation=parent1.generation + 1
        )
        self.compute_individual_sharing_factor(self.population, offspring2)
        self.compute_individual_fit(individual=offspring2)
        
        return offspring1, offspring2
    
    def mutate(self, individual):
        """
            Performs mutation on an individual by replacing a random subtree.
        """
        # Clone individual to avoid modifying original
        mutated_tree = copy.deepcopy(individual.tree)

        # Select a random node to mutate
        node_to_mutate = self.select_random_node(mutated_tree)
        if node_to_mutate is None:
            return  # Cannot mutate without a node

        # Replace the subtree with a new random subtree
        new_subtree = individual.full_tree(self.initial_depth) 
        
        # Ensure that the new_subtree has the correct arity
        required_children = individual.get_function_arity(new_subtree.value)
        if len(new_subtree.children) != required_children:
            return  # Discard mutation

        node_to_mutate.value = new_subtree.value
        node_to_mutate.children = new_subtree.children
        
        # Ensure the mutated tree does not exceed max depth
        if self.tree_depth(mutated_tree) > self.max_depth:
            return  # Discard mutation

        # Update individual
        individual.tree = mutated_tree

    def measure_diversity(self, population):
        """
            Calculate diversity based on tree structures.
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
            population[i].sharing_factor = S_i
            population[i].diversity = individual_diversity
            
        if count == 0:
            return 0
        diversity = total_distance / count
        
        # Set sigma share for next iteration to be:
        self.sigma_share = self.sigma_share_weight * diversity
        return diversity
    
    # ----------------- Main execution loop ------------------------- #
    
    def run(self, fitness_function):
        """
            Main loop to run the genetic algorithm.
        """
        # Assign fitness function to in-class variable
        self.fitness_function = fitness_function
        
        # Init population
        self.initialize_population()
        
        # Initialize lists to store bloat metrics
        self.average_size_list = []
        self.sigma_share_list = []
        
        for gen in range(self.generations):

            # Start timing
            start_time = time.time()
            
            # Calculate fitness using multiprocessing
            self.calculate_fitness_with_sharing(gen)
            
            # Update best fitness list
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_fitness_list.append(best_individual.fitness)
    
            # Measure diversity and progress of sigma share
            diversity = self.measure_diversity(self.population)
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
                        self.compute_individual_fit(individual=new_individual)
                        next_population.append(new_individual)
                                                
                        if len(next_population) < self.pop_size:
                            new_individual = Individual(self.args, fitness_function=self.fitness_function)
                            self.compute_individual_sharing_factor(self.population, new_individual) 
                            self.compute_individual_fit(individual=new_individual)
                            next_population.append(new_individual)
    
                i += 2

            # Check if individual of next population is already successful. No need to recombination as it will always have largest fitness
            self.check_success_new_pop(gen+1, next_population)
            
            if self.poulation_success == True:
                
                # Update best fitness list
                best_individual = max(next_population, key=lambda ind: ind.fitness)
                self.best_fitness_list.append(best_individual.fitness)

                # Measure diversity
                diversity = self.measure_diversity(next_population)
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
            
            # Print progress
            if (gen + 1) % 10 == 0:
                
                # Measure Size, Depth statistics
                self.compute_population_size_depth()
           
                print(f"\nInbreeding threshold set to: {self.inbred_threshold}.")
                print(f"Generation {gen + 1}: Best Fitness = {best_individual.fitness:.3f}\n"
                      f"Diversity = {self.diversity_list[gen]:.3f}\n"
                      f"Sigma Share: {self.sigma_share_list[gen]:.3f}\n"
                      f"Avg Size = {self.average_size_list[-1]:.3f}\n")
                
                # End timing
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"\nTime taken to run 10 gen: {elapsed_time:.4f} seconds")

        return self.best_fitness_list, self.diversity_list, self.sigma_share_list, gen+1

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
    finally:
        # Ensure that the pool is properly closed
        if gp_landscape.pool is not None:
            gp_landscape.pool.close()
            gp_landscape.pool.join()
