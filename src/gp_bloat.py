"""
    Author: Lute Lillo
    
    Definition
    ------------
        Genetic Programming Algorithm (mu + lambda) class. 
            - Initializes Population
            - Evolves Algorithm through mutation and crossover
            - Tracks metrics
"""

import numpy as np
import copy
import random
import util
import experiments as exp
import time
import gp_introns
from gp_node import Individual

class GeneticAlgorithmGPBloat:
    
    def __init__(self, args, mut_rate, inbred_threshold=None):
        self.args = args
        self.pop_size = args.pop_size
        self.generations = args.generations
        self.mutation_rate = mut_rate
        self.inbred_threshold = inbred_threshold
        self.max_depth = args.max_depth
        self.initial_depth = args.initial_depth
        self.intron_fraction = args.intron_fraction
        self.poulation_success = False
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []
        
        # TODO: Testing dynamic mutation type
        self.mutation_type = args.mutation_type
        
        if self.mutation_type == "random":
            self.div_checkpoint = 0
        else:
            self.div_checkpoint = 15 # TODO: This has to be re-thought, because it is harder to define a value to start going down at the beginning
        
    def collect_all_stats(self):
        
        # Compute population intros at failure.
        self.compute_population_size_depth()
        self.compute_introns_lists(self.intron_fraction)
        
        # Then collect them
        intron_lists = util.pack_intron_lists(self.pop_ratio_intron_list, self.avg_ratio_intron_list, self.pop_total_intron_list, self.pop_total_nodes_list)
        measures_lists = util.pack_measures_lists(self.average_size_list, self.average_depth_list)
        metrics_lists = util.pack_metrics_lists(self.best_fitness_list, self.diversity_list)
        
        return metrics_lists, measures_lists, intron_lists
        
    # -------------------- Intron computations ----------------------- #
    
    def compute_introns_lists(self, sample_fraction=1.0):
        
        # TODO: suubset because too expensive
        sample_size = int(len(self.population) * sample_fraction)
        sampled_population = random.sample(self.population, sample_size)
        
        # Measure intron metrics
        intron_metrics = self.landscape.measure_introns(sampled_population)
        self.pop_ratio_intron_list.append(intron_metrics['population_intron_ratio'])
        self.avg_ratio_intron_list.append(intron_metrics['average_intron_ratio'])
        self.pop_total_intron_list.append(intron_metrics['population_total_intron_nodes'])
        self.pop_total_nodes_list.append(intron_metrics['population_total_nodes'])
    
    def compute_population_size_depth(self):
        
        # Get Tree sizes for entire population (nº nodes)
        tree_sizes = [self.count_nodes(ind.tree) for ind in self.population]
        average_size = sum(tree_sizes) / len(tree_sizes)
        self.average_size_list.append(average_size)
        
        # Get Tree depths for entire population
        tree_depths = [self.tree_depth(ind.tree) for ind in self.population]
        average_depth = sum(tree_depths) / len(tree_depths)
        self.average_depth_list.append(average_depth)  
            
    # ---------------------- Diversity computations -------------------------------#
    
    def measure_diversity(self, population):
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
        
        # Track diversity value
        self.diversity_list.append(diversity)

    # ----------------------- Individial Tree ~ Node computations ------------------ #
    
    def generate_intron_subtree_for_context(self, context_operator=None):
        """
            Definition
            -----------
                Returns a subtree that behaves as an intron, i.e., it does not affect the parent operator's outcome.
        
            Parameters
            -----------
                - context_operator(str or None): The operator in the parent node where this subtree is being inserted.
                                                If None, we'll randomly choose from all intron patterns.
            
            Returns
            -----------
                - Node (node): A subtree that yields 0 in add/sub contexts or 1 in mul/div contexts, or random if context is None.
        """
        # Distinguish the two main contexts:
        #  - add/sub => yields zero
        #  - mul/div => yields one
        
        if context_operator in ['+', '-']:
            # We want a subtree that yields 0
            return util.subtree_yields_zero()
        elif context_operator in ['*', '/']:
            # We want a subtree that yields 1
            return util.subtree_yields_one()
        else:
            # If we don't know the context, pick from either one randomly
            # or build a combined library of all patterns
            patterns = []
            
            # All zero-yielding
            zero_subtree = util.subtree_yields_zero()
            # All one-yielding
            one_subtree = util.subtree_yields_one()
            
            # We can just pick from [zero_subtree, one_subtree], or expand it further
            # to pick from multiple patterns each time
            patterns.append(zero_subtree)
            patterns.append(one_subtree)
            
            return np.random.choice(patterns)

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

    def select_random_node_and_parent_operator(self, tree):
        """
        Traverses the tree and collects (node, parent's operator).
        Then returns one randomly chosen (node, parent_operator).
        
        Parameters
        ----------
        root : Node
            The root of the individual's tree.
        
        Returns
        -------
        (Node, str or None)
            The randomly chosen node and the operator in the parent node 
            that uses this node as a child. If the node is the root, 
            parent_operator will be None.
            
            Returns (None, None) if the tree is empty.
        """
        if tree is None:
            return None, None  # Empty tree
        
        # We'll do a BFS or DFS, collecting (child, parent's operator).
        # For the root, there's no parent, so parent_operator is None.
        
        queue = [(tree, None)]  # (current_node, parent_operator)
        collected = []
        
        while queue:
            current_node, parent_op = queue.pop()
            
            # Record this node and the parent's operator
            collected.append((current_node, parent_op))
            
            # If this node is not terminal, enqueue its children
            if not current_node.is_terminal():
                for child in current_node.children:
                    queue.append((child, current_node.value))
        
        if not collected:
            return None, None
        
        # Randomly select one entry from 'collected'
        chosen_node, chosen_parent_op = random.choice(collected)
        return chosen_node, chosen_parent_op

    
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
        
    def average_node_arity(self, node):
        """
        Calculates the average arity (number of children) of nodes in a tree.
        """
        total_nodes, total_children = self._sum_node_arities(node)
        if total_nodes == 0:
            return 0
        return total_children / total_nodes

    def _sum_node_arities(self, node):
        if node is None:
            return 0, 0
        total_nodes = 1
        total_children = len(node.children)
        for child in node.children:
            nodes, children = self._sum_node_arities(child)
            total_nodes += nodes
            total_children += children
        return total_nodes, total_children

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
            
    def initialize_population(self):
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
        for individual in self.population:
            individual.fitness, individual.success = self.fitness_function(individual.tree)
            if individual.success:
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
    
    def mating_selection(self, selected):
        can_selected = []
        
        # Select only individuals with nº nodes > inbred_threshold value
        if self.inbred_threshold is not None:
                can_mate_selected = 0
                
                for ind in selected:
                    tree_size = self.count_nodes(ind.tree)
        
                    if tree_size >= self.inbred_threshold:
                        can_mate_selected +=  1
                        can_selected.append(ind)
                            
        else:
            can_selected = selected
    
        return can_selected
    
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

        offspring1 = Individual(
            self.args,
            fitness_function=self.fitness_function,
            tree=child1,
        )
        offspring2 = Individual(
            self.args,
            fitness_function=self.fitness_function,
            tree=child2,
        )

        return offspring1, offspring2
    
    def mutate_intron(self, individual):
        """
            Definition
            -----------
                Applies Random Subtree mutation to an individual, specifically inserting guaranteed intron subtrees in the correct operator context (if known).
        """
        mutated_tree = copy.deepcopy(individual.tree)

        # Select a random node to mutate
        node_to_mutate, parent_op = self.select_random_node_and_parent_operator(mutated_tree) # Note: Do not use the other prent, node function as this specifically selects for 1 parent node. Ensuring introns.
        if node_to_mutate is None:
            return  # No mutation possible

        # Generate an intron subtree based on the parent's operator or None if you don't track it
        new_subtree = self.generate_intron_subtree_for_context(parent_op)
                
        # Optionally check arity, or just override
        required_children = individual.get_function_arity(new_subtree.value)
        if len(new_subtree.children) != required_children:
            return  # Discard mutation if it doesn't match
        
        # Replace
        node_to_mutate.value = new_subtree.value
        node_to_mutate.children = new_subtree.children

        # Validate tree depth
        if self.tree_depth(mutated_tree) > self.max_depth:
            return  # Exceeding depth, discard
        
        # Commit mutation
        individual.tree = mutated_tree
        
    def mutate(self, individual):
        """
            Definition
            -----------
                Applies Random Subtree mutation to an individual.
        """
        
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
        
    def control_mutation_type(self, gen):
        
        # Trim last 15 generations
        temp_list = self.diversity_list[gen - 15:gen]
        
        # Take average of last 15 generations
        avg_div = np.mean(temp_list)
        
        # Compare to last checkpoint
        if avg_div < self.div_checkpoint:
            
            # Introduce exploration if diversity is lower
            self.mutation_type = "intron"
            
        else:
            self.mutation_type = "random"
            
        print(f"\nMutation type at generation: {gen} is {self.mutation_type}. Current Average (last 15 gens) Diversity is {avg_div}. Previous was {self.div_checkpoint}.")
        
        # Update
        self.div_checkpoint = avg_div
        
    # ----------------- Main execution loop ------------------------- #
    
    def run(self, landscape):
        
        # Assign fitness function to in class variable
        self.landscape = landscape
        self.fitness_function = self.landscape.symbolic_fitness_function
        
        # Init population
        self.initialize_population()
        
        # Initialize lists to store bloat metrics
        self.average_size_list = []
        self.average_depth_list = []
        
        # Initialize lists to store intron statistical metrics
        self.pop_ratio_intron_list = []
        self.avg_ratio_intron_list = []
        self.pop_total_intron_list = []
        self.pop_total_nodes_list = []
        
        for gen in range(self.generations):

            # Calculate fitness
            self.calculate_fitness(gen)
            
            # Update best fitness list
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_fitness_list.append(best_individual.fitness)
    
            # Measure diversity
            self.measure_diversity(self.population)
            
            if (gen + 1) % 15 == 0:
                self.control_mutation_type(gen + 1)
            
            # Start timing
            start_time = time.time()
            
            # Collect all metrics
            metrics_lists, measures_lists, intron_lists = self.collect_all_stats()
            
            # End timing
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"\nGen: {gen+1}. InbreedThreshold: {self.inbred_threshold}. Time taken to collect all data: {elapsed_time:.4f} seconds")
            
            # Early Stopping condition if successful individual has been found
            if self.poulation_success == True:
                  
                return metrics_lists, measures_lists, intron_lists, gen + 1
    
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
                      
            # -------------- No specific mating selection ------------ #
            while i < lambda_pop: 
                parent1 = selected[i % len(selected)]
                parent2 = selected[(i + 1) % len(selected)]
                offspring = self.crossover(parent1, parent2)
    
                if offspring[0] is not None and offspring[1] is not None:
                    
                    # --- Mutate introducing random subtrees --- #
                    # self.mutate(offspring[0])
                    # self.mutate(offspring[1])
       
                    # --- Mutate introducing specific introns --- #
                    # self.mutate_intron(offspring[0])
                    # self.mutate_intron(offspring[1])
                    
                    # ---- Half-n-half mutation type population ---- #
                    self.mutate(offspring[0])
                    self.mutate_intron(offspring[1])
                    
                    # ---- half-n-half with quartiles mutation type population Random p(0.75) and Intron p(0.25) -> random_plus_... ---- #
                    # if random.random() <= 0.75:
                    #     self.mutate(offspring[0])
                    #     self.mutate(offspring[1])
                    # else:
                    #     self.mutate_intron(offspring[0])
                    #     self.mutate_intron(offspring[1])
                        
                    # ---- half-n-half with quartiles mutation type population Random p(0.25) and Intron p(0.75) -> intron_plus_...---- #
                    # if random.random() > 0.75:
                    #     self.mutate(offspring[0])
                    #     self.mutate(offspring[1])
                    # else:
                    #     self.mutate_intron(offspring[0])
                    #     self.mutate_intron(offspring[1])
                        
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

            # Combine the population (mu+lambda)
            combined_population = next_population[:lambda_pop] + self.population     
            combined_population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Update the population
            self.population = combined_population[:self.pop_size]
            self.pop_size = len(self.population)
            
            if self.poulation_success == True:
                
                # Update best fitness list
                best_individual = max(self.population, key=lambda ind: ind.fitness)
                self.best_fitness_list.append(best_individual.fitness)

                # Measure diversity
                self.measure_diversity(self.population)
                
                # Collect all
                metrics_lists, measures_lists, intron_lists = self.collect_all_stats()

                # Returns 2 + gens because technically we are just shortcutting the crossover of this current generation. 
                # So, +1 for 0th-indexed offset, and +1 for skipping some steps.
                # This added values will have been returned in the next gen loop iteration.
                return metrics_lists, measures_lists, intron_lists, gen + 2
        
            # Print progress
            if (gen + 1) % 10 == 0:
                
                print(f"Generation {gen + 1}: Best Fitness = {best_individual.fitness:.3f}\n"
                      f"Diversity = {self.diversity_list[gen]:.3f}\n"
                      f"Avg Size = {self.average_size_list[-1]:.3f}\n"
                      f"Population Intron Ratio = {self.pop_ratio_intron_list[-1]:.3f}\n"
                      f"Current Mutation Type = {self.mutation_type}\n")
                
        # Collect all if failed run
        metrics_lists, measures_lists, intron_lists = self.collect_all_stats()
        
        return metrics_lists, measures_lists, intron_lists, gen+1
        
if __name__ == "__main__":
    
    # Get args
    args = util.set_args()
    
    # Create Landscape
    landscape = gp_introns.GPIntronAnalyzer(args)

    # -------------------------------- Experiment: Multiple Runs w/ fixed population and fixed mutation rate --------------------------- #
    
    term1 = f"genetic_programming/{args.benchmark}/"
    term2 = "bloat/"
    term3 = f"PopSize:{args.pop_size}_InThres:{args.inbred_threshold}_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_MaxD:{args.max_depth}_InitD:{args.initial_depth}" 
    args.config_plot = term1 + term2 + term3
    
    print("Running GA with NO Inbreeding Mating...")
    results_no_inbreeding = exp.test_multiple_runs_function_bloat(args, landscape, args.inbred_threshold)
    util.save_accuracy(results_no_inbreeding, f"{args.config_plot}_no_inbreeding.npy")