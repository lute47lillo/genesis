"""

    TODO: Need to compare Novelty against fitness methods
"""


import numpy as np
import copy
from scipy.spatial import distance
import random

class Individual:
    def __init__(self, genes, id=None, parents=None, generation=0):
        """
            Definition
            -----------
                - genes: The gene vector representing the solution.
                - id: A unique identifier for each individual.
                - parents=None: A set containing the IDs of all ancestors.
        """
        # self.genes = genes
        self.genes = genes if genes is not None else []
        self.fitness = None
        self.novelty = None
        self.total_fitness = None
        self.behavior = None
        self.id = id if id is not None else np.random.randint(1e9)
        self.parents = parents if parents is not None else []
        self.ancestors = set() # IDs of all ancestors
        self.generation = generation

        # TODO: Calculating ancestry for all ancestors is really computatioinally expensive so limit it to 6 generations?
        # Update ancestors with depth limitation
        max_depth = 10  # Set the desired ancestry depth
        self.ancestors = set()
        for parent in self.parents:
            if parent.generation >= self.generation - max_depth:
                self.ancestors.update(parent.ancestors)
                self.ancestors.add(parent.id)
                
        # Update ancestors based on parents
        # for parent in self.parents:
        #     self.ancestors.update(parent.ancestors)
        #     self.ancestors.add(parent.id) 

# -------------------------------------------------------------------------------------------------------- #

class NoveltyArchive:
    def __init__(self, threshold=0.1, k=5, max_size=500):
        """
        Initialize the Novelty Archive.

        Parameters:
        - threshold (float): Distance threshold for considering behaviors as novel.
        - k (int): Number of nearest neighbors to consider for novelty calculation.
        - max_size (int): Maximum size of the archive.
        """
        self.archive = []
        self.threshold = threshold
        self.k = k
        self.max_size = max_size

    def compute_novelty(self, behavior, population_behaviors):
        """
        Compute the novelty of a given behavior.

        Parameters:
        - behavior (list or numpy.ndarray): Behavior descriptor of the individual.
        - population_behaviors (list): List of behavior descriptors of the current population.

        Returns:
        - novelty (float): Calculated novelty score.
        """
        # Combine archive and current population behaviors
        all_behaviors = self.archive + population_behaviors

        if len(all_behaviors) == 0:
            return float('inf')  # Maximum novelty for the first individual

        # Compute distances to all behaviors
        distances = distance.cdist([behavior], all_behaviors, 'euclidean')[0]
        # Exclude distance to self if present
        distances = distances[distances != 0]

        # Get k nearest distances
        if len(distances) >= self.k:
            nearest_distances = np.partition(distances, self.k)[:self.k]
        else:
            nearest_distances = distances

        novelty = np.mean(nearest_distances)
        return novelty

    def add(self, behavior):
        """
        Add a new behavior to the archive if it is novel enough.

        Parameters:
        - behavior (list or numpy.ndarray): Behavior descriptor of the individual.
        """
        if len(self.archive) == 0:
            self.archive.append(behavior)
            return

        # Compute distance to all behaviors in the archive
        distances = distance.cdist([behavior], self.archive, 'euclidean')[0]
        min_distance = np.min(distances)

        if min_distance > self.threshold:
            if len(self.archive) >= self.max_size:
                # Remove the oldest entry to maintain the archive size
                self.archive.pop(0)
            self.archive.append(behavior)

class LandscapeGA:
    def __init__(self, args, landscape, bounds, inbred_threshold=None):
        self.args = args
        self.pop_size = args.pop_size
        self.dimensions = args.dimensions
        self.bounds = bounds
        self.generations = args.generations
        self.mutation_rate = args.mutation_rate
        self.tournament_size = args.tournament_size
        self.inbred_threshold = inbred_threshold
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []
        self.collapse_events = []
        self.initial_pop_size = self.pop_size
        self.landscape = landscape
        
        # Existing initialization code...
        self.max_kinship = 0.5 # max_kinship  # Threshold for kinship coefficient. 0.125 is first cousings. 0.0625 is second cousins
        
        # Initialize the Novelty Archive
        # TODO: Create hyperparameters for the novelty
        self.novelty_archive = NoveltyArchive(threshold=0.1, k=10, max_size=self.pop_size)
        
    def log_ancestry(self, generation_number):
        if self.current_generation == generation_number:
            print(f"Ancestry Information at Generation {generation_number}:")
            for ind in self.population:
                print(f"Individual ID: {ind.id}, Parents: {[parent.id for parent in ind.parents]}")
                print(f"Ancestors: {ind.ancestors}")
                print("---")

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            genes = np.random.randint(2, size=self.dimensions)
            individual = Individual(genes)
            self.population.append(individual)

    def calculate_fitness(self):
        for individual in self.population:
            individual.fitness = self.landscape.get_fitness(individual.genes)
            
    def calculate_fitness_and_novelty(self):
        """
            TODO: Implement 
                            alpha = 1.0  # Weight for fitness
                            beta = 1.0   # Weight for novelty           
        """
        # Collect behaviors for the current population
        population_behaviors = []
        for individual in self.population:
            # Calculate fitness
            individual.fitness = self.landscape.get_fitness(individual.genes)
            # Extract behavior
            behavior = self.extract_behavior(individual.genes)
            individual.behavior = behavior
            population_behaviors.append(behavior)

        # Compute novelty scores and update fitness
        for individual in self.population:
            novelty = self.novelty_archive.compute_novelty(individual.behavior, population_behaviors)
            
            # You can adjust the weighting between fitness and novelty as needed
            individual.novelty = novelty
            
            # Combine fitness and novelty 
            # TODO: after normalizing over maximum global fitness in MBP
            novelty_fitness = novelty / self.landscape.m
            individual.total_fitness = individual.fitness + novelty_fitness
            
            # Add to the archive
            self.novelty_archive.add(individual.behavior)
            
    def kinship_coefficient(self, ind1, ind2):
        """
            # TODO: Simple version
            The kinship coefficient (f) between two individuals is the probability that a randomly selected allele from both individuals is identical by descent (IBD).
            
            - initial population is unrelated.
        """
        
        shared_ancestors = ind1.ancestors.intersection(ind2.ancestors)
        total_ancestors = ind1.ancestors.union(ind2.ancestors)

        if not total_ancestors:
            return 0.0  # No ancestors, unrelated

        # Simple approximation: ratio of shared ancestors to total ancestors
        f = len(shared_ancestors) / len(total_ancestors)
        return f

    def tournament_selection(self, k=3):
        selected = []
        for i in range(self.pop_size):
            participants = np.random.choice(self.population, k)
            winner = max(participants, key=lambda ind: ind.total_fitness) # ind.total_fitness for novelty + fitness / fitness for just fitness
            selected.append(winner)
        return selected

    def genetic_distance(self, ind1, ind2):
        return np.sum(ind1.genes != ind2.genes)  # Hamming distance
    
    def compute_distance_to_nearest_peak(self, genome):
        """
        Compute the Euclidean distance from a position to the nearest peak.

        Parameters:
        - position (numpy.ndarray): The position of the individual in the search space.

        Returns:
        - distance (float): The shortest Euclidean distance to any peak.
        """
        position = np.sum(genome)
        peaks = self.landscape.peaks  # Access peaks directly from the landscape
        if not peaks:
            return float('inf')  # No peaks defined

        distances = [
            min(abs(peak.position - position), self.landscape.n - abs(peak.position - position))
            for peak in peaks
        ]
        return min(distances)
    
    def extract_behavior(self, genome):
        """
        Extract the behavior descriptor for an individual based on its genes.

        Parameters:
        - genes (numpy.ndarray): The genes of the individual.

        Returns:
        - behavior (list or numpy.ndarray): The behavior descriptor.
        """
        distance = self.compute_distance_to_nearest_peak(genome)
        behavior = [distance]  # Behavior can be a list with the distance
        return behavior

    def crossover(self, parent1, parent2):
        if self.inbred_threshold is not None:
            # Calculate kinship coefficient
            f = self.kinship_coefficient(parent1, parent2)
            if f > self.max_kinship:
                # Prevent mating
                return None, None
            
            # distance = self.genetic_distance(parent1, parent2)
            # if distance < self.inbred_threshold: # the bigger the allowed distance, the farther apart the parents need to be
            #     return None, None

        # One-point crossover
        crossover_point = np.random.randint(1, self.dimensions)
        child_genes1 = np.concatenate([parent1.genes[:crossover_point], parent2.genes[crossover_point:]])
        child_genes2 = np.concatenate([parent2.genes[:crossover_point], parent1.genes[crossover_point:]])


        # Create offspring with updated ancestry
        child1 = Individual(
            genes=child_genes1,
            parents=[parent1, parent2],
            generation=max(parent1.generation, parent2.generation) + 1
        )
        child2 = Individual(
            genes=child_genes2,
            parents=[parent1, parent2],
            generation=max(parent1.generation, parent2.generation) + 1
        )

        return child1, child2
    
        # No Pedigree implementation
        # ancestors1 = parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id})
        # ancestors2 = copy.deepcopy(ancestors1)

        # child1 = Individual(child_genes1, ancestors=ancestors1, generation=parent1.generation + 1)
        # child2 = Individual(child_genes2, ancestors=ancestors2, generation=parent1.generation + 1)

        # return child1, child2

    def mutate(self, individual):
        for i in range(self.dimensions):
            if np.random.rand() < self.mutation_rate:
                individual.genes[i] = 1 - individual.genes[i]  # Flip bit

    def measure_diversity(self):
        """
            Definition
            -----------
                Heterozygosity: Measures the probability that two alleles at a locus are different.
                
                Results Intepretation of diversity
                    0: No diversity at a locus (all individuals have the same allele).
                    0.5: Maximum diversity at a locus (alleles are equally frequent).
                    
                Based of the Hardy-Weinberg Equilibrium (HWE):
                    Describes a state in which allele and genotype frequencies in a population remain
                    constant from generation to generation in the absence of evolutionary influences.
                
            Returns
            --------
                Allelic Diversity (Da): The average proportion of different alleles at each gene locus.
             
        """
        # Initi Allele frequency
        total_loci = self.dimensions # Genome Length
        allele_frequencies = np.zeros((total_loci, 2))  # For binary genes (0 and 1)
        
        # Count the occurrences of alleles 0 and 1 at each locus across the population.
        for ind in self.population:
            for locus, allele in enumerate(ind.genes):
                allele_frequencies[locus, allele] += 1
        
        # Calculate heterozygosity at each locus
        heterozygosities = []
        for locus in range(total_loci):
            freq0 = allele_frequencies[locus, 0] / self.pop_size # Freq of allele 0
            freq1 = allele_frequencies[locus, 1] / self.pop_size
            
            # General Formula will be 1 - sum over all freq_i^2 but in this case is binary
            # freq0^2 + freq1^1 is the probability of selection 2 identical alleles
            heterozygosity = 1 - (freq0**2 + freq1**2)  # freq0 is frequency of genotype 00
            heterozygosities.append(heterozygosity)
        
        diversity = np.mean(heterozygosities)
        return diversity

    def run(self, collapse_threshold=0.2, collapse_fraction=0.1):
        
        # Initialize the population
        self.initialize_population()
        
        # Data structures to store lineage information
        self.lineage_data = []
        global_optimum_fitness_list = []

        for gen in range(self.generations):
            self.current_generation = gen + 1
            
            #  # Log ancestry at generation 5
            # if self.current_generation == 3:
            #     self.log_ancestry(3)
            #     exit()
            
            # Add-on needed only for MovingPeaksLandscape
            if self.args.bench_name == 'MovingPeaksLandscape':
                if gen % self.landscape.shift_interval == 0 and gen != 0:
                    self.landscape.shift_peaks()
            
            # Get the fitness of the current population
            # Calculate fitness and novelty
            if self.args.bench_name == 'MovingPeaksLandscape':
                # Calculate fitness and novelty
                self.calculate_fitness_and_novelty()
            
            else:
                self.calculate_fitness() # TODO disable for other methods
            
                best_fitness = max(self.population, key=lambda ind: ind.fitness).fitness
                self.best_fitness_list.append(best_fitness)
                
            # Record best fitness
            best_individual = max(self.population, key=lambda ind: ind.total_fitness)
            self.best_fitness_list.append(best_individual.total_fitness)
            
            # Record global optimum fitness
            if self.args.bench_name == 'MovingPeaksLandscape':
                global_optimum_fitness = self.landscape.get_current_global_optimum_fitness()
                global_optimum_fitness_list.append(global_optimum_fitness)

            # Calculate diversity by the heterozygosity at each locus
            diversity = self.measure_diversity()
            self.diversity_list.append(diversity)
            
            # Collect lineage data for visualization
            for ind in self.population:
                self.lineage_data.append({
                    'id': ind.id,
                    'ancestors': ind.ancestors,
                    'generation': gen,
                    'fitness': ind.total_fitness
                })
                
            # # Check for population collapse
            # if self.args.bench_name == 'MovingPeaksLandscape':
            #     # Check for population collapse condition
            #     if diversity < collapse_threshold:
            #         num_to_replace = int(self.pop_size * collapse_fraction)
            #         # Replace the least fit individuals
            #         self.population.sort(key=lambda ind: ind.total_fitness)
            #         for _ in range(num_to_replace):
            #             genes = np.random.randint(2, size=self.dimensions)
            #             new_individual = Individual(genes=genes, generation=gen+1)
            #             self.population.pop(0)  # Remove the least fit
            #             self.population.append(new_individual)
                    
            #         # Re-calculate fitness as new individuals have been added
            #         self.calculate_fitness_and_novelty()
            #         # record the generations where a population collapse (or diversity restoration) event occurs.
            #         # TODO: "Population collapse" refers to genetic diversity loss rather than population size reduction
            #         # So, this is technically a diversity restoration mechanism. 
            #         # Genetic Collapse: Occurs when diversity is significantly reduced, making the population vulnerable to being trapped in local optima
            #         self.collapse_events.append(gen + 1)
            #         print(f"Generation {gen + 1}: Diversity {diversity:.4f} below threshold. Replaced {num_to_replace} individuals.")

            # Tournament selection.
            selected = self.tournament_selection(self.tournament_size)
        
            # # Selection            
            # next_population = []
            # failed_parents = set()  # Use a set to avoid duplicates
            # i = 0
            # while i < len(selected):
            #     parent1 = selected[i]
            #     parent2 = selected[(i + 1) % len(selected)]
            #     offspring = self.crossover(parent1, parent2)

            #     if offspring[0] is not None and offspring[1] is not None:
            #         # Successful mating
            #         self.mutate(offspring[0])
            #         self.mutate(offspring[1])
            #         next_population.extend(offspring)
            #     else:
            #         # Mating failed due to inbreeding prevention
            #         # Collect parents for potential inclusion in next generation
            #         failed_parents.add(parent1)
            #         failed_parents.add(parent2)
            #     i += 2
            
            # Selection
            individuals_needing_mates = selected.copy()
            next_population = []
            failed_parents = set()

            # Shuffle the list to ensure randomness
            random.shuffle(individuals_needing_mates)

            while individuals_needing_mates:
                parent1 = individuals_needing_mates.pop(0)  # Take the first individual needing a mate
                mate_found = False  # Flag to track if a suitable mate is found

                # Create a copy to iterate over potential mates
                potential_mates = individuals_needing_mates.copy()
                random.shuffle(potential_mates)  # Shuffle potential mates for randomness

                # Try to find a mate for parent1
                for potential_mate in potential_mates:
                    if parent1 == potential_mate:
                        continue  # Skip if same individual

                    offspring = self.crossover(parent1, potential_mate)
                    if offspring[0] is not None and offspring[1] is not None:
                        # Successful mating
                        self.mutate(offspring[0])
                        self.mutate(offspring[1])
                        next_population.extend(offspring)

                        # Remove the mate from the list, as they have mated
                        individuals_needing_mates.remove(potential_mate)
                        mate_found = True
                        break  # Exit the loop as we have found a mate
                    
                if not mate_found:
                    # Parent1 could not find a suitable mate
                    failed_parents.add(parent1)

            # Convert the set to a list for sorting
            failed_parents = list(failed_parents)

            # Determine the number of elites to retain from failed parents
            elite_size = int(0.20 * self.pop_size)  # Adjust the percentage as needed

            # Sort failed parents by total_fitness in descending order
            failed_parents.sort(key=lambda ind: ind.total_fitness, reverse=True)

            # Select the top individuals
            elites_from_failed_parents = failed_parents[:elite_size]

            # Add elites to the next generation
            next_population.extend(elites_from_failed_parents)
            
            # Update the population
            self.population = next_population
            
            # After updating the population
            min_population_size = 20
            if len(self.population) < min_population_size:
                print(f"Generation {gen + 1}: Population size {len(self.population)} below minimum threshold {min_population_size}. Introducing new individuals.")
                num_to_add = min_population_size - len(self.population)
                while len(self.population) < self.initial_pop_size:
                    genes = np.random.randint(2, size=self.dimensions)
                    new_individual = Individual(genes=genes, generation=gen+1)
                    self.population.append(new_individual)
            
            self.pop_size = len(self.population)
            print(f"New pop size: {self.pop_size}")

        return self.best_fitness_list, self.diversity_list, global_optimum_fitness_list, self.collapse_events