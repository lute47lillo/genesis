"""

    TODO: Need to compare Novelty against fitness methods
"""


import numpy as np
import copy
from scipy.spatial import distance

class Individual:
    def __init__(self, genes, id=None, ancestors=None, generation=0):
        """
            Definition
            -----------
                - genes: The gene vector representing the solution.
                - id: A unique identifier for each individual.
                - ancestors: A set containing the IDs of all ancestors.
        """
        # self.genes = genes
        self.genes = genes if genes is not None else []
        self.fitness = None
        self.novelty = None
        self.total_fitness = None
        self.behavior = None
        self.id = id if id is not None else np.random.randint(1e9)
        self.ancestors = ancestors if ancestors is not None else set()
        self.generation = generation # Track the generation of the individual  

class GeneticAlgorithm:
    def __init__(self, landscape, pop_size, dimensions, bounds, generations, mutation_rate, allowed_distance=None):
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.allowed_distance = allowed_distance  # None means no inbreeding prevention
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []
        self.landscape = landscape # Could be NK landscape or any optimization function

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            genes = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
            individual = Individual(genes)
            self.population.append(individual)

    def calculate_fitness(self):
        for individual in self.population:
            individual.fitness = self.landscape(individual.genes)

    def tournament_selection(self, k=10):
        selected = []
        for _ in range(self.pop_size):
            participants = np.random.choice(self.population, k)
            winner = min(participants, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected

    def genetic_distance(self, ind1, ind2):
        """
            Definitions
            ------------
                Quantify genetic similarity between individuals. Euclidean Distance between 2 points in space.
        """
        return np.linalg.norm(ind1.genes - ind2.genes)

    def crossover(self, parent1, parent2):
        if self.allowed_distance is not None:
            distance = self.genetic_distance(parent1, parent2)
            if distance < self.allowed_distance: # Minimum genetic distance (threshold) between individuals to prevent inbreeding
                return None, None

        mask = np.random.rand(self.dimensions) < 0.5
        child_genes1 = np.where(mask, parent1.genes, parent2.genes)
        child_genes2 = np.where(mask, parent2.genes, parent1.genes)

        # offspring inherit the union of their ancestors' IDs plus the parents' own IDs
        ancestors1 = parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id})
        ancestors2 = copy.deepcopy(ancestors1)

        child1 = Individual(child_genes1, ancestors=ancestors1, generation=parent1.generation + 1)
        child2 = Individual(child_genes2, ancestors=ancestors2, generation=parent1.generation + 1)

        return child1, child2

    def mutate(self, individual):
        for i in range(self.dimensions):
            if np.random.rand() < self.mutation_rate:
                individual.genes[i] = np.random.uniform(self.bounds[0], self.bounds[1])

    def measure_diversity(self):
        """
            Definition
            -----------
                Measure diversity in the population by counting the total number of unique ancestor IDs in the current population.
                The, normalize by dividing by the product of population size and generations.
                Finally, the result is a measure of genetic diversity based on the variety of lineages present.
        """
        ancestor_ids = set()
        for ind in self.population:
            ancestor_ids.update(ind.ancestors)
            ancestor_ids.add(ind.id)
        diversity = len(ancestor_ids) / (self.pop_size * self.generations)
        return diversity

    def run(self):
        self.initialize_population()

        for gen in range(self.generations):
            self.calculate_fitness()
            best_fitness = min(self.population, key=lambda ind: ind.fitness).fitness
            self.best_fitness_list.append(best_fitness)

            diversity = self.measure_diversity()
            self.diversity_list.append(diversity)

            selected = self.tournament_selection()

            next_population = []
            i = 0
            while len(next_population) < self.pop_size:
                parent1 = selected[i % len(selected)]
                parent2 = selected[(i+1) % len(selected)]
                offspring = self.crossover(parent1, parent2)

                if offspring[0] is not None and offspring[1] is not None:
                    self.mutate(offspring[0])
                    self.mutate(offspring[1])
                    next_population.extend(offspring)
                else:
                    if self.allowed_distance is None:
                        next_population.append(copy.deepcopy(parent1))
                        next_population.append(copy.deepcopy(parent2))
                    else:
                        genes = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
                        individual = Individual(genes, generation=gen+1)
                        next_population.append(individual)
                        if len(next_population) < self.pop_size:
                            genes = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
                            individual = Individual(genes, generation=gen+1)
                            next_population.append(individual)
                i += 2

            self.population = next_population[:self.pop_size]

        return self.best_fitness_list, self.diversity_list

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
        self.landscape = landscape
        
        # Initialize the Novelty Archive
        # TODO: Create hyperparameters for the novelty
        self.novelty_archive = NoveltyArchive(threshold=0.1, k=5, max_size=self.pop_size)

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
            # For example, combine fitness and novelty
            individual.total_fitness = individual.fitness + novelty
            # Optionally, add to the archive
            self.novelty_archive.add(individual.behavior)

    def tournament_selection(self, k=3):
        selected = []
        for i in range(self.pop_size):
            participants = np.random.choice(self.population, k)
            # for idx, part in enumerate(participants):
                # print(f"Round ({i}). Participant ({idx}) total fitness: {part.total_fitness}")
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
            distance = self.genetic_distance(parent1, parent2)
            if distance < self.inbred_threshold: # the bigger the allowed distance, the farther apart the parents need to be
                return None, None

        # One-point crossover
        crossover_point = np.random.randint(1, self.dimensions)
        child_genes1 = np.concatenate([parent1.genes[:crossover_point], parent2.genes[crossover_point:]])
        child_genes2 = np.concatenate([parent2.genes[:crossover_point], parent1.genes[crossover_point:]])

        ancestors1 = parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id})
        ancestors2 = copy.deepcopy(ancestors1)

        child1 = Individual(child_genes1, ancestors=ancestors1, generation=parent1.generation + 1)
        child2 = Individual(child_genes2, ancestors=ancestors2, generation=parent1.generation + 1)

        return child1, child2

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
                
            # Check for population collapse
            if self.args.bench_name == 'MovingPeaksLandscape':
                # Check for population collapse condition
                if diversity < collapse_threshold:
                    num_to_replace = int(self.pop_size * collapse_fraction)
                    # Replace the least fit individuals
                    self.population.sort(key=lambda ind: ind.total_fitness)
                    for _ in range(num_to_replace):
                        genes = np.random.randint(2, size=self.dimensions)
                        new_individual = Individual(genes=genes, generation=gen+1)
                        self.population.pop(0)  # Remove the least fit
                        self.population.append(new_individual)
                    
                    # Re-calculate fitness as new individuals have been added
                    self.calculate_fitness_and_novelty()
                    # record the generations where a population collapse (or diversity restoration) event occurs.
                    # TODO: "Population collapse" refers to genetic diversity loss rather than population size reduction
                    # So, this is technically a diversity restoration mechanism. 
                    # Genetic Collapse: Occurs when diversity is significantly reduced, making the population vulnerable to being trapped in local optima
                    self.collapse_events.append(gen + 1)
                    print(f"Generation {gen + 1}: Diversity {diversity:.4f} below threshold. Replaced {num_to_replace} individuals.")

            # Tournament selection. TODO: move down 
            selected = self.tournament_selection(self.tournament_size)

            next_population = []
            i = 0
            while len(next_population) < self.pop_size:
                
                # crossover
                parent1 = selected[i % len(selected)]
                parent2 = selected[(i+1) % len(selected)]
                offspring = self.crossover(parent1, parent2)

                # Mutate
                if offspring[0] is not None and offspring[1] is not None:
                    self.mutate(offspring[0])
                    self.mutate(offspring[1])
                    next_population.extend(offspring)
                else:
                    # Introduce new random individuals to maintain population size
                    genes = np.random.randint(2, size=self.dimensions)
                    individual = Individual(genes, generation=gen+1)
                    next_population.append(individual)
                    if len(next_population) < self.pop_size:
                        genes = np.random.randint(2, size=self.dimensions)
                        individual = Individual(genes, generation=gen+1)
                        next_population.append(individual)
                i += 2

            self.population = next_population[:self.pop_size]

        return self.best_fitness_list, self.diversity_list, global_optimum_fitness_list, self.collapse_events