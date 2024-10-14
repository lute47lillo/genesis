import numpy as np
import copy

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
        self.landscape = landscape

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            genes = np.random.randint(2, size=self.dimensions)
            individual = Individual(genes)
            self.population.append(individual)

    def calculate_fitness(self):
        for individual in self.population:
            individual.fitness = self.landscape.get_fitness(individual.genes)

    def tournament_selection(self, k=3):
        selected = []
        for _ in range(self.pop_size):
            participants = np.random.choice(self.population, k)
            winner = max(participants, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected

    def genetic_distance(self, ind1, ind2):
        return np.sum(ind1.genes != ind2.genes)  # Hamming distance

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
                
            Returns
            --------
                Allelic Diversity (Da): The average proportion of different alleles at each gene locus.
             
        """
        # Initi Allele frequency
        total_loci = self.dimensions
        allele_frequencies = np.zeros((total_loci, 2))  # For binary genes (0 and 1)
        
        # Count the occurrences of alleles 0 and 1 at each locus across the population.
        for ind in self.population:
            for locus, allele in enumerate(ind.genes):
                allele_frequencies[locus, allele] += 1
        
        # Calculate heterozygosity at each locus
        heterozygosities = []
        for locus in range(total_loci):
            freq0 = allele_frequencies[locus, 0] / self.pop_size
            freq1 = allele_frequencies[locus, 1] / self.pop_size
            heterozygosity = 1 - (freq0**2 + freq1**2)
            heterozygosities.append(heterozygosity)
        
        diversity = np.mean(heterozygosities)
        return diversity

    def run(self):
        
        # Initialize the population
        self.initialize_population()
        
        # Data structures to store lineage information
        self.lineage_data = []

        for gen in range(self.generations):
            
            # Add-on needed only for MovingPeaksLandscape
            if self.args.bench_name == 'MovingPeaksLandscape':
                if gen % self.landscape.shift_interval == 0 and gen != 0:
                    self.landscape.shift_peaks()
            
            # Get the fitness of the current population
            self.calculate_fitness()
            best_fitness = max(self.population, key=lambda ind: ind.fitness).fitness
            self.best_fitness_list.append(best_fitness)

            # Calculate diversity by the heterozygosity at each locus
            diversity = self.measure_diversity()
            self.diversity_list.append(diversity)
            
            # Collect lineage data for visualization
            for ind in self.population:
                self.lineage_data.append({
                    'id': ind.id,
                    'ancestors': ind.ancestors,
                    'generation': gen,
                    'fitness': ind.fitness
                })

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

        return self.best_fitness_list, self.diversity_list