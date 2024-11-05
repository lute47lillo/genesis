"""

"""


import numpy as np
import copy
import plotting as plot
import util
import experiments as exp

class Individual:
    def __init__(self, genes, ancestors=None, id=None, parents=None, generation=0):
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
        self.ancestors = ancestors if ancestors is not None else set() # IDs of all ancestors
        self.generation = generation
                
class GeneticAlgorithm:
    def __init__(self, args, landscape, pop_size, mutation_rate, inbred_threshold=None):
        self.args = args
        self.pop_size = pop_size
        self.dimensions = args.dimensions
        self.bounds = args.bounds
        self.generations = args.generations
        self.mutation_rate = mutation_rate
        self.tournament_size = args.tournament_size
        self.inbred_threshold = inbred_threshold  # None means no inbreeding prevention
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []
        self.landscape = landscape # optimization function

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
        if self.inbred_threshold is not None:
            distance = self.genetic_distance(parent1, parent2)
            if distance < self.inbred_threshold: # Minimum genetic distance (threshold) between individuals to prevent inbreeding
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
    
    def measure_variance_diversity(self):
        """Measure diversity based on variance of gene values."""
        gene_matrix = np.array([ind.genes for ind in self.population])
        variances = np.var(gene_matrix, axis=0)
        diversity = np.mean(variances)
        return diversity
    
    def measure_allelic_diversity(self):
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

    def run(self):
        self.initialize_population()

        for gen in range(self.generations):
            self.calculate_fitness()
            best_fitness = min(self.population, key=lambda ind: ind.fitness).fitness
            self.best_fitness_list.append(best_fitness)

            # diversity = self.measure_diversity()
            diversity = self.measure_variance_diversity()
            # diversity = self.measure_allelic_diversity()
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
                    if self.inbred_threshold is None:
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
    
if __name__ == "__main__":
    
    # Get args
    args = util.set_args()
    
    # Select landscape based on the function optimzation chose, set bounds as well
    landscape = util.select_benchmark(args)
    
    # ------------------------- Population Sizes Experiments --------------------------- #
    
    print(f"\nPopulation Size Experiments")
    
    pop_sizes = [200, 300, 400]
    # pop_sizes = [20, 40]
    
    print("\nRunning GA with NO Inbreeding mating...")
    results_no_inbreeding = exp.run_multiple_pop_sizes(args, pop_sizes, landscape, args.inbred_threshold)
    
    print("\nRunning GA with Inbreeding Mating...")
    results_inbreeding = exp.run_multiple_pop_sizes(args, pop_sizes, landscape, None)

    # Create plot name
    args.config_plot = f"Variance_PopSize:{pop_sizes}_InThres:{args.inbred_threshold}_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}" 
    plot.plot_opt_fn_parameters(args, "PopSize", pop_sizes, results_inbreeding, results_no_inbreeding)
    
    # ------------------------- Mutation Rate Experiments --------------------------- #
    
    print(f"\nMutation Rate Experiments")
    
    # mutation_rates = [0.0005, 0.005, 0.05, 0.01, 0.1]
    mutation_rates = [0.08, 0.1, 0.2]
    
    print("\nRunning GA with NO Inbreeding mating...")
    results_no_inbreeding = exp.run_inbreeding_mutation_rates(args, mutation_rates, landscape, args.inbred_threshold)
    
    print("\nRunning GA with Inbreeding mating...")
    results_inbreeding = exp.run_inbreeding_mutation_rates(args, mutation_rates, landscape, None)
    
    args.config_plot = f"Variance__PopSize:{args.pop_size}_InThres:{args.inbred_threshold}_Mrates:{mutation_rates}_Gens:{args.generations}_TourSize:{args.tournament_size}" 
    plot.plot_opt_fn_parameters(args, "MutRates", mutation_rates, results_inbreeding, results_no_inbreeding)
