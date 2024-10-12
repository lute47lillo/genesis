import numpy as np
import copy
import os 
import matplotlib.pyplot as plt

# Your Landscape Class
class Landscape:
    """ N-K Fitness Landscape """
    def __init__(self, n=10, k=2):
        self.n = n  # The number of genes (loci) in the genome. Genome length
        self.k = k  # Number of other loci interacting with each gene. For each gene 2^(k+1) possible combinations of gene states that affect its fitness contribution
        self.gene_contribution_weight_matrix = np.random.rand(n, 2**(k+1)) 
        
    def get_contributing_gene_values(self, genome, gene_num):
        contributing_gene_values = ""
        for i in range(self.k+1):
            contributing_gene_values += str(genome[(gene_num + i) % self.n])
        return contributing_gene_values
    
    def get_fitness(self, genome):
        """
            Higher values of K increases the epistatic interactions (dependencies among genes). 
            Therefore, making the landscape more rugged.
            
            Maximum that one gene can contribute is 1. 
            Fitness of a genome is the average over the sum of the fitness contirbution of each gene.
        """
        gene_values = np.zeros(self.n)
        for gene_num in range(len(genome)):
            contributing_gene_values = self.get_contributing_gene_values(genome, gene_num)
            index = int(contributing_gene_values, 2)
            gene_values[gene_num] = self.gene_contribution_weight_matrix[gene_num, index]
        return np.mean(gene_values)

# Individual and GA Classes
class Individual:
    def __init__(self, genes=None, id=None, ancestors=None):
        self.genes = genes if genes is not None else []
        self.fitness = None
        self.id = id if id is not None else np.random.randint(1e9)
        self.ancestors = ancestors if ancestors is not None else set()

class GeneticAlgorithm:
    def __init__(self, pop_size, dimensions, bounds, generations, mutation_rate, allowed_distance=None):
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.allowed_distance = allowed_distance
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            genes = np.random.randint(2, size=self.dimensions)
            individual = Individual(genes)
            self.population.append(individual)

    def calculate_fitness(self):
        for individual in self.population:
            individual.fitness = landscape.get_fitness(individual.genes)

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
        if self.allowed_distance is not None:
            distance = self.genetic_distance(parent1, parent2)
            if distance < self.allowed_distance:
                return None, None

        # One-point crossover
        crossover_point = np.random.randint(1, self.dimensions)
        child_genes1 = np.concatenate([parent1.genes[:crossover_point], parent2.genes[crossover_point:]])
        child_genes2 = np.concatenate([parent2.genes[:crossover_point], parent1.genes[crossover_point:]])

        ancestors1 = parent1.ancestors.union(parent2.ancestors, {parent1.id, parent2.id})
        ancestors2 = copy.deepcopy(ancestors1)

        child1 = Individual(child_genes1, ancestors=ancestors1)
        child2 = Individual(child_genes2, ancestors=ancestors2)

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

        for gen in range(self.generations):
            
            # Get the fitness of the current population
            self.calculate_fitness()
            best_fitness = max(self.population, key=lambda ind: ind.fitness).fitness
            self.best_fitness_list.append(best_fitness)

            # Calculate diversity by the heterozygosity at each locus
            diversity = self.measure_diversity()
            self.diversity_list.append(diversity)

            # Tournament selection. TODO: move down 
            selected = self.tournament_selection()

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
                    individual = Individual(genes)
                    next_population.append(individual)
                    if len(next_population) < self.pop_size:
                        genes = np.random.randint(2, size=self.dimensions)
                        individual = Individual(genes)
                        next_population.append(individual)
                i += 2

            self.population = next_population[:self.pop_size]

        return self.best_fitness_list, self.diversity_list

# Parameters
N = 100  # Genome length
K = 14   # Number of interactions
pop_size = 50
generations = 200
mutation_rate = 0.01
allowed_distance = 5  # Adjust as needed

# Create an instance of the Landscape
landscape = Landscape(n=N, k=K)

def run_inbreeding_pop_sizes(pop_sizes, dimensions, generations, mutation_rate, allowed_distance):
    
    # Run GA without Inbreeding Prevention
    results_inbreeding = {}
    print("Running GA with Inbreeding Mating...")
    for pop_size in pop_sizes:
        # Run GA with Inbreeding Prevention
        ga = GeneticAlgorithm(
            pop_size=pop_size,
            dimensions=dimensions,
            bounds=None,
            generations=generations,
            mutation_rate=mutation_rate,
            allowed_distance=None
        )
        best_fitness_list, diversity_list = ga.run()
        results_inbreeding[pop_size] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list
        }
        print(f"Population Size {pop_size}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results_inbreeding

def run_no_inbreeding_pop_sizes(pop_sizes, dimensions, generations, mutation_rate, allowed_distance):
    
    # Run GA with Inbreeding Prevention
    results_no_inbreeding = {}
    print("\nRunning GA with NO Inbreeding mating...")
    for pop_size in pop_sizes:
        # Run GA with Inbreeding Prevention
        ga = GeneticAlgorithm(
            pop_size=pop_size,
            dimensions=dimensions,
            bounds=None,
            generations=generations,
            mutation_rate=mutation_rate,
            allowed_distance=allowed_distance
        )
        best_fitness_list, diversity_list = ga.run()
        results_no_inbreeding[pop_size] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list
        }
        print(f"Population Size {pop_size}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results_no_inbreeding


# up to 100 with 5 distance allowed and 0.01 and genome length 100, it seems to be favoring not inbreeding
pop_sizes = [150, 200, 250, 300]
results_inbreeding = run_inbreeding_pop_sizes(pop_sizes, N, generations, mutation_rate, allowed_distance)
results_no_inbreeding = run_no_inbreeding_pop_sizes(pop_sizes, N, generations, mutation_rate, allowed_distance)


# Plot Best Fitness Comparison
plt.figure(figsize=(16, 9))
for element in pop_sizes:
    plt.plot(
        results_inbreeding[element]['best_fitness'],
        label=f'Inbreeding, MutRates {element}'
    )
    plt.plot(
        results_no_inbreeding[element]['best_fitness'],
        label=f'No Inbreeding, MutRates {element}',
        linestyle='--'
    )
plt.title('Best Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.legend()
plt.grid(True)
plt.savefig(f"{os.getcwd()}/BigPOPSIZEs_test_nk_fit.png")
plt.close()

# Plot Diversity Comparison
plt.figure(figsize=(16, 9))
for element in pop_sizes:
    plt.plot(
        results_inbreeding[element]['diversity'],
        label=f'Inbreeding, MutRates {element}'
    )
    plt.plot(
        results_no_inbreeding[element]['diversity'],
        label=f'No Inbreeding, MutRates {element}',
        linestyle='--'
    )
plt.title('Genetic Diversity over Generations')
plt.xlabel('Generation')
plt.ylabel('Diversity')
plt.legend()
plt.grid(True)
plt.savefig(f"{os.getcwd()}/BigPOPSIZEs_test_nk_div.png")
plt.close()


# def run_inbreeding_mutation_rates(pop_size, dimensions, generations, mutation_rates, allowed_distance):
    
#     # Run GA without Inbreeding Prevention
#     results_inbreeding = {}
#     print("Running GA with Inbreeding Mating...")
#     for rate in mutation_rates:
#         # Run GA with Inbreeding Prevention
#         ga = GeneticAlgorithm(
#             pop_size=pop_size,
#             dimensions=dimensions,
#             bounds=None,
#             generations=generations,
#             mutation_rate=rate,
#             allowed_distance=None
#         )
#         best_fitness_list, diversity_list = ga.run()
#         results_inbreeding[rate] = {
#             'best_fitness': best_fitness_list,
#             'diversity': diversity_list
#         }
#         print(f"Mutation Rate {rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
#     return results_inbreeding

# def run_no_inbreeding_mutation_rates(pop_size, dimensions, generations, mutation_rates, allowed_distance):
    
#     # Run GA with Inbreeding Prevention
#     results_no_inbreeding = {}
#     print("\nRunning GA with NO Inbreeding mating...")
#     for rate in mutation_rates:
#         # Run GA with Inbreeding Prevention
#         ga = GeneticAlgorithm(
#             pop_size=pop_size,
#             dimensions=dimensions,
#             bounds=None,
#             generations=generations,
#             mutation_rate=rate,
#             allowed_distance=allowed_distance
#         )
#         best_fitness_list, diversity_list = ga.run()
#         results_no_inbreeding[rate] = {
#             'best_fitness': best_fitness_list,
#             'diversity': diversity_list
#         }
#         print(f"Mutation Rate {rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
#     return results_no_inbreeding

# mutation_rates = [0.01, 0.1, 0.2, 0.3]
# results_inbreeding = run_inbreeding_mutation_rates(pop_size, N, generations, mutation_rates, allowed_distance)
# results_no_inbreeding = run_no_inbreeding_mutation_rates(pop_size, N, generations, mutation_rates, allowed_distance)


# # Plot Best Fitness Comparison
# plt.figure(figsize=(16, 9))
# for element in mutation_rates:
#     plt.plot(
#         results_inbreeding[element]['best_fitness'],
#         label=f'Inbreeding, MutRates {element}'
#     )
#     plt.plot(
#         results_no_inbreeding[element]['best_fitness'],
#         label=f'No Inbreeding, MutRates {element}',
#         linestyle='--'
#     )
# plt.title('Best Fitness over Generations')
# plt.xlabel('Generation')
# plt.ylabel('Best Fitness')
# plt.legend()
# plt.grid(True)
# plt.savefig(f"{os.getcwd()}/MutRates_test_nk_fit.png")
# plt.close()

# # Plot Diversity Comparison
# plt.figure(figsize=(16, 9))
# for element in mutation_rates:
#     plt.plot(
#         results_inbreeding[element]['diversity'],
#         label=f'Inbreeding, MutRates {element}'
#     )
#     plt.plot(
#         results_no_inbreeding[element]['diversity'],
#         label=f'No Inbreeding, MutRates {element}',
#         linestyle='--'
#     )
# plt.title('Genetic Diversity over Generations')
# plt.xlabel('Generation')
# plt.ylabel('Diversity')
# plt.legend()
# plt.grid(True)
# plt.savefig(f"{os.getcwd()}/MutRates_test_nk_div.png")
# plt.close()