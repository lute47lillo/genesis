"""

    
"""


import numpy as np
from scipy.spatial import distance

import util as util
import copy
import plotting as plot
import benchmark_factory as bf
import experiments as exp

class Individual:
    
    # TODO: Landscape could be initialize since terminal line
    def __init__(self, args, genes, curr_pop_behave, novelty_archive, landscape=None, id=None, parents=None, generation=0):
        """
            Definition
            -----------
                - genes: The gene vector representing the solution.
                - id: A unique identifier for each individual.
                - parents=None: A set containing the IDs of all ancestors.
        """
        self.args = args
        self.genes = genes if genes is not None else []
        self.fitness = None
        self.novelty = None
        self.total_fitness = None
        self.behavior = None
        self.id = id if id is not None else np.random.randint(1e9)
        self.parents = parents if parents is not None else []
        self.ancestors = set() # IDs of all ancestors
        self.generation = generation
        self.landscape = landscape
        self.novelty_max = 2  # Max behavior distance (1 for peak index + 1 for normalized distance)

        # Initial generation individual
        if not self.parents:
            self.ancestors.add(self.id)
            
        # Update ancestors with depth limitation
        max_depth = 10  # Set the desired ancestry depth. Edit for computational issues
        for parent in self.parents:
            if parent.generation >= self.generation - max_depth:
                self.ancestors.update(parent.ancestors)
                self.ancestors.add(parent.id)
        
        # Set the initial fitness and novelty of the individual wrt novelty archive and the population behaviors
        self.get_novel_fit_indiv(curr_pop_behave, novelty_archive)

                
    def get_novel_fit_indiv(self, population_behaviors, novelty_archive):
        
        # Set importance values. Default to 1.0
        alpha = self.args.fit_weight        # Weight for fitness
        beta = self.args.novelty_weight     # Weight for novelty
        
        fitness_max = max(peak.height for peak in self.landscape.peaks)  # Max peak height

        # Compute scaling factor
        scaling_factor = (fitness_max - 0) / (self.novelty_max - 0) # 0s represent minimum value
        
        # Get landscape-based fitness
        self.fitness = self.landscape.get_fitness(self.genes)
        
        # Get novelty
        self.behavior = util.extract_behavior(self.genes, self.landscape)
        self.novelty = novelty_archive.compute_novelty(self.behavior, population_behaviors)

        # Scale novelty to fitness range
        novelty_scaled = self.novelty * scaling_factor

        # Combine fitness and scaled novelty
        self.total_fitness = alpha * self.fitness + beta * novelty_scaled
        
        # Add to the archive
        novelty_archive.add(self.behavior)

# -------------------------------------------------------------------------------------------------------- #

class NoveltyArchive:
    def __init__(self, threshold=0.1, k=5, max_size=500):
        """
            Definition:
            ------------
                Initialize the Novelty Archive.

            Parameters:
            ------------
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
            Definition
            ------------
                Compute the novelty of a given behavior.

            Parameters:
            ------------
                - behavior (list or numpy.ndarray): Behavior descriptor of the individual.
                - population_behaviors (list): List of behavior descriptors of the current population.

            Returns:
            ------------
                - novelty (float): Calculated novelty score.
        """
        # Combine archive and current population behaviors
        all_behaviors = self.archive + population_behaviors

        if len(all_behaviors) == 0:
            return float('inf')  # Maximum novelty for the first individual

        # Compute distances to all behaviors
        distances = [util.behavior_distance(behavior, b) for b in all_behaviors]
        
        # Exclude distance to self if present
        distances = [d for d in distances if d != 0]

        # Check for behaviors to compare in the archive
        effective_k = min(self.k, len(distances))
        if effective_k == 0: # No more behaviors to compare to.
            return 0.0  # Or some default novelty value
        
        nearest_distances = np.partition(distances, effective_k - 1)[:effective_k]
        novelty = np.mean(nearest_distances)
        
        return novelty

    def add(self, behavior):
        """
            Definition:
            ------------
                Add a new behavior to the archive if it is novel enough.

            Parameters:
            ------------
                - behavior (list or numpy.ndarray): Behavior descriptor of the individual.
        """
        # Check archive is not empty
        if len(self.archive) == 0:
            self.archive.append(behavior)
            return

        # Compute distance to all behaviors in the archive
        distances = [util.behavior_distance(behavior, b) for b in self.archive]
        min_distance = np.min(distances)

        if min_distance > self.threshold:
            if len(self.archive) >= self.max_size:
                
                # Remove the oldest entry to maintain the archive size
                self.archive.pop(0)
            self.archive.append(behavior)

class GeneticAlgorithmRugged:
    
    def __init__(self, args, landscape, bounds, max_kinship=None):
        
        # Hyperparameters
        self.args = args
        self.pop_size = args.pop_size
        self.dimensions = args.dimensions
        self.bounds = bounds
        self.generations = args.generations
        self.mutation_rate = args.mutation_rate
        self.tournament_size = args.tournament_size
        self.max_kinship = max_kinship
        self.initial_pop_size = self.pop_size
        self.max_kinship = args.max_kinship  # Threshold for kinship coefficient. 0.125 is first cousings. 0.0625 is second cousins

        # Metrics and Lists
        self.population = []
        self.best_fitness_list = []
        self.population_behaviors = []
        self.diversity_list = []
        self.collapse_events = []
        self.global_optimum_fitness_list = []
        
        # Others
        self.landscape = landscape
        self.args.landscape = landscape
        
        # Initialize the Novelty Archive 
        self.novelty_archive = NoveltyArchive(threshold=args.archive_threshold, k=args.archive_nn, max_size=self.pop_size)
        self.novelty_max = 2 # Maximum value of Novelty. Hardcoded value for scaling
        
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
            individual = Individual(self.args, genes, self.population_behaviors, self.novelty_archive, self.landscape)
            self.population.append(individual)

    def calculate_fitness(self):
        for individual in self.population:
            individual.fitness = self.landscape.get_fitness(individual.genes)

    def calculate_fitness_and_novelty(self):
        """
            Definition
            -----------
                Calculate combined fitness of Peak distance and novelty for a given population.         
        """
        # Set importance values. Default to 1.0
        alpha = self.args.fit_weight        # Weight for fitness
        beta = self.args.novelty_weight     # Weight for novelty

        # Recalculate fitness_max based on current peak heights
        fitness_max = max(peak.height for peak in self.landscape.peaks)

        # Compute scaling factor
        scaling_factor = fitness_max / self.novelty_max if self.novelty_max > 0 else 0
        
        # Collect behaviors for the current population
        population_behaviors = []
        for individual in self.population:
            
            # Calculate fitness
            individual.fitness = self.landscape.get_fitness(individual.genes)
            
            # Extract behavior
            behavior = util.extract_behavior(individual.genes, self.landscape)
            individual.behavior = behavior
            population_behaviors.append(behavior)

        # Compute novelty scores and update fitness
        for individual in self.population:
            individual.novelty = self.novelty_archive.compute_novelty(individual.behavior, population_behaviors)
                        
            # Scale novelty to fitness scale
            novelty_fitness = individual.novelty * scaling_factor
            individual.total_fitness = alpha * individual.fitness + beta * novelty_fitness
            
            # Add to the archive
            self.novelty_archive.add(individual.behavior)
            
        self.population_behaviors = population_behaviors
            
    def kinship_coefficient(self, ind1, ind2):
        """
            Definition
            -----------
                The kinship coefficient (f) between two individuals is the probability that a randomly selected allele from both individuals is identical by descent (IBD).
        """
        
        shared_ancestors = ind1.ancestors.intersection(ind2.ancestors)
        total_ancestors = ind1.ancestors.union(ind2.ancestors)

        # No ancestors, unrelated
        if not total_ancestors:
            return 0.0  

        # Ratio of shared ancestors to total ancestors. TODO: Can be made more complex
        f = len(shared_ancestors) / len(total_ancestors)
        return f

    def tournament_selection(self, k=3):
        selected = []
        for i in range(self.pop_size):
            participants = np.random.choice(self.population, k, replace=True) # Each participant is unique
            winner = max(participants, key=lambda ind: ind.total_fitness) # ind.total_fitness for novelty + fitness / fitness for just fitness
            selected.append(winner)
        return selected

    def genetic_distance(self, ind1, ind2):
        return np.sum(ind1.genes != ind2.genes)  # Hamming distance
    
    def crossover(self, parent1, parent2):
        # Calculate kinship coefficient
        if self.max_kinship is not None:
            f = self.kinship_coefficient(parent1, parent2)
            if f > self.max_kinship: # Large F kinship ratio prevents crossover. Prevent Mating
                return None, None

        # One-point crossover
        crossover_point = np.random.randint(1, self.dimensions)
        child_genes1 = np.concatenate([parent1.genes[:crossover_point], parent2.genes[crossover_point:]])
        child_genes2 = np.concatenate([parent2.genes[:crossover_point], parent1.genes[crossover_point:]])

        # Create offspring with updated ancestry
        child1 = Individual(
            args=self.args,
            genes=child_genes1,
            parents=[parent1, parent2],
            generation=max(parent1.generation, parent2.generation) + 1,
            landscape=self.landscape,
            curr_pop_behave=self.population_behaviors, 
            novelty_archive=self.novelty_archive
        )
        child2 = Individual(
            args=self.args,
            genes=child_genes2,
            parents=[parent1, parent2],
            generation=max(parent1.generation, parent2.generation) + 1,
            landscape=self.landscape,
            curr_pop_behave=self.population_behaviors, 
            novelty_archive=self.novelty_archive
        )

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
    
    def run(self):
        
        # Initialize the population
        self.initialize_population()
        cross_count = {}
        inmigration_count = {}

        for gen in range(self.generations):
            # self.study_inbred_chances(gen)
            self.current_generation = gen + 1
            
            # Add-on needed only for MovingPeaksLandscape
            if self.args.bench_name == 'MovingPeaksLandscape':
                if gen % self.landscape.shift_interval == 0:# and gen != 0:
                    self.landscape.apply_shift_peaks(gen) # When using same peaks.
                    # self.landscape.shift_peaks() # When using independent peaks for inbreeding and no inbreeding treatments
                # elif gen == 0:
                #     self.landscape.peaks = self.landscape.original_peaks

            # Calculate fitness and novelty
            if self.args.bench_name == 'MovingPeaksLandscape':
                
                # Best individual is ranked for Pure-Fitness -> Distance to Peak.
                self.calculate_fitness_and_novelty()
                best_individual = max(self.population, key=lambda ind: ind.fitness)
                self.best_fitness_list.append(best_individual.fitness)
                # print(f"Generation {gen+1}. Fitness: {best_individual.fitness}. Novelty: {best_individual.total_fitness - best_individual.fitness}. Total Finess: {best_individual.total_fitness}")
                            
                # Record global optimum fitness after shifting
                self.global_optimum_fitness_list.append(self.landscape.global_optimum.height)
            else:
                
                self.calculate_fitness() # TODO Other methods
                best_fitness = max(self.population, key=lambda ind: ind.fitness).fitness
                self.best_fitness_list.append(best_fitness)
         
        
            # Calculate diversity by the heterozygosity at each locus
            diversity = self.measure_diversity()
            self.diversity_list.append(diversity)

            # Tournament selection.
            selected = self.tournament_selection(self.tournament_size)
        
            # Selection            
            next_population = []
            lambda_pop = self.pop_size * 2
            i = 0
            count_true, count_false = 0,0
            while i < lambda_pop: # Used to be len(selected)
            # while len(next_population) < self.pop_size:
                parent1 = selected[i % len(selected)]
                parent2 = selected[(i+1) % len(selected)]
                offspring = self.crossover(parent1, parent2)

                if offspring[0] is not None and offspring[1] is not None:
                    
                    # Mutate offspring
                    self.mutate(offspring[0])
                    self.mutate(offspring[1])        
                    next_population.extend(offspring)
                    count_true += 1
                    
                else:
                    # Append parents if Inbreeding is allowed
                    if self.max_kinship is None: 
                        next_population.append(copy.deepcopy(parent1))
                        next_population.append(copy.deepcopy(parent2))
                        count_false += 1
                    else:
                        # Introduce new individuals if inbreeding is not allowed
                        genes = np.random.randint(2, size=self.dimensions)
                        individual = Individual(self.args, genes, generation=gen+1, 
                                                landscape=self.landscape,
                                                curr_pop_behave=self.population_behaviors, 
                                                novelty_archive=self.novelty_archive)

                        next_population.append(individual)
                        count_false += 1
                        
                        if len(next_population) < self.pop_size:
                            genes = np.random.randint(2, size=self.dimensions)
                            individual = Individual(self.args, genes, generation=gen+1, 
                                                    landscape=self.landscape,
                                                    curr_pop_behave=self.population_behaviors, 
                                                    novelty_archive=self.novelty_archive)
                            next_population.append(individual)
                            count_false += 1
                i += 2
                
            cross_count[gen] = count_true
            inmigration_count[gen] = count_false
            # BUG: Probably few of next population are being chosen, given the combination
            # of novelty and fitness and that we are just ranking. Maybe we have to select based on one thing?
            # Combine the population (mu+lambda)
            # self.population = next_population[:self.pop_size]
            combined_population = next_population[:lambda_pop] + self.population # BUG: I had self.population before, but it had to be selected right?          
            combined_population.sort(key=lambda ind: ind.total_fitness, reverse=True)

            # # print(f"Lenght of new population: {len(next_population)} + Length of population selected: {len(selected)}. Allowed {self.pop_size}")
            # # exit()
            # # Update the population
            self.population = combined_population[:self.pop_size]
            self.pop_size = len(self.population)

        # print(f"\nCrossovers")
        # for k,v in cross_count.items():
        #     print(f"Gen {k} ~ Offspring Crossovers: {v}")
            
        # print("\nAdding indiv")
        # for k,v in inmigration_count.items():
        #     print(f"Gen {k} ~ Individuals added: {v}")

        return self.best_fitness_list, self.diversity_list, self.global_optimum_fitness_list, self.collapse_events

if __name__ == "__main__":
    
    # ---------------------- MovingPeaksLandscape ---------------------- #
    
    # Get args
    args = util.set_args()
    
    landscape = bf.MovingPeaksLandscape(args)
    
    # Set file plotting name
    term1 = f"{args.bench_name}/NoveltyCollapseKinship/Pop:{args.pop_size}_MR:{args.mutation_rate}_"
    term2 = f"G:{args.generations}_Kc:{args.max_kinship}_TSize:{args.tournament_size}_Dim:{args.dimensions}_"
    term3 = f"MPLShift:{args.mpl_shift_interval}_FitW:{args.fit_weight}_NovW:{args.novelty_weight}_Ann:{args.archive_nn}_"
    term4 = f"ArcT:{args.archive_threshold}" 
    
    args.config_plot = term1 + term2 + term3 + term4
    # Run Individual
    # args.inbred_threshold = 10
    # individual_ga_run(args, landscape, args.inbred_threshold)
    
    # # Run experiments
    print("\n#---------- Running GA with NO Inbreeding Mating... ----------#")
    results_no_inbreeding = exp.multiple_runs_experiment(args, landscape, args.max_kinship)

    print("\n#---------- Running GA with Inbreeding Mating... ----------#")
    results_inbreeding = exp.multiple_runs_experiment(args, landscape, None)

    # # # Plot experiments
    gs_list, fit_list, div_list, label_list = plot.collect_bootstrapping_data(args, results_no_inbreeding, results_inbreeding)
    plot.plot_multiple_runs_MPL_global_optima(args, gs_list, fit_list, div_list, label_list)
    
    # plot.plot_all(args, gs_list, fit_list, label_list, x_label='Generations', y_label='Fitness', title=f'Inbreeding vs no Inbreeding w/ PopSize: {args.pop_size} & MutRate: {args.mutation_rate}')
    # plot.plot_all(args, gs_list, div_list, label_list, x_label='Generations', y_label='Diversity', title=f'Inbreeding vs no Inbreeding w/ PopSize: {args.pop_size} & MutRate: {args.mutation_rate}')
    
    # ---------------------- Other ---------------------- #
    
    # Create an instance of the Landscape
    # landscape = bf.NKLandscape(n=args.N_NKlandscape, k=args.K_NKlandscape)
    # landscape = bf.Jump(args)
    # landscape = bf.DeceptiveLeadingBlocks(args)
    # landscape = bf.Rastrigin(args)
    
# # Check for population collapse
            # if self.args.bench_name == 'MovingPeaksLandscape':
            #     # Check for population collapse condition
            #     if diversity < self.args.collapse_threshold:
            #         num_to_replace = int(self.pop_size * self.args.collapse_fraction)
                    
            #         # Replace the least fit individuals
            #         self.population.sort(key=lambda ind: ind.total_fitness)
            #         # self.population.sort(key=lambda ind: ind.fitness)
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
    
    
# ---------------------- Complex selection with collapse ---------------- #
            
            # Selection
            # individuals_needing_mates = selected.copy()
            # next_population = []
            # failed_parents = set()

            # # Shuffle the list to ensure randomness
            # random.shuffle(individuals_needing_mates)

            # # Dating mating process
            # while individuals_needing_mates:
                
            #     parent1 = individuals_needing_mates.pop(0)  # Take the first individual needing a mate
            #     mate_found = False                          # Flag to track if a suitable mate is found

            #     # Create a copy to iterate over potential mates
            #     potential_mates = individuals_needing_mates.copy()
            #     random.shuffle(potential_mates)  # Shuffle potential mates for randomness

            #     # Try to find a mate for parent1
            #     for potential_mate in potential_mates:
            #         if parent1 == potential_mate:
            #             continue  # Skip if same individual

            #         offspring = self.crossover(parent1, potential_mate)
            #         if offspring[0] is not None and offspring[1] is not None:
            #             # Successful mating
            #             self.mutate(offspring[0])
            #             self.mutate(offspring[1])
            #             next_population.extend(offspring)

            #             # Remove the mate from the list, as they have mated
            #             individuals_needing_mates.remove(potential_mate)
            #             mate_found = True
            #             break  # Exit the loop as we have found a mate
                    
            #     if not mate_found:
            #         # Parent1 could not find a suitable mate
            #         failed_parents.add(parent1)

            # # Convert the set to a list for sorting
            # failed_parents = list(failed_parents)

            # # Determine the number of elites to retain from failed parents
            # elite_size = int(0.20 * self.pop_size)  # Adjust the percentage as needed

            # # Sort failed parents by total_fitness in descending order
            # failed_parents.sort(key=lambda ind: ind.total_fitness, reverse=True) # Novelty search
            # # failed_parents.sort(key=lambda ind: ind.fitness, reverse=True)

            # # Select the top individuals
            # elites_from_failed_parents = failed_parents[:elite_size]

            # # Add elites to the next generation
            # next_population.extend(elites_from_failed_parents)
            
            # # Update the population
            # self.population = next_population
            
            # # After updating the population
            # min_population_size = 20
            # if len(self.population) < min_population_size:
            #     print(f"Generation {gen + 1}: Population size {len(self.population)} below minimum threshold {min_population_size}. Introducing new individuals.")
        
            #     # Add until initial population is fixed again
            #     while len(self.population) < self.initial_pop_size:
            #         genes = np.random.randint(2, size=self.dimensions)
            #         new_individual = Individual(genes=genes, generation=gen+1)
            #         self.population.append(new_individual)
            
            # self.pop_size = len(self.population)
            # print(f"New pop size: {self.pop_size}")
            
        
            # distance = self.genetic_distance(parent1, parent2)
            # if distance < self.inbred_threshold: # the bigger the allowed distance, the farther apart the parents need to be
            #     return None, None
            
            
        # def study_inbred_chances(self, gen):
        
        # print(f"GEN: {gen+1}. Studying inbreeding relationship")
        # i = 0
        # true_mating = 0
        # false_mating = 0
        # while i < len(self.population):
        #     parent1 = self.population[i]
        #     parent2 = self.population[(i + 1) % len(self.population)]
        #     f = self.kinship_coefficient(parent1, parent2) 
        #     mate = f < self.max_kinship
        #     if mate:
        #         true_mating += 1
        #     else:
        #         false_mating += 1
        #     print(f"Individual {i+1} and {i+2} have kinship coeff: {f}. Can they mate? {mate}")
            
        #     i += 2
            
        # print(f"There were {true_mating} pairs that can mate.")
        # print(f"There were {false_mating} pairs that cannot mate.\n\n")