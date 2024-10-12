import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import math
import benchmark_fn_factory as fns


class Individual:
    def __init__(self, genes, id=None, ancestors=None):
        """
            Definition
            -----------
                - genes: The gene vector representing the solution.
                - id: A unique identifier for each individual.
                - ancestors: A set containing the IDs of all ancestors.
        """
        self.genes = genes
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
        self.allowed_distance = allowed_distance  # None means no inbreeding prevention
        self.population = []
        self.best_fitness_list = []
        self.diversity_list = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            genes = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
            individual = Individual(genes)
            self.population.append(individual)

    def calculate_fitness(self, opt_fn):
        for individual in self.population:
            individual.fitness = opt_fn(individual.genes)

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

        child1 = Individual(child_genes1, ancestors=ancestors1)
        child2 = Individual(child_genes2, ancestors=ancestors2)

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

    def run(self, optim_fn=fns.rastrigin):
        self.initialize_population()

        for gen in range(self.generations):
            self.calculate_fitness(optim_fn)
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
                        individual = Individual(genes)
                        next_population.append(individual)
                        if len(next_population) < self.pop_size:
                            genes = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
                            individual = Individual(genes)
                            next_population.append(individual)
                i += 2

            self.population = next_population[:self.pop_size]

        return self.best_fitness_list, self.diversity_list

