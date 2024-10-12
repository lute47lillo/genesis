import numpy as np
import matplotlib.pyplot as plt
import os

# Rastrigin Function (Same as before)
def rastrigin(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

# Initialize Population with Lineage Tracking
def init_population(pop_size, dimensions, bounds):
    population = []
    for _ in range(pop_size):
        individual = {
            'genes': np.random.uniform(bounds[0], bounds[1], dimensions),
            'id': np.random.randint(1e6),  # Unique identifier
            'ancestors': set()
        }
        population.append(individual)
    return population

# Calculate Fitness
def calculate_fitness(population):
    return np.array([rastrigin(individual['genes']) for individual in population])

# Genetic Distance Function
def genetic_distance(ind1, ind2):
    return np.linalg.norm(ind1['genes'] - ind2['genes'])

# Selection (Tournament)
def tournament_selection(population, fitness, k=3):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        indices = np.random.choice(pop_size, k, replace=False)
        selected.append(population[indices[np.argmin(fitness[indices])]])
    return selected

# Crossover with Inbreeding Prevention
def crossover(parent1, parent2, allowed_distance, crossover_rate=0.7):
    # Check genetic distance
    distance = genetic_distance(parent1, parent2)
    if distance < allowed_distance:
        # Skip mating due to inbreeding prevention
        return None, None
    if np.random.rand() < crossover_rate:
        mask = np.random.rand(len(parent1['genes'])) < 0.5
        child_genes1 = np.where(mask, parent1['genes'], parent2['genes'])
        child_genes2 = np.where(mask, parent2['genes'], parent1['genes'])
    else:
        child_genes1 = parent1['genes'].copy()
        child_genes2 = parent2['genes'].copy()
    # Create offspring with updated lineage
    child1 = {
        'genes': child_genes1,
        'id': np.random.randint(1e6),
        'ancestors': parent1['ancestors'].union(parent2['ancestors'], {parent1['id'], parent2['id']})
    }
    child2 = {
        'genes': child_genes2,
        'id': np.random.randint(1e6),
        'ancestors': parent1['ancestors'].union(parent2['ancestors'], {parent1['id'], parent2['id']})
    }
    return child1, child2

# Mutation
def mutate(individual, mutation_rate, bounds):
    for i in range(len(individual['genes'])):
        if np.random.rand() < mutation_rate:
            individual['genes'][i] = np.random.uniform(bounds[0], bounds[1])
    return individual

# Genetic Algorithm with Inbreeding Prevention
def genetic_algorithm(pop_size, dimensions, bounds, generations, mutation_rate, allowed_distance):
    population = init_population(pop_size, dimensions, bounds)
    best_fitness_list = []
    diversity_list = []

    for gen in range(generations):
        fitness = calculate_fitness(population)
        best_fitness = np.min(fitness)
        best_fitness_list.append(best_fitness)

        # Measure Diversity (using unique ancestor IDs)
        unique_ancestors = set()
        for individual in population:
            unique_ancestors.update(individual['ancestors'])
            unique_ancestors.add(individual['id'])
        diversity = len(unique_ancestors) / (pop_size * generations)
        diversity_list.append(diversity)

        # Selection
        selected = tournament_selection(population, fitness)

        # Crossover and Mutation
        next_population = []
        i = 0
        while len(next_population) < pop_size and i < len(selected):
            parent1 = selected[i % len(selected)]
            parent2 = selected[(i+1) % len(selected)]
            child1, child2 = crossover(parent1, parent2, allowed_distance)
            if child1 and child2:
                child1 = mutate(child1, mutation_rate, bounds)
                child2 = mutate(child2, mutation_rate, bounds)
                next_population.extend([child1, child2])
            i += 2
        # If not enough offspring were produced due to inbreeding prevention, fill the rest randomly
        while len(next_population) < pop_size:
            random_individual = {
                'genes': np.random.uniform(bounds[0], bounds[1], dimensions),
                'id': np.random.randint(1e6),
                'ancestors': set()
            }
            next_population.append(random_individual)
        population = next_population[:pop_size]

    return best_fitness_list, diversity_list

# Parameters
pop_size = 20
dimensions = 10
bounds = (-5.12, 5.12)
generations = 100
mutation_rate = 0.4
allowed_distance = 1.0  # Adjust this threshold based on problem scale

# Run GA with Inbreeding Prevention
best_fitness, diversity = genetic_algorithm(pop_size, dimensions, bounds, generations, mutation_rate, allowed_distance)

# Plotting
plt.plot(best_fitness)
plt.title('Best Fitness over Generations with Inbreeding Prevention')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.savefig(f"{os.getcwd()}/test_fit.png")
plt.close()
# plt.show()

plt.plot(diversity)
plt.title('Diversity over Generations with Inbreeding Prevention')
plt.xlabel('Generation')
plt.ylabel('Genetic Diversity')
plt.savefig(f"{os.getcwd()}/test_div.png")
plt.close()
# plt.show()
