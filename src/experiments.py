import benchmark_fn_factory as fns
import matplotlib.pyplot as plt
import numpy as np
from genetic_algorithms import GeneticAlgorithm, LandscapeGA
import plotting as plot
import util as util

def get_bounds(benchmark):
    if benchmark == 'ackley':
        bounds = (-32.768, 32.768)
    elif benchmark == 'rastrigin':
        bounds = (-5.12, 5.12)
    elif benchmark == 'rosenbrock':
        bounds = (-2.048, 2.048)
    elif benchmark == 'schwefel':
        bounds = (-500, 500)
    elif benchmark == 'griewank':
        bounds = (-600, 600)   
    elif benchmark == 'sphere':
        bounds = (-5.12, 5.12)
    
    return bounds
    
def set_config_parameters(benchmark):
    
    # Experiment Parameters
    pop_sizes = [25, 50, 100, 200]
    dimensions = 10
    bounds = get_bounds(benchmark)
    generations = 200
    mutation_rate = 0.2
    allowed_distance = 1.0
    
    return pop_sizes, dimensions, bounds, generations, mutation_rate, allowed_distance

def run_inbreeding_pop_sizes(optim_fn, pop_sizes, dimensions, bounds, generations, mutation_rate, allowed_distance):
    
    # Run GA without Inbreeding Prevention
    results_inbreeding = {}
    print("Running GA with Inbreeding mating...")
    for pop_size in pop_sizes:
        ga = GeneticAlgorithm(
            landscape=optim_fn,
            pop_size=pop_size,
            dimensions=dimensions,
            bounds=bounds,
            generations=generations,
            mutation_rate=mutation_rate,
            allowed_distance=None  # No inbreeding prevention
        )
        best_fitness_list, diversity_list = ga.run()
        results_inbreeding[pop_size] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list
        }
        print(f"Population Size {pop_size}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results_inbreeding

def run_no_inbreeding_pop_sizes(optim_fn, pop_sizes, dimensions, bounds, generations, mutation_rate, allowed_distance):
    
    # Run GA with Inbreeding Prevention
    results_no_inbreeding= {}
    print("\nRunning GA with NO Inbreeding Mating...")
    for pop_size in pop_sizes:
        ga = GeneticAlgorithm(
            landscape=optim_fn,
            pop_size=pop_size,
            dimensions=dimensions,
            bounds=bounds,
            generations=generations,
            mutation_rate=mutation_rate,
            allowed_distance=allowed_distance  # Inbreeding prevention active
        )
        best_fitness_list, diversity_list = ga.run()
        results_no_inbreeding[pop_size] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list
        }
        print(f"Population Size {pop_size}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results_no_inbreeding

def run_inbreeding_mutation_rates(optim_fn, pop_size, dimensions, bounds, generations, mutation_rates, allowed_distance):
    
    # Run GA without Inbreeding Prevention
    results_inbreeding = {}
    print("Running GA with Inbreeding Mating...")
    for rate in mutation_rates:
        ga = GeneticAlgorithm(
            landscape=optim_fn,
            pop_size=pop_size,
            dimensions=dimensions,
            bounds=bounds,
            generations=generations,
            mutation_rate=rate,
            allowed_distance=None  # No inbreeding prevention
        )
        best_fitness_list, diversity_list = ga.run()
        results_inbreeding[rate] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list
        }
        print(f"Mutation Rate {rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results_inbreeding

def run_no_inbreeding_mutation_rates(optim_fn, pop_size, dimensions, bounds, generations, mutation_rates, allowed_distance):
    
    # Run GA with Inbreeding Prevention
    results_no_inbreeding = {}
    print("\nRunning GA with NO Inbreeding mating...")
    for rate in mutation_rates:
        ga = GeneticAlgorithm(
            landscape=optim_fn,
            pop_size=pop_size,
            dimensions=dimensions,
            bounds=bounds,
            generations=generations,
            mutation_rate=rate,
            allowed_distance=allowed_distance  # Inbreeding prevention active
        )
        best_fitness_list, diversity_list = ga.run()
        results_no_inbreeding[rate] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list
        }
        print(f"Mutation Rate {rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results_no_inbreeding
    
def multiple_runs_experiment(args, landscape, inbred_threshold):
    """
        Definition
        -----------
            Run basic GA with hyperparameters of your choice for multiple runs. Landscape based algorithm.
    """
        
    results = {}
    for run in range(args.exp_num_runs):
        ga = LandscapeGA(
            args=args,
            landscape=landscape,
            bounds=None,
            inbred_threshold=inbred_threshold
        )
        best_fitness_list, diversity_list = ga.run()
        results[run] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list, 
        }
        print(f"Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results
    
if __name__ == "__main__":
    
    # Get args
    args = util.set_args()
    
    benchmarks = {"ackley": fns.ackley_function, "rosenbrock":fns.rosenbrock_function,
                  "rastrigin": fns.rastrigin, "schwefel": fns.schwefel_function,
                  "griewank" :fns.griewank_function, "sphere": fns.sphere_function}
    
    pop_sizes, dimensions, bounds, generations, mutation_rate, allowed_distance = set_config_parameters(args.benchmark)
    # results_inbreeding = run_inbreeding_pop_sizes(benchmarks.get(args.benchmark), pop_sizes, dimensions, bounds, generations, mutation_rate, allowed_distance)
    # results_no_inbreeding = run_no_inbreeding_pop_sizes(benchmarks.get(args.benchmark), pop_sizes, dimensions, bounds, generations, mutation_rate, allowed_distance)

    
    # args.config_plot = f"MutRate:{mutation_rate}_Threshold:{allowed_distance}_Gens:{generations}_TourSize:10" # TODO: SEt tournament as another variable
    # plot.plot_fitness_comparison_populations(args, pop_sizes, "PopSize", results_inbreeding, results_no_inbreeding)
    # plot.plot_diversity_comparison_populations(args, pop_sizes, "PopSize", results_inbreeding, results_no_inbreeding)
    
    # ------------------------- Mutation Rate Experiments --------------------------- #
    
    mutation_rates = [0.01, 0.1, 0.2, 0.3, 0.4]
    pop_size = 300
    results_inbreeding = run_inbreeding_mutation_rates(benchmarks.get(args.benchmark), pop_size, dimensions, bounds, generations, mutation_rates, allowed_distance)
    results_no_inbreeding = run_no_inbreeding_mutation_rates(benchmarks.get(args.benchmark), pop_size, dimensions, bounds, generations, mutation_rates, allowed_distance)
    
    args.config_plot = f"PopSize:{pop_size}_Threshold:{allowed_distance}_Gens:{generations}_TourSize:10" # TODO: SEt tournament as another variable
    plot.plot_fitness_comparison_populations(args, mutation_rates, "MutRates", results_inbreeding, results_no_inbreeding)
    plot.plot_diversity_comparison_populations(args, mutation_rates, "MutRates", results_inbreeding, results_no_inbreeding)
    


