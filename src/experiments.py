import benchmark_factory as bf
import matplotlib.pyplot as plt
import numpy as np
from genetic_algorithms import LandscapeGA, NoveltyArchive
from ga_optimization import GeneticAlgorithm
from genetic_programming import GeneticAlgorithmGP
import plotting as plot
import util as util

# -------------------------- Optimization Functions ------------------------------ #

def run_multiple_pop_sizes(args, pop_sizes, landscape, inbred_threshold):
    
    # Run GA with different population sizes
    results = {}
    for pop_size in pop_sizes:
        ga = GeneticAlgorithm(
            args=args,
            mutation_rate=args.mutation_rate,
            pop_size=pop_size,
            landscape=landscape,
            inbred_threshold=inbred_threshold 
        )
        best_fitness_list, diversity_list = ga.run()
        results[pop_size] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list
        }
        print(f"Population Size {pop_size}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results

def run_inbreeding_mutation_rates(args, mutation_rates, landscape, inbred_threshold):
    
    # Run GA with different mutation rates
    results = {}
    for rate in mutation_rates:
        ga = GeneticAlgorithm(
            args=args,
            mutation_rate=rate,
            pop_size=args.pop_size,
            landscape=landscape,
            inbred_threshold=inbred_threshold 
        )
        best_fitness_list, diversity_list = ga.run()
        results[rate] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list
        }
        print(f"Mutation Rate {rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results
    
# ---------------------------- Rugged Landscape Functions -------------------------- #

def multiple_runs_experiment(args, landscape, inbred_threshold):
    """
        Definition
        -----------
            Run basic GA with hyperparameters of your choice for multiple runs. Landscape based algorithm.
    """
    
    # Initialize Novelty Archive
    results = {}
    for run in range(args.exp_num_runs):
        ga = LandscapeGA(
            args=args,
            landscape=landscape,
            bounds=None,
            inbred_threshold=inbred_threshold
        )
        best_fitness_list, diversity_list, global_optimum_fitness_list, collapse_events = ga.run(collapse_threshold=0.2, collapse_fraction=0.1)
        results[run] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list, 
            'global_optimum': global_optimum_fitness_list, 
            'collapse_events': collapse_events
        }
        print(f"Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")
        
    return results

def individual_ga_run(args, landscape, inbred_threshold):

    if inbred_threshold == None:
        print("Running GA with Inbreeding Mating...")
    else:
        print("Running GA with NO Inbreeding Mating...")
    
    # Run
    ga = LandscapeGA(
        args=args,
        landscape=landscape,
        bounds=None,
        inbred_threshold=inbred_threshold
    )
    best_fitness_list, diversity_list, global_optimum_fitness_list, collapse_events = ga.run(collapse_threshold=0.2, collapse_fraction=0.5)
    plot.plot_individual_MPL_global_optima(args, best_fitness_list, diversity_list, global_optimum_fitness_list, collapse_events, collapse_threshold=0.2)
    
# ----------------------------------- Genetic Programming Experiments -------------------------- #

def multiple_runs_function_gp(args, landscape, inbred_threshold):
    """
        TODO: Generalize to be for all function experiments and choose within. As of now only work with ackley
    """
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGP(
            args=args,
            inbred_threshold=inbred_threshold  # Adjust based on inbreeding prevention
        )
        # Run GP-based GA for Given Function
        best_fitness_list, diversity_list, gen_success = ga_gp.run(landscape.symbolic_fitness_function)
        
        results[run] = {
                'best_fitness': best_fitness_list,
                'diversity': diversity_list, 
                'generation_success': gen_success
            }
        
        # Sanity Save of results
        if inbred_threshold == None:
            util.save_accuracy(results, f"{args.config_plot}_inbreeding_RUN:{run}_{gen_success}.npy")
        else:
            util.save_accuracy(results, f"{args.config_plot}_no_inbreeding_RUN:{run}_{gen_success}.npy")
            
        print(f"Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")

    return results