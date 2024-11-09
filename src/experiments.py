from ga_rugged import GeneticAlgorithmRugged
from ga_optimization import GeneticAlgorithmOpt
from genetic_programming import GeneticAlgorithmGP
import plotting as plot
import util as util
import random

# -------------------------- Optimization Functions ------------------------------ #

def run_multiple_pop_sizes(args, pop_sizes, landscape, inbred_threshold):
    
    # Run GA with different population sizes
    results = {}
    for pop_size in pop_sizes:
        ga = GeneticAlgorithmOpt(
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
        ga = GeneticAlgorithmOpt(
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

def multiple_runs_experiment(args, landscape, max_kinship):
    """
        Definition
        -----------
            Run basic GA with hyperparameters of your choice for multiple runs. Landscape based algorithm.
    """
    
    # Initialize Novelty Archive
    results = {}
    for run in range(args.exp_num_runs):
        args.current_run = run
        ga = GeneticAlgorithmRugged(
            args=args,
            landscape=landscape,
            bounds=None,
            max_kinship=max_kinship
        )
        best_fitness_list, diversity_list, global_optimum_fitness_list, collapse_events = ga.run()
        results[run] = {
            'best_fitness': best_fitness_list,
            'diversity': diversity_list, 
            'global_optimum': global_optimum_fitness_list, 
            'collapse_events': collapse_events
        }

        print(f"\nExperiment Run {run+1}. Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}")
        
        # TODO: Catch error for some number of shifts with respect to the print
        n_shifts = int(args.generations / args.mpl_shift_interval)
        shifts_fit_avg, shifts_gl_avf, shifts_div_avg = 0, 0, 0
        for n in range(1, n_shifts+1): 
            idx = (args.mpl_shift_interval)*n - 1
            if idx == args.generations:
                n_shifts -= 1 # for printing purposes
                break
            
            shifts_fit_avg += best_fitness_list[idx]
            shifts_gl_avf += global_optimum_fitness_list[idx]
            shifts_div_avg += diversity_list[idx]
            print(f"\tPeak Shift {n}. Best Fitness {best_fitness_list[idx]:.4f} ({global_optimum_fitness_list[idx]:.4f}) ~ Best Diversity {diversity_list[idx]:.4f}")
        
        print(f"Average Across Peak Shifts. Fitness: {(shifts_fit_avg/n_shifts):.4f} ({(shifts_gl_avf/n_shifts):.4f}) ~ Diversity: {(shifts_div_avg/n_shifts):.4f}")
        
    return results

def individual_ga_run(args, landscape, max_kinship):

    if max_kinship == None:
        print("Running GA with Inbreeding Mating...")
    else:
        print("Running GA with NO Inbreeding Mating...")
    
    # Run
    ga = GeneticAlgorithmRugged(
        args=args,
        landscape=landscape,
        bounds=None,
        max_kinship=max_kinship
    )
    best_fitness_list, diversity_list, global_optimum_fitness_list, collapse_events = ga.run()
    plot.plot_individual_MPL_global_optima(args, best_fitness_list, diversity_list, global_optimum_fitness_list, collapse_events, collapse_threshold=0.2)
    
def multiple_mrates_rugged_ga(args, mutation_rates, landscape, max_kinship):
    """
        TODO: Generalize to be for all function experiments and choose within. As of now only work with ackley
    """
    
    # Initialize Rugged-based GA for Any given function
    results = {}
    for rate in mutation_rates:
        # Initialize lists to store data across all runs for this mutation rate
        results[rate] = {
            'best_fitness': [],
            'diversity': [],
            'global_optimum': [],
            'collapse_events': []
        }
        for run in range(args.exp_num_runs):
            print(f"Running experiment nº {run} w/ Mutation Rate: {rate}")
            ga_gp = GeneticAlgorithmRugged(
                args=args,
                landscape=landscape,
                bounds=None,
                max_kinship=max_kinship
            )
            
            best_fitness_list, diversity_list, global_optimum_fitness_list, collapse_events = ga_gp.run()
            results[rate]['best_fitness'].append(best_fitness_list)
            results[rate]['diversity'].append(diversity_list)
            results[rate]['global_optimum'].append(global_optimum_fitness_list)
            results[rate]['collapse_events'].append(collapse_events)
        
            print(f"Population Size {args.pop_size} & Mutation Rate: {rate}: Best Fitness {best_fitness_list[-1]:.4f} ({global_optimum_fitness_list[-1]:.4f}) ~ Best Diversity {diversity_list[-1]:.4f}")
                
    return results
    
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
            mut_rate=args.mutation_rate,
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
        # if inbred_threshold == None:
        #     util.save_accuracy(results, f"{args.config_plot}_inbreeding_RUN:{run}_{gen_success}.npy")
        # else:
        #     util.save_accuracy(results, f"{args.config_plot}_no_inbreeding_RUN:{run}_{gen_success}.npy")
            
        print(f"Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")

    return results

def multiple_mrates_function_gp(args, mutation_rates, landscape, inbred_threshold):
    """
        TODO: Generalize to be for all function experiments and choose within. As of now only work with ackley
    """
    
    # Initialize GP-based GA for Any given function
    results = {}
    for rate in mutation_rates:
        # Initialize lists to store data across all runs for this mutation rate
        results[rate] = {
            'generation_successes': [],   # List of gen_success from each run
            'diversity': [],
            'fitness': []
        }
        for run in range(args.exp_num_runs):
            print(f"Running experiment nº {run} w/ Mutation Rate: {rate}")
            ga_gp = GeneticAlgorithmGP(
                args=args,
                mut_rate=rate,
                inbred_threshold=inbred_threshold  # Adjust based on inbreeding prevention
            )
            # Run GP-based GA for Given Function
            best_fitness_list, diversity_list, gen_success = ga_gp.run(landscape.symbolic_fitness_function)
            
            results[rate]['generation_successes'].append(gen_success)
            results[rate]['diversity'].append(diversity_list)
            results[rate]['fitness'].append(best_fitness_list)
                
            print(f"Population Size {args.pop_size} & Mutation Rate: {rate}: Generation Success {gen_success} ~ Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")

    return results