from gp_bloat import GeneticAlgorithmGPBloat  # Run Intron Study and bloat experiments
from gp_sharing_parallel import GeneticAlgorithmGPSharingParallel # Run Fitness sharing experiments
from gp_base import GeneticAlgorithmGPPerformance # Run performance-vanilla experiments
from gp_semantics import GeneticAlgorithmGPSemantics
import util as util
import numpy as np
import random

# ----------------------------------- Genetic Programming Performance -------------------------- #

def test_multiple_runs_function_gp_based(args, landscape, inbred_threshold):
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        # TODO: Running 750 performance algorithms
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGPPerformance(
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
        
        print(f"Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}: Best Fitness {best_fitness_list[-1]:.3f} ~ Best Diversity {diversity_list[-1]:.3f}")

    return results

# ----------------------------------- Genetic Programming Semantics (SSC, SAC) -------------------------- #

def test_multiple_runs_function_gp_semantics(args, landscape, inbred_threshold):
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        # TODO: Running 750 performance algorithms
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGPSemantics(
            args=args,
            mut_rate=args.mutation_rate,
            inbred_threshold=inbred_threshold  # Adjust based on inbreeding prevention
        )
        # Run GP-based GA for Given Function
        best_fitness_list, diversity_list, average_tree_list, gen_success = ga_gp.run(landscape.symbolic_fitness_function, landscape.evaluate_semantics_tree)
        
        results[run] = {
                'best_fitness': best_fitness_list,
                'diversity': diversity_list, 
                'generation_success': gen_success,
                'avg_tree_size': average_tree_list,
            }
        
        print(f"Avg. Tree size {np.mean(average_tree_list):.3f} ~ Avg. Diversity {np.mean(diversity_list):.3f}. Generation Success: {gen_success}.")

    return results

# ----------------------------------- Genetic Programming Fitness Sharing -------------------------- #

def test_multiple_runs_function_sharing(args, landscape, inbred_threshold):
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        print(f"Running experiment nº: {run}")
        # TODO: Running parallel
        ga_gp = GeneticAlgorithmGPSharingParallel(
            args=args,
            mut_rate=args.mutation_rate,
            inbred_threshold=inbred_threshold  # Adjust based on inbreeding prevention
        )
        # Run GP-based GA for Given Function
        best_fitness_list, diversity_list, sigma_share_list, gen_success = ga_gp.run(landscape.symbolic_fitness_function)
        
        results[run] = {
                'best_fitness': best_fitness_list,
                'diversity': diversity_list, 
                'generation_success': gen_success,
                'sigma_share': sigma_share_list
            }
        
        print(f"Inbred Threshold: {inbred_threshold}. Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}\n"
                f"Generation Success: {gen_success}. Best Fitness = {best_fitness_list[-1]:.3f}\n"
                f"Diversity = {diversity_list[-1]:.3f}\n"
                f"Sigma Share = {sigma_share_list[-1]:.3f}\n")
        
    return results

# ----------------------------------- Genetic Programming Bloat and Introns -------------------------- #

def test_multiple_runs_function_bloat(args, landscape, inbred_threshold):
    """
        Definition
        -----------
            TESTING EXPERIMENTAL FUNCTIONS.
                - Currently bloat effects.
    """
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGPBloat(
            args=args,
            mut_rate=args.mutation_rate,
            inbred_threshold=inbred_threshold  # Adjust based on inbreeding prevention
        )
        # Run GP-based GA for Given Function
        metrics_lists, measures_lists, intron_lists, gen_success = ga_gp.run(landscape)
        
        # Split lists
        best_fitness_list, diversity_list = metrics_lists
        average_size_list, average_depth_list = measures_lists
        pop_ratio_intron_list, avg_ratio_intron_list, pop_total_intron_list, pop_total_nodes_list = intron_lists
        
        results[run] = {
                'best_fitness': best_fitness_list,
                'diversity': diversity_list, 
                'generation_success': gen_success,
                'avg_tree_size': average_size_list,
                'avg_tree_depth': average_depth_list,
                'pop_intron_ratio': pop_ratio_intron_list,
                'avg_intron_ratio': avg_ratio_intron_list,
                'pop_total_introns': pop_total_intron_list,
                'pop_total_nodes': pop_total_nodes_list
            }
        
        print(f"Inbred Threshold: {inbred_threshold}. Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}\n"
                f"Generation Success: {gen_success}. Best Fitness = {best_fitness_list[-1]:.3f}\n"
                f"Diversity = {diversity_list[-1]:.3f}\n"
                f"Avg Size = {average_size_list[-1]:.3f}\n"
                f"Avg Depth = {average_depth_list[-1]:.3f}\n"
                f"Population Intron Ratio = {pop_ratio_intron_list[-1]:.3f}\n"
                f"Avg Intron Ratio per Individual = {avg_ratio_intron_list[-1]:.3f}\n"
                f"Population Total Intron Nodes = {pop_total_intron_list[-1]:.3f}\n" 
                f"Population Total Nodes = {pop_total_nodes_list[-1]:.3f}\n")
        
    return results