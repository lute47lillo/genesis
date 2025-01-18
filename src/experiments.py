from gp_testing import GeneticAlgorithmGPTesting
from gp_bloat import GeneticAlgorithmGPBloat
from gp_unblock import GeneticAlgorithmGPUnblock
from gp_sharing import GeneticAlgorithmGPSharing
from gp_only_mut import GeneticAlgorithmGPMutation
from gp_only_cross import GeneticAlgorithmGPOnlyCrossover
from gp_dynamic import GeneticAlgorithmGPDynamic
from gp_linear import GeneticAlgorithmGPLinear
from gp_performance import GeneticAlgorithmGPRamped
from gp_sharing_parallel import GeneticAlgorithmGPSharingParallel
from gp_base import GeneticAlgorithmGPPerformance
import util as util
import random
    
# ----------------------------------- Genetic Programming Experiments -------------------------- #

def test_multiple_runs_function_gp(args, landscape, inbred_threshold):
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGPTesting(
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

# ----------------------------------- Genetic Programming Performance w/ ramped half-n-half initialization -------------------------- #

def test_multiple_runs_function_gp_ramped(args, landscape, inbred_threshold):
    
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

# ----------------------------------- Genetic Programming Dynamic-------------------------- #

def test_multiple_runs_function_dynamic(args, landscape, inbred_threshold):
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGPDynamic(
            args=args,
            mut_rate=args.mutation_rate,
            inbred_threshold=inbred_threshold  # Adjust based on inbreeding prevention
        )
        # Run GP-based GA for Given Function
        best_fitness_list, diversity_list, inbred_threshold_list, gen_success = ga_gp.run(landscape.symbolic_fitness_function)
        
        results[run] = {
                'best_fitness': best_fitness_list,
                'diversity': diversity_list, 
                'generation_success': gen_success,
                'inbred_threshold': inbred_threshold_list
            }
        
        print(f"Inbred Threshold: {inbred_threshold}. Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}\n"
                f"Generation Success: {gen_success}. Best Fitness = {best_fitness_list[-1]:.3f}\n"
                f"Diversity = {diversity_list[-1]:.3f}\n"
                f"Inbreed Threshold = {inbred_threshold_list[-1]:.3f}\n")
        
    return results

# ----------------------------------- Genetic Programming Exploration vs Exploitation -------------------------- #

def test_multiple_runs_function_linear(args, landscape, inbred_threshold):
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGPLinear(
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
        
        print(f"Linear with slope: {args.slope_threshold}. Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}\n"
                f"Generation Success: {gen_success}. Best Fitness = {best_fitness_list[-1]:.3f}\n"
                f"Diversity = {diversity_list[-1]:.3f}\n")
        
    return results


# ----------------------------------- Genetic Programming InbreedUnblock -------------------------- #

def test_multiple_runs_function_unblock(args, landscape, inbred_threshold):
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGPUnblock(
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

# ----------------------------------- Genetic Programming Mutation -------------------------- #

def test_multiple_runs_function_mutation(args, landscape, inbred_threshold):
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGPMutation(
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

# ----------------------------------- Genetic Programming Only Crossover -------------------------- #

def test_multiple_runs_function_only_cross(args, landscape, inbred_threshold):
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
        # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
        print(f"Running experiment nº: {run}")
        ga_gp = GeneticAlgorithmGPOnlyCrossover(
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
        metrics_lists, measures_lists, intron_lists, _, gen_success = ga_gp.run(landscape)
        
        # Split lists
        best_fitness_list, diversity_list = metrics_lists
        average_size_list, average_depth_list = measures_lists
        pop_ratio_intron_list, avg_ratio_intron_list, pop_total_intron_list, pop_total_nodes_list = intron_lists
        # avg_kinship_list, t_close_list, t_far_list = kinship_lists
        
        results[run] = {
                'best_fitness': best_fitness_list,
                'diversity': diversity_list, 
                'generation_success': gen_success,
                'avg_tree_size': average_size_list,
                'avg_tree_depth': average_depth_list,
                'pop_intron_ratio': pop_ratio_intron_list,
                'avg_intron_ratio': avg_ratio_intron_list,
                'pop_total_introns': pop_total_intron_list,
                'pop_total_nodes': pop_total_nodes_list,
                # 'avg_kinship': avg_kinship_list,
                # 't_close': t_close_list, 
                # 't_far': t_far_list
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
                # f"Avg Population Tree Kinship = {avg_kinship_list[-1]:.3f}\n"
                # f"Most Related Tree Kinship = {t_close_list[-1][1]:.3f} with {len(t_close_list[-1][0])} ancestors\n"
                # f"Least Related Tree Kinship = {t_far_list[-1][1]:.3f} with {len(t_far_list[-1][0])} ancestors.")
                
        # # Sanity Save of results
        # if inbred_threshold == None:
        #     util.save_accuracy(results, f"{args.config_plot}_inbreeding_RUN:{run}_{gen_success}.npy")
        # else:
        #     util.save_accuracy(results, f"{args.config_plot}_no_inbreeding_RUN:{run}_{gen_success}.npy")
        
    return results


# ----------------------------------- Legacy  -------------------------- #
def multiple_mrates_function_gp(args, mutation_rates, landscape, inbred_threshold):
    
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
           
            # Reset the seed for every run
            util.set_seed(random.randint(0, 999999))
            print(f"Running experiment nº {run} w/ Mutation Rate: {rate}")
            ga_gp = GeneticAlgorithmGPTesting(
                args=args,
                mut_rate=rate,
                inbred_threshold=inbred_threshold  # Adjust based on inbreeding prevention
            )
            # Run GP-based GA for Given Function
            best_fitness_list, diversity_list, gen_success = ga_gp.run(landscape.symbolic_fitness_function)
            
            results[rate]['generation_successes'].append(gen_success)
            results[rate]['diversity'].append(diversity_list)
            results[rate]['fitness'].append(best_fitness_list)
            
            # Sanity Save of results
            if inbred_threshold == None:
                util.save_accuracy(results, f"{args.config_plot}_inbreeding_RUN:{run}_{gen_success}_MR:{rate}.npy")
            else:
                util.save_accuracy(results, f"{args.config_plot}_no_inbreeding_RUN:{run}_{gen_success}_MR:{rate}.npy")
            
                
            print(f"Inbred Threshold: {inbred_threshold}. Population Size {args.pop_size} & Mutation Rate: {rate}: Generation Success {gen_success} ~ Best Fitness {best_fitness_list[-1]:.3f} ~ Best Diversity {diversity_list[-1]:.3f}")

    return results