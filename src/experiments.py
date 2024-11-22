from genetic_programming import GeneticAlgorithmGP
from gp_testing import GeneticAlgorithmGPTesting
from gp_bloat import GeneticAlgorithmGPBloat
import util as util
import random
    
# ----------------------------------- Genetic Programming Experiments -------------------------- #

def multiple_runs_function_gp(args, landscape, inbred_threshold):
    """
        TODO: Generalize to be for all function experiments and choose within. As of now only work with ackley
    """
    
    # Initialize GP-based GA for Any given function
    results = {}
    for run in range(args.exp_num_runs):
         # Reset the seed for every run
        util.set_seed(random.randint(0, 999999))
       
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

def test_multiple_runs_function_gp(args, landscape, inbred_threshold):
    """
        TODO: Generalize to be for all function experiments and choose within. As of now only work with ackley
    """
    
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
        
        print(f"Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}: Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")

    return results

def test_multiple_runs_function_bloat(args, landscape, inbred_threshold):
    """
        TODO: Generalize to be for all function experiments and choose within. As of now only work with ackley
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
        best_fitness_list, diversity_list, average_size_list, average_depth_list, population_intron_ratio_list, average_intron_ratio_list, gen_success = ga_gp.run(landscape)
        
        results[run] = {
                'best_fitness': best_fitness_list,
                'diversity': diversity_list, 
                'generation_success': gen_success,
                'avg_tree_size': average_size_list,
                'avg_tree_depth': average_depth_list,
                'population_intron_ratio': population_intron_ratio_list,
                'average_intron_ratio': average_intron_ratio_list
            }
        
        print(f"Population Size {args.pop_size} & Mutation Rate: {args.mutation_rate}. "
                f"Generation {gen_success}: Best Fitness = {best_fitness_list[-1]:.4f}, "
                f"Diversity = {diversity_list[-1]:.4f}, "
                f"Avg Size = {average_size_list[-1]:.2f}, "
                f"Avg Depth = {average_depth_list[-1]:.2f}, "
                f"Population Intron Ratio = {population_intron_ratio_list[-1]:.4f}, "
                f"Avg Intron Ratio per Individual = {average_intron_ratio_list[-1]:.4f}")
        
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
           
            # Reset the seed for every run
            util.set_seed(random.randint(0, 999999))
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
            
            # Sanity Save of results
            if inbred_threshold == None:
                util.save_accuracy(results, f"{args.config_plot}_inbreeding_RUN:{run}_{gen_success}_MR:{rate}.npy")
            else:
                util.save_accuracy(results, f"{args.config_plot}_no_inbreeding_RUN:{run}_{gen_success}_MR:{rate}.npy")
            
                
            print(f"Population Size {args.pop_size} & Mutation Rate: {rate}: Generation Success {gen_success} ~ Best Fitness {best_fitness_list[-1]:.4f} ~ Best Diversity {diversity_list[-1]:.4f}")

    return results