import argparse
import numpy as np
import torch
import os
import pandas as pd
import benchmark_factory as bf
from sympy import symbols, sympify, simplify, expand
import sympy
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def set_args():
    
    # Define arguments
    argparser = argparse.ArgumentParser()
    
    # Global Experimental variables
    argparser.add_argument('--seed', type=int, help='Seed for random', default=99)
    argparser.add_argument('--device', type=str, help='CPU/GPU usage', default="cpu")
    argparser.add_argument('--benchmark', type=str, help='Optimization function to run', default="rastrigin")
    argparser.add_argument('--bench_name', type=str, help='Problem landscape name (ie: MovingPeaksLandscape)', default="none")
    argparser.add_argument('--config_plot', type=str, help='plot info details', default="none")
    argparser.add_argument('--current_run', type=int, help='Current experimental run if multiple runs', default=0)
    
    # Experimental Hyperparameters
    argparser.add_argument('--generations', type=int, help='Nº of generations to run the GA.', default=150)
    argparser.add_argument('--pop_size', type=int, help='Population Size (could be used as fixed parameter in many settings)', default=300)
    argparser.add_argument('--mutation_rate', type=float, help='Mutation Rate (could be used as fixed parameter in many settings)', default=0.0005)
    argparser.add_argument('--inbred_threshold', type=int, help='Inbreeding Threshold. Below threshold is considered inbreeding. \
                            Minimum genetic (tree-edit) distance required to allow mating', default=5)
    argparser.add_argument('--tournament_size', type=int, help='Nº of individuals to take part in the Tournament selection', default=15)
    argparser.add_argument('--exp_num_runs', type=int, help='Nº of experimental runs. (Fixed hyperparameters)', default=15) # intron_fraction
    argparser.add_argument('--intron_fraction', type=float, help='Fraction of the population to compute introns from.', default=1.0) # intron_fraction

    # Genetic Programming variables
    argparser.add_argument('--max_depth', type=int, help='GP Tree maximum depth', default=10)
    argparser.add_argument('--initial_depth', type=int, help='GP Tree maximum depth', default=3) 
    argparser.add_argument('--fitness_weight', type=float, help='Proportional importance weight in the total fitness calculation for abs. error fitness', default=1.0)
    argparser.add_argument('--diversity_weight', type=float, help='Proportional importance weight in the total fitness calculation for diversity', default=0.0) 
    argparser.add_argument('--sigma_share', type=float, help='The sharing radius. It determines how far the sharing effect extends.', default=0.1)
    
    # Parse all arguments
    args = argparser.parse_args()
    
    # Set the seed for reproducibility
    args.seed = random.randint(0, 999999)
    set_seed(args.seed)
    
    return args

# ---------------- Write/Read helper functions --------------------- #

def save_accuracy(array, file_path_temp):
    file_path = f"{os.getcwd()}/saved_data/" + file_path_temp
    mode = 'wb'  # Write mode (overwrite or create if it doesn't exist)
    with open(file_path, mode) as f:
        np.save(f, array)
    print(f"\nAccuracy data saved to {file_path}")

# ---------------- Genetic Programming helper functions --------------------- #

def get_function_bounds(benchmark):
    
    if benchmark == 'nguyen1' or benchmark == 'nguyen2' or benchmark == 'nguyen3' or benchmark == 'nguyen4' or benchmark == 'nguyen5' or benchmark == 'nguyen6':
        bounds = (-4.0, 4.0)
    elif benchmark == 'nguyen7' or benchmark == 'nguyen8':
        bounds = (0.0, 8.0)
    
    return bounds

def select_gp_benchmark(args):
    """
        Definition
        -----------
            Select the benchmark for Genetic Programming experiments.
    """
    benchmarks = {"nguyen1": bf.nguyen1, "nguyen2": bf.nguyen2,
                  "nguyen3": bf.nguyen3, "nguyen4": bf.nguyen4,
                  "nguyen5": bf.nguyen5, "nguyen6": bf.nguyen6,
                  "nguyen7": bf.nguyen7, "nguyen8": bf.nguyen8}
    
    # Get function
    gp_bench_fn = benchmarks.get(args.benchmark)
    
    # Set the specific bounds
    args.bounds = get_function_bounds(args.benchmark)
    
    return gp_bench_fn

# --------------------------- Intron Helper Functions ------------------------------- #

def pack_intron_lists(pop_ration_in, avg_ratio_in, pop_total_in, pop_total_nodes):
    intron_lists = (pop_ration_in, avg_ratio_in, pop_total_in, pop_total_nodes)
    return intron_lists

def pack_kinship_lists(avg_kinship, t_close, t_far):
    kinship_lists = (avg_kinship, t_close, t_far)
    return kinship_lists

def pack_measures_lists(average_size_list, average_depth_list):
    measures_lists = (average_size_list, average_depth_list)
    return measures_lists

def pack_metrics_lists(best_fitness_list, diversity_list):
    metrics_lists = (best_fitness_list, diversity_list)
    return metrics_lists

# ---------------------- Diversity in fitness ---------------- #

def compute_min_max_fit(population, max_fitness, min_fitness):
    
    # Get min - max fitness values for normalization
    for individual in population:
        min_fitness = min(min_fitness, individual.fitness)
        max_fitness = max(max_fitness, individual.fitness)
        
    return max_fitness, min_fitness

def compute_min_max_div(population, max_div, min_div):
    # Compute min - max diversity for normalization.
    for individual in population:
        min_div = min(min_div, individual.diversity)
        max_div = max(max_div, individual.diversity)
        
    return max_div, min_div
            
def scale_fitness_values(fitness_individual, max_fitness, min_fitness):
    
    # Avoid division by zero
    if max_fitness == min_fitness:
        fitness_individual = 1.0
    else:
        fitness_individual = (fitness_individual - min_fitness) / (max_fitness - min_fitness)
        
    return fitness_individual

def scale_diversity_values(diversity_individual, max_div, min_div):
    
    # Avoid division by zero
    if max_div == min_div:
        diversity_individual = 1.0
    else:
        diversity_individual = (diversity_individual - min_div) / (max_div - min_div)
        
    return diversity_individual
    
# -------------- Plotting helper functions --------------- #
        
def create_padded_df(data, metric, run_ids):
    
    # Determine the maximum length for the metric across all runs
    max_length = max(len(run_data[metric]) for run_data in data.values())
    
    # Initialize a DataFrame with NaNs
    df = pd.DataFrame(index=range(max_length), columns=run_ids, dtype=float)
    
    # Populate the DataFrame
    for run_id, run_data in data.items():
        values = run_data[metric]
        df[run_id].iloc[:len(values)] = values
    
    return df

def pad_sublist(sublist, target_length):
    """
    Pads the sublist to the target_length by repeating the last element.
    
    Parameters:
    - sublist (list): The original sublist.
    - target_length (int): The desired length after padding.
    
    Returns:
    - list: The padded sublist.
    """
    current_length = len(sublist)
    if current_length < target_length:
        padding = [sublist[-1]] * (target_length - current_length)
        return sublist + padding
    else:
        return sublist
    
# Flatten the Data
def flatten_results_depths(treatment_name, depths):
    data_df = []
    for depth in depths:
        # Load dict data
        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen2/gp_lambda/PopSize:300_InThres:4_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:3_{treatment_name}.npy"
        data = np.load(file_path_name, allow_pickle=True)
        data_dict = data.item()
        
        # Iterate over and create DataFram
        for run, metrics in data_dict.items():
            # diversity = metrics['diversity']# TODO: For another plto
            generation_success = metrics['generation_success']
            data_df.append({
                'Treatment': treatment_name,
                'Depth': depth,
                'Run': run,
                'Generation_Success': generation_success
            })
    return pd.DataFrame(data_df)

# Flatten the Data
def flatten_results_thresholds(treatment_name, thresholds):
    data_df = []
    for thres in thresholds:
        # Load dict data
        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen2/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:9_InitD:3_{treatment_name}.npy"
        data = np.load(file_path_name, allow_pickle=True)
        data_dict = data.item()
        
        # Iterate over and create DataFram
        for run, metrics in data_dict.items():
            # diversity = metrics['diversity']# TODO: For another plto
            generation_success = metrics['generation_success']
            data_df.append({
                'Treatment': treatment_name,
                'Thresholds': thres,
                'Run': run,
                'Generation_Success': generation_success
            })
    return pd.DataFrame(data_df)

def flatten_results_in_max_depth_diversity(bench_name, treatment_name, thresholds, depths, init_depth):
    data_df = []
    for thres in thresholds:
        for depth in depths:
            # Load dict data
            file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:{init_depth}_{treatment_name}.npy"
            data = np.load(file_path_name, allow_pickle=True)
            data_dict = data.item()
            for run, metrics in data_dict.items():
                # TODO: If Wanted al values needs to padd:
                # padded_diversity = [pad_sublist(sublist, target_length) for sublist in metrics['diversity']]
                # metrics['diversity'] = padded_diversity
                diversity = metrics['diversity'][-1]
                generation_success = metrics['generation_success']
        
                # Update
                data_df.append({
                    'Treatment': treatment_name,
                    'Max_Depth': depth,
                    'Inbred_Threshold': thres,
                    'Run': run,
                    'Generation_Success': generation_success,
                    'Diversity': diversity
                })
    return pd.DataFrame(data_df)

# Determine Global Maximum Depth
def get_global_max_depth(*results_dicts):
    max_depth = 0
    for results in results_dicts:
        for run in results:
            current_depth = len(results[run]['diversity'])
            if current_depth > max_depth:
                max_depth = current_depth
    return max_depth

# Pad 'diversity' Lists
def pad_diversity_lists(results_dict, target_length):
    for run in results_dict:
        original_length = len(results_dict[run]['diversity'])
        padded_diversity = [pad_sublist(sublist, target_length) for sublist in results_dict[run]['diversity']]
        results_dict[run]['diversity'] = padded_diversity
        print(f"Run {run}: Padded Diversity Lengths = {[len(s) for s in results_dict[run]['diversity']]}")
    return results_dict

# Create DF for all attributes for the given dictionary treatment
def pad_dict_and_create_df(results, attributes, global_max_length, n_runs):
    
    data = {}
    for attr in attributes:
        attr_list = [results[run][attr] for run in range(n_runs)]
        # print(len(attr_list))
        
        # Pad up to max length of any run for 1:1 comparison
        attr_padded = [pad_sublist(sublist, global_max_length) for sublist in attr_list]
        for run in range(n_runs):
            results[run][attr] = attr_padded[run]
        
        # Initialize a list to collect data from all runs for this attribute
        attr_data = []
        
        # Iterate through each run
        for run in results:
            # Extract the first 150 elements for the current attribute
            # Convert to NumPy array for efficient computation
            attr_values = np.array(results[run][attr][:150])
            attr_data.append(attr_values)
        
        # Stack the data vertically to create a 2D NumPy array (runs x elements)
        stacked_data = np.vstack(attr_data)  # Shape: (15, 150)
        
        # Compute the mean across the runs (axis=0)
        mean_values = np.mean(stacked_data, axis=0)  # Shape: (150,)
        
        # Store the averaged data in the 'data' dictionary
        data[attr] = mean_values
        
    df = pd.DataFrame(data)
    
    return df
