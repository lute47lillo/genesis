import argparse
import numpy as np
import torch
import os
import pandas as pd
import benchmark_factory as bf
from gp_node import Node
import random
import copy

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
    
    # Fitness Sharing Experiment variables
    argparser.add_argument('--sigma_share', type=float, help='The sharing radius. It determines how far the sharing effect extends.', default=3)
    argparser.add_argument('--sigma_share_weight', type=float, help='Percentage of Diversity as distance metric to  be used to calculate sigma share. ss = ss_weight * ss', default=0.2)
    
    # Dynamic threshold variables
    argparser.add_argument('--slope_threshold', type=int, help='Type of increase of inbreeding threhsold. 1 or -1', default=1)
    argparser.add_argument('--gen_change', type=int, help='Generation X at which changin rate of change of threshold', default=25) 
    argparser.add_argument('--linear_type', type=str, help='Whether the change is abrupt or continuous', default='continuous') 
    
    # Dynamic Mutation Type
    argparser.add_argument('--mutation_type', type=str, help='Whether the mutation subtrees are intron or random', default='random')     
    
    # Semantics experiments
    argparser.add_argument('--semantics_type', type=str, help='Choose the type of semantic crossover used. (SAC, SSC, or None)', default='SAC')    
    argparser.add_argument('--low_sensitivity', type=float, help='Lower Bound sensitivity for similarity. Used in SAC and SSC', default=0.02)
    argparser.add_argument('--high_sensitivity', type=float, help='Higher Bound sensitivity for similarity. Used in SSC', default=8)
    
    # non-HBC experiments
    argparser.add_argument('--random_injection', type=float, help='Population percentage of random individuals to inject in population. AFPO-style', default=0.3)


    
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

def pack_measures_lists(average_size_list, average_depth_list):
    measures_lists = (average_size_list, average_depth_list)
    return measures_lists

def pack_metrics_lists(best_fitness_list, diversity_list):
    metrics_lists = (best_fitness_list, diversity_list)
    return metrics_lists

def subtree_yields_one():
    """
    Returns a subtree guaranteed to yield 1 for all x.
    Useful as an intron in multiplication or division contexts.
    """
    patterns = []
    
    # 1) Direct one
    patterns.append(Node(1.0))
    
    # 2) cos(0)
    patterns.append(Node('cos', [Node(0.0)]))
    
    # 3) Node / Node (identical)
    #   Create a small subtree for the child, then duplicate it.
    
    child = None
    while True:
        candidate = small_random_terminal_subtree()
        # If candidate can evaluate to 0, it might cause division by zero issues
        # But let's allow 'x' or 1.0. If candidate is Node(0.0), let's regenerate.
        if not (isinstance(candidate.value, float) and candidate.value == 0.0):
            child = candidate
            break
    
    child_copy = copy.deepcopy(child)
    patterns.append(Node('/', [child, child_copy]))  # e.g. x / x or 1 / 1
    
    return np.random.choice(patterns)

def subtree_yields_zero():
    """
    Returns a subtree guaranteed to yield 0 for all x.
    Useful as an intron in addition or subtraction contexts.
    """
    patterns = []
    
    # 1) Direct zero
    patterns.append(Node(0.0))
    
    # 2) sin(0)
    patterns.append(Node('sin', [Node(0.0)]))
    
    # 3) log(1.0)
    patterns.append(Node('log', [Node(1.0)]))
    
    # 4) Node - Node (identical)
    #   We'll create a small subtree for the child, then duplicate it.
    child = small_random_terminal_subtree()
    child_copy = copy.deepcopy(child)
    patterns.append(Node('-', [child, child_copy]))  # e.g. x - x
    
    return np.random.choice(patterns)

def small_random_terminal_subtree():
    """
    Returns a Node that is either:
      - Node('x'), or
      - Node(0.0), or
      - Node(1.0).
    """
    choice = np.random.choice(['x', '0', '1'])
    if choice == 'x':
        return Node('x')
    elif choice == '0':
        return Node(0.0)
    else:
        return Node(1.0)

# ---------------------- Diversity in fitness ---------------- #

def compute_min_max_fit(population, max_fitness, min_fitness):
    
    # Get min - max fitness values for normalization
    for individual in population:
        min_fitness = min(min_fitness, individual.fitness)
        max_fitness = max(max_fitness, individual.fitness)
        
    return max_fitness, min_fitness
            
def scale_fitness_values(fitness_individual, max_fitness, min_fitness):
    
    # Avoid division by zero
    if max_fitness == min_fitness:
        fitness_individual = 1.0
    else:
        fitness_individual = (fitness_individual - min_fitness) / (max_fitness - min_fitness)
        
    return fitness_individual
    
# -------------- Plotting and analysis helper functions --------------- #

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
        padding_value = sublist[-1] if sublist else 0
        padding = [padding_value] * (target_length - current_length)
        return sublist + padding
    else:
        return sublist

# Determine Global Maximum Depth
def get_global_max_depth(*results_dicts):
    max_depth = 0
    for results in results_dicts:
        for run in results:
            current_depth = len(results[run]['diversity'])
            if current_depth > max_depth:
                max_depth = current_depth
    return max_depth

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


# ---------------- Result printing and visualization -------------- #
def compute_composite_score_for_eval(sr_dfs, sr, results, suc_w=2, div_w=0, gen_w=-1):
    """
        Definition
        ------------
            Computes and ranks the different hyperparameter combination for a given experiment based off 3 metrics:
                - Total successful runs.
                - Diversity
                - Mean Generation Success (speed of convergence)
                
        Parameters
        -------------
            - sr_dfs (dict): Contains the diversity and best_fitness parameters ready to be accessed per SR function
            - sr (str): The SR function.
            - results (dict): Contains the statistical metrics.
            - suc_w, div_w, gen_w (int): The weigth of each of the metrics to compute the composite score.
    """
# Extract keys and diversity lists
    keys = sr_dfs[sr]['diversity'][2]                # List of keys
    diversity_lists = sr_dfs[sr]['diversity'][1]    # List of diversity lists

    # Prepare data for DataFrame
    data = []
    for i, key in enumerate(keys):
        diversity_list = diversity_lists[i]
        try:
            diversity_value = diversity_list[-1][-1]  # Last element of the last sublist
        except IndexError:
            diversity_value = float('-inf')  # Handle empty lists if necessary
        n_successes = results[key]['n_successes']
        mean_gen_success = np.mean(results[key]['generation_successes'])
        
        data.append({
            'key': key,
            'n_successes': n_successes,
            'diversity': round(diversity_value, 2),
            'mean_gen_success': round(mean_gen_success, 2)
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Define Weights. Playing with these tells me different interpretations. 
    # Example: With 0 diversity, we can observe the pattern of how diversity is spread when the other two are the important things
    # And then for the top-3 best, and top-3 worse we can plot the evolution of diversity to see how it might go up then down.
    weights = {
        'n_successes': suc_w,
        'diversity': div_w,
        'mean_gen_success': gen_w  # Negative weight because lower is better
    }

    # Calculate Composite Score
    df['composite_score'] = (
        df['n_successes'] * weights['n_successes'] +
        df['diversity'] * weights['diversity'] +
        df['mean_gen_success'] * weights['mean_gen_success']
    )
    df['composite_score'] = round(df['composite_score'], 3)

    # Sort DataFrame based on Composite Score
    df_sorted = df.sort_values(
        by='composite_score',
        ascending=False
    ).reset_index(drop=True)

    # Display the sorted DataFrame
    print(f"\nUsing weights -> SucW: {suc_w}. DivW: {div_w}. GenW: {gen_w}")
    print("Sorted Results Based on Composite Score:")
    print(df_sorted[['key', 'n_successes', 'diversity', 'mean_gen_success', 'composite_score']])
    
    return df_sorted
        
