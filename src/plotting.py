import matplotlib.pyplot as plt
import os 
import numpy as np
from scipy.stats import bootstrap
import util
import pandas as pd

def plot_opt_multiple_runs(args, parameter, ks_list, best_fitness_list, diversity_list, label_list):
    """
        Definition
        -----------
            Plot the Fitness results of inbreeding prevention mechanism against no prevention over different populations sizes
            
        Parameters
        -----------
            - elem_list (List): Contains the hyperparameters over to what the experiment was run. Ex: Population Sizes, Mutation Rates, ...
            - Parameter (str): The Hyperparameter studied. For plotting and reference 
    """
    # Plot Best Fitness Comparison
    plt.figure(figsize=(16, 9))
    linestyles = ['-', ':', '-', ':'] # One for each methods

    # Best Fitness Over Generations 
    plt.subplot(1, 2, 1)
    for idx, ks in enumerate(ks_list):
        plt.plot(ks, best_fitness_list[idx][0], label=label_list[idx], linestyle=linestyles[idx])
        plt.fill_between(ks, best_fitness_list[idx][1], best_fitness_list[idx][2], alpha=0.5)
    
    # Plot the global_optimum_fitness_list per generation
    plt.title('Best Fitness vs. Global Optimum Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Genetic Diversity Over Generations and the collapses
    plt.subplot(1, 2, 2)
    for idx, ks in enumerate(ks_list):
        plt.plot(ks, diversity_list[idx][0], label=label_list[idx])
        plt.fill_between(ks, diversity_list[idx][1], diversity_list[idx][2], alpha=0.5)
    
    plt.title('Allelic Diversity Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"{os.getcwd()}/figures/{parameter}/{args.benchmark}/{args.config_plot}.png")
    plt.close()

def plot_opt_fn_parameters(args, parameter, elem_list, results_inbreeding, results_no_inbreeding):
    """
        Definition
        -----------
            Plot the Fitness results of inbreeding prevention mechanism against no prevention over different populations sizes
            
        Parameters
        -----------
            - elem_list (List): Contains the hyperparameters over to what the experiment was run. Ex: Population Sizes, Mutation Rates, ...
            - Parameter (str): The Hyperparameter studied. For plotting and reference 
    """

    # Create a figure and two subplots
    fig, axes = plt.subplots(1, 2, figsize=(32, 14))

    # First subplot: Best Fitness over Generations
    ax1 = axes[0]
    for element in elem_list:
        ax1.plot(
            results_inbreeding[element]['best_fitness'],
            label=f'Inbreeding, {parameter} {element}'
        )
        ax1.plot(
            results_no_inbreeding[element]['best_fitness'],
            label=f'No Inbreeding, {parameter} {element}',
            linestyle='--'
        )
    ax1.set_title('Best Fitness over Generations')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.grid(True)

    # Place the legend below the first subplot
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize='large')

    # Second subplot: Genetic Diversity over Generations
    ax2 = axes[1]
    for element in elem_list:
        ax2.plot(
            results_inbreeding[element]['diversity'],
            label=f'Inbreeding, {parameter} {element}'
        )
        ax2.plot(
            results_no_inbreeding[element]['diversity'],
            label=f'No Inbreeding, {parameter} {element}',
            linestyle='--'
        )
    ax2.set_title('Genetic Diversity over Generations')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Diversity')
    ax2.grid(True)

    # Place the legend below the second subplot
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize='large')

    # Adjust layout to make space for the legends
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.3)  # Increase bottom margin to accommodate legends

    # Save the figure with tight bounding box to include legends
    plt.savefig(
        f"{os.getcwd()}/figures/{parameter}/{args.benchmark}/{args.config_plot}.png",
        bbox_inches='tight'
    )
    plt.close()

    
def plot_individual_MPL_global_optima(args, best_fitness_list, diversity_list, global_optimum_fitness_list, collapse_events, collapse_threshold):
    
    # Plotting Results
    plt.figure(figsize=(14, 6))

    # Best Fitness Over Generations
    plt.subplot(1, 2, 1)
    plt.plot(best_fitness_list, label='Best Fitness')
    plt.plot(global_optimum_fitness_list, label='Global Optimum Fitness', linestyle='--')
    plt.title('Best Fitness vs. Global Optimum Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Genetic Diversity Over Generations
    plt.subplot(1, 2, 2)
    plt.plot(diversity_list, label='Allelic Diversity', color='orange')
    plt.axhline(y=collapse_threshold, color='red', linestyle='--', label='Collapse Threshold')
    
    # Mark collapse events
    for event in collapse_events:
        plt.axvline(x=event, color='red', linestyle='--', label='Collapse Event' if event == collapse_events[0] else '')
    plt.title('Allelic Diversity Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.legend()
    

    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}.png")
    plt.close()
    
def plot_multiple_runs_MPL_global_optima(args, ks_list, best_fitness_list, diversity_list, label_list, collapse_threshold=0.2):
    
    # Plotting Results
    plt.figure(figsize=(14, 6))
    linestyles = ['-', ':', '-', ':'] # One for each methods

    # Best Fitness Over Generations 
    plt.subplot(1, 2, 1)
    for idx, ks in enumerate(ks_list):
        plt.plot(ks, best_fitness_list[idx][0], label=label_list[idx], linestyle=linestyles[idx])
        plt.fill_between(ks, best_fitness_list[idx][1], best_fitness_list[idx][2], alpha=0.5)
    
    # Plot the global_optimum_fitness_list per generation
    plt.title('Best Fitness vs. Global Optimum Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Genetic Diversity Over Generations and the collapses
    label_list = label_list[:2] # Trim the Global Optimums
    ks_list = ks_list[:2]
    plt.subplot(1, 2, 2)
    for idx, ks in enumerate(ks_list):
        plt.plot(ks, diversity_list[idx][0], label=label_list[idx])
        plt.fill_between(ks, diversity_list[idx][1], diversity_list[idx][2], alpha=0.5)
    
    plt.axhline(y=collapse_threshold, color='red', linestyle='--', label='Collapse Threshold')
    plt.title('Allelic Diversity Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}.png")
    plt.close()
    
def plot_multiple_runs_GP_functions(args,  ks_list, best_fitness_list, diversity_list, label_list,):
    # Plotting Results
    plt.figure(figsize=(14, 6))
    linestyles = ['-', ':', '-', ':'] # One for each methods

    # Best Fitness Over Generations 
    plt.subplot(1, 2, 1)
    for idx, ks in enumerate(ks_list):
        plt.plot(ks, best_fitness_list[idx][0], label=label_list[idx], linestyle=linestyles[idx])
        plt.fill_between(ks, best_fitness_list[idx][1], best_fitness_list[idx][2], alpha=0.5)
    
    # Plot the global_optimum_fitness_list per generation
    plt.title('Best Fitness vs. Global Optimum Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Genetic Diversity Over Generations and the collapses
    plt.subplot(1, 2, 2)
    for idx, ks in enumerate(ks_list):
        plt.plot(ks, diversity_list[idx][0], label=label_list[idx])
        plt.fill_between(ks, diversity_list[idx][1], diversity_list[idx][2], alpha=0.5)
    
    plt.title('Allelic Diversity Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}.png")
    plt.close()


# ---------------------- Bootstrapping methods ---------------------------------- #

def collect_bootstrapping_data(args, results_no_inbreeding, results_inbreeding):
    """
        Definition
        -----------
            Helper method that collects the data from the results dictionary of the GA and sets it up for plotting.
            
            TODO: Currently only tested to work when executing multiple_run_experiment fn.
    """
    
    # Create trackers for metrics
    gs_list = []
    fit_list = []
    div_list = []
    label_list = []
    
    # Collect for No Inbreeding
    g_noI, fit_noI = plot_mean_and_bootstrapped_ci(results_no_inbreeding, key='best_fitness')
    _, div_noI = plot_mean_and_bootstrapped_ci(results_no_inbreeding, key='diversity')
    gs_list.append(g_noI)
    fit_list.append(fit_noI)
    div_list.append(div_noI)
    label_list.append("No Inbreeding")

    # Collect for Inbreeding
    g_I, fit_I = plot_mean_and_bootstrapped_ci(results_inbreeding, key='best_fitness')
    _, div_I = plot_mean_and_bootstrapped_ci(results_inbreeding, key='diversity')
    gs_list.append(g_I)
    fit_list.append(fit_I)
    div_list.append(div_I)
    label_list.append("Inbreeding")
    
    # Collect for Global Optimum Fitness
    if args.bench_name == 'MovingPeaksLandscape':
        
        # Global Optimums of fitness
        gen_noI, gopt_noI = plot_mean_and_bootstrapped_ci(results_no_inbreeding, key='global_optimum')
        gs_list.append(gen_noI)
        fit_list.append(gopt_noI)
        label_list.append("Global Optimum Fitness NO Inbreeding")
        
        gen_I, gopt_I = plot_mean_and_bootstrapped_ci(results_inbreeding, key='global_optimum')
        gs_list.append(gen_I)
        fit_list.append(gopt_I)
        label_list.append("Global Optimum Fitness Inbreeding")
        
        # # Genetic Collapse
        # gen_col_no, col_no = plot_mean_and_bootstrapped_ci(results_no_inbreeding, key='collapse_events')
        # gs_list.append(gen_col_no)
        # div_list.append(col_no)
        # label_list.append("Genetic Collapse NO Inbreeding")
        
        # gen_col_yes, col_yes = plot_mean_and_bootstrapped_ci(results_inbreeding, key='collapse_events')
        # gs_list.append(gen_col_yes)
        # div_list.append(col_yes)
        # label_list.append("Genetic Collapse Inbreeding")
    
    return gs_list, fit_list, div_list, label_list

# Function to calculate bootstrapped confidence intervals
def compute_bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for 1D data array."""
    data = np.asarray(data)
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper

def plot_mean_and_bootstrapped_ci(experimental_results, key):
    """
    Plot mean fitness over generations with bootstrapped confidence intervals.

    Parameters: 
    experimental_results: (dict or list) Contains 'fitness_over_gens' for each run.
    title: (string) Title of the plot.
    x_label: (string) X-axis label.
    y_label: (string) Y-axis label.
    plot: (bool) Whether to display the plot.

    Returns:
    generations: (numpy array) Array of generation indices.
    stats: (tuple) Tuple containing mean fitness, lower CI, and upper CI.
    """
    # Determine number of runs and generations
    num_runs = len(experimental_results)
    
    # Assuming all runs have the same number of generations
    num_gens = len(experimental_results[0][key])
    generations = np.arange(num_gens)
    
    # Collect fitness data into a 2D array of shape (num_runs, num_gens)
    fitness_data = np.zeros((num_runs, num_gens))
    
    for k in range(num_runs):
        fitness_data[k, :] = experimental_results[k][key]
    
    # Initialize lists to store mean fitness and confidence intervals over generations
    mean_fitness = []
    fit_ci_low = []
    fit_ci_high = []
    
    # Loop over each generation
    for gen in range(num_gens):
        # Extract fitness values at this generation across runs
        fitness_values = fitness_data[:, gen]
        
        # Compute mean fitness at this generation
        mean_fit = np.mean(fitness_values)
        
        # Compute bootstrap confidence intervals
        fit_ci_l, fit_ci_h = compute_bootstrap_ci(fitness_values)
        
        mean_fitness.append(mean_fit)
        fit_ci_low.append(fit_ci_l)
        fit_ci_high.append(fit_ci_h)
    
    return generations, (mean_fitness, fit_ci_low, fit_ci_high)

# ------------------ Bootstrapping Irregular lengths GP ---------------------- #
def compute_bootstrap_ci(df, metric_name, confidence_level=0.95, n_bootstraps=1000):
    """
    Compute bootstrap confidence intervals for each time step.

    Parameters:
    - df: pandas DataFrame containing the metric data with runs as columns.
    - metric_name: Name of the metric (for labeling purposes).
    - confidence_level: Confidence level for the intervals.
    - n_bootstraps: Number of bootstrap resamples.

    Returns:
    - summary_df: DataFrame containing mean, lower CI, and upper CI for each time step.
    """
    means = []
    cis_lower = []
    cis_upper = []
    indices = df.index

    for idx in indices:
        # Extract available data at this time step, ignoring NaNs
        data_at_step = df.iloc[idx].dropna().values

        if len(data_at_step) < 2:
            # Not enough data to perform bootstrap
            if len(data_at_step) == 1:
                mean = data_at_step[0]
                cis_low = np.nan
                cis_high = np.nan
                print(f"Time Step {idx}: Only one observation available. Confidence intervals set to NaN.")
            else:
                mean = np.nan
                cis_low = np.nan
                cis_high = np.nan
                print(f"Time Step {idx}: No observations available. Mean and confidence intervals set to NaN.")
                
            means.append(mean)
            cis_lower.append(cis_low)
            cis_upper.append(cis_high)
            continue

        try:
            # Perform bootstrap
            res = bootstrap(
                (data_at_step,),
                np.mean,
                confidence_level=confidence_level,
                n_resamples=n_bootstraps,
                method='percentile'
            )

            # Compute the mean of the actual data
            mean = np.mean(data_at_step)

            # Extract confidence interval
            cis_lower.append(res.confidence_interval.low)
            cis_upper.append(res.confidence_interval.high)
            means.append(mean)

        except Exception as e:
            # Handle unexpected exceptions
            print(f"Time Step {idx}: Bootstrap failed with error: {e}. Mean and confidence intervals set to NaN.")
            
            means.append(np.nan)
            cis_lower.append(np.nan)
            cis_upper.append(np.nan)

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'mean': means,
        'ci_lower': cis_lower,
        'ci_upper': cis_upper
    }, index=indices)

    summary_df.index.name = 'Time Step'
    return summary_df

def plot_all(args, ks_list, fit_list, label_list, x_label="Generations", y_label="N", title='Comparisons'):

    plt.figure(figsize=(20, 10))
    for idx, ks in enumerate(ks_list):
        plt.plot(ks, fit_list[idx][0], label=label_list[idx])
        plt.fill_between(ks, fit_list[idx][1], fit_list[idx][2], alpha=0.5, label='95% CI')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}: {y_label} vs. {x_label}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}_{y_label}.png")
    plt.close()
    
def plot_gen_vs_run(args, results_no_inbreeding, results_inbreeding):
    
    # Create a figure
    plt.figure(figsize=(20, 10))

    # Get number of runs
    n_runs = args #args.exp_num_runs
    run_numbers = np.arange(1, n_runs + 1)

    # Colect the generations
    generation_success_inbreeding    = [results_inbreeding[run]['generation_success'] for run in range(n_runs)]
    generation_success_no_inbreeding = [results_no_inbreeding[run]['generation_success'] for run in range(n_runs)]
    
    # Plot scatter
    plt.plot(run_numbers, generation_success_inbreeding, marker='o', linestyle='-', color='blue', label='Inbreeding')
    plt.plot(run_numbers, generation_success_no_inbreeding, marker='x', linestyle='--', color='red', label='NO Inbreeding')
    
    # Show grid
    plt.title('Successful generation over Experimental Runs')
    plt.xlabel('Runs')
    plt.xticks(run_numbers)
    plt.ylabel('Generations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/0.01TESTIN100pop.png")
    # plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}.png")
    
def plot_runs_with_ci(df, summary_df, metric_name, color='blue'):
    plt.figure(figsize=(10, 6))
    
    # Plot individual runs
    for run_id in df.columns:
        plt.plot(df.index, df[run_id], marker='o', linestyle='--', alpha=0.5, label=f'Run {run_id}')
    
    # Plot mean and confidence intervals
    plt.plot(summary_df.index, summary_df['mean'], label=f'Mean {metric_name}', color=color, linewidth=2)
    plt.fill_between(summary_df.index, summary_df['ci_lower'], summary_df['ci_upper'], color=color, alpha=0.3, label='95% CI')
    
    plt.xlabel('Time Step')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Time with Runs and 95% Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{os.getcwd()}/{metric_name}_test.png")
    plt.close()
    
def plot_generation_successes(results, mutation_rates, plot_save_title):
    """
    Plots the generation successes for each mutation rate over the runs.

    :param results: Dictionary containing results from multiple_mrates_function_gp function.
    :param mutation_rates: List of mutation rates used in the experiments.
    """
    num_runs = len(next(iter(results.values()))['generation_successes'])  # Get the number of runs

    plt.figure(figsize=(12, 6))

    for rate in mutation_rates:
        generation_successes = results[rate]['generation_successes']  # List of gen_success per run
        generation_successes = sorted(generation_successes)
        gen_suc_mean = np.mean(generation_successes)
        print(f"Mutatio Rate: {rate}. Mean Generation success: {gen_suc_mean}")
        runs = range(1, num_runs + 1)
        plt.plot(runs, generation_successes, marker='o', label=f'Mutation Rate {rate}')

    plt.xlabel('Run Number')
    plt.ylabel('Generation of Success')
    plt.title('Generation Successes Across Runs for Different Mutation Rates')
    plt.legend(title='Mutation Rates')
    plt.xticks(runs)  # Ensure all run numbers are shown on the x-axis
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{plot_save_title}")
    plt.show()
    

if __name__ == "__main__":
    
    # expression = "(- (+ (* (* x x) (+ 1.0 x)) (+ (- 1.0 1.0) (+ (/ 1.0 1.0) (+ x x)))) (+ 1.0 x))"
 
    # print("Indented Tree Structure:")
    # expr = util.convert_tree_to_expression(expression)
    # print(expr)
    
    print("\nNO Inbreeding")
    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/mut_rates/Mrates:[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]_PopSize:100_InThres:4_Gens:150_TourSize:10_MaxD:8_InitD:3_no_inbreeding.npy"
    data = np.load(file_path_name, allow_pickle=True)
    data_dict_no = data.item()
    mutation_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    plot_generation_successes(data_dict_no, mutation_rates, "TEST_noInbreeding")
    
    print("\nInbreeding")
    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/mut_rates/Mrates:[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]_PopSize:100_InThres:4_Gens:150_TourSize:10_MaxD:8_InitD:3_inbreeding.npy"
    data = np.load(file_path_name, allow_pickle=True)
    data_dict_no = data.item()
    mutation_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    plot_generation_successes(data_dict_no, mutation_rates, "TEST_Inbreeding")
    
    
    # print("NO Inbreeding")
    # # config_plot = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/experimental/PopSize:100_InThres:5_Mrates:0.01_Gens:100_TourSize:10_MaxD:8_InitD:2_no_inbreeding.npy" 

    # data = np.load(config_plot, allow_pickle=True)
    
    # # print(data)
    # data_dict_no = data.item()
    
    # # for key, value in data_dict.items():
    # #     for list_key, values in value.items():
    # #         temp_v = np.array(values)
    # #         print(key, list_key, temp_v.shape)
    # #         print(temp_v)
            
    # # # Create DataFrames
    # # run_ids = data_dict.keys()
    # # best_fitness_df = util.create_padded_df(data_dict, 'best_fitness', run_ids)
    # # diversity_df = util.create_padded_df(data_dict, 'diversity', run_ids)

    # # print("Best Fitness DataFrame:")
    # # print(best_fitness_df)
    # # print("\nDiversity DataFrame:")
    # # print(diversity_df)
    
    # # # Compute bootstrap confidence intervals
    # # best_fitness_ci = compute_bootstrap_ci(best_fitness_df, 'Best Fitness')
    # # diversity_ci = compute_bootstrap_ci(diversity_df, 'Diversity')

    # # print("Best Fitness Confidence Intervals:")
    # # print(best_fitness_ci)
    # # print("\nDiversity Confidence Intervals:")
    # # print(diversity_ci)
    
    # # # Plot Best Fitness with Individual Runs
    # # plot_runs_with_ci(best_fitness_df, best_fitness_ci, 'Best Fitness', color='green')

    # # # Plot Diversity with Individual Runs
    # # plot_runs_with_ci(diversity_df, diversity_ci, 'Diversity', color='orange')
            
            
    # print("Inbreeding")
    # config_plot = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/experimental/PopSize:100_InThres:5_Mrates:0.01_Gens:100_TourSize:10_MaxD:8_InitD:2_inbreeding.npy" 

    # data = np.load(config_plot, allow_pickle=True)
    
    # # print(data)
    # data_dict = data.item()
    
    # for key, value in data_dict.items():
    #     for list_key, values in value.items():
    #         temp_v = np.array(values)
    #         print(key, list_key, temp_v.shape)
            
    # plot_gen_vs_run(5, data_dict_no, data_dict)