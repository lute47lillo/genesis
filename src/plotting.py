import matplotlib.pyplot as plt
import os 
import numpy as np
from scipy.stats import bootstrap
import util
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
    # plt.axhline(y=collapse_threshold, color='red', linestyle='--', label='Collapse Threshold') # TODO: Not being used currently
    
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
    plt.figure(figsize=(24, 12))
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
    
    # plt.axhline(y=collapse_threshold, color='red', linestyle='--', label='Collapse Threshold') # TODO: Not being used currently
    plt.title('Allelic Diversity Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}.png")
    plt.close()
    
def plot_multiple_parameters_MPL(args, params_list, gs_list, best_fitness_list, diversity_list, label_list):
    
    # Plotting Results
    plt.figure(figsize=(24, 12))
    linestyles = ['-', ':', '-.'] # One for each methods

    # Best Fitness Over Generations 
    plt.subplot(1, 2, 1)
    for param in params_list:
        for idx, ks in enumerate(gs_list):
            print(f"\nParameter: {param} and idx: {idx}. Mean Fitness: {np.mean(best_fitness_list[idx][param]['mean_fitness'])}")
            plt.plot(ks, best_fitness_list[idx][param]['mean_fitness'], label=label_list[idx] + f' ~ MR: {param}')
            plt.fill_between(ks, best_fitness_list[idx][param]['ci_low'], best_fitness_list[idx][param]['ci_high'], alpha=0.5)
    
    # Plot the global_optimum_fitness_list per generation
    plt.title('Best Fitness vs. Global Optimum Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Genetic Diversity Over Generations and the collapses
    trim_gopts = 2 + len(params_list) - 1
    label_list = label_list[:trim_gopts] # Trim the Global Optimums
    gs_list = gs_list[:2]
    plt.subplot(1, 2, 2)
    for param in params_list:
        for idx, ks in enumerate(gs_list):
            plt.plot(ks, diversity_list[idx][param]['mean_fitness'], label=label_list[idx] + f' ~ MR: {param}')
            plt.fill_between(ks, diversity_list[idx][param]['ci_low'], diversity_list[idx][param]['ci_high'], alpha=0.5)
    
    plt.title('Allelic Diversity Over Generations (MPB)')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}_TEEEEMP.png")
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
        label_list.append("Global Optimum Fitness")
        
        # Only used if different global optimums per experiment
        # gen_I, gopt_I = plot_mean_and_bootstrapped_ci(results_inbreeding, key='global_optimum')
        # gs_list.append(gen_I)
        # fit_list.append(gopt_I)
        # label_list.append("Global Optimum Fitness Inbreeding")
        
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
def compute_bootstrap_ci_rugged(data, n_bootstrap=1000, ci=0.95):
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
        fit_ci_l, fit_ci_h = compute_bootstrap_ci_rugged(fitness_values)
        
        mean_fitness.append(mean_fit)
        fit_ci_low.append(fit_ci_l)
        fit_ci_high.append(fit_ci_h)
    
    return generations, (mean_fitness, fit_ci_low, fit_ci_high)

# ---------------------- Various Hyperparameters * Runs Bootstrapping functions ----------------- #

def compute_bootstrap_ci_multiple_params(data, confidence=0.95, n_bootstraps=1000):
    """
    Compute bootstrapped confidence intervals for the mean of the data.

    Parameters:
        data (array-like): Data to compute the confidence interval for.
        confidence (float): Confidence level.
        n_bootstraps (int): Number of bootstrap samples.

    Returns:
        (float, float): Lower and upper bounds of the confidence interval.
    """
    data = np.asarray(data)
    if data.size == 0:
        return np.nan, np.nan

    res = bootstrap(
        (data,),
        np.mean,
        confidence_level=confidence,
        n_resamples=n_bootstraps,
        method='percentile'
    )
    return res.confidence_interval.low, res.confidence_interval.high

def plot_mean_and_bootstrapped_ci_multiple_parameters(args, parameters_list, experimental_results, treatment, key='best_fitness'):
    """
    Plot mean metric over generations with bootstrapped confidence intervals for multiple parameters.

    Parameters: 
        parameters_list (list): List of parameter values (e.g., mutation rates) to plot.
        experimental_results (dict): Nested dictionary containing experimental data.
        key (str): The key to extract from the innermost dictionaries (e.g., 'best_fitness').

    Returns:
        None
    """
    if len(set(parameters_list)) != len(parameters_list):
        print("Warning: Duplicate parameters detected in parameters_list.")

    print("Parameters List:", parameters_list)
    print("Number of unique parameters:", len(set(parameters_list)))

    num_runs = len(experimental_results)
    num_params = len(parameters_list)
    
    if num_runs == 0 or num_params == 0:
        print("No runs or parameters to plot.")
        return {}
    
    sample_run = next(iter(experimental_results.values()))
    sample_param = next(iter(sample_run.keys()))
    num_gens = len(sample_run[sample_param][key])
    generations = np.arange(num_gens)
    
    # print(f"Number of runs: {num_runs}")
    # print(f"Parameters (e.g., mutation rates): {parameters_list}")
    # print(f"Number of generations: {num_gens}")
    
    # Initialize fitness_data
    fitness_data = {param: np.zeros((num_runs, num_gens)) for param in parameters_list}
    
    # Populate fitness_data with debug prints
    for run_idx, run in experimental_results.items():
        for param in parameters_list:
            if param not in run:
                raise ValueError(f"Parameter {param} not found in run {run_idx}.")
            fitness_values = run[param][key]
            if len(fitness_values) != num_gens:
                raise ValueError(f"Inconsistent number of generations in run {run_idx}, parameter {param}.")
            fitness_data[param][run_idx, :] = fitness_values
            print(f"Run {run_idx}, Param {param}, Fitness Values (first 5): {fitness_values[:5]}")
    
    
    # Initialize plot
    plt.figure(figsize=(12, 8))
    
    # Colors for different parameters
    colors = plt.cm.viridis(np.linspace(0, 1, num_params))
    
    # Dictionary to store bootstrap results
    param_bootstrap = {}
    
    # Plot mean and confidence intervals for each parameter
    for idx, param in enumerate(parameters_list):
        param_fitness = fitness_data[param]  # Shape: (num_runs, num_gens)
        
        mean_fit = np.mean(param_fitness, axis=0)
        ci_lower = np.zeros(num_gens)
        ci_upper = np.zeros(num_gens)
        
        print(f"\nProcessing Parameter: {param}")
        print(f"Mean Fitness (first 5): {mean_fit[:5]}")
        
        for gen in range(num_gens):
            gen_fitness = param_fitness[:, gen]
            
            # Compute bootstrap confidence intervals
            fit_ci_l, fit_ci_h = compute_bootstrap_ci_multiple_params(gen_fitness)
            ci_lower[gen] = fit_ci_l
            ci_upper[gen] = fit_ci_h
            
            if gen < 5:  # Print first few for debugging
                print(f"Generation {gen}: CI_low={fit_ci_l}, CI_high={fit_ci_h}")
        
        param_bootstrap[param] = {
            'mean_fitness': mean_fit,
            'ci_low': ci_lower,
            'ci_high': ci_upper
        }
        
        # Plotting
        plt.plot(generations, mean_fit, label=f'Param: {param}', color=colors[idx], linestyle='-', linewidth=2)
        plt.fill_between(generations, ci_lower, ci_upper, color=colors[idx], alpha=0.3)
    
    plt.title(f'{treatment} ~ Mean {key.replace("_", " ").title()} Over Generations with 95% CI')
    plt.xlabel('Generation')
    plt.ylabel(f'{key.replace("_", " ").title()}')
    plt.legend(title='Parameters')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}_{treatment}.png")
    plt.close()


# ------------------ Bootstrapping Irregular lengths GP ---------------------- #

def compute_bootstrap_ci(df, confidence_level=0.95, n_bootstraps=1000):
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
    
def autolabel(bars):
    """
        Definition
        ------------
            Attach a text label above each bar displaying its height.
    """
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', fontsize=8
        )
        
def plot_gen_vs_run(args, results_no_inbreeding, results_inbreeding):
    """
        Definition
        -----------
            For a given symbolic regression problem in Genetic Programming plot the generation of success for each run.
            Also, plot the final diversity value of each run.
            
            TODO: Could plot how the diversity over time changes. 
    """
    
    # Create a figure
    plt.figure(figsize=(28, 12))

    # Get number of runs
    n_runs = args.exp_num_runs
    run_numbers = np.arange(1, n_runs + 1)
    
    # Colect the generations
    generation_success_inbreeding    = [results_inbreeding[run]['generation_success'] for run in range(n_runs)]
    generation_success_no_inbreeding = [results_no_inbreeding[run]['generation_success'] for run in range(n_runs)]
    
    # Collect the final diversity values for both treatments
    diversity_inbreeding = [results_inbreeding[run]['diversity'][-1] for run in range(n_runs)]
    diversity_no_inbreeding = [results_no_inbreeding[run]['diversity'][-1] for run in range(n_runs)]
    
    # Zip together to sort
    paired_inbreeding = list(zip(generation_success_inbreeding, diversity_inbreeding))
    paired_no_inbreeding = list(zip(generation_success_no_inbreeding, diversity_no_inbreeding))
    
    sorted_paired_inbreeding = sorted(paired_inbreeding, key=lambda x: x[0])
    sorted_paired_no_inbreeding = sorted(paired_no_inbreeding, key=lambda x: x[0])
    
    # Unzip
    generation_success_inbreeding, diversity_inbreeding = zip(*sorted_paired_inbreeding)
    generation_success_no_inbreeding, diversity_no_inbreeding = zip(*sorted_paired_no_inbreeding)
    
    # NEED TO KEEP for final at gen is best
    # ------
    # Colect the generations
    # generation_success_inbreeding    = [results_inbreeding[run]['generation_success'] for run in range(n_runs)]
    # generation_success_no_inbreeding = [results_no_inbreeding[run]['generation_success'] for run in range(n_runs)]
    
    # diversity_inbreeding = [results_inbreeding[run]['diversity'] for run in range(n_runs)]
    # diversity_no_inbreeding = [results_no_inbreeding[run]['diversity'] for run in range(n_runs)]
    
    # paired_inbreeding = list(zip(generation_success_inbreeding, diversity_inbreeding))
    # paired_no_inbreeding = list(zip(generation_success_no_inbreeding, diversity_no_inbreeding))
    
    # sorted_paired_inbreeding = sorted(paired_inbreeding, key=lambda x: x[0])
    # sorted_paired_no_inbreeding = sorted(paired_no_inbreeding, key=lambda x: x[0])
    
    # # Unzip
    # generation_success_inbreeding, diversity_inbreeding = zip(*sorted_paired_inbreeding)
    # generation_success_no_inbreeding, diversity_no_inbreeding = zip(*sorted_paired_no_inbreeding)
    
    # print(generation_success_no_inbreeding)
    # print(generation_success_inbreeding)
    # print(generation_success_no_inbreeding[n_runs-1])
    # print(diversity_inbreeding[0][generation_success_no_inbreeding[0]])
    # diversity_inbreeding = [diversity_inbreeding[idx][generation_success_no_inbreeding[idx]-1] for idx in range(n_runs)]
    # diversity_no_inbreeding = [diversity_no_inbreeding[idx][-1] for idx in range(n_runs)]
    

    # ----------------------------
    # Subplot 1: Generation Success
    # ----------------------------
    plt.subplot(1, 2, 1)
    
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
    
    # ----------------------------
    # Subplot 2: Final Diversity
    # ----------------------------
    plt.subplot(1, 2, 2)
    
    # Define the width of each bar
    bar_width = 0.35  # Adjust as needed
    
    # Calculate y-axis limits with epsilon
    combined_min = min(min(diversity_inbreeding), min(diversity_no_inbreeding)) - 5
    combined_max = max(max(diversity_inbreeding), max(diversity_no_inbreeding)) + 5
    
    # Plot bars for Inbreeding
    bars_a = plt.bar(run_numbers - bar_width/2, diversity_inbreeding, bar_width,  label='Inbreeding', color='skyblue') #skyblue
    
    # Plot bars for No Inbreeding
    # bars_b = plt.bar(run_numbers + bar_width/2, diversity_inbreeding_1, bar_width, label='Inbreeding Final', color='red') # salmon
    bars_b = plt.bar(run_numbers + bar_width/2, diversity_no_inbreeding, bar_width, label='NO Inbreeding', color='salmon') # salmon

    
    # Customize the subplot
    plt.xlabel('Experimental Run')
    plt.ylabel('Final Diversity')
    plt.title('Comparison of Diversity: Inbreeding vs No Inbreeding')
    # plt.title('Comparison of Diversity: Inbreeding (at no-Inbreeding Gen solution) vs Inbreeding Final')
    plt.xticks(run_numbers)
    plt.ylim(combined_min, combined_max)
    plt.legend()
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    
    autolabel(bars_a)
    autolabel(bars_b)
    
    # Close
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}_keep_last.png")
    # plt.savefig(f"{os.getcwd()}/figures/genetic_programming/nguyen2/meeting/PopSize:300_InThres:5_Mrates:0.0005_Gens:150_TourSize:15_MaxD:9_InitD:3_keep_last.png")
    plt.close()
    
def plot_time_of_convergence_vs_diversity(args, results_no_inbreeding, results_inbreeding, temp_runs=15):
    """
        Definition
        -----------
            For a given symbolic regression problem in Genetic Programming plot the generation of success for each run.
            Also, plot the final diversity value of each run.
            
            TODO: Could plot how the diversity over time changes. 
    """
    
    # Create a figure
    plt.figure(figsize=(16, 12))
    
    # Get number of runs
    n_runs = args.exp_num_runs
    
    # For testing plotting
    # n_runs = temp_runs
    
    # Colect the generations
    generation_success_inbreeding    = [results_inbreeding[run]['generation_success'] for run in range(n_runs)]
    generation_success_no_inbreeding = [results_no_inbreeding[run]['generation_success'] for run in range(n_runs)]
    
    # Collect the final diversity values for both treatments
    diversity_inbreeding = [results_inbreeding[run]['diversity'][-1] for run in range(n_runs)]
    diversity_no_inbreeding = [results_no_inbreeding[run]['diversity'][-1] for run in range(n_runs)]
    
    combined_min = min(min(diversity_inbreeding), min(diversity_no_inbreeding)) + 1
    combined_max = max(max(diversity_inbreeding), max(diversity_no_inbreeding)) + 1
    
    plt.scatter(generation_success_inbreeding, diversity_inbreeding, marker='o', color='blue', label='Inbreeding')
    plt.scatter(generation_success_no_inbreeding, diversity_no_inbreeding, marker='x', color='red', label='NO Inbreeding')
    
    plt.xlabel('Time to Convergence (Generation of Solution)')
    plt.ylabel('Final Diversity')
    plt.title('Comparison of Final Diversity by time of Convergence: Inbreeding vs No Inbreeding')
    plt.ylim(combined_min, combined_max)
    plt.legend()
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    plt.tight_layout()
    # plt.savefig(f"{os.getcwd()}/figures/div_time_TEST.png")
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}_diversity_vs_time_convergence.png")
    plt.close()
    
def plot_diversity_generation_over_time(args, results_no_inbreeding, results_inbreeding, temp_runs=15, temp_gens=150):
    
    # Create a figure
    plt.figure(figsize=(20, 12))
    
    # Get number of runs
    n_runs = args.exp_num_runs
    n_gens = args.generations
    
    # For testing plotting
    # n_runs = temp_runs
    # n_gens = temp_gens
    
    gs_list = []
    div_list = []
    label_list = []
    
    # Get diversity
    diversity_inbreeding = [results_inbreeding[run]['diversity'] for run in range(n_runs)]
    diversity_no_inbreeding = [results_no_inbreeding[run]['diversity'] for run in range(n_runs)]
    
    # Find maximum length list and padd the rest to be final diversity at point of convergence
    max_length_inbreeding = max(len(sublist) for sublist in diversity_inbreeding)
    max_length_no_inbreeding = max(len(sublist) for sublist in diversity_no_inbreeding)
    global_max_length = max(max_length_inbreeding, max_length_no_inbreeding)
    
    # Pad all sublists in diversity_inbreeding
    diversity_inbreeding_padded = [util.pad_sublist(sublist, global_max_length) for sublist in diversity_inbreeding]

    # Pad all sublists in diversity_no_inbreeding
    diversity_no_inbreeding_padded = [util.pad_sublist(sublist, global_max_length) for sublist in diversity_no_inbreeding]

    # Update results_inbreeding with padded diversity lists
    for run in range(n_runs):
        original_length = len(results_inbreeding[run]['diversity'])
        results_inbreeding[run]['diversity'] = diversity_inbreeding_padded[run]
        # print(f"Run {run} Inbreeding: Original Length = {original_length}, Padded Length = {len(results_inbreeding[run]['diversity'])}")

    # Update results_no_inbreeding with padded diversity lists
    for run in range(n_runs):
        original_length = len(results_no_inbreeding[run]['diversity'])
        results_no_inbreeding[run]['diversity'] = diversity_no_inbreeding_padded[run]
        # print(f"Run {run} No Inbreeding: Original Length = {original_length}, Padded Length = {len(results_no_inbreeding[run]['diversity'])}")
        
    # ----- Bootstrap ------ #
    # Collect for No Inbreeding
    g_noI, div_noI = plot_mean_and_bootstrapped_ci(results_no_inbreeding, key='diversity')
    gs_list.append(g_noI)
    div_list.append(div_noI)
    label_list.append("No Inbreeding")

    # Collect for Inbreeding
    g_I, div_I = plot_mean_and_bootstrapped_ci(results_inbreeding, key='diversity')
    gs_list.append(g_I)
    div_list.append(div_I)
    label_list.append("Inbreeding")
   
    # Plot
    for idx, ks in enumerate(gs_list):
        plt.plot(ks, div_list[idx][0], label=label_list[idx])
        plt.fill_between(ks, div_list[idx][1], div_list[idx][2], alpha=0.5)
    
    # plt.axhline(y=collapse_threshold, color='red', linestyle='--', label='Collapse Threshold') # TODO: Not being used currently
    plt.title('Diversity Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.legend()
    
    plt.xlabel('Time to Convergence (Generation of Solution)')
    plt.ylabel('Final Diversity')
    plt.title('Comparison of Final Diversity by time of Convergence: Inbreeding vs No Inbreeding')
    plt.xticks(np.arange(1,n_gens+2, step=int(n_gens/10)))
    plt.legend()
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    plt.tight_layout()
    # plt.savefig(f"{os.getcwd()}/figures/div_over_gens_TEST.png")
    plt.savefig(f"{os.getcwd()}/figures/{args.config_plot}_diversity_vs_gen_overtime.png")
    plt.close()
    
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
    
def gen_success_vs_mas_depth(args, depths):
    """
        Definition
        -----------
            Plots the 
    """
    
    # Sample Data Creation
    df_inbreeding = util.flatten_results_depths('inbreeding', depths)
    df_no_inbreeding = util.flatten_results_depths('no_inbreeding', depths)
        
    df_combined = pd.concat([df_inbreeding, df_no_inbreeding], ignore_index=True)

    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(14, 8))

    # Create a boxplot
    box = sns.boxplot(x='Depth', y='Generation_Success', hue='Treatment', data=df_combined, palette="Set2")

    # Overlay with a swarm plot
    swarm = sns.swarmplot(x='Depth', y='Generation_Success', hue='Treatment', data=df_combined, dodge=True, color='0.25', alpha=0.6)

    # Customize the plot
    plt.title('Generation of Success by Tree Depth and Treatment', fontsize=16)
    plt.xlabel('Maximum Depth of Tree', fontsize=14)
    plt.ylabel('Generation of Success', fontsize=14)

    # Adjust the legend
    handles, labels = box.get_legend_handles_labels()
    
    # Remove duplicate labels
    l = plt.legend(handles[0:2], labels[0:2], title='Treatment', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()
    plt.savefig(f"{os.getcwd()}/figures/gen_success_vs_mas_depth.png")
    
def gen_success_vs_inbreeding_threshold(args, thresholds):
    """
        Definition
        -----------
            Plots both treatments in GP with different inbreeding threshold against the generation of success.
            The initial and maximum depth, the tournament size and the population and mutation rate is fixed.
    """
    
    # Sample Data Creation
    df_inbreeding = util.flatten_results_thresholds('inbreeding', thresholds)
    df_no_inbreeding = util.flatten_results_thresholds('no_inbreeding', thresholds)
        
    df_combined = pd.concat([df_inbreeding, df_no_inbreeding], ignore_index=True)

    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(14, 8))

    # Create a boxplot
    box = sns.boxplot(x='Thresholds', y='Generation_Success', hue='Treatment', data=df_combined, palette="Set2")

    # Overlay with a swarm plot
    swarm = sns.swarmplot(x='Thresholds', y='Generation_Success', hue='Treatment', data=df_combined, dodge=True, color='0.25', alpha=0.6)

    # Customize the plot
    plt.title('Generation of Success by Inbreeding Threshold and Treatment', fontsize=16)
    plt.xlabel('Inbreeding Threshold', fontsize=14)
    plt.ylabel('Generation of Success', fontsize=14)

    # Adjust the legend
    handles, labels = box.get_legend_handles_labels()
    
    # Remove duplicate labels
    l = plt.legend(handles[0:2], labels[0:2], title='Treatment', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()
    plt.savefig(f"{os.getcwd()}/figures/gen_success_vs_inbreeding_threshold.png")
    
def plot_threshold_vs_max_depth_by_diversity(args, thresholds, depths):
      
    # Flatten both treatments
    df_inbreeding = util.flatten_results_in_max_depth_diversity('inbreeding', thresholds, depths)
    df_no_inbreeding = util.flatten_results_in_max_depth_diversity('no_inbreeding', thresholds, depths)
    
    # Combine into a single DataFrame
    df_combined = pd.concat([df_inbreeding, df_no_inbreeding], ignore_index=True)
    
    # Aggregate diversity by treatment, max_depth, and inbred_threshold
    heatmap_data = df_combined.groupby(['Treatment', 'Max_Depth', 'Inbred_Threshold'])['Diversity'].mean().reset_index()

    # Pivot the data for heatmap
    pivot_inbreeding = heatmap_data[heatmap_data['Treatment'] == 'inbreeding'].pivot_table( index='Inbred_Threshold', columns='Max_Depth', values='Diversity', aggfunc='mean')  # or another aggregation function as needed)
    pivot_no_inbreeding = heatmap_data[heatmap_data['Treatment'] == 'no_inbreeding'].pivot_table(index='Inbred_Threshold', columns='Max_Depth', values='Diversity', aggfunc='mean')  # or another aggregation function as needed)
    
    overall_min = min(pivot_inbreeding.min().min(), pivot_no_inbreeding.min().min())
    overall_max = max(pivot_inbreeding.max().max(), pivot_no_inbreeding.max().max())
        
    # Create a figure with a specified size
    plt.figure(figsize=(32, 16))

    # Create annotations DataFrame
    annotations = pivot_no_inbreeding.copy()
    for i in annotations.index:
        for j in annotations.columns:
            val_no_inbreeding = pivot_no_inbreeding.loc[i, j]
            val_inbreeding = pivot_inbreeding.loc[i, j]
            annotations.loc[i, j] = f"{val_no_inbreeding:.2f} ({val_inbreeding:.2f})"
            
    # Plot heatmap for No Inbreeding Treatment with annotations
    ax = plt.subplot(111)
    sns.heatmap(
        pivot_no_inbreeding,
        annot=annotations,
        fmt="",
        cmap="YlOrRd",
        vmin=overall_min,
        vmax=overall_max,
        cbar=True,
        ax=ax,
        linewidths=.5,
        linecolor='gray',
        annot_kws={
            "size": 15,        # Increase font size
            #"weight": "bold",  # Make text bold
            "color": "black"   # Change text color to black
        }  
    )
    ax.set_title('Average Final Diversity - No Inbreeding (Inbreeding in parentheses)', fontsize=14)
    ax.set_xlabel('Maximum Depth of Tree', fontsize=14)
    ax.set_ylabel('Inbred Threshold', fontsize=14)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/heatmaps_depth_vs_inbreeding_threshold_by_diversity.png")
    plt.close()
    
def plot_threshold_vs_max_depth_by_gen_success(args, thresholds, depths):
      
    # Flatten both treatments
    df_inbreeding = util.flatten_results_in_max_depth_diversity('inbreeding', thresholds, depths)
    df_no_inbreeding = util.flatten_results_in_max_depth_diversity('no_inbreeding', thresholds, depths)
    
    # Combine into a single DataFrame
    df_combined = pd.concat([df_inbreeding, df_no_inbreeding], ignore_index=True)
    
    # Aggregate diversity by treatment, max_depth, and inbred_threshold
    heatmap_data = df_combined.groupby(['Treatment', 'Max_Depth', 'Inbred_Threshold'])['Generation_Success'].mean().reset_index()

    # Pivot the data for heatmap
    pivot_inbreeding = heatmap_data[heatmap_data['Treatment'] == 'inbreeding'].pivot_table( index='Inbred_Threshold', columns='Max_Depth', values='Generation_Success', aggfunc='mean')  # or another aggregation function as needed)
    pivot_no_inbreeding = heatmap_data[heatmap_data['Treatment'] == 'no_inbreeding'].pivot_table(index='Inbred_Threshold', columns='Max_Depth', values='Generation_Success', aggfunc='mean')  # or another aggregation function as needed)
    
    overall_min = min(pivot_inbreeding.min().min(), pivot_no_inbreeding.min().min())
    overall_max = max(pivot_inbreeding.max().max(), pivot_no_inbreeding.max().max())
        
    # Create a figure with a specified size
    plt.figure(figsize=(20, 12))

    # Use GridSpec to allocate space for two heatmaps and a shared colorbar
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    # Plot heatmap for Inbreeding Treatment
    ax1 = plt.subplot(gs[0])
    sns.heatmap(
        pivot_inbreeding,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",  # Choose a single color map for consistency
        vmin=overall_min,
        vmax=overall_max,
        cbar=False,  # Disable individual colorbars
        ax=ax1,
        linewidths=.5,
        linecolor='gray'
    )
    ax1.set_title('Average Generation of Success - Inbreeding', fontsize=14)
    ax1.set_xlabel('Maximum Depth of Tree', fontsize=12)
    ax1.set_ylabel('Inbred Threshold', fontsize=12)

    # Plot heatmap for No Inbreeding Treatment
    ax2 = plt.subplot(gs[1])
    sns.heatmap(
        pivot_no_inbreeding,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",  # Use the same color map
        vmin=overall_min,
        vmax=overall_max,
        cbar=False,  # Disable individual colorbars
        ax=ax2,
        linewidths=.5,
        linecolor='gray'
    )
    ax2.set_title('Average Generation of Success - No Inbreeding', fontsize=14)
    ax2.set_xlabel('Maximum Depth of Tree', fontsize=12)
    ax2.set_ylabel('Inbred Threshold', fontsize=12)

    # Create a single colorbar for both heatmaps
    cbar_ax = plt.subplot(gs[2])
    norm = plt.Normalize(vmin=overall_min, vmax=overall_max)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])  # Only needed for older versions of Matplotlib
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Average Generation of Success', fontsize=14)

    # Adjust layout and save the figure
    plt.suptitle('Average Diversity by Max Depth and Inbred Threshold across Treatments', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/heatmaps_depth_vs_threshold_by_gen_success.png")
    plt.close()
    
if __name__ == "__main__":
    
    # expression = "(- (+ (* (* x x) (+ 1.0 x)) (+ (- 1.0 1.0) (+ (/ 1.0 1.0) (+ x x)))) (+ 1.0 x))"
 
    # print("Indented Tree Structure:")
    # expr = util.convert_tree_to_expression(expression)
    # print(expr)
    
    # print("\nNO Inbreeding")
    # file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen2/meeting/PopSize:300_InThres:8_Mrates:0.0005_Gens:150_TourSize:15_MaxD:9_InitD:3_no_inbreeding.npy"
    # data = np.load(file_path_name, allow_pickle=True)
    # data_dict_no = data.item()
    # mutation_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # # plot_generation_successes(data_dict_no, mutation_rates, "Mrates:[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]_PopSize:100_InThres:4_Gens:150_TourSize:10_MaxD:8_InitD:3_no_inbreeding.png")
    
    # print("\nInbreeding")
    # file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen2/meeting/PopSize:300_InThres:8_Mrates:0.0005_Gens:150_TourSize:15_MaxD:9_InitD:3_inbreeding.npy"
    # data = np.load(file_path_name, allow_pickle=True)
    # data_dict = data.item()
    # mutation_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # # plot_generation_successes(data_dict_no, mutation_rates, "Mrates:[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]_PopSize:100_InThres:4_Gens:150_TourSize:10_MaxD:8_InitD:3_inbreeding.png")
    
    # # plot_time_of_convergence_vs_diversity([], data_dict_no, data_dict)
    # plot_diversity_generation_over_time([], data_dict_no, data_dict)
    # # plot_gen_vs_run(15, data_dict_no, data_dict)
    
    depths = [6, 7, 8, 9, 10]  # Example maximum depths
    # gen_success_vs_mas_depth([], depths)
    
    # thresholds = [4,5,6,7,8]
    thresholds = [4, 5, 6, 7, 8]
    # gen_success_vs_inbreeding_threshold([], thresholds)
    
    # plot_threshold_vs_max_depth_by_gen_success([], thresholds, depths)
    plot_threshold_vs_max_depth_by_diversity([], thresholds, depths)

    
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