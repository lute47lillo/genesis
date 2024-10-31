import matplotlib.pyplot as plt
import os 
import numpy as np

def plot_fitness_comparison_populations(args, elem_list, parameter, results_inbreeding, results_no_inbreeding):
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
    for element in elem_list:
        plt.plot(
            results_inbreeding[element]['best_fitness'],
            label=f'Inbreeding, {parameter} {element}'
        )
        plt.plot(
            results_no_inbreeding[element]['best_fitness'],
            label=f'No Inbreeding, {parameter} {element}',
            linestyle='--'
        )
    plt.title('Best Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{os.getcwd()}/figures/{parameter}/{args.benchmark}/Fit_{args.config_plot}.png")
    plt.close()

def plot_diversity_comparison_populations(args, elem_list, parameter, results_inbreeding, results_no_inbreeding):
    """
        Definition
        -----------
            Plot the Diversity results of inbreeding prevention mechanism against no prevention over different populations sizes
    """
    # Plot Diversity Comparison
    plt.figure(figsize=(16, 9))
    for element in elem_list:
        plt.plot(
            results_inbreeding[element]['diversity'],
            label=f'Inbreeding, {parameter} {element}'
        )
        plt.plot(
            results_no_inbreeding[element]['diversity'],
            label=f'No Inbreeding, {parameter} {element}',
            linestyle='--'
        )
    plt.title('Genetic Diversity over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{os.getcwd()}/figures/{parameter}/{args.benchmark}/Div_{args.config_plot}.png")
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