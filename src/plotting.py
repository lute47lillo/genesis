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

# ---------------------- Bootstrapping methods ---------------------------------- #

def collect_bootstrapping_data(results_no_inbreeding, results_inbreeding):
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