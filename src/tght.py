import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

# Define the bootstrap function
def compute_bootstrap_ci(data, confidence=0.95, n_bootstraps=1000):
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

# Define the plotting function with debugging
def plot_mean_and_bootstrapped_ci_multiple_parameters(parameters_list, experimental_results, key):
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
    
    print(f"Number of runs: {num_runs}")
    print(f"Parameters (e.g., mutation rates): {parameters_list}")
    print(f"Number of generations: {num_gens}")
    
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
            fit_ci_l, fit_ci_h = compute_bootstrap_ci(gen_fitness)
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
    
    plt.title(f'Mean {key.replace("_", " ").title()} Over Generations with 95% CI')
    plt.xlabel('Generation')
    plt.ylabel(f'{key.replace("_", " ").title()}')
    plt.legend(title='Parameters')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"stst.png")
    
    return param_bootstrap

# Example usage with sample data
if __name__ == "__main__":
    # Sample experimental_results
    sample_results = {
        0: {
            0.01: {
                'best_fitness': [100 + np.random.normal(0, 1) for _ in range(50)],
                'diversity': [50 + np.random.normal(0, 1) for _ in range(50)],
                'global_optimum': [100 for _ in range(50)],
            },
            0.05: {
                'best_fitness': [110 + np.random.normal(0, 1.5) for _ in range(50)],
                'diversity': [55 + np.random.normal(0, 1) for _ in range(50)],
                'global_optimum': [110 for _ in range(50)],
            },
            0.1: {
                'best_fitness': [120 + np.random.normal(0, 2) for _ in range(50)],
                'diversity': [60 + np.random.normal(0, 1) for _ in range(50)],
                'global_optimum': [120 for _ in range(50)],
            },
        },
        1: {
            0.01: {
                'best_fitness': [100 + np.random.normal(0, 1) for _ in range(50)],
                'diversity': [50 + np.random.normal(0, 1) for _ in range(50)],
                'global_optimum': [100 for _ in range(50)],
            },
            0.05: {
                'best_fitness': [110 + np.random.normal(0, 1.5) for _ in range(50)],
                'diversity': [55 + np.random.normal(0, 1) for _ in range(50)],
                'global_optimum': [110 for _ in range(50)],
            },
            0.1: {
                'best_fitness': [120 + np.random.normal(0, 2) for _ in range(50)],
                'diversity': [60 + np.random.normal(0, 1) for _ in range(50)],
                'global_optimum': [120 for _ in range(50)],
            },
        },
        # Add more runs as needed
    }

    # Define mutation rates
    mutation_rates = [0.01, 0.05, 0.1]

    # Call the plotting function
    bootstrap_results = plot_mean_and_bootstrapped_ci_multiple_parameters(
        parameters_list=mutation_rates,
        experimental_results=sample_results,
        key='best_fitness'
    )

    # Optionally, inspect bootstrap_results
    for param, stats in bootstrap_results.items():
        print(f"\nParameter: {param}")
        print(f"Mean Fitness (first 5): {stats['mean_fitness'][:5]}")
        print(f"CI Low (first 5): {stats['ci_low'][:5]}")
        print(f"CI High (first 5): {stats['ci_high'][:5]}")
