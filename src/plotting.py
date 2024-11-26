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
    plt.savefig(f"{os.getcwd()}/figures/gp_lambda/gen_success_vs_mas_depth.png")
    
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
    plt.savefig(f"{os.getcwd()}/figures/gp_lambda/gen_success_vs_inbreeding_threshold.png")
    
def plot_threshold_vs_max_depth_by_diversity(args, bench_name, thresholds, depths, init_depth):
      
    # Flatten both treatments
    df_inbreeding = util.flatten_results_in_max_depth_diversity(bench_name, 'inbreeding', thresholds, depths, init_depth)
    df_no_inbreeding = util.flatten_results_in_max_depth_diversity(bench_name, 'no_inbreeding', thresholds, depths, init_depth)
    
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
    no_in_count = 0
    in_count = 0
    min_no_in = 150
    min_in = 150
    max_no_in = 0
    max_in = 0
    annotations = pivot_no_inbreeding.copy()
    for i in annotations.index:
        for j in annotations.columns:
            val_no_inbreeding = pivot_no_inbreeding.loc[i, j]
            val_inbreeding = pivot_inbreeding.loc[i, j]
            annotations.loc[i, j] = f"{val_no_inbreeding:.2f} ({val_inbreeding:.2f})"
            if val_no_inbreeding <= val_inbreeding:
                no_in_count += 1
                min_no_in = min(min_no_in, val_no_inbreeding)
                max_no_in = max(max_no_in, val_no_inbreeding)
            else:
                in_count +=1 
                min_in = min(min_in, val_inbreeding)
                max_in = max(max_in, val_inbreeding)
                
    n_comb = no_in_count+in_count
    print(f"\nFor a total of {n_comb} combinations.")
    print(f"NO Inbreeding ({no_in_count}): {(no_in_count/n_comb)*100:.3f}%. Minimum gen: {min_no_in} and Max gen: {max_no_in}.")
    print(f"Inbreeding ({in_count}): {(in_count/n_comb)*100:.3f}%. Minimum gen: {min_in} and Max gen: {max_in}.")
            
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
    plt.savefig(f"{os.getcwd()}/figures/gp_lambda/{bench_name}_heatmaps_depth_vs_inbreeding_threshold_by_diversity_InitD:{init_depth}.png")
    plt.close()
    
def plot_threshold_vs_max_depth_by_gen_success(args, bench_name, thresholds, depths, init_depth):
      
    # Flatten both treatments
    df_inbreeding = util.flatten_results_in_max_depth_diversity(bench_name, 'inbreeding', thresholds, depths, init_depth)
    df_no_inbreeding = util.flatten_results_in_max_depth_diversity(bench_name, 'no_inbreeding', thresholds, depths, init_depth)
    
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
    plt.savefig(f"{os.getcwd()}/figures/gp_lambda/{bench_name}_heatmaps_depth_vs_threshold_by_gen_success_InitD:{init_depth}.png")
    plt.close()
    
def plot_all_heatmap(args, bench_name, thresholds, depths, init_depth):
      
    # Flatten both treatments
    df_inbreeding = util.flatten_results_in_max_depth_diversity(bench_name, 'inbreeding', thresholds, depths, init_depth)
    df_no_inbreeding = util.flatten_results_in_max_depth_diversity(bench_name, 'no_inbreeding', thresholds, depths, init_depth)
    
    # Combine into a single DataFrame
    df_combined = pd.concat([df_inbreeding, df_no_inbreeding], ignore_index=True)
    
    # Aggregate diversity by treatment, max_depth, and inbred_threshold
    heatmap_data_div = df_combined.groupby(['Treatment', 'Max_Depth', 'Inbred_Threshold'])['Diversity'].mean().reset_index()

    # Pivot the data for heatmap
    pivot_inbreeding = heatmap_data_div[heatmap_data_div['Treatment'] == 'inbreeding'].pivot_table( index='Inbred_Threshold', columns='Max_Depth', values='Diversity', aggfunc='mean')  # or another aggregation function as needed)
    pivot_no_inbreeding = heatmap_data_div[heatmap_data_div['Treatment'] == 'no_inbreeding'].pivot_table(index='Inbred_Threshold', columns='Max_Depth', values='Diversity', aggfunc='mean')  # or another aggregation function as needed)
    
    overall_min = min(pivot_inbreeding.min().min(), pivot_no_inbreeding.min().min())
    overall_max = max(pivot_inbreeding.max().max(), pivot_no_inbreeding.max().max())
    
    # Get the Generation Success Grouping
    # Aggregate diversity by treatment, max_depth, and inbred_threshold
    heatmap_data_gen = df_combined.groupby(['Treatment', 'Max_Depth', 'Inbred_Threshold'])['Generation_Success'].mean().reset_index()

    # Pivot the data for heatmap
    pivot_inbreeding_gen = heatmap_data_gen[heatmap_data_gen['Treatment'] == 'inbreeding'].pivot_table( index='Inbred_Threshold', columns='Max_Depth', values='Generation_Success', aggfunc='mean')  # or another aggregation function as needed)
    pivot_no_inbreeding_gen = heatmap_data_gen[heatmap_data_gen['Treatment'] == 'no_inbreeding'].pivot_table(index='Inbred_Threshold', columns='Max_Depth', values='Generation_Success', aggfunc='mean')  # or another aggregation function as needed)
    
        
    # Create a figure with a specified size
    plt.figure(figsize=(32, 16))

    no_in_count = 0
    in_count = 0
    min_no_in = 150
    min_in = 150
    max_no_in = 0
    max_in = 0
    annotations = pivot_no_inbreeding_gen.copy()
    for i in annotations.index:
        for j in annotations.columns:
            val_no_inbreeding = pivot_no_inbreeding_gen.loc[i, j]
            val_inbreeding = pivot_inbreeding_gen.loc[i, j]
            annotations.loc[i, j] = f"{val_no_inbreeding:.2f} ({val_inbreeding:.2f})"
            if val_no_inbreeding <= val_inbreeding:
                no_in_count += 1
                
            else:
                in_count +=1 

            
            min_no_in = min(min_no_in, val_no_inbreeding)
            max_no_in = max(max_no_in, val_no_inbreeding)
            
            min_in = min(min_in, val_inbreeding)
            max_in = max(max_in, val_inbreeding)
                
    n_comb = no_in_count+in_count
    print(f"\nFor a total of {n_comb} combinations.")
    print(f"NO Inbreeding ({no_in_count}): {(no_in_count/n_comb)*100:.3f}%. Minimum gen: {min_no_in} and Max gen: {max_no_in}.")
    print(f"Inbreeding ({in_count}): {(in_count/n_comb)*100:.3f}%. Minimum gen: {min_in} and Max gen: {max_in}.")
            
            
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
    ax.set_title('Average Final Diversity [HeatMap] and Average Generation of Success [Value] - No Inbreeding (Inbreeding)', fontsize=14)
    ax.set_xlabel('Maximum Depth of Tree', fontsize=14)
    ax.set_ylabel('Inbred Threshold', fontsize=14)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/gp_lambda/{bench_name}_heatmaps_all_InitD:{init_depth}.png")
    plt.close()
    
def plot_any_attr_vs_gen(args, results_no_inbreeding, results_inbreeding, config_plot="test", temp_runs=15, temp_gens=150, attribute="avg_tree_size"):
    """
        Definition
        -----------
            Plot any given attribute against longest total number of generations it took either of the treatments.
                attr: best_fitness
                attr: diversity
                attr: generation_success
                attr: avg_tree_size
                attr: avg_tree_depth
                attr: pop_intron_ratio
                attr: avg_intron_ratio
                attr: pop_total_introns
                attr: pop_total_nodes
                attr: avg_kinship
                attr: t_close
                attr: t_far
    """
    
    # Create a figure
    plt.figure(figsize=(20, 12))
    
    # Get number of runs
    if args is None:
        n_runs = temp_runs
        config_plot = config_plot
    else:
        n_runs = args.exp_num_runs
        config_plot = args.config_plot
   
    gs_list = []
    div_list = []
    label_list = []
    
    # Get diversity
    avg_tree_size_inbreeding = [results_inbreeding[run][attribute] for run in range(n_runs)]
    avg_tree_size_no_inbreeding = [results_no_inbreeding[run][attribute] for run in range(n_runs)]
    
    # Find maximum length list and padd the rest to be final diversity at point of convergence
    max_length_inbreeding = max(len(sublist) for sublist in avg_tree_size_inbreeding)
    max_length_no_inbreeding = max(len(sublist) for sublist in avg_tree_size_no_inbreeding)
    global_max_length = max(max_length_inbreeding, max_length_no_inbreeding)
    
    # Pad all sublists in diversity_inbreeding
    avg_tree_size_inbreeding_padded = [util.pad_sublist(sublist, global_max_length) for sublist in avg_tree_size_inbreeding]

    # Pad all sublists in diversity_no_inbreeding
    avg_tree_sizeno_inbreeding_padded = [util.pad_sublist(sublist, global_max_length) for sublist in avg_tree_size_no_inbreeding]

    # Update results_inbreeding with padded diversity lists
    for run in range(n_runs):
        original_length = len(results_inbreeding[run][attribute])
        results_inbreeding[run][attribute] = avg_tree_size_inbreeding_padded[run]
        # print(f"Run {run} Inbreeding: Original Length = {original_length}, Padded Length = {len(results_inbreeding[run][attribute])}.")

    # Update results_no_inbreeding with padded diversity lists
    for run in range(n_runs):
        original_length = len(results_no_inbreeding[run][attribute])
        results_no_inbreeding[run][attribute] = avg_tree_sizeno_inbreeding_padded[run]
        # print(f"Run {run} No Inbreeding: Original Length = {original_length}, Padded Length = {len(results_no_inbreeding[run][attribute])}.")

    # ----- Bootstrap ------ #
    
    # Collect for No Inbreeding
    g_noI, div_noI = plot_mean_and_bootstrapped_ci(results_no_inbreeding, key=attribute)
    gs_list.append(g_noI)
    div_list.append(div_noI)
    label_list.append("No Inbreeding")

    # Collect for Inbreeding
    g_I, div_I = plot_mean_and_bootstrapped_ci(results_inbreeding, key=attribute)
    gs_list.append(g_I)
    div_list.append(div_I)
    label_list.append("Inbreeding")
   
    # Plot
    for idx, ks in enumerate(gs_list):
        plt.plot(ks, div_list[idx][0], label=label_list[idx])
        plt.fill_between(ks, div_list[idx][1], div_list[idx][2], alpha=0.5)
    
    plt.title('Diversity Over Generations')
    plt.xlabel('Generation')
    plt.ylabel(f'{attribute}')
    plt.legend()
    
    plt.xlabel('Time to Convergence (Generation of Solution)')
    plt.ylabel(f'Final {attribute}')
    plt.title(f'Comparison of Final {attribute} by time of Convergence: Inbreeding vs No Inbreeding')
    plt.xticks(np.arange(0, global_max_length, step=int(global_max_length/10)))
    plt.legend()
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{config_plot}_{attribute}.png")
    plt.close()
    
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def compute_correlations(results_no_inbreeding, results_inbreeding, attributes, config_plot="test", temp_runs=15, temp_gens=150, attribute_1="avg_tree_size", attribute_2="avg_tree_depth"):
    """
        Definition
        -----------
           TODO: Need to automatize for all attribute pairs and both treatments
           TODO: Need to compute the mean and CI of the runs, as 1 is not enough.
    
    """

    n_runs = temp_runs
    
    # Pick attribute at random to get whats the run that had the most generations out of all.
    max_length_attr = util.get_global_max_depth(results_inbreeding)
    max_length_no_attr = util.get_global_max_depth(results_no_inbreeding)
    global_max_length = max(max_length_attr, max_length_no_attr)
    
    # Create DF for all attributes for the given dictionary treatment
    df_inbreeding    = util.pad_dict_and_create_df(results_inbreeding, attributes, global_max_length, n_runs)
    df_no_inbreeding = util.pad_dict_and_create_df(results_no_inbreeding, attributes, global_max_length, n_runs)

    # Get diversity
    attr_1 = [results_no_inbreeding[run][attribute_1] for run in range(n_runs)]
    attr_2 = [results_no_inbreeding[run][attribute_2] for run in range(n_runs)]
    
    # Find maximum length list and padd the rest to be final diversity at point of convergence
    max_length_attr_1 = max(len(sublist) for sublist in attr_1)
    max_length_attr_2 = max(len(sublist) for sublist in attr_2)
    global_max_length = max(max_length_attr_1, max_length_attr_2)
    
    # Pad all sublists in diversity_inbreeding
    attr_1_padded = [util.pad_sublist(sublist, global_max_length) for sublist in attr_1]

    # Pad all sublists in diversity_no_inbreeding
    attr_2_padded = [util.pad_sublist(sublist, global_max_length) for sublist in attr_2]
    
    for run in range(n_runs):
        original_length = len(results_no_inbreeding[run][attribute_1])
        results_no_inbreeding[run][attribute_1] = attr_1_padded[run]
        # print(f"Run {run} Inbreeding: Original Length = {original_length}, Padded Length = {len(results_inbreeding[run][attribute])}.")

    # Update results_no_inbreeding with padded diversity lists
    for run in range(n_runs):
        original_length = len(results_no_inbreeding[run][attribute_2])
        results_no_inbreeding[run][attribute_2] = attr_2_padded[run]
        
    # Create a DataFrame
    df = pd.DataFrame({
        attribute_1: results_no_inbreeding[0][attribute_1],
        attribute_2: results_no_inbreeding[0][attribute_2]
    })

    # Compute Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(df[attribute_1], df[attribute_2])
    print(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")

    # Compute Spearman's correlation
    spearman_corr, spearman_p = stats.spearmanr(df[attribute_1], df[attribute_2])
    print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

    # Compute Kendall's Tau
    kendall_corr, kendall_p = stats.kendalltau(df[attribute_1], df[attribute_2])
    print(f"Kendall's Tau: {kendall_corr:.4f} (p-value: {kendall_p:.4f})")


    # Using Seaborn for enhanced visualization
    # TODO: I could plot only 2 attributes against each other, but I can probably plot two treatments here (inbreeding vs no inbreeding)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.regplot(x=attribute_1, y=attribute_2, data=df, ci=None, scatter_kws={'s': 50, 'alpha':0.7})

    plt.title('Scatter Plot with Regression Line')
    plt.xlabel(attribute_1)
    plt.ylabel(attribute_2)
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{config_plot}_Scatter_plot_{attribute_1}_vs_{attribute_2}.png")
    plt.close()
    
def plot_correlation_heatmap(df_no_inbreeding, df_inbreeding, attributes, config_plot):
    """
    Plots the correlation heatmaps of two DataFrames side by side with a shared color scale.

    Parameters:
    - df_no_inbreeding (pd.DataFrame): DataFrame without inbreeding.
    - df_inbreeding (pd.DataFrame): DataFrame with inbreeding.
    - attributes (list): List of attributes to include in the correlation.
    - config_plot (str): Configuration identifier for saving the plot.
    """
    
    # Ensure only the specified attributes are used
    df_no_inbreeding = df_no_inbreeding[attributes]
    df_inbreeding = df_inbreeding[attributes]
    
    # Compute correlation matrices
    corr_no_inbreeding = df_no_inbreeding.corr(method='pearson')
    corr_inbreeding = df_inbreeding.corr(method='pearson')
    
    # Determine the overall min and max for the color scale
    # This ensures both heatmaps use the same scale
    combined_corr = pd.concat([corr_no_inbreeding.stack(), corr_inbreeding.stack()])
    vmin = combined_corr.min()
    vmax = combined_corr.max()
    
    # Set up the matplotlib figure with two subplots side by side
    fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
    
    # Define the colormap
    cmap = 'coolwarm'
    
    # Plot the first heatmap: No Inbreeding
    sns.heatmap(
        corr_no_inbreeding, 
        annot=True, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        ax=axes[0],
        cbar=False,  # We'll add a single colorbar later
        square=True,
        linewidths=.5,
        fmt=".2f"
    )
    axes[0].set_title('Correlation Matrix - No Inbreeding')
    
    # Plot the second heatmap: Inbreeding
    sns.heatmap(
        corr_inbreeding, 
        annot=True, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        ax=axes[1],
        cbar=False,  # We'll add a single colorbar later
        square=True,
        linewidths=.5,
        fmt=".2f"
    )
    axes[1].set_title('Correlation Matrix - Inbreeding')
    
    # Add a single colorbar to the right of both heatmaps
    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=15)
    
    # Set a main title for the entire figure
    # fig.suptitle('Comparison of Correlation Matrices', fontsize=16, y=0.95)
    
    # Ensure the 'figures' directory exists
    figures_dir = os.path.join(os.getcwd(), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the combined figure
    save_path = os.path.join(figures_dir, f"{config_plot}_CorrelationHeatmap_Comparison.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    
    attributes = ['best_fitness', 'diversity', 'avg_tree_size',
                  'avg_tree_depth', 'pop_intron_ratio', 'avg_intron_ratio', 'pop_total_introns',
                  'pop_total_nodes', 'avg_kinship']
    
    # TODO: 't_close', 't_far' since they represent a single kinship value. Is just for reference. Think how to maybe use it
    
    # ----- BLOAT STUDY: Read files ------ #  
    
    print("\nNO Inbreeding")
    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen2/bloat/PopSize:300_InThres:8_Mrates:0.0005_Gens:150_TourSize:15_MaxD:6_InitD:3_no_inbreeding.npy"
    data = np.load(file_path_name, allow_pickle=True)
    results_no_inbreeding = data.item()

    print("\nInbreeding")
    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen2/bloat/PopSize:300_InThres:8_Mrates:0.0005_Gens:150_TourSize:15_MaxD:6_InitD:3_inbreeding.npy"
    data = np.load(file_path_name, allow_pickle=True)
    results_inbreeding = data.item()

    # plot_any_attr_vs_gen(None, results_no_inbreeding, results_inbreeding, temp_runs=5, attribute="pop_total_nodes")
    # compute_correlations(results_no_inbreeding, results_inbreeding, attributes, config_plot="test", temp_runs=5, temp_gens=150, attribute_1="avg_tree_size", attribute_2="pop_total_introns")
    
    # Pick attribute at random to get whats the run that had the most generations out of all.
    max_length_attr = util.get_global_max_depth(results_inbreeding)
    max_length_no_attr = util.get_global_max_depth(results_no_inbreeding)
    global_max_length = max(max_length_attr, max_length_no_attr)
    
    # Create DF for all attributes for the given dictionary treatment
    df_inbreeding    = util.pad_dict_and_create_df(results_inbreeding, attributes, global_max_length, 5)
    df_no_inbreeding = util.pad_dict_and_create_df(results_no_inbreeding, attributes, global_max_length, 5)
    
    plot_correlation_heatmap(df_no_inbreeding, df_inbreeding, attributes, config_plot="test")
    
    # --------- Heatmaps and statistics ----------- #
    # depths = [6, 7, 8, 9]#, 10]  # Example maximum depths
    # gen_success_vs_mas_depth([], depths) # TODO: They are for an static threshold
    
    # thresholds = [5,6,7,8,9] # for InitD:2
    # thresholds = [5, 6, 7, 8, 9, 10, 11, 12, 13]#, 14] # InitD:3
    
    
    # gen_success_vs_inbreeding_threshold([], thresholds) # TODO: they are for an static depth
    
    # plot_threshold_vs_max_depth_by_gen_success([], "nguyen2", thresholds, depths, 3)
    # plot_threshold_vs_max_depth_by_diversity([], "nguyen3", thresholds, depths, 3)
    
    # plot_all_heatmap([], "nguyen5", thresholds, depths, 3)
    
    # ------------ Legacy experimental -------------- #
    # generations = range(1, len(self.average_size_list) + 1)
    # plt.figure()
    # plt.plot(generations, self.average_size_list, label='Average Size')
    # plt.plot(generations, self.average_depth_list, label='Average Depth')
    # plt.plot(generations, self.average_arity_list, label='Average Arity')
    # plt.xlabel('Generation')
    # plt.ylabel('Bloat Metric')
    # plt.title('Bloat Metrics Over Generations')
    # plt.legend()
    # if self.inbred_threshold == None:
    #     plt.savefig(f"{os.getcwd()}/figures/genetic_programming/bloat/test_inbreeding.png")
    # else:
    #     plt.savefig(f"{os.getcwd()}/figures/genetic_programming/bloat/test_NO_inbreeding.png")

    