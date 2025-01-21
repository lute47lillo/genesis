import matplotlib.pyplot as plt
import os 
import numpy as np
from scipy.stats import bootstrap
import util
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from scipy import stats
import json
from collections import defaultdict

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ---------------------- Bootstrapping methods ---------------------------------- #

# Function to calculate bootstrapped confidence intervals
def compute_bootstrap_ci(data, n_bootstrap=1000, ci=0.95, window_size=50):
    """Compute bootstrap confidence interval for 1D data array."""

    data = np.asarray(data)
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    means_sorted = np.sort(means)
    
    # Apply moving average smoothing
    smoothed_means = np.convolve(means_sorted, np.ones(window_size)/window_size, mode='valid')
    
    lower = np.percentile(smoothed_means, (1 - ci) / 2 * 100)
    upper = np.percentile(smoothed_means, (1 + ci) / 2 * 100)
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

def compute_correlations(results_inbreeding, temp_runs=15, attribute_1="avg_tree_size", attribute_2="avg_tree_depth", threshold=None):
    """
        Definition
        -----------
           TODO: Need to automatize for all attribute pairs and both treatments
           TODO: Need to compute the mean and CI of the runs, as 1 is not enough.
    
    """

    n_runs = temp_runs
    
    # Get diversity
    attr_1 = [results_inbreeding[run][attribute_1] for run in range(n_runs)]
    attr_2 = [results_inbreeding[run][attribute_2] for run in range(n_runs)]
    
    # nº total success
    count = 0
    avg_gen = 0
    gen_success = [results_inbreeding[run]['generation_success'] for run in range(n_runs)]
    for gen in gen_success:
        if gen < 300:
            count += 1
        avg_gen += gen
            
    avg_gen = avg_gen / n_runs
    diversity_print = [results_inbreeding[run]['diversity'] for run in range(n_runs)]
    
    # Find maximum length list and padd the rest to be final diversity at point of convergence
    max_length_attr_1 = max(len(sublist) for sublist in attr_1)
    max_length_attr_2 = max(len(sublist) for sublist in attr_2)
    global_max_length = max(max_length_attr_1, max_length_attr_2)
    
    # Pad all sublists in diversity_inbreeding
    attr_1_padded = [util.pad_sublist(sublist, global_max_length) for sublist in attr_1]

    # Pad all sublists in diversity_no_inbreeding
    attr_2_padded = [util.pad_sublist(sublist, global_max_length) for sublist in attr_2]
    
    attr_div_padded = [util.pad_sublist(sublist, global_max_length) for sublist in diversity_print]

    
    for run in range(n_runs):
        # original_length = len(results_no_inbreeding[run][attribute_1])
        results_inbreeding[run][attribute_1] = attr_1_padded[run]
        # print(f"Run {run} Inbreeding: Original Length = {original_length}, Padded Length = {len(results_inbreeding[run][attribute])}.")

    # Update results_no_inbreeding with padded diversity lists
    for run in range(n_runs):
        # original_length = len(results_no_inbreeding[run][attribute_2])
        results_inbreeding[run][attribute_2] = attr_2_padded[run]
        
    for run in range(n_runs):
        # original_length = len(results_no_inbreeding[run][attribute_2])
        results_inbreeding[run]['diversity'] = attr_div_padded[run]
            
    attr_1 = [results_inbreeding[run][attribute_1] for run in range(n_runs)]
    attr_2 = [results_inbreeding[run][attribute_2] for run in range(n_runs)] # (15, 151)
    attr_div = [results_inbreeding[run]['diversity'] for run in range(n_runs)]
    
    # Compute the mean across the runs (axis=0)
    mean_values_attr1 = np.mean(np.array(attr_1), axis=0) # Resulting shape: (N,)
    mean_values_attr2 = np.mean(np.array(attr_2), axis=0) # Resulting shape: (N,)
    final_div = np.mean(np.array(attr_div), axis=0)[-1] # final diversity averaged across all runs
    
    final_average = np.mean(np.mean(np.array(attr_div), axis=0))
    # print(f"Average: {np.mean(np.mean(np.array(attr_div), axis=0))}, final: {final_div}")

    
    # Create a DataFrame
    df = pd.DataFrame({
        attribute_1: mean_values_attr1,
        attribute_2: mean_values_attr2
    })

    # Compute Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(df[attribute_1], df[attribute_2])
    # print(f"Threshold: {threshold}. Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f}). Total nº success: {count}. Final diversity: {final_div:.2f}. Average gen: {avg_gen:.2f}.")
    
    # suc_div_thresh = f"{count} ({final_div})"
    suc_div_thresh = f"{count} ({final_average})" # TODO: Doing it with the final average to use averages for all.
    return suc_div_thresh
    
# Define the new plotting function
def plot_combined_corr_heatmaps(dfs, thresholds, attributes, config_plot):
    
    # Compute correlation matrices and find global min and max
    corrs = []
    min_corr = 1.0
    max_corr = -1.0
    for df in dfs:
        df = df[attributes]
        corr = df.corr(method='pearson')
        corrs.append(corr)
        min_corr = min(min_corr, corr.values.min())
        max_corr = max(max_corr, corr.values.max())
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Define the colormap
    cmap = 'coolwarm'
    
    # Plot each heatmap
    for i, (corr, thres) in enumerate(zip(corrs, thresholds)):
        sns.heatmap(
            corr,
            annot=True,
            cmap=cmap,
            cbar=False,  # We'll add a single colorbar later
            square=True,
            linewidths=.6,
            fmt=".2f",
            vmin=min_corr,
            vmax=max_corr,
            ax=axes[i]
        )
        axes[i].set_title(f'Inbreeding Threshold = {thres}')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar
    
    # Create a single colorbar
    cbar_ax = fig.add_axes([.91, .3, .03, .4])  # Adjust position as needed
    norm = plt.Normalize(vmin=min_corr, vmax=max_corr)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    
    # Ensure the 'figures/correlations' directory exists
    figures_dir = os.path.join(os.getcwd(), "figures", "correlations")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the combined figure
    save_path = os.path.join(figures_dir, f"{config_plot}CombinedHeatmap.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def collect_plot_values(dfs, value_label, bloat_thresholds, attribute_1, n_runs=15):
    
    # Metrics for CI
    gs_list = [] # Generations list
    value_list = [] # specific-attribute values
    label_list = [] # label list
    
    for idx, results in enumerate(dfs):
        
        if value_label == '':
            threshold_label = str(bloat_thresholds[idx])
        else:
            threshold_label = f"{value_label}: " + str(bloat_thresholds[idx])
        
        # Get Attribute
        attribute_lists = [results[run][attribute_1] for run in range(n_runs)]

        # Find maximum length list and padd the rest to be final attr at point of convergence
        max_length = max(len(sublist) for sublist in attribute_lists)
        global_max_length = min(150, max_length) # Capping at 150
        global_max_length = 300 #150

        # Pad all sublists in diversity_inbreeding
        attribute_padded = [util.pad_sublist(sublist, global_max_length) for sublist in attribute_lists]

        # Update results_inbreeding with padded diversity lists
        for run in range(n_runs):

            # results[run][attribute_1] = attribute_padded[run][:150]
            results[run][attribute_1] = attribute_padded[run][:300]

            # print(f"Run {run} Inbreeding: Original Length = {original_length}, Padded Length = {len(results_inbreeding[run][attribute])}.")

        # ----- Bootstrap ------ #
        
        # Collect for No Inbreeding
        g_noI, div_noI = plot_mean_and_bootstrapped_ci(results, key=attribute_1)
        gs_list.append(g_noI)
        value_list.append(div_noI)
        label_list.append(threshold_label)
        
    return (gs_list, value_list, label_list)

def plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot, global_max_length=150):
    """
    TODO: Automatically adjust the legend to not overlap
    Plot multiple attributes in separate columns for each Symbolic Regression (SR) problem,
    with each SR problem occupying its own row and a centralized, enlarged legend.
    
    Parameters:
    -----------
    sr_dfs : dict
        Dictionary where each key is an SR function name, and each value is another dictionary
        mapping attribute names to their corresponding (gs_list, div_list, label_list).
        
        Example:
        sr_dfs = {
            'Nguyen1': {
                'pop_intron_ratio': (gs_list, div_list, label_list),
                'diversity': (gs_list, div_list, label_list),
                ...
            },
            'Nguyen2': {
                'pop_intron_ratio': (gs_list, div_list, label_list),
                'diversity': (gs_list, div_list, label_list),
                ...
            },
            ...
        }
        
    sr_fns : list
        List of SR function names corresponding to the keys in sr_dfs.
        
    attributes : list
        List of attribute names to plot. Each attribute will occupy its own column.
        
    config_plot : str
        Configuration identifier to include in the plot filename.
        
    global_max_length : int
        The maximum generation number to set as the upper limit on the X-axis.
    """
    
    sns.set_style("darkgrid")
    num_sr_problems = len(sr_fns)
    num_attributes = len(attributes)
    
    # Define the figure size based on the number of SR problems and attributes
    # Adjust the height and width per subplot as needed (e.g., 3 inches per subplot)
    fig_height = 7 * num_sr_problems
    fig_width = 8 * num_attributes  # 7 inches per attribute column
    fig, axes = plt.subplots(nrows=num_sr_problems, ncols=num_attributes, figsize=(fig_width, fig_height), sharex=True)
    
    # If there's only one SR problem, axes is a 1D array. Make it a 2D array for consistency.
    if num_sr_problems == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Initialize lists to collect legend handles and labels
    handles = []
    labels = []
    legend_added = False  # Flag to ensure we collect handles only once
    
    # Iterate over each SR problem and its corresponding SR function name
    limits = {}
    for sr_idx, sr_fn in enumerate(sr_fns):
        sr_attributes = sr_dfs[sr_fn]  # Get the attribute dict for this SR problem
        
        limits[sr_fn] = {}
        for attr_idx, attribute in enumerate(attributes):
            # Iterate over each attribute to plot
            ax = axes[sr_idx, attr_idx]  # Access the appropriate subplot
            
            # Retrieve the plotting data for this attribute
            if attribute not in sr_attributes:
                raise ValueError(f"Attribute '{attribute}' not found in SR problem '{sr_fn}'.")
            
            gs_list, div_list, label_list = sr_attributes[attribute]
            
            # Plot each SR run within the current SR problem for the current attribute
            run_y_min = 999999
            run_y_max = 0
            for run_idx, ks in enumerate(gs_list):

                diversity = div_list[run_idx][0]  # Assuming div_list contains tuples/lists with at least one element
                line, = ax.plot(ks, diversity, label=label_list[run_idx])
                                
                # Collect handles and labels only once for the centralized legend
                if not legend_added:
                    handles.append(line)
                    labels.append(label_list[run_idx])
                    
                run_y_min = min(diversity) if min(diversity) < run_y_min else run_y_min
                run_y_max = max(diversity) if max(diversity) > run_y_max else run_y_max
                
            # Get global y lims
            print(f"SR: {sr_fn} - {attribute} with x-min: {run_y_min} and x-max: {run_y_max}.")    
            limits[sr_fn].update({attribute :(run_y_min, run_y_max)})
        print()
        legend_added = True  # Ensure we don't collect handles again
        
    result = defaultdict(lambda: [float('inf'), float('-inf')])

    for sub_dict in limits.values():
        for attr, (min_val, max_val) in sub_dict.items():
            result[attr][0] = min(result[attr][0], min_val)
            result[attr][1] = max(result[attr][1], max_val)

    # Convert to regular dict with tuples
    limits = {attr: tuple(vals) for attr, vals in result.items()}
    
    # Set global y-lim for all subplots
    for sr_idx, _ in enumerate(sr_fns):
        for attr_idx, attr in enumerate(attributes):
            # axes[sr_idx, attr_idx].set_ylim([limits[attr][0], limits[attr][1]])
            axes[sr_idx, attr_idx].tick_params(axis='both', which='major', labelsize=15) 
    
    # Set the X-axis label for the bottom subplots
    for ax in axes[-1, :]:
        ax.set_xlabel('Generation', fontsize=15)
        
    # Set y-labels as titles
    for idx, attribute in enumerate(attributes):
        if attributes[idx] == "pop_intron_ratio":
            axes[0, idx].set_title("Intron Ratio", fontsize=15)
        elif attributes[idx] == "diversity":
            axes[0, idx].set_title("Diversity", fontsize=15)
        else:
            axes[0, idx].set_title("Avg. Tree Size", fontsize=15)
        
    # Set SR names as y-labels
    for idx, attribute in enumerate(sr_fns):
        axes[idx, 0].set_ylabel(f'{sr_fns[idx]}', fontsize=15)
    
    # # Set X-axis ticks based on global_max_length
    xticks_step = max(1, int(global_max_length / 10))  # Ensure step is at least 1
    for ax in axes[-1, :]:
        ax.set_xticks(np.arange(0, global_max_length + xticks_step, step=xticks_step))
        
    # # Set the font size of the y and x-axis for all subplots
    # for ax in axes:
    #     ax.tick_params(axis='both', which='major', labelsize=15) 
    
    # Create a centralized, enlarged legend
    # Remove duplicate labels by using a dictionary
    unique_labels = dict(zip(labels, handles))
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', ncol=min(len(unique_labels), 5), fontsize='20', frameon=True, bbox_to_anchor=(0.5, 0.95))
    
    # Adjust layout to make room for the centralized legend
    plt.subplots_adjust(top=0.90)  # Adjust as needed (e.g., 0.90 places the legend at 90% of the figure height)
    
    # Ensure the 'figures' directory exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"{config_plot}_{'_'.join(attributes)}_vs_generations.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
    # Close the plot to free up memory
    plt.close()

    
if __name__ == "__main__":
    
    # ------ Independent Bloat Study ------------- #
    
    print("\nBloat ~ intron study.")
    attributes = ['diversity', 'avg_tree_size', 'pop_intron_ratio']
    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]
    bloat_thresholds = ["None", 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    max_depth = 10
    
    # Create the final diversity dict by threshold where 'Thresh': [succ (div), ...]
    succ_div_dict = {}
    
    # Append first the sr
    succ_div_dict.update({"Function": sr_fns})
    
    # Iterate over all functions to get values, file with other attributes and plot.
    # types = ["random_mut", "intron_mutation", "half_mut", "intron_plus", "random_plus"]
    types = ["half_mut"]#, "intron_mutation"]
    
    for type_run in types:
        
        # Trackers for plot and intron attribute files
        sr_dfs = {}
        output_data = []
        
        # Temporary list for success diversity dictionary
        threshold_temp = [[] for _ in range(len(bloat_thresholds))]
        for sr_idx, sr in enumerate(sr_fns):
            
            print(f"\nSymbolic Regression Function: {sr}")
            dict_results = []
            dfs = []
            
            sr_temp = []
            for thres in bloat_thresholds:
                
                # --------  Subtree mutations types -------- #
                if type_run == "random_mut":
                    type_run = "introns"
                
                # Read file
                if thres == "None": # Read the inbreeding treatment without threshold
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/bloat/{type_run}_PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:300_TourSize:15_MaxD:10_InitD:3_inbreeding.npy"
                else:
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/bloat/{type_run}_PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:300_TourSize:15_MaxD:10_InitD:3_no_inbreeding.npy"
                
                if type_run == "introns":
                    type_run = "random_mut"
                    
                # Load the data dict
                data = np.load(file_path_name, allow_pickle=True)
                results_inbreeding = data.item()
                dict_results.append(results_inbreeding)
                
                # Pad the data for correct plotting
                global_max_length = util.get_global_max_depth(results_inbreeding)
                df_no_inbreeding = util.pad_dict_and_create_df(results_inbreeding, attributes, global_max_length, 15)
                dfs.append(df_no_inbreeding)
                
                # Compute PEARSON correlations between all attributes or plot an indivdual heatmap. Returns suc_div_thresh = f"{count} ({final_div})"
                succ_div_thres = compute_correlations(results_inbreeding, temp_runs=15, attribute_1="pop_intron_ratio", attribute_2="avg_tree_size", threshold=thres)
                sr_temp.append(succ_div_thres)
    
            # Mutate order from SR-index to Threshold-index. Invert
            for idx, i in enumerate(sr_temp):
                threshold_temp[idx].append(i)
            
            # Append to the diversity success dict
            for idx, thres in enumerate(bloat_thresholds):
                succ_div_dict[thres] = threshold_temp[idx]
                
            # # Write to a file the success - diversity dictionary - Specify file paths
            # dict_file_path = f"{os.getcwd()}/saved_data/introns_study/{type_run}_succ_div_dict_300Gen.json"

            # # Write the dictionary to a file
            # with open(dict_file_path, 'w') as f:
            #     json.dump(succ_div_dict, f)
                
            # Gather data to plot for all SR
            sr_dfs[sr] = {
                'pop_intron_ratio': collect_plot_values(dict_results, 'Inbred Threshold', bloat_thresholds, 'pop_intron_ratio', n_runs=15), 
                'diversity': collect_plot_values(dict_results, 'Inbred Threshold', bloat_thresholds, 'diversity', n_runs=15),
                'avg_tree_size': collect_plot_values(dict_results, 'Inbred Threshold', bloat_thresholds, 'avg_tree_size', n_runs=15),
            }
            
            for idx, thres in enumerate(bloat_thresholds):

                # Extract final values
                intron_ratio = sr_dfs[sr]['pop_intron_ratio'][1][idx][0][-1]
                avg_tree_size = sr_dfs[sr]['avg_tree_size'][1][idx][0][-1]
                diversity = sr_dfs[sr]['diversity'][1][idx][0][-1]
                
                # Extract mean values
                mean_intron_ratio = np.mean(sr_dfs[sr]['pop_intron_ratio'][1][idx][0])
                mean_avg_tree_size = np.mean(sr_dfs[sr]['avg_tree_size'][1][idx][0])
                mean_diversity = np.mean(sr_dfs[sr]['diversity'][1][idx][0])
                
                
                print(f"Threshold: {thres}. Total nº success: {succ_div_dict[thres][sr_idx][:2]}. Average diversity: {mean_diversity:.3f}. Average intron ratio: {mean_intron_ratio:.3f}. Average tree size: {mean_avg_tree_size:.3f}.")

                
            #     # Append to the output data
            #     output_data.append((sr, thres, intron_ratio, avg_tree_size, diversity, mean_intron_ratio, mean_avg_tree_size, mean_diversity))


        # # Save the data to a CSV file. Legend -> ... _ALL means that mean values have been included
        # output_file_path = f"{os.getcwd()}/saved_data/introns_study/{type_run}_symbolic_regression_data_300Gen.csv"
        # columns = ["Function", "Threshold", "Intron Ratio", "Average Tree Size", "Diversity", "Mean Intron Ratio", "Mean Average Tree Size", "Mean Diversity"]
        # output_df = pd.DataFrame(output_data, columns=columns)
        # output_df.to_csv(output_file_path, index=False)
        
        # # Plot 
        # attributes =["diversity", "pop_intron_ratio", "avg_tree_size"]
        # plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot=f"genetic_programming/bloat/{type_run}/300gen_structure_div_intrRatio_treeSize", global_max_length=300)
        
        # Plot all heatmaps in a 2x2 grid
        # plot_combined_corr_heatmaps(dfs, bloat_thresholds, attributes, config_plot=f"{sr}_")