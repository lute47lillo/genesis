import os
import numpy as np
import matplotlib.pyplot as plt
import util
import seaborn as sns
from plotting import collect_plot_values
from collections import defaultdict
import scipy.stats as stats

def get_gp_statistics_fit_study(bench_name, thres, folder_name):
    """
        Definition
        -----------
            Compute the total number of successful runs per treatment for a given set-up of hyperparameters.
    """    
    
    dict_results = []
    success = 0

    # Load dict data
    if thres == "None":
        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/{folder_name}/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:10_InitD:3_inbreeding.npy"
    else:
        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/{folder_name}/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:10_InitD:3_no_inbreeding.npy"
        
    # Load the data dict
    data = np.load(file_path_name, allow_pickle=True)
    data_dict = data.item()
    dict_results.append(data_dict)
    
    gens_succ = []
    for run, metrics in data_dict.items():
        generation_success = metrics['generation_success']
        
        if folder_name == 'gp_fit_study':
            percents = metrics['best_percents']
        
        if generation_success < 150:
            success += 1
            gens_succ.append(generation_success)
            
    if folder_name == 'gp_fit_study':
        bootstrapped_values = {'best_percents': collect_plot_values(dict_results, 'Threshold', [thres], 'best_percents', n_runs=15)}
        gen_suc = np.mean(gens_succ)
        return success, percents, bootstrapped_values, gen_suc
    
    return success     
   
    

def plot_percents(sr, thres, thres_percent, percents):
    
    # Sample data from NONE
    none_parent = []
    none_off = []
    for i in percents:
        off_value = i * 100
        par_value = np.abs(100 - off_value)
        
        # Track values
        none_off.append(off_value)
        none_parent.append(par_value)
                
    # Sample data from Threshold
    thres_parent = []
    thres_off = []
    for i in thres_percent:
        off_value = i * 100
        par_value = np.abs(100 - off_value)
        
        # Track values
        thres_off.append(off_value)
        thres_parent.append(par_value)
                
    # Success at generation
    none_success = len(none_off) - 1
    thres_success = len(thres_off) - 1
        
    # Pad the lists
    global_max_length = max(len(thres_off), len(none_off))
    
    if len(thres_off) > len(none_off):
        none_off = util.pad_sublist(none_off, global_max_length)
        none_parent = util.pad_sublist(none_parent, global_max_length)
    else:
        thres_off = util.pad_sublist(thres_off, global_max_length)
        thres_parent = util.pad_sublist(thres_parent, global_max_length)

    # Create a line plot
    x = [i for i in range(len(thres_parent))]
    plt.figure(figsize=(20, 12))
    plt.plot(x, none_off, marker='o', linestyle='-', color='b', label='T: None. Offspring (%)')
    plt.plot(x, none_parent, marker='o', linestyle='-', color='g', label='T: None. Parents (%)')
    plt.axvline(x=none_success, color='r', linestyle='-', linewidth=2, label='None Success Gen')

    
    plt.plot(x, thres_off, marker='x', linestyle='--', color='r', label=f'T: {thres}. Offspring (%)')
    plt.plot(x, thres_parent, marker='x', linestyle='--', color='y', label=f'T: {thres}. Parents (%)')
    plt.axvline(x=thres_success, color='r', linestyle='--', linewidth=2, label=f'T: {thres} Success Gen')

    # Add title and labels
    plt.title(f'{sr}. Ratio of offspring vs parents and viceversa better than other (%)')
    plt.xlabel('Generations')
    plt.ylabel('Fitness (%)')

    # Add legend
    plt.legend()
    
    # Show grid
    plt.grid(True)
    
    # Save the adjusted plot
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Normal visualization save
    plot_filename = f"genetic_programming/fit_study/test_{sr}_{thres}.jpg"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

def plot_bootstrap(thres_best, sr_fns, sr_dict, sr_none_dict, gen_success):
    """
    Plots bootstrapped diversity metrics for different SR problems, comparing
    scenarios with a best threshold and with no threshold (None).

    Parameters:
    - thres_best (float): The best threshold value used in the plots.
    - sr_fns (list): A list of SR function names.
    - sr_dict (dict): Dictionary containing data for SR problems with the best threshold.
    - sr_none_dict (dict): Dictionary containing data for SR problems with no threshold.
    """
    
    # Set Seaborn style for better aesthetics
    sns.set_style("darkgrid")
    
    # Create a figure with two subplots sharing the y-axis
    fig, axs = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
    
    # Titles for the subplots
    subplot_titles = ['With Best Threshold', 'With No Threshold (None)']
    
    # Data sources for the subplots
    data_sources = [sr_dict, sr_none_dict]
    
    # Labels for the plots
    threshold_labels = [f"T:{thres_best}", "T:None"]
    
    # Iterate over the two subplots
    for ax_idx, (ax, data_source, threshold_label, gen_suc) in enumerate(zip(axs, data_sources, threshold_labels, gen_success)):
        for sr_fn in sr_fns:
            sr_attributes = data_source[sr_fn]  # Get the attribute dict for this SR problem
            sr_gen_succ = gen_suc[sr_fn]
            # Get bootstrapped data
            gs_list, div_list, _ = sr_attributes['best_percents']
            
            # Plot each SR run within the current SR problem for the current attribute
            for run_idx, ks in enumerate(gs_list):
                # Extract diversity metrics
                diversity = div_list[run_idx][0]
                upper_diversity = div_list[run_idx][1]
                lower_diversity = div_list[run_idx][2]
                
                # Plot diversity
                ax.plot(ks, diversity, label=f"{sr_fn}. {threshold_label}")
                # ax.axvline(x=sr_gen_succ, linestyle='-', linewidth=2, label=f'{sr_fn}. {threshold_label}. Avg. Gen Success')

                # Optionally, fill between upper and lower diversity bounds
                # Uncomment the following line if you want to show the confidence interval
                # ax.fill_between(ks, lower_diversity, upper_diversity, alpha=0.35)
        
        # Set subplot title and labels
        ax.set_title(subplot_titles[ax_idx], fontsize=18)
        ax.set_xlabel('Generations', fontsize=15)
        if ax_idx == 0:
            ax.set_ylabel('Diversity', fontsize=15)
        
        # Optional: Customize x and y limits if needed
        # ax.set_xlim([min_x, max_x])
        # ax.set_ylim([min_y, max_y])
    
    # Handle legends
    # To avoid duplicate labels in the legend, use a dictionary to keep unique labels
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=15)
    
    # If needed, you can also add a separate legend for the second subplot
    # However, in this case, since legends are shared, it's not necessary
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Ensure the 'figures/genetic_programming/fit_study' directory exists
    figures_dir = os.path.join(os.getcwd(), 'figures', 'genetic_programming', 'fit_study')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Define the plot filename
    plot_filename = "TESTOffspring_vs_parent_fitness_ratio_All.png"
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
    # Close the plot to free up memory
    plt.close(fig)
    
def  evaluate_p_tests(list1, list2):
    
    # Check for Normality
    shapiro_list1 = stats.shapiro(list1)
    shapiro_list2 = stats.shapiro(list2)

    print(f"non-HBC Shapiro-Wilk Test: Statistics={shapiro_list1.statistic:.4f}, p-value={shapiro_list1.pvalue:.4f}")
    print(f"HBC Shapiro-Wilk Test: Statistics={shapiro_list2.statistic:.4f}, p-value={shapiro_list2.pvalue:.4f}")

    # Determine which test to use
    alpha = 0.05
    if shapiro_list1.pvalue > alpha and shapiro_list2.pvalue > alpha:
        print("Both distributions are normal. Using Independent Two-Sample t-Test.")
        t_stat, p_value = stats.ttest_ind(list1, list2, equal_var=False)  # Welch's t-test
        print(f"t-statistic={t_stat:.4f}, p-value={p_value:.4f}")
        
        # Calculate Effect Size (Cohen's d)
        mean1, mean2 = np.mean(list1), np.mean(list2)
        std1, std2 = np.std(list1, ddof=1), np.std(list2, ddof=1)
        pooled_std = np.sqrt(((std1**2) + (std2**2)) / 2)
        cohen_d = (mean1 - mean2) / pooled_std
        print(f"Cohen's d = {cohen_d:.4f}")
        
    else:
        print("At least one distribution is not normal. Using Mann-Whitney U Test.")
        u_stat, p_value = stats.mannwhitneyu(list1, list2, alternative='two-sided')
        print(f"U-statistic={u_stat}, p-value={p_value:.4f}")
        
        # Calculate Effect Size (Rank-Biserial Correlation)
        n1, n2 = len(list1), len(list2)
        rbc = 1 - (2 * u_stat) / (n1 * n2)
        print(f"Rank-Biserial Correlation = {rbc:.4f}")

    # Interpretation
    if 't_stat' in locals():
        if p_value < alpha:
            print("Result: Statistically significant difference between the two groups.")
        else:
            print("Result: No statistically significant difference between the two groups.")
    elif 'u_stat' in locals():
        if p_value < alpha:
            print("Result: Statistically significant difference between the two groups.")
        else:
            print("Result: No statistically significant difference between the two groups.")

if __name__ == "__main__":
    
    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]
    thresholds = ["None", 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    sr_dict = {}
    sr_none_dict = {}
    
    none_gen_suc = {}
    thres_gen_suc = {}
    
    # Best tracker to print single scalar value
    best = {}
    
    for sr in sr_fns:
        print(sr)

        # Trackers
        best_thres_succes = 0
        thres_name = 0
        best[sr] = {}
        for thres in thresholds:
    
            succes, percents, bootstrapped_values, gen_suc = get_gp_statistics_fit_study(sr, thres, "gp_fit_study")
    
            # Save None percents
            if thres == "None":
                # continue
                s_rate = np.round((succes/15)*100, 3)
                best[sr][thres] = (percents, s_rate, gen_suc)
                sr_none_dict[sr] = bootstrapped_values
                none_gen_suc[sr] = gen_suc
            else:
                if succes >= best_thres_succes:
                        
                    # Get the boostrapped values to plot
                    sr_dict[sr] = bootstrapped_values
                    thres_gen_suc[sr] = gen_suc
                    
                    # Update best
                    thres_name = thres
                    best_thres_succes = succes
                    s_rate = np.round((succes/15)*100, 3)
                    
                    best[sr][thres] = (percents, s_rate, gen_suc)
            
            # off_avg_best = np.mean(percents) * 100
            # par_avg_per = np.abs(100 - off_avg_best)
            # print(f"Threshold: {thres}.\n"
            #     f"Success rate: {(succes/15)*100:.3f}%. Avg. Offspring {off_avg_best:.3f}%. Avg. Parent {par_avg_per:.3f}%.\n")
            
    # Get data about the percentages 
    for sr, t_vals in best.items():
        print(f"\nSymbolic Regression: {sr}.")
        thres_none = list(t_vals.keys())[0]
        v_none = t_vals["None"]
        print(f"non-HBC. Offspring Positive: {(np.mean(v_none[0]) * 100):.3f}%.")

        thres = list(t_vals.keys())[-1]
        v = t_vals[thres]
        print(f"HBC. Offspring Positive: {(np.mean(v[0]) * 100):.3f}%.")

        evaluate_p_tests(v_none[0], v[0]) # Evaluate p-values of such runs.
        
    # Average Gen success
    # gen_success = [thres_gen_suc, none_gen_suc]
    
    # PLOT
    # plot_bootstrap(thres_name, sr_fns, sr_dict, sr_none_dict, gen_success)
    
    # # ------ Multiple ---- # # NOTE: Used in the experiment to allow multiple attempts at finding suitable mates using HBC
    # folder_names = ["gp_lambda", "gp_multiple"]
    # thresholds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # print(f"Running multiple chances HBC Mate selection.")
    # for sr in sr_fns:
        
    #     print(sr)

    #     for folder in folder_names:
    #         best_thres_succes = 0
    #         thres_name = 0
    #         for thres in thresholds:
            
    #             succes = get_gp_statistics_fit_study(sr, thres, folder)
    #             if succes >= best_thres_succes:
    #                 thres_name = thres
    #                 best_thres_succes = succes
     
                
    #         print(f"Type: {folder}. Threshold: {thres_name}.\n"
    #             f"Success rate: {(best_thres_succes/15)*100:.3f}%.")