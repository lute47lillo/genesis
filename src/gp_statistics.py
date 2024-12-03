"""

"""
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

treatments = ["inbreeding", "no_inbreeding"]

def get_gp_statistics(bench_name, depths, thresholds, treatment_name, init_depth=3):
    """
        Definition
        -----------
            Compute the total number of successful runs per treatment for a given set-up of hyperparameters.
    """
        
    success = 0
    no_suc = 0
    # For all Threshold values
    for thres in thresholds:
        
        # For all depths
        for depth in depths:
            
            # Load dict data
            file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:{init_depth}_{treatment_name}.npy"
            data = np.load(file_path_name, allow_pickle=True)
            data_dict = data.item()
            for run, metrics in data_dict.items():
                diversity = metrics['diversity'][-1]
                generation_success = metrics['generation_success']
                
                if generation_success < 150:
                    success += 1
                else:
                    no_suc += 1
              
    return success,  no_suc

def get_gen_avg_inbreed(bench_name, depths, thresholds, treatment_name):
    """
        Definition
        -----------
            Inbreeding does not change behavior wrt Inbred Threshold. Therefore, to make a fair comparison, let's average over total Nº of runs per threshold.
            That way, for each Threshold we have an avg generation success value to compare against no inbreeding treatment.
            
            
    """

    thresholds_gens = {}
    min_max_gens = {}
    
    
    # For all depths
    for depth in depths:
        
        gen_avgs_depths = 0
        min_gens = 0
        max_gens = 0
        
        # For all Threshold values
        for thres in thresholds:
            
            # Gather average running for that depth over that threshold (15 runs)
            gen_avgs_t = 0
            temp_gens_thres = []
            
            # Load dict data
            file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:3_{treatment_name}.npy"
            data = np.load(file_path_name, allow_pickle=True)
            data_dict = data.item()
            for run, metrics in data_dict.items():
                
                # Can do for all metrics
                diversity = metrics['diversity'][-1]
                
                generation_success = metrics['generation_success']
                gen_avgs_t += generation_success
                
                temp_gens_thres.append(generation_success)
                
            # Normalize over number of runs and add to the depths count
            gen_avgs_depths += gen_avgs_t 
            
            # print(f"\nDepth: {depth}. Threshold: {thres}. Range: {temp_gens_thres}.")
            
            # Get ranges
            min_gens += min(temp_gens_thres)
            max_gens += max(temp_gens_thres)
            
            # print(f"\nDepth: {depth}. Threshold: {thres}. Range: min -> {min(temp_gens_thres)}, max -> {max(temp_gens_thres)}.")
            
        # Normalize over all thresholds        
        gen_avgs_depths = gen_avgs_depths / (len(thresholds) * 15) # divide by all runs
        
        # print(f"\nDepth: {depth}. Avg generation success: {gen_avgs_depths}. Range: {min_gens} ~ {max_gens}.")
        # Normalize over all threshold iterations.
        min_gens = min_gens / len(thresholds)
        max_gens = max_gens / len(thresholds)

        # print(f"\nDepth: {depth}. Avg generation success: {gen_avgs_depths}. Range: {min_gens} ~ {max_gens}.")
 
        
        thresholds_gens[depth] = gen_avgs_depths
        min_max_gens[depth] = (min_gens, max_gens)

    return thresholds_gens, min_max_gens
                
def get_gen_no_inbred(bench_name, depths, thresholds, treatment_name):
    """
        Definition
        -----------
            Compute the total number of successful runs per treatment for a given set-up of hyperparameters.
    """
        
    threshold_depths = {}
    min_max_gens_depth = {}
    min_gens = 150
    max_gens = 0
    
    # For all depths
    for depth in depths:
        
        threshold_results = {}
        gen_avgs_depths = 0
        
        min_gens_d = 150
        max_gens_d = 0
        
        # For all Threshold values
        for thres in thresholds:
            gen_avgs_t = 0

            # Load dict data
            file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:3_{treatment_name}.npy"
            data = np.load(file_path_name, allow_pickle=True)
            data_dict = data.item()
            for run, metrics in data_dict.items():

                generation_success = metrics['generation_success']
                
                gen_avgs_t += generation_success
                
            # Normalize over number of runs and add to the depths count
            gen_avgs_depths = gen_avgs_t / 15
            
            # Save for tht threshold the depths
            threshold_results[thres] = gen_avgs_depths
            
            min_gens = min(threshold_results[thres], min_gens)
            max_gens = max(threshold_results[thres], max_gens)
            
            min_gens_d = min(threshold_results[thres], min_gens_d)
            max_gens_d = max(threshold_results[thres], max_gens_d)

            # print(f"\nDepth: {depth}. Threshold: {thres}. Avg generation success: {gen_avgs_depths}. Range: {min_gens_d} ~ {max_gens_d}.")
            
        threshold_depths[depth] = threshold_results
        min_max_gens_depth[depth] = (min_gens_d, max_gens_d)
        # print(f"\nDepth: {depth}. Avg generation success: {gen_avgs_depths}. Range: {min_gens_d} ~ {max_gens_d}.")

        min_max_gens = (min_gens, max_gens)
        
        
    return threshold_depths, min_max_gens, min_max_gens_depth
                
def check_dynamic(srs, depths):
    
    # Define threshold values
    thresholds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 'None'] #TODO: Use maybe dynamoc in appendix: "Dynamic", "Dynamic2"]
    
    # Create dict to store plotting values for all sr
    plot_data = {}
    
    for bench_name in srs:
        
        print(f"\n{bench_name}")
        
        # Create metric to collect
        results = {}
        plot_data[bench_name] = {
                'Threshold': [],
                'Avg_Gen': [],
                'Solved': [],
                'Solved_Percentage': []
        }
        for thres in thresholds:
            results[thres] = {}
            avg_per_thres = 0
            solved_per_thres = 0
    
            for depth in depths:
                results[thres][depth] = {
                    'gens': [],
                    'solved': int
                }
                
                # Read file. None is the inbreeding condition.
                if thres == 'None':
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:3_inbreeding.npy"
                else:
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:3_no_inbreeding.npy"
                
                # Load file
                data = np.load(file_path_name, allow_pickle=True)
                data_dict = data.item()
                yes = 0
                
                # For each run in each depth count how many solved and the generation.
                for run, metrics in data_dict.items():

                    generation_success = metrics['generation_success']
                    
                    if generation_success < 150:
                        yes += 1
                        
                    results[thres][depth]['gens'].append(generation_success)
                    results[thres][depth]['solved'] = yes

                # Do the averages per run, to show by threshold value
                avg_per_thres += np.mean(results[thres][depth]['gens'])
                solved_per_thres += results[thres][depth]['solved']
                # print(f"\nThreshold: {thres}. Depth: {depth}. Avg Gen: {np.mean(results[thres][depth]['gens'])}. Solved: {results[thres][depth]['solved']}")
            
            avg_per_thres = avg_per_thres / len(depths)
            percent = (solved_per_thres / (len(depths) * 15)) * 100
            print(f"\nThreshold: {thres}. Avg Gen per Threshold (all depths): {avg_per_thres:.3f}. Solved: {solved_per_thres} ({percent:.3f}%)")
            
            # Collect by bench name
            plot_data[bench_name]['Threshold'].append(thres)
            plot_data[bench_name]['Avg_Gen'].append(avg_per_thres)
            plot_data[bench_name]['Solved'].append(solved_per_thres)
            plot_data[bench_name]['Solved_Percentage'].append(percent)
          
        # convert to dfs
        
        # df = pd.DataFrame(plot_data)  
        
        # print(df[bench_name]['Threshold'])
        
        # # Convert all 'Threshold' to string to treat 'None' as a category
        # df[bench_name]['Threshold'] = df[bench_name]['Threshold'].astype(str)

        # # Replace 'None' string with a more descriptive label
        # df[bench_name]['Threshold'] = df[bench_name]['Threshold'].replace('None', 'No Threshold') 
        # temp_plot(df, bench_name)
        
        # Process and plot each benchmark
    for bench_name, bench_data in plot_data.items():
        
        # Convert list of dictionaries to DataFrame
        df_bench = pd.DataFrame(bench_data)
        
        # Convert 'Threshold' to string and handle 'None'
        # df_bench['Threshold'] = df_bench['Threshold'].astype(str)
        df_bench['Threshold'] = df_bench['Threshold'].replace('None', 'No Threshold')
        
        # Plot
        temp_plot2(df_bench, bench_name)
            

def temp_plot(df_bench, bench_name):
    """
    Plots Avg Gen and Solved vs. Threshold for a given benchmark.

    Parameters:
    - df_bench (pd.DataFrame): DataFrame containing 'Threshold', 'Avg_Gen', 'Solved', 'Solved_Percentage'.
    - bench_name (str): Name of the benchmark.
    """
    
    # Define category order, placing 'No Threshold' at the end
    sorted_thresholds = sorted(
        [x for x in df_bench['Threshold'].unique() if x != 'No Threshold'],
        key=lambda x: int(x) if x.isdigit() else x
    )
    if 'No Threshold' in df_bench['Threshold'].unique():
        sorted_thresholds.append('No Threshold')
    
    # Create a mapping from category to numerical position
    category_to_pos = {category: pos for pos, category in enumerate(sorted_thresholds)}
    
    # Map 'Threshold' to numerical positions for plotting
    df_bench['Threshold_Pos'] = df_bench['Threshold'].map(category_to_pos)
    
    # Initialize the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Avg_Gen as a line
    color = 'tab:blue'
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Avg Gen per Threshold (all depths)', color=color)
    ax1.plot(
        df_bench['Threshold_Pos'],
        df_bench['Avg_Gen'],
        color=color,
        marker='o',
        label='Avg Gen'
    )
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Set x-ticks to category labels
    ax1.set_xticks(list(category_to_pos.values()))
    ax1.set_xticklabels(list(category_to_pos.keys()), rotation=45)
    
    # Create a second y-axis for Solved
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Solved', color=color)
    
    # Plot Solved as a bar chart
    ax2.bar(
        df_bench['Threshold_Pos'],
        df_bench['Solved'],
        color=color,
        alpha=0.6,
        label='Solved'
    )
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add percentage labels on top of the bars
    for idx, row in df_bench.iterrows():
        pos = row['Threshold_Pos']
        solved = row['Solved']
        percent = row['Solved_Percentage']
        ax2.text(
            pos,
            solved + 0.5,  # Adjust the y-position as needed
            f"{percent:.1f}%",
            ha='center',
            color=color,
            fontsize=8
        )
    
    # Add a title and legends
    plt.title(f'Avg Gen and Nº Solved vs. Threshold for {bench_name}')
    fig.tight_layout()  # Adjust layout to prevent clipping
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc='upper left')
    
    # Ensure the 'figures' directory exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Determine plot filename
    plot_filename = f"{bench_name}_no_temp.png"
    
    # Save the figure
    plot_path = os.path.join(figures_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to {plot_path}")
    
import pandas as pd
import matplotlib.pyplot as plt
import os

def temp_plot2(df_bench, bench_name):
    """
    Plots Avg Gen and Solved vs. Threshold for a given benchmark.

    Parameters:
    - df_bench (pd.DataFrame): DataFrame containing 'Threshold', 'Avg_Gen', 'Solved', 'Solved_Percentage'.
    - bench_name (str): Name of the benchmark.
    """
    
    # Define category order, placing 'No Threshold' at the end
    # Separate numerical and categorical thresholds
    numerical_thresholds = [th for th in df_bench['Threshold'] if th != 'No Threshold']
    numerical_thresholds_sorted = sorted(numerical_thresholds, key=lambda x: int(x))
    
    if 'No Threshold' in df_bench['Threshold'].unique():
        sorted_thresholds = numerical_thresholds_sorted + ['No Threshold']
    else:
        sorted_thresholds = numerical_thresholds_sorted
    
    # Create a mapping from category to numerical position
    category_to_pos = {category: pos for pos, category in enumerate(sorted_thresholds)}
    
    # Map 'Threshold' to numerical positions for plotting
    df_bench['Threshold_Pos'] = df_bench['Threshold'].map(category_to_pos)
    
    # Sort the DataFrame based on 'Threshold_Pos' to ensure correct plotting order
    df_bench_sorted = df_bench.sort_values('Threshold_Pos')
    
    # Initialize the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Avg_Gen as a line
    color = 'tab:blue'
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Avg Gen per Threshold (all depths)', color=color)
    ax1.plot(
        df_bench_sorted['Threshold_Pos'],
        df_bench_sorted['Avg_Gen'],
        color=color,
        marker='o',
        label='Avg Gen'
    )
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Set x-ticks to category labels
    ax1.set_xticks(list(category_to_pos.values()))
    ax1.set_xticklabels(list(category_to_pos.keys()), rotation=45)
    
    # Create a second y-axis for Solved
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Solved', color=color)
    
    # Plot Solved as a bar chart
    ax2.bar(
        df_bench_sorted['Threshold_Pos'],
        df_bench_sorted['Solved'],
        color=color,
        alpha=0.6,
        label='Solved'
    )
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add percentage labels on top of the bars
    for idx, row in df_bench_sorted.iterrows():
        pos = row['Threshold_Pos']
        solved = row['Solved']
        percent = row['Solved_Percentage']
        ax2.text(
            pos,
            solved + 0.5,  # Adjust the y-position as needed
            f"{percent:.1f}%",
            ha='center',
            color=color,
            fontsize=8
        )
    
    # Add a title and legends
    plt.title(f'Avg Gen and Nº Solved vs. Threshold for {bench_name}')
    fig.tight_layout()  # Adjust layou
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc='upper left')
    
    # Ensure the 'figures' directory exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Determine plot filename
    plot_filename = f"Threshold_AvgByDepths/{bench_name}.png"
    
    # Save the figure
    plot_path = os.path.join(figures_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to {plot_path}")


        

if __name__ == "__main__":
    
    thresholds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    depths = [6, 7, 8, 9, 10] 
    
    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5"]
    
    # TODO: This is an Experimental checking for dynamical inbreeding threshold
    # check_dynamic(sr_fns, depths)
    # exit()
    
    for sr in sr_fns:
        
        inbreed_quick = 0
        no_inbreed_quick = 0
        
        print(f"\n{sr}")
        thresholds_gens, min_max_gens = get_gen_avg_inbreed(sr, depths, thresholds, "inbreeding")
        no_threshold_results, no_min_max_gens, no_min_max_gens_depth = get_gen_no_inbred(sr, depths, thresholds, "no_inbreeding")
        
        # Metrics for range.
        total_min_d = 0
        total_max_d = 0
        
        no_total_min_d = 0
        no_total_max_d = 0
        
        # Count for each threshold which one beats in depth
        threshold_gens_counts = []
        
        for dep, thresholds in no_threshold_results.items():
            # print(f"\nMax Depth: {dep}")
            
            
            for threshold, value in thresholds.items():
                
                # Generation Success Speed convergence count.
                if thresholds_gens[dep] <= value:
                    inbreed_quick += 1
                else:
                    no_inbreed_quick += 1
                    threshold_gens_counts.append(threshold)
                    
                # print(f"Inbreeding: {thresholds_gens[dep]} ~ No Inbreeding: {value}")

            # Min-Max range of inbreeding
            (min_d, max_d) = min_max_gens[dep]
            total_min_d += min_d
            total_max_d += max_d
            
            # Min-Max range of no-inbreeding
            (no_min_d, no_max_d) = no_min_max_gens_depth[dep]
            no_total_min_d += no_min_d
            no_total_max_d += no_max_d
            
        #     print("\nBy Depth Generation Range:")
        #     print(f"Depth: {dep}")
        #     print(f"No Inbreeding: {no_min_d:.3f} ~ {no_max_d:.3f}.")
        #     print(f"Inbreeding: {min_d:.3f} ~ {max_d:.3f}.")
            
        # Get the global generation range
        print("\nGlobal Generation Range:")
        total_min_d = total_min_d / len(depths)
        total_max_d = total_max_d / len(depths)   
        
        print(f"No Inbreeding: {no_min_max_gens[0]:.3f} ~ {no_min_max_gens[1]:.3f}.")
        print(f"Inbreeding: {total_min_d:.3f} ~ {total_max_d:.3f}.")
           
        # Get the convergence speed ratio globally
        total_inb = (inbreed_quick / 50) * 100
        total_no_in = (no_inbreed_quick / 50) * 100
        print(f"\nConvergence Speed Ratio (%):")
        print(f"Inbreeding: {total_inb:.3f}% ({inbreed_quick}) ~ No Inbreeding: {total_no_in:.3f}% ({no_inbreed_quick})\n")
        
        # # TODO: I do not think is too relevant
        # print(f"For a specific Inbreeding Threshold. In how many different maximum depths was no_inbreeding quicker?\n")
        # counter = Counter(threshold_gens_counts)
        # sorted_counts = dict(sorted(counter.items()))
        # for k, v in sorted_counts.items():
        #     print(f"Threshold {k}, no_inbreeding quicker in {v} different Max Depths.")

#TODO: Note, there are 750 total different runs. In inbreeding they are divided only by max depth. So, 15 runs * 10 thresholds = 150 runs per depth.
# Note, Each depth is averaged to get a representative succesful generation average. Comparing that to the 50 combinations of no_inbreeding.
# Note, so each Depth~Threshold combo has 15 runs that are averaged. That X number is compared against the Y value of the 150 runs per that depth in inbreeding treatment.
# And there are a total of 50 different comparisons (50 no inbreeding values) paired up with 5 different inbreeding values (1 for each depth)
# We do this because the inbreeding threshold does not make the inbreeding treatment change at all, is a fixed variable. it all varies for maximum depth.

# Therefore, that is what we see in table 1 in convergence speed.
# The total number of runs successes does not matter at all in Table 1 because there are 750 total runs per each.
# TODO: There will be another table where for each depth 150 runs each treatment, what is the convergence, in that we will see how different inbreeding thresholds affect the depth

    # ----------- Legacy ------------- #
    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5"]
    for sr in sr_fns:
        print(sr)
        succes1, no_suc1 = get_gp_statistics(sr, depths, thresholds, "inbreeding", 3)
        succes2, no_suc2 = get_gp_statistics(sr, depths, thresholds, "no_inbreeding", 3)
    
        n_combs = len(depths)*len(thresholds)*15
        print(f"For a total of {n_combs} experimental runs")
        print(f"NO Inbreeding success: {succes2}. Success rate: {(succes2/n_combs)*100:.2f}%")
        print(f"Inbreeding success: {succes1}. Success rate: {(succes1/n_combs)*100:.2f}%\n")
