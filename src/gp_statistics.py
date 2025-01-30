"""

"""
import os
import numpy as np
import pandas as pd
import pandas as pd
from scipy.stats import fisher_exact

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
            NOTE: Inbreeding does not change behavior wrt Inbred Threshold file ending terminations.
            Therefore, to make a fair comparison, let's average over total Nº of runs per threshold.
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

if __name__ == "__main__":
    """
        Gets the Tables for HBC performance data.
            - Comparison vs Random subtree crossover.
            - Comparison between depths and thresholds.
       
    """
    
    thresholds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    depths = [10] # Used to get Main Experiments material results

    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5","nguyen6", "nguyen7", "nguyen8"]
    
    for sr in sr_fns:
        
        inbreed_quick = 0
        no_inbreed_quick = 0
        
        print(f"\n{sr}")
        no_threshold_results, no_min_max_gens, no_min_max_gens_depth = get_gen_no_inbred(sr, depths, thresholds, "no_inbreeding")
        thresholds_gens, min_max_gens = get_gen_avg_inbreed(sr, depths, thresholds, "inbreeding")
        
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
                
            # Min-Max range of inbreeding
            (min_d, max_d) = min_max_gens[dep]
            total_min_d += min_d
            total_max_d += max_d
            
            # Min-Max range of no-inbreeding
            (no_min_d, no_max_d) = no_min_max_gens_depth[dep]
            no_total_min_d += no_min_d
            no_total_max_d += no_max_d
            
        # Get the global generation range
        print("\nGlobal Generation Range:")
        total_min_d = total_min_d / len(depths)
        total_max_d = total_max_d / len(depths)   
        
        print(f"No Inbreeding: {no_min_max_gens[0]:.3f} ~ {no_min_max_gens[1]:.3f}.")
        print(f"Inbreeding: {total_min_d:.3f} ~ {total_max_d:.3f}.")
           
        # Get the convergence speed ratio globally
        n_combinations = len(depths) * len(thresholds)
        total_inb = (inbreed_quick / n_combinations) * 100
        total_no_in = (no_inbreed_quick / n_combinations) * 100
        print(f"\nConvergence Speed Ratio (%):")
        print(f"Inbreeding: {total_inb:.3f}% ({inbreed_quick}) ~ No Inbreeding: {total_no_in:.3f}% ({no_inbreed_quick})\n")
        

    # ----------- Legacy ------------- #
    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]

    for sr in sr_fns:
        print(sr)
        # depths = [6, 7, 8, 9, 10] # Used to get Supplemental material results
        depths = [10] # Used to get Main Experiments material results

        thresholds_none = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14] #["None"]
        succes1, no_suc1 = get_gp_statistics(sr, depths, thresholds_none, "inbreeding", 3)
        
        thresholds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        succes2, no_suc2 = get_gp_statistics(sr, depths, thresholds, "no_inbreeding", 3)
    
        n_combs = len(depths)*len(thresholds)*15
        print(f"For a total of {n_combs} experimental runs")
        print(f"NO Inbreeding success: {succes2}. Success rate: {(succes2/n_combs)*100:.2f}%")
        
        n_combs_none = len(depths)*len(thresholds_none)*15
        print(f"For a total of {n_combs_none} experimental runs")
        print(f"Inbreeding success: {succes1}. Success rate: {(succes1/n_combs_none)*100:.2f}%\n")
        
    