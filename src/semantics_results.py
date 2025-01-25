import os
import numpy as np
import util

def compute_success(results_inbreeding, temp_runs=15):
    """
        Definition
        -----------
           Compute semantics results and plots. Used in Figures and Tables accros section results.
    
    """

    n_runs = temp_runs
    
    # nº total success
    count = 0
    avg_gen = 0
    gen_success = [results_inbreeding[run]['generation_success'] for run in range(n_runs)]
    for gen in gen_success:
        if gen < 150:
            count += 1
        avg_gen += gen
            
    avg_gen = avg_gen / n_runs
    
    # Pad all sublists in diversity_no_inbreeding
    attr = [results_inbreeding[run]['avg_tree_size'] for run in range(n_runs)] # (15, 151) 
    max_length_attr = max(len(sublist) for sublist in attr if len(sublist) != 0)         
    attr_padded = [util.pad_sublist(sublist, max_length_attr) for sublist in attr]

    # Update results_no_inbreeding with padded diversity lists
    for run in range(n_runs):
        results_inbreeding[run]['avg_tree_size'] = attr_padded[run]
    
    attr_size = [results_inbreeding[run]['avg_tree_size'] for run in range(n_runs)] # (15, 151)  
    final_average_tree_size = np.mean(np.mean(np.array(attr_size), axis=0))

    # ---------------------------------------- #
    
    # Collect diversity data
    diversity_print = [results_inbreeding[run]['diversity'] for run in range(n_runs)]
    max_length_attr_div = max(len(sublist) for sublist in diversity_print)
    attr_div_padded = [util.pad_sublist(sublist, max_length_attr_div) for sublist in diversity_print]
        
    for run in range(n_runs):
        results_inbreeding[run]['diversity'] = attr_div_padded[run]   
          
    attr_div = [results_inbreeding[run]['diversity'] for run in range(n_runs)]
    final_average_div = np.mean(np.mean(np.array(attr_div), axis=0))

    # Put as 1 value
    suc_div_thresh = f"{count} ({final_average_div})"
    
    return suc_div_thresh, final_average_tree_size

if __name__ == "__main__":
    
    # ------ Independent Bloat Study ------------- #
    
    print("\nSemantics Study")
    attributes = ['diversity', 'avg_tree_size']
    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]
    thresholds = ["None", 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    low_sensitivities=[0.02, 0.04, 0.06, 0.08]
    high_sensitivities=[8.0, 10.0, 12.0]
   
    
    # Create the final diversity dict by threshold where 'Thresh': [succ (div), ...]
    succ_div_dict = {}
    
    # Append first the sr
    succ_div_dict.update({"Function": sr_fns})
    
    # Iterate over all functions to get values, file with other attributes and plot.
    semantic_types = ["SSC", "SAC"]
    bests = {}
    for semantics in semantic_types:
        
        # Trackers for plot and intron attribute files
        sr_dfs = {}
        output_data = []
        
        # Temporary list for success diversity dictionary
        threshold_temp = [[] for _ in range(len(thresholds))]
        bests[semantics] = {}
        for sr_idx, sr in enumerate(sr_fns):
            
            print(f"\nSymbolic Regression Function: {sr}")
            dict_results = []
            dfs = []
            sr_temp = []
            
            # Track to pring only best value
            best_value = 0
            best_tree_size = 999999
            for thres in thresholds:
                for lows in low_sensitivities:
                    prev_high=1
                    for highs in high_sensitivities:
                
                        # --------  Semantic Crossover types -------- #
                    
                        # Read file
                        if thres == "None":
                            if semantics == "SAC":
                                file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/gp_semantics/Semantics:{semantics}_LowS:{lows}_InThres:None_inbreeding.npy" 
                            else:
                                file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/gp_semantics/Semantics:{semantics}_LowS:{lows}_HighS:{highs}_InThres:None_inbreeding.npy" 

                        else:
                            if semantics == "SAC":
                                file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/gp_semantics/Semantics:{semantics}_LowS:{lows}_InThres:{thres}_no_inbreeding.npy" 
                            else:
                                file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/gp_semantics/Semantics:{semantics}_LowS:{lows}_HighS:{highs}_InThres:{thres}_no_inbreeding.npy" 
                    
                        # Load the data dict
                        data = np.load(file_path_name, allow_pickle=True)
                        results_inbreeding = data.item()
                        succ_div_thres, avg_tree_size = compute_success(results_inbreeding, temp_runs=15)
                        
                        if int(succ_div_thres[:1]) > best_value:
                            bests[semantics][sr] = (thres, int(succ_div_thres[:1]), succ_div_thres[2:], avg_tree_size, lows, highs)
                            best_value = int(succ_div_thres[:1])
                        if best_value == int(succ_div_thres[:1]) and avg_tree_size < best_tree_size:
                            bests[semantics][sr] = (thres, int(succ_div_thres[:1]), succ_div_thres[2:], avg_tree_size, lows, highs)
                            best_tree_size = avg_tree_size
                            
    for sems, vals in bests.items():
        print(f"\nSemantics: {sems}.")
        for funcs, attrs in vals.items():
            if sems == "SSC":
                print(f"SR: {funcs}: Threshold: {attrs[0]}.\tLow Sensitivity: {attrs[4]}. High Sensitivity: {attrs[5]}.\tSuccess rate (%): {(attrs[1]/15)*100:.2f}. Avg. Diversity: {attrs[2]}. Avg. Tree Size: {attrs[3]}")
            else:
                print(f"SR: {funcs}: Threshold: {attrs[0]}.\tLow Sensitivity: {attrs[4]}.\tSuccess rate (%): {(attrs[1]/15)*100:.2f}. Avg. Diversity: {attrs[2]}. Avg. Tree Size: {attrs[3]}")