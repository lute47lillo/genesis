import os
import numpy as np

def get_gp_statistics_afpo(bench_name, injection):
    """
        Definition
        -----------
            Compute the total number of successful runs per treatment for a given set-up of hyperparameters.
    """    
    
    dict_results = []
    success = 0

    if injection == 0.0:
        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/gp_lambda/PopSize:300_InThres:None_Mrates:0.0005_Gens:150_TourSize:15_MaxD:10_InitD:3_inbreeding.npy"  
    else:
        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/gp_afpo/RandomInjection_{injection}_PopSize:300_InThres:None_Mrates:0.0005_Gens:150_TourSize:15_MaxD:10_InitD:3_inbreeding.npy"
        
    # Load the data dict
    data = np.load(file_path_name, allow_pickle=True)
    data_dict = data.item()
    dict_results.append(data_dict)
    
    gens_succ = []
    for run, metrics in data_dict.items():
        generation_success = metrics['generation_success']
        
        if generation_success < 150:
            success += 1
            gens_succ.append(generation_success)
    

    return success, gens_succ 
   
if __name__ == "__main__":
    
    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]
    thresholds = ["None", 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    injections = [0.0, 0.1, 0.2, 0.3, 0.5]
    # injections = [0.0, 0.1, 0.2, 0.3]
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
        for inj in injections:
    
            succes, gen_suc = get_gp_statistics_afpo(sr, inj)
    
            print(f"Random Injecction: {inj}.\n"
                  f"Success rate: {(succes/15)*100:.3f}%.\n")
            