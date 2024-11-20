"""

"""
import os
import numpy as np

treatments = ["inbreeding", "no_inbreeding"]

def get_gp_statistics(bench_name, depths, thresholds, treatment_name, init_depth=3):
        
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
                
                
if __name__ == "__main__":
    
    # thresholds = [6, 7, 8, 9, 10, 11, 12, 13, 14]
    thresholds = [4, 5, 6, 7]
    depths = [6, 7, 8, 9, 10] 
    succes1, no_suc1 = get_gp_statistics("nguyen3", depths, thresholds, "inbreeding", 2)
    succes2, no_suc2 = get_gp_statistics("nguyen3", depths, thresholds, "no_inbreeding", 2)
    
    print(f"Inbreeding success: {succes1}. NO Inbreeding success: {succes2}.")
    print(f"Inbreeding no success: {no_suc1}. NO Inbreeding no success: {no_suc2}.")