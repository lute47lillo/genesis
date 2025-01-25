import os
import plotting as plot
import numpy as np
import util
import pandas as pd
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)   

# --------- Fitness Sharing Experiments ----------------- #

def test_fit_sharing_performance():
    """
        Definition
        -----------
           Compute fitness sharing results. Used in Experimental section results.
           
    """
    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]
    thresholds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, "None"]
    sigma_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    keys = []

    sr_dfs = {}
    dataframes = []
    for sr in sr_fns:
        print(f"\nSymbolic Regression Function: {sr} with Max Depth: 10")
        dict_results = []
        results = {}
        
        for weight in sigma_weights:
            treatment = "no_inbreeding"
            for thres in thresholds:
                success_count = 0
                gens_succs = []
                
                if thres == "None":
                    treatment = "inbreeding"
                    
                # Read file
                if weight != 0.0:
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/sharing/SigmaShare:{weight}_PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:10_InitD:3_{treatment}.npy"
                else:
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:10_InitD:3_{treatment}.npy"
                    
                try:
                    data = np.load(file_path_name, allow_pickle=True)  
                    
                    results_rate = data.item()
                
                    for k, v in results_rate.items():
                        gen = results_rate[k]['generation_success']
                        gens_succs.append(gen)
                        
                        if gen < 150:
                            success_count +=1     
                            
                except FileNotFoundError:
                    print(f"Warning: The file '{file_path_name}' does not exist. Loading default data.")
                    gens_succs = [150] * 15
                    success_count = 0
            
                key = "W:" + str(weight) + "_T:" + str(thres)
                results[key] = {
                    'generation_successes' : gens_succs,
                    'n_successes': success_count
                    }
                keys.append(key)
                # print(f"{key} - Mean Gen. Success: {np.mean(gens_succs)}. Nº total success: {success_count}")
                
                dict_results.append(results_rate)
                
        
        sr_dfs[sr] = {
            'diversity': plot.collect_plot_values(dict_results, '', keys, 'diversity', n_runs=15),
            'best_fitness': plot.collect_plot_values(dict_results, '', keys, 'best_fitness', n_runs=15)
        }        
        
        data = util.compute_composite_score_for_eval(sr_dfs, sr, results)
        
        # Preprocess the data: split the 'key' column into 'W' and 'T'
        data[['W', 'T']] = data['key'].str.extract(r'W:(\d+\.\d+)_T:(.*)')
        data['W'] = data['W'].astype(float)  # Convert W to float
        data['T'] = pd.to_numeric(data['T'], errors='coerce')  # Convert T to float; handle 'None'
        
        
        # Append to dataframes list
        data['function_name'] = f'{sr}'  # Create name for each file
        dataframes.append(data)

        # Reorder individual columns for clarity
        data = data[['W', 'T', 'n_successes', 'diversity', 'mean_gen_success', 'composite_score']]

        # Save the data to a CSV file. Legend -> ... 
        output_file_path = f"{os.getcwd()}/saved_data/sharing/{sr}_sharing_data.csv"
        output_df = pd.DataFrame(data)
        output_df.to_csv(output_file_path, index=False)
        
    # Combine all DataFrames
    combined_data = pd.concat(dataframes, ignore_index=True)
    
    # Reorder columns for clarity
    combined_data = combined_data[['function_name', 'W', 'T', 'n_successes', 'diversity', 'mean_gen_success', 'composite_score']]
    combined_data['T'] = combined_data['T'].fillna('None')

    # Save the combined data for future use
    combined_file_path = f"{os.getcwd()}/saved_data/sharing/TEST_combined_sharing_data.csv"
    combined_data.to_csv(combined_file_path, index=False)
        
    # attributes =["best_fitness", "diversity"]
    # plot.plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot=f"genetic_programming/{sr}/sharing/Depth:{max_depth}_performance", global_max_length=150)

if __name__ == "__main__":
     
    # -------- Fitness Sharing ------- #
    print("\nFitness Sharing")
    test_fit_sharing_performance(10)
    
    
