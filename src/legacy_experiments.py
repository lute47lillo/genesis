import os
import plotting as plot
import numpy as np
import util
import pandas as pd
import legacy_plotting as leg_plot
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)   


# --------- Only Mutation treatment ~ different rates ----------------- #

def test_only_mutation_performance():
    """
        Definition
        -----------
            Test if using only mutation (no Crossover) with different mutation rates is better or worse than using 
            any type of crossover (InbreedBlock or not).
            
        Results
        -----------
            Mutatio Rate: 0.1. Mean Generation success: 138.66666666666666
            Mutatio Rate: 0.01. Mean Generation success: 136.06666666666666
            Mutatio Rate: 0.001. Mean Generation success: 126.8
            Mutatio Rate: 0.0005. Mean Generation success: 120.0
            Mutatio Rate: CrossoverNoIn. Mean Generation success: 23.866666666666667
            Mutatio Rate: CrossoverYesIn. Mean Generation success: 42.8
            
        Answer
        -----------
        
            Removing crossover clearly hurts performance. We also see that when only using mutation rate,
            the best rate is 0.0005 which is what we have been using so far.
    """
    sr_fns = ["nguyen1"]#, "nguyen2", "nguyen3", "nguyen4", "nguyen5"]
    mut_rates = [0.1, 0.01, 0.001, 0.0005, "CrossoverNoIn", "CrossoverYesIn"]
    sr_dfs = {}
    keys = []
    
    for sr in sr_fns:
        print(f"\nSymbolic Regression Function: {sr}")
        dict_results = []
        results = {}
        for rate in mut_rates:
            success_count = 0
            gens_succs = []
        
            # Read file
            if rate == "CrossoverNoIn":
                file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/gp_lambda/PopSize:300_InThres:5_Mrates:0.0005_Gens:150_TourSize:15_MaxD:6_InitD:3_no_inbreeding.npy"
            elif rate == "CrossoverYesIn":
                file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/gp_lambda/PopSize:300_InThres:5_Mrates:0.0005_Gens:150_TourSize:15_MaxD:6_InitD:3_inbreeding.npy"
            else:
                file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/only_mut/FW:1.0_DW:0.0_PopSize:300_InThres:None_Mrates:{rate}_Gens:150_TourSize:15_MaxD:6_InitD:3_mutation.npy"
            
            data = np.load(file_path_name, allow_pickle=True)
            results_rate = data.item()
            
            for k, v in results_rate.items():
                gen = results_rate[k]['generation_success']
                gens_succs.append(gen)
                
                if gen < 150:
                    success_count +=1
                        
            key = "Rate:" + str(rate)
            results[key] = {
                'generation_successes' : gens_succs,
                'n_successes': success_count
                }
            keys.append(key)
            dict_results.append(results_rate)
        
        sr_dfs[sr] = {
            'diversity': plot.collect_plot_values(dict_results, '', keys, 'diversity', n_runs=15),
            'best_fitness': plot.collect_plot_values(dict_results, '', keys, 'best_fitness', n_runs=15)
        }
        
        util.compute_composite_score_for_eval(sr_dfs, sr, results)
        
        
    # plot.plot_generation_successes(results, mut_rates, f"genetic_programming/{sr}/only_mut/gens_vs_runs")

    # attributes =["best_fitness", "diversity"]
    # plot.plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot=f"genetic_programming/{sr}/only_mut/performance", global_max_length=150)

# --------- InbreedUnblock experiments ----------------- #

def test_dynamic_performance():
    """
        Definition
        -----------
            Test if using Dynamic Inbreeding Threshold Adaptation
            
        Results
        -----------

            
        Answer
        -----------
        
    """
    sr_fns = ["nguyen1"]#, "nguyen2", "nguyen3", "nguyen4", "nguyen5"]
    depths = [6, 7, 8, 9, 10]
    thresholds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, "None"]
    keys = []

    sr_dfs = {}
    for sr in sr_fns:
        print(f"\nSymbolic Regression Function: {sr} with max depth: {max_depth}")
        dict_results = []
        results = {}

        for type in ["InbreedBlock", "Dynamic"]:
            
            if type == 'Dynamic': # Files are saved to default of 5
                thresholds = [5]
                
            for thres in thresholds:
                treatment = "no_inbreeding"
                
                for depth in depths:
                    success_count = 0
                    gens_succs = []
                    
                    if thres == "None":
                        treatment = "inbreeding"
                    
                    # Read file
                    if type == "InbreedBlock":
                        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:3_{treatment}.npy"
                    else:
                        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/dynamic/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:3_no_inbreeding.npy"
                    
                    data = np.load(file_path_name, allow_pickle=True)
                    results_rate = data.item()
                    
                    for k, v in results_rate.items():
                        gen = results_rate[k]['generation_success']
                        gens_succs.append(gen)
                        
                        if gen < 150:
                            success_count +=1
                            
                    if type == "Dynamic":
                        key = type + "_D:" + str(depth)
                    else:
                        key = type + "_T:" + str(thres) + "_D:" + str(depth)
                        
                    results[key] = {
                        'generation_successes' : gens_succs,
                        'n_successes': success_count
                        }
                    keys.append(key)
                    
                    dict_results.append(results_rate)
        
        sr_dfs[sr] = {
            'diversity': plot.collect_plot_values(dict_results, '', keys, 'diversity', n_runs=15),
            'best_fitness': plot.collect_plot_values(dict_results, '', keys, 'best_fitness', n_runs=15)
        }
        
        util.compute_composite_score_for_eval(sr_dfs, sr, results)
        
    # plot.plot_generation_successes(results, keys, f"genetic_programming/{sr}/unblock/Depth:{depth}_gens_vs_runs")

    # attributes =["best_fitness", "diversity"]
    # plot.plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot=f"genetic_programming/{sr}/unblock/Depth:{max_depth}_performance", global_max_length=150)


# --------- InbreedUnblock experiments ----------------- #

def test_InbreedUnblock_performance(max_depth=6):
    """
        Definition
        -----------
            Test if using InbreedUnblock (opposite to InbreedBlock) is better or worse than InbreedBlock.
            Is an experiment to determine if allowing really similar individuals to mate is better than 
            completely different ones.
            
        Results
        -----------
            For Max Depth 6:
                Variable: InbreedBlock_5. Mean Generation success: 23.866666666666667
                Variable: InbreedBlock_10. Mean Generation success: 25.533333333333335
                Variable: InbreedBlock_14. Mean Generation success: 15.333333333333334
                Variable: InbreedUnblock_5. Mean Generation success: 61.93333333333333
                Variable: InbreedUnblock_10. Mean Generation success: 73.46666666666667
                Variable: InbreedUnblock_14. Mean Generation success: 18.066666666666666
                
            For Max Depth 10:
                Variable: InbreedBlock_5. Mean Generation success: 29.133333333333333
                Variable: InbreedBlock_10. Mean Generation success: 10.6
                Variable: InbreedBlock_14. Mean Generation success: 24.666666666666668
                Variable: InbreedUnblock_5. Mean Generation success: 75.6
                Variable: InbreedUnblock_10. Mean Generation success: 44.8
                Variable: InbreedUnblock_14. Mean Generation success: 52.53333333333333
            
        Answer
        -----------
            We observe that crossing over similar individuals is worse in terms of performance than applying the 
            InbreedBlock, reinforcing the idea that we need to look for syntactically different individuals.
            
            We also have plotted diversity under figures/{sr}/unblock and we observe that it tanks when doing InbreedUnblock.
        
    """
    sr_fns = ["nguyen1"]#, "nguyen2", "nguyen3", "nguyen4", "nguyen5"]
    thresholds = [5, 6 ,7, 8, 9, 10, 11, 12, 13, 14]
    keys = []

    sr_dfs = {}
    for sr in sr_fns:
        print(f"\nSymbolic Regression Function: {sr} with max depth: {max_depth}")
        dict_results = []
        results = {}

        for type in ["InbreedBlock", "InbreedUnblock"]:
            for thres in thresholds:
                success_count = 0
                gens_succs = []
                
                # Read file
                if type == "InbreedBlock":
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{max_depth}_InitD:3_no_inbreeding.npy"
                else:
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/unblock/FW:1.0_DW:0.0_PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{max_depth}_InitD:3_no_inbreeding.npy"
                
                data = np.load(file_path_name, allow_pickle=True)
                results_rate = data.item()
                
                for k, v in results_rate.items():
                    gen = results_rate[k]['generation_success']
                    gens_succs.append(gen)
                    
                    if gen < 150:
                        success_count +=1
                        
                key = type + "_" + str(thres)
                results[key] = {
                    'generation_successes' : gens_succs,
                    'n_successes': success_count
                    }
                keys.append(key)
                
                dict_results.append(results_rate)
        
        sr_dfs[sr] = {
            'diversity': plot.collect_plot_values(dict_results, '', keys, 'diversity', n_runs=15),
            'best_fitness': plot.collect_plot_values(dict_results, '', keys, 'best_fitness', n_runs=15)
        }
        
        df_sorted = util.compute_composite_score_for_eval(sr_dfs, sr, results)
        
    # plot.plot_generation_successes(results, keys, f"genetic_programming/{sr}/unblock/Depth:{max_depth}_gens_vs_runs")

    # attributes =["best_fitness", "diversity"]
    # plot.plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot=f"genetic_programming/{sr}/unblock/Depth:{max_depth}_performance", global_max_length=150)

    return df_sorted
# --------- Fitness + Diversity selection Experiments ----------------- #

def test_fit_and_div_selection_performance():
    """
        Definition
        -----------
           Testing if using Absolute Error fitness + diversity as total individual fitness is better than just using abs. error fitness.
           
        Results
        -----------
          
        Answer
        -----------
            We observe that introducing diversity as part of the guiding selection is hurtful for the performance.
           
    """
    sr_fns = ["nguyen1"]#, "nguyen2", "nguyen3", "nguyen4", "nguyen5"]
    deiversity_weights = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    thresholds = [5, 10, 14]
    keys = []

    sr_dfs = {}
    for sr in sr_fns:
        print(f"\nSymbolic Regression Function: {sr}")
        dict_results = []
        results = {}
        for thres in thresholds:
            for weight in deiversity_weights:
                success_count = 0
                gens_succs = []
                
                # REad the file
                file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/diversity/FW:1.0_DW:{weight}_PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:6_InitD:3_no_inbreeding.npy"
                
                data = np.load(file_path_name, allow_pickle=True)
                results_rate = data.item()
                
                for k, v in results_rate.items():
                    gen = results_rate[k]['generation_success']
                    gens_succs.append(gen)
                    
                    if gen < 150:
                        success_count +=1
                
                key = f"T:{thres}_W:{weight}"
                results[key] = {
                    'generation_successes' : gens_succs,
                    'n_successes': success_count
                    }
                keys.append(key)
                
                dict_results.append(results_rate)
        
        sr_dfs[sr] = {
            'diversity': plot.collect_plot_values(dict_results, '', keys, 'diversity', n_runs=15),
            'best_fitness': plot.collect_plot_values(dict_results, '', keys, 'best_fitness', n_runs=15)
        }
        
        util.compute_composite_score_for_eval(sr_dfs, sr, results)
    # plot.plot_generation_successes(results, keys, f"genetic_programming/{sr}/diversity/gens_vs_runs")

    # attributes =["best_fitness", "diversity"]
    # plot.plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot=f"genetic_programming/{sr}/diversity/performance", global_max_length=150)

# --------- Only Crossover ~ NO Mutation Experiments ----------------- #

def test_only_crossover_performance(sr_fns, thresholds, max_depth=6):
    """
        Definition
        -----------
            TODO: Missing None threhsold
            Test if only using crossover without mutation is beneficial.
                - Test the effect of mutation in the InbreedBlock crossover by removing mutation or not.
           
        Results
        -----------
            For Max Depth 6:
                Variable: UseMutation_5. Mean Generation success: 23.866666666666667
                Variable: UseMutation_10. Mean Generation success: 8.666666666666666
                Variable: UseMutation_14. Mean Generation success: 15.333333333333334
                
                Variable: NoUseMutation_5. Mean Generation success: 32.46666666666667
                Variable: NoUseMutation_10. Mean Generation success: 36.2
                Variable: NoUseMutation_14. Mean Generation success: 32.733333333333334
                
            For Max Depth 10:
                Variable: UseMutation_5. Mean Generation success: 29.133333333333333
                Variable: UseMutation_10. Mean Generation success: 10.6
                Variable: UseMutation_14. Mean Generation success: 24.666666666666668
                
                Variable: NoUseMutation_5. Mean Generation success: 57.666666666666664
                Variable: NoUseMutation_10. Mean Generation success: 17.666666666666668
                Variable: NoUseMutation_14. Mean Generation success: 13.533333333333333
            
        Answer
        -----------
            Inconclusive results. 
                - For MaxDepth 6 it clearly seems that usign mutation is better when appliedo to InbreedBlock
                - For MaxDepth 10 it is worse to use mutation in one case where Threshold is 14 (?)
                    - It also seems that as the threshold increasig it gets faster to not use the mutation.
                    
                    • Run for More threhsolds.
                    • Run all depths with threshlold 14.
                    • Run for Nguyen2 function.
           
    """
    keys = []
    sr_dfs = {}

    
    for sr in sr_fns:
        print(f"\nSymbolic Regression Function: {sr} with max depth: {max_depth}")
        dict_results = []
        results = {}
        for type in ["UseMutation", "NoUseMutation"]: # NoUseMutation is like only_cross
            treatment = "no_inbreeding" 
            for thres in thresholds:
                success_count = 0
                gens_succs = []
                
                if thres == "None":
                    treatment = "inbreeding"
                  
                # Read file
                if type == "UseMutation":  
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{max_depth}_InitD:3_{treatment}.npy"
                else:
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/only_cross/PopSize:300_InThres:{thres}_Mrates:0.0_Gens:150_TourSize:15_MaxD:{max_depth}_InitD:3_{treatment}.npy"
                
                data = np.load(file_path_name, allow_pickle=True)
                results_rate = data.item()
                
                for k, v in results_rate.items():
                    gen = results_rate[k]['generation_success']
                    gens_succs.append(gen)
                    
                    if gen < 150:
                        success_count +=1
                
                key = type + "_" + str(thres)
                results[key] = {
                    'generation_successes' : gens_succs,
                    'n_successes': success_count
                    }
                keys.append(key)
                dict_results.append(results_rate)
                # print(f"{key} - Mean Gen. Success: {np.mean(gens_succs)}. Nº total success: {success_count}")
        
        sr_dfs[sr] = {
            'diversity': plot.collect_plot_values(dict_results, '', keys, 'diversity', n_runs=15),
            'best_fitness': plot.collect_plot_values(dict_results, '', keys, 'best_fitness', n_runs=15)
        }
        
        util.compute_composite_score_for_eval(sr_dfs, sr, results)


    # plot.plot_generation_successes(results, keys, f"genetic_programming/{sr}/only_cross/MaxDepth:{max_depth}_gens_vs_runs")

    # attributes =["best_fitness", "diversity"]
    # plot.plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot=f"genetic_programming/{sr}/only_cross/MaxDepth:{max_depth}_performance", global_max_length=150)


# --------- Fitness Sharing Experiments ----------------- #

def test_fit_sharing_performance(max_depth=10):
    """
        Definition
        -----------
           TODO
        Results
        -----------
            
        Answer
        -----------
           
    """
    sr_fns = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]
    thresholds = [5, 10, 14, "None"]
    sigma_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    keys = []

    sr_dfs = {}
    for sr in sr_fns:
        print(f"\nSymbolic Regression Function: {sr} with Max Depth: {max_depth}")
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
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/sharing/SigmaShare:{weight}_PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{max_depth}_InitD:3_{treatment}.npy"
                else:
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{sr}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{max_depth}_InitD:3_{treatment}.npy"
                    
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
        
        util.compute_composite_score_for_eval(sr_dfs, sr, results)
        
    plot.plot_generation_successes(results, keys, f"genetic_programming/{sr}/sharing/Depth:{max_depth}_gens_vs_runs")

    attributes =["best_fitness", "diversity"]
    plot.plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot=f"genetic_programming/{sr}/sharing/Depth:{max_depth}_performance", global_max_length=150)

# --------- Exploration vs Exploitation Experiments ----------------- #

def test_fit_explore_vs_exploit_performance(max_depth=6):
    """
        Definition
        -----------
           TODO: Need to change some files names
        Results
        -----------
            
        Answer
        -----------
           
    """
    sr_fns = ["nguyen1"]#, "nguyen2", "nguyen3", "nguyen4", "nguyen5"]
    slopes = ["Positive_continuous", "Negative_continuous", "Positive_abrupt", "Negative_abrupt", "None", 5, 10, 14]
    gens_change = [10, 25, 50, 75]
    keys = []

    sr_dfs = {}
    for sr in sr_fns:
        print(f"\nSymbolic Regression Function: {sr} with Max Depth: {max_depth}")
        dict_results = []
        results = {}
        
        for slope in slopes:
            treatment = "no_inbreeding"
            
            if type(slope) == int or slope == "None":
                gens_change = [1]
                
            for gen_change in gens_change:
                success_count = 0
                gens_succs = []
                
                # Read the file
                if type(slope) == int:
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/gp_lambda/PopSize:300_InThres:{slope}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{max_depth}_InitD:3_no_inbreeding.npy"
                elif slope == "None":
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/gp_lambda/PopSize:300_InThres:{slope}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{max_depth}_InitD:3_inbreeding.npy"
                else:
                    file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/linear/GenChangeL{gen_change}_PopSize:300_InThres:{slope}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{max_depth}_InitD:3.npy"
                data = np.load(file_path_name, allow_pickle=True)
                results_rate = data.item()
                
                for k, v in results_rate.items():
                    gen = results_rate[k]['generation_success']
                    gens_succs.append(gen)
                    
                    if gen < 150:
                        success_count +=1
                
                if slope == "Positive_continuous":
                    key = "Pos_ExploitThenExplore" + "_G:" + str(gen_change)
                elif slope == "Positive_abrupt":
                    key = "Pos_ExploreThenExploit" + "_G:" + str(gen_change)
                elif slope == "Negative_continuous":
                    key = "Neg_ExploreThenExploit" + "_G:" + str(gen_change)
                elif slope == "Negative_abrupt":
                    key = "Neg_ExploitThenExplore" + "_G:" + str(gen_change)
                else:
                    key = "T:" + str(slope)
        
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
        
        util.compute_composite_score_for_eval(sr_dfs, sr, results)
        
    plot.plot_generation_successes(results, keys, f"genetic_programming/{sr}/linear/Depth:{max_depth}_gens_vs_runs")

    attributes =["best_fitness", "diversity"]
    plot.plot_all_sr_in_columns(sr_dfs, sr_fns, attributes, config_plot=f"genetic_programming/{sr}/linear/Depth:{max_depth}_performance", global_max_length=150)



if __name__ == "__main__":
    
    # -------- Only Mutation ~ No Crossover ------- #
    # print("\nOnly Mutation ~ No Crossover")
    # test_only_mutation_performance()
    
    # -------- InbreedUnblock ------- #
    # print("\nInbreedUnblock")
    # unblock_dfs = []
    # df_6 = test_InbreedUnblock_performance(6)
    # unblock_dfs.append(df_6)
    # df_10 = test_InbreedUnblock_performance(10)
    # unblock_dfs.append(df_10)
    
    # leg_plot.plot_scatter_InbreedUnblock(unblock_dfs, f"genetic_programming/nguyen1/unblock/scatter", [6, 10], titles=None)
    
    # -------- Abs Error Fitness + Diversity selection ------- #
    # print("\nAbs Error Fitness + Diversity selection")
    # test_fit_and_div_selection_performance()
    
    
    # -------- Only Crossover - No mutation ------- #
    # print("\nOnly Crossover - No mutation")
    # sr_fns = ["nguyen1", "nguyen2"]#, "nguyen3", "nguyen4", "nguyen5"]
    
    # Only_cross_1.txt
    # thresholds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, "None"]
    # test_only_crossover_performance(sr_fns, thresholds, 6)
    # test_only_crossover_performance(sr_fns, thresholds, 10)
    
    # Only_cross_2.txt
    # sr_fns = ["nguyen1"]
    # thresholds = [14, "None"] # Only NG1
    # test_only_crossover_performance(sr_fns, thresholds, 6)
    # test_only_crossover_performance(sr_fns, thresholds, 7) # Only NG1 and 14 - None
    # test_only_crossover_performance(sr_fns, thresholds, 8) # Only NG1 and 14 - None
    # test_only_crossover_performance(sr_fns, thresholds, 9) # Only NG1 and 14 - None
    # test_only_crossover_performance(sr_fns, thresholds, 10)
    
    
    # Only_cross_3.txt
    # sr_fns = ["nguyen1", "nguyen2"]
    # thresholds = [5, 10, 14, "None"] # All SR
    # test_only_crossover_performance(sr_fns, thresholds, 6)
    # test_only_crossover_performance(sr_fns, thresholds, 10)

    
    # -------- Fitness Sharing ------- #
    print("\nFitness Sharing")
    # test_fit_sharing_performance(6)
    # test_fit_sharing_performance(7)
    # test_fit_sharing_performance(8)
    # test_fit_sharing_performance(9)
    # test_fit_sharing_performance(10)
    
    # -------- Exploration vs Exploitation ------- #
    # print("\nExplore vs Exploit")
    # test_fit_explore_vs_exploit_performance(6)
    # test_fit_explore_vs_exploit_performance(10)
    
    # file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/gp_lambda/PopSize:300_InThres:14_Mrates:0.0005_Gens:150_TourSize:15_MaxD:6_InitD:3_no_inbreeding.npy"
    # data = np.load(file_path_name, allow_pickle=True)
    # results_rate = data.item()
    
    # attribute_lists = np.mean([results_rate[run]['generation_success'] for run in range(15)])
    # div = np.mean([results_rate[run]['diversity'][-1] for run in range(15)])
    # print(f"Avg Gen: {attribute_lists}. Final Diversity: {div}")
    
    # file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen1/diversity/FW:1.0_DW:0.0_PopSize:300_InThres:14_Mrates:0.0005_Gens:150_TourSize:15_MaxD:6_InitD:3_no_inbreeding.npy"
    # data = np.load(file_path_name, allow_pickle=True)
    # results_rate = data.item()
    
    # attribute_lists = np.mean([results_rate[run]['generation_success'] for run in range(15)])
    # div = np.mean([results_rate[run]['diversity'][-1] for run in range(15)])
    # print(f"Avg Gen: {attribute_lists}. Final Diversity: {div}")
    
    
    # --------- Dynamic Inbreeding Threshold ----- #
    # print("\nDynamic Inbreeding Threshold")
    # test_dynamic_performance()
    
    
