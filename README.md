GECCO 2025 submission

  To reproduce the results there are multiple ways to proceed. 
  
  You can manually run each experiment by setting the specific parameters or you can use the bash script bloat_jobs.sh 

  Manually:

    Refer to the source file that will run the specific experiment, and execute such as

      Example:
      time python src/gp_base.py --benchmark nguyen5 --inbred_threshold 5

    This will execute HBC with threshold 5. Value 1 is used to indicate that we want to run a non-HBC version.

    NOTE: in the util.py file there is a list of arguments and hyperparameters and their descriptions. There you will find all of the important variations with which we ran our experiments.


  Automatically:

    If you have access to run sbatch files. 
    
      1) Go to bloat_jobs.sh, uncomment the experiment that you would like to run.
      2) Go to bloat.sbatch, set your variables, uncomment the specific experiment that you wanted to run and is associated in bloat_jobs.sh
      3) Execute.

    Note: The bash script is modifiable. If you do not have access to running sbatch files. Then, you can simply change the file to which bloat_jobs.sh file is sending the hyperparameters for the experiment and you could run them.

  Plots and collection results.

    In order to collect results. Once we have executed all the necessary run for a specific experiment, then we need to go to specific files and get the figures and results used in the paper.
  
    Performance
      - gp_statistics.py will provide the performance results for HBC vs non-HBC.
      - afpo_results.py will get results provided in the Appendix.
      - gp_fit_study.py will provide results about offspring rates wrt parents.
      - semantics_results.py will provide results about SSC and SAC in all forms.
  
    Introns
      1) gp_plotting.py will combine the data and produce the 3x8 plot with population intron ratio, average tree size and diversity over time.
      2) Then, we need to run bloat_results_2.py in order to merge that data, get some results used in the tables and prepare for plotting.
      3) The intron 5 subplot Figure with intron ratios and average tree sizes will be obtained after following the previous steps and executing bloat_results_3.py


For reference, all the experiments are shown below. $ARGS$ Indicate where we need to specify different hyperparameters as they have been explored in the paper.
  
#####################
#750-run tests      #--------- For performance evaluation ----------- w/ fixed max depth of 10
#####################

<!-- echo "Running job with Inbred Threshold: $ARG2 and Maximum Depth: 10" -->

time python src/gp_base.py --benchmark nguyen1 --inbred_threshold "$ARG2" --max_depth "$ARG1"
time python src/gp_base.py --benchmark nguyen2 --inbred_threshold "$ARG2" --max_depth "$ARG1"
time python src/gp_base.py --benchmark nguyen3 --inbred_threshold "$ARG2" --max_depth "$ARG1"
time python src/gp_base.py --benchmark nguyen4 --inbred_threshold "$ARG2" --max_depth "$ARG1"
time python src/gp_base.py --benchmark nguyen5 --inbred_threshold "$ARG2" --max_depth "$ARG1"
time python src/gp_base.py --benchmark nguyen6 --inbred_threshold "$ARG2" --max_depth "$ARG1"
time python src/gp_base.py --benchmark nguyen7 --inbred_threshold "$ARG2" --max_depth "$ARG1"
time python src/gp_base.py --benchmark nguyen8 --inbred_threshold "$ARG2" --max_depth "$ARG1"

#####################
#non-HBC AFPO       #--------- Injecting Random Individuals ----------- w/ fixed max depth of 10
#####################

<!-- echo "Running job with random injection: $ARG2 and Maximum Depth: 10" -->

AFPO non-HBC
time python src/afpo_gp_base.py --benchmark nguyen1 --inbred_threshold 1 --random_injection "$ARG2" 
time python src/afpo_gp_base.py --benchmark nguyen2 --inbred_threshold 1 --random_injection "$ARG2" 
time python src/afpo_gp_base.py --benchmark nguyen3 --inbred_threshold 1 --random_injection "$ARG2" 
time python src/afpo_gp_base.py --benchmark nguyen4 --inbred_threshold 1 --random_injection "$ARG2" 
time python src/afpo_gp_base.py --benchmark nguyen5 --inbred_threshold 1 --random_injection "$ARG2" 
time python src/afpo_gp_base.py --benchmark nguyen6 --inbred_threshold 1 --random_injection "$ARG2" 
time python src/afpo_gp_base.py --benchmark nguyen7 --inbred_threshold 1 --random_injection "$ARG2" 
time python src/afpo_gp_base.py --benchmark nguyen8 --inbred_threshold 1 --random_injection "$ARG2" 

#####################
#Semantics          #--------- For Semantics evaluation ----------- w/ fixed max depth of 10
#####################

<!-- echo "Running job with HBC Threshold: $ARG1 and Semantics Type: SSC and Low Sensitivity: $ARG2 and High Sensitivity $ARG3" -->

time python src/gp_semantics.py --benchmark nguyen1 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
time python src/gp_semantics.py --benchmark nguyen2 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
time python src/gp_semantics.py --benchmark nguyen3 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
time python src/gp_semantics.py --benchmark nguyen4 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
time python src/gp_semantics.py --benchmark nguyen5 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
time python src/gp_semantics.py --benchmark nguyen6 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
time python src/gp_semantics.py --benchmark nguyen7 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
time python src/gp_semantics.py --benchmark nguyen8 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"

<!-- 
echo "Running job with HBC Threshold: $ARG1 and Semantics Type: SAC and Low Sensitivity: $ARG2" -->

time python src/gp_semantics.py --benchmark nguyen1 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
time python src/gp_semantics.py --benchmark nguyen2 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
time python src/gp_semantics.py --benchmark nguyen3 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
time python src/gp_semantics.py --benchmark nguyen4 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
time python src/gp_semantics.py --benchmark nguyen5 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
time python src/gp_semantics.py --benchmark nguyen6 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
time python src/gp_semantics.py --benchmark nguyen7 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
time python src/gp_semantics.py --benchmark nguyen8 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"


#####################
#Fit sharing   #--------- Fitness Sharing Experiments ----------- w/ different inbreeding thresholds and depths and sigma shares
#####################

<!-- echo "Running job with Inbred Threshold: $ARG2 and Maximum Depth: $ARG1 and sigma share weight: $ARG3" -->

time python src/gp_sharing_parallel.py --benchmark nguyen1 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
time python src/gp_sharing_parallel.py --benchmark nguyen2 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
time python src/gp_sharing_parallel.py --benchmark nguyen3 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
time python src/gp_sharing_parallel.py --benchmark nguyen4 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
time python src/gp_sharing_parallel.py --benchmark nguyen5 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
time python src/gp_sharing_parallel.py --benchmark nguyen6 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
time python src/gp_sharing_parallel.py --benchmark nguyen7 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
time python src/gp_sharing_parallel.py --benchmark nguyen8 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"


#####################
#Introns / Bloat #--------- Intron ratio and bloat effect ----------- w/ different inbreeding thresholds and depths and sigma shares
#####################

<!-- echo "Running job with Inbred Threshold: $ARG2" -->

time python src/gp_introns.py --benchmark nguyen1 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
time python src/gp_introns.py --benchmark nguyen2 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
time python src/gp_introns.py --benchmark nguyen3 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
time python src/gp_introns.py --benchmark nguyen4 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
time python src/gp_introns.py --benchmark nguyen5 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
time python src/gp_introns.py --benchmark nguyen6 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
time python src/gp_introns.py --benchmark nguyen7 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
time python src/gp_introns.py --benchmark nguyen8 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random

  
