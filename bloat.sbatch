#!/bin/bash
# Specify a partition. dggpu (deepgreen) -> GPUS, bdgpu -> quicker prototyping (blackdiamond) on GPUs, bluemoon -> basic job without GPU usage.
#SBATCH --partition=bluemoon
# Request physical nodes (usually 1).
#SBATCH --nodes=1
# Request tasks (usually 1).
#SBATCH --ntasks=1
# Request processor cores (only if your program has multithreading/multiprocessing).
#SBATCH --cpus-per-task=14
# Request GPUs (delete if not needed).
# SBATCH --gres=gpu:2
# Specify memory.
#SBATCH --mem=20G
# Maximum time limit of 10 minutes.
# Format: D-HH:MM:SS. Leading zeroes can be omitted, but included here for explanation.
#SBATCH --time=30:00:00
# Decide the name of the job.
#SBATCH --job-name=SSC7
# Name of the slurm output. x = job name.
#SBATCH --output=%j_%x.out

echo 'Running in:' $(pwd)
source ~/.bash_profile

# Activate conda env.
conda activate neurobotics

# Print GPU information
nvidia-smi

# Exit immediately if any command has a non-zero return code.
set -e
set -o pipefail

# Go to necessary folder
cd ~/diversity

# NOW, DO SOMETHING:
# For fun, echo some useful and interesting information 
echo "Starting sbatch script myscript.sh at:`date`"
echo "  running host:    ${SLURMD_NODENAME}"
echo "  assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "  partition used:  ${SLURM_JOB_PARTITION}"
echo "  jobid:           ${SLURM_JOBID}"


#####################
## 750-run tests   ## # --------- For performance evaluation ----------- # w/ fixed max depth of 10
#####################

# echo "Running job with Inbred Threshold: $ARG2 and Diversity Weight: $ARG1 and Maximum Depth: 10"

# time python src/gp_base.py --benchmark nguyen1 --inbred_threshold "$ARG2" --max_depth "$ARG1"
# time python src/gp_base.py --benchmark nguyen2 --inbred_threshold "$ARG2" --max_depth "$ARG1"
# time python src/gp_base.py --benchmark nguyen3 --inbred_threshold "$ARG2" --max_depth "$ARG1"
# time python src/gp_base.py --benchmark nguyen4 --inbred_threshold "$ARG2" --max_depth "$ARG1"
# time python src/gp_base.py --benchmark nguyen5 --inbred_threshold "$ARG2" --max_depth "$ARG1"
# time python src/gp_base.py --benchmark nguyen6 --inbred_threshold "$ARG2" --max_depth "$ARG1"
# time python src/gp_base.py --benchmark nguyen7 --inbred_threshold "$ARG2" --max_depth "$ARG1"
# time python src/gp_base.py --benchmark nguyen8 --inbred_threshold "$ARG2" --max_depth "$ARG1"

#####################
## Semantics   ## # --------- For Semantics evaluation ----------- # w/ fixed max depth of 10
#####################

# echo "Running job with HBC Threshold: $ARG1 and Semantics Type: SSC and Low Sensitivity: $ARG2 and High Sensitivity $ARG3"

# time python src/gp_semantics.py --benchmark nguyen1 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
# time python src/gp_semantics.py --benchmark nguyen2 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
# time python src/gp_semantics.py --benchmark nguyen3 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
# time python src/gp_semantics.py --benchmark nguyen4 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
# time python src/gp_semantics.py --benchmark nguyen5 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
# time python src/gp_semantics.py --benchmark nguyen6 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
# time python src/gp_semantics.py --benchmark nguyen7 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"
# time python src/gp_semantics.py --benchmark nguyen8 --inbred_threshold "$ARG1" --semantics_type "SSC" --low_sensitivity "$ARG2" --high_sensitivity "$ARG3"

# echo "Running job with HBC Threshold: $ARG1 and Semantics Type: SAC and Low Sensitivity: $ARG2"

# time python src/gp_semantics.py --benchmark nguyen1 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
# time python src/gp_semantics.py --benchmark nguyen2 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
# time python src/gp_semantics.py --benchmark nguyen3 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
# time python src/gp_semantics.py --benchmark nguyen4 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
# time python src/gp_semantics.py --benchmark nguyen5 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
# time python src/gp_semantics.py --benchmark nguyen6 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
# time python src/gp_semantics.py --benchmark nguyen7 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"
# time python src/gp_semantics.py --benchmark nguyen8 --inbred_threshold "$ARG1" --semantics_type "SAC" --low_sensitivity "$ARG2"


#####################
## Fit sharing   ## # --------- Fitness Sharing Experiments ----------- # w/ different inbreeding thresholds and depths and sigma shares
#####################

# echo "Running job with Inbred Threshold: $ARG2 and Maximum Depth: $ARG1 and sigma share weight: $ARG3"

# time python src/gp_sharing_parallel.py --benchmark nguyen1 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
# time python src/gp_sharing_parallel.py --benchmark nguyen2 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
# time python src/gp_sharing_parallel.py --benchmark nguyen3 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
# time python src/gp_sharing_parallel.py --benchmark nguyen4 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
# time python src/gp_sharing_parallel.py --benchmark nguyen5 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
# time python src/gp_sharing_parallel.py --benchmark nguyen6 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
# time python src/gp_sharing_parallel.py --benchmark nguyen7 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"
# time python src/gp_sharing_parallel.py --benchmark nguyen8 --inbred_threshold "$ARG2" --max_depth "$ARG1" --sigma_share_weight "$ARG3"


#####################
## Introns / Bloat ## # --------- Intron ratio and bloat effect ----------- # w/ different inbreeding thresholds and depths and sigma shares
#####################

# echo "Running job with Inbred Threshold: $ARG2"

# time python src/gp_introns.py --benchmark nguyen1 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
# time python src/gp_introns.py --benchmark nguyen2 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
# time python src/gp_introns.py --benchmark nguyen3 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
# time python src/gp_introns.py --benchmark nguyen4 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
# time python src/gp_introns.py --benchmark nguyen5 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
# time python src/gp_introns.py --benchmark nguyen6 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
# time python src/gp_introns.py --benchmark nguyen7 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
# time python src/gp_introns.py --benchmark nguyen8 --generations 150 --pop_size 300 --mutation_rate 0.0005 --inbred_threshold "$ARG2" --exp_num_runs 15 --max_depth 10 --initial_depth 3 --intron_fraction 1.0 --mutation_type random
