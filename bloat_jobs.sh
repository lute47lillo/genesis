#!/bin/bash

# --------- Pure runs performance (750 runs per SR fn)  ----------- #

# # Array of inbred_threshold values and maximum depths
# inbred_thresholds=(6 7 8 9 10 11 12 13 14)
# # inbred_thresholds=(1)
# depths=(6 7 8 9 10) # Missign 10


# # TODO: TESTING HOW NG5 performs with the dedicated selection.

# for depth in "${depths[@]}"; do
#     for inbred_threshold in "${inbred_thresholds[@]}"; do

#         # Submit the job with the specified depth and inbred_threshold
#         sbatch --export=ARG1="$depth",ARG2="$inbred_threshold" bloat.sbatch

#         # Print confirmation message
#         echo "Submitted job with Max Depth=$depth and Inbreeding Threshold=$inbred_threshold"
#     done
# done

# --------- For Bloat studies with fixed Inbreeding Threshold ----------- #

# # # Fixed depth
# depth=10

# # Array of inbred_threshold values
# # inbred_thresholds=(1 5 10 14)
# # inbred_thresholds=(6 7 8 9 11 12 13)

# inbred_thresholds=(1 5 6 7 8 9 10 11 12 13 14)

# # Iterate over each inbred_threshold value
# for inbred_threshold in "${inbred_thresholds[@]}"; do
#     # Submit the job with the specified depth and inbred_threshold
#     sbatch --export=ARG1="$depth",ARG2="$inbred_threshold" bloat.sbatch
    
#     # Print confirmation message
#     echo "Submitted job with Max Depth=$depth and Inbreeding Threshold=$inbred_threshold"
# done


# --------- Fitness Sharing ----------- #

# Array of values
# depths=(10)
# # inbred_thresholds=(1 5 10 14)
# inbred_thresholds=(6 7 8 9 11 12 13)
# sigma_share_weights=(0.1 0.2 0.3 0.5 0.8)

# for depth in "${depths[@]}"; do
#     for threshold in "${inbred_thresholds[@]}"; do
#         for sigma in "${sigma_share_weights[@]}"; do
#             sbatch --export=ARG1="$depth",ARG2="$threshold",ARG3="$sigma" bloat.sbatch
#             echo "Submitted job with Max Depth=\"$depth\" and Inbreeding Threshold=\"$threshold\" and Sigma Share weights=\"$sigma\""
#         done
#     done
# done

# --------------------- Semantics Studies ------------------------- #

# Fixed HBC Threshold to 1 -> None as there is no crossover

# Values
# threshold=(1 5 6 7 8 9 10 11 12 13 14)
# low_sensitivities=(0.02 0.04 0.06 0.08)
# high_sensitivities=(8 10 12)

# # # # Running for SSC
# for thres in "${threshold[@]}"; do
#     for low in "${low_sensitivities[@]}"; do
#         for high in "${high_sensitivities[@]}"; do
#             sbatch --export=ARG1="$thres",ARG2="$low",ARG3="$high" bloat.sbatch
#             echo "Submitted job for SSC with Threshold=\"$thres\" and Low Sensitivity=\"$low\" and High Sensitivity=\"$high\""
#         done
#     done
# done

# Running for SAC
# for thres in "${threshold[@]}"; do
#     for low in "${low_sensitivities[@]}"; do
#         sbatch --export=ARG1="$thres",ARG2="$low" bloat.sbatch
#         echo "Submitted job for SAC with Threshold=\"$thres\" and Low Sensitivity=\"$low\""
#     done
# done
