"""
    GENESIS
    
    Running Genetic algorithm
    
    Author: Lute Lillo
    
    Date: 13/10/2024
    
    # TODO: Merge both GENETIC Algorithms to work with both landscapes and optimization functions

"""

import util as util
import plotting as plot
import benchmark_factory as bf
from experiments import multiple_runs_experiment, individual_ga_run


if __name__ == "__main__":
    
    # -------------------------------------------- #
    
    # Get args
    args = util.set_args()
    
    # Double-check special case dimension hyperparameter
    args.dimensions = args.N_NKlandscape  # Genome length

    # Create an instance of the Landscape
    # landscape = bf.NKLandscape(n=args.N_NKlandscape, k=args.K_NKlandscape)
    # landscape = bf.Jump(args)
    # landscape = bf.DeceptiveLeadingBlocks(args)
    # landscape = bf.Rastrigin(args)
    landscape = bf.MovingPeaksLandscape(args)
    
    # Set file plotting name
    args.config_plot = f"{args.bench_name}/NEWNovelty_PopSize:{args.pop_size}_InThres:{args.inbred_threshold}_Mrates:{args.mutation_rate}_Gens:{args.generations}_TourSize:{args.tournament_size}_N:{args.N_NKlandscape}_K:{args.K_NKlandscape}" 

    # Run Individual
    # args.inbred_threshold = 10
    # individual_ga_run(args, landscape, args.inbred_threshold)
    
    # # Run experiments
    # print("Running GA with NO Inbreeding Mating...")
    results_no_inbreeding = multiple_runs_experiment(args, landscape, args.inbred_threshold)

    # print("Running GA with Inbreeding Mating...")
    results_inbreeding = multiple_runs_experiment(args, landscape, None)

    # # Plot experiments
    gs_list, fit_list, div_list, label_list = plot.collect_bootstrapping_data(args, results_no_inbreeding, results_inbreeding)
    plot.plot_multiple_runs_MPL_global_optima(args, gs_list, fit_list, div_list, label_list)
    
    # plot.plot_all(args, gs_list, fit_list, label_list, x_label='Generations', y_label='Fitness', title=f'Inbreeding vs no Inbreeding w/ PopSize: {args.pop_size} & MutRate: {args.mutation_rate}')
    # plot.plot_all(args, gs_list, div_list, label_list, x_label='Generations', y_label='Diversity', title=f'Inbreeding vs no Inbreeding w/ PopSize: {args.pop_size} & MutRate: {args.mutation_rate}')