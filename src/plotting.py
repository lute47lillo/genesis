import matplotlib.pyplot as plt
import os

def plot_fitness_comparison_populations(args, elem_list, parameter, results_inbreeding, results_no_inbreeding):
    """
        Definition
        -----------
            Plot the Fitness results of inbreeding prevention mechanism against no prevention over different populations sizes
            
        Parameters
        -----------
            - elem_list (List): Contains the hyperparameters over to what the experiment was run. Ex: Population Sizes, Mutation Rates, ...
            - Parameter (str): The Hyperparameter studied. For plotting and reference 
    """
    # Plot Best Fitness Comparison
    plt.figure(figsize=(16, 9))
    for element in elem_list:
        plt.plot(
            results_inbreeding[element]['best_fitness'],
            label=f'Inbreeding, {parameter} {element}'
        )
        plt.plot(
            results_no_inbreeding[element]['best_fitness'],
            label=f'No Inbreeding, {parameter} {element}',
            linestyle='--'
        )
    plt.title('Best Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{os.getcwd()}/figures/{parameter}/{args.benchmark}/Fit_{args.config_plot}.png")
    plt.close()

def plot_diversity_comparison_populations(args, elem_list, parameter, results_inbreeding, results_no_inbreeding):
    """
        Definition
        -----------
            Plot the Diversity results of inbreeding prevention mechanism against no prevention over different populations sizes
    """
    # Plot Diversity Comparison
    plt.figure(figsize=(16, 9))
    for element in elem_list:
        plt.plot(
            results_inbreeding[element]['diversity'],
            label=f'Inbreeding, {parameter} {element}'
        )
        plt.plot(
            results_no_inbreeding[element]['diversity'],
            label=f'No Inbreeding, {parameter} {element}',
            linestyle='--'
        )
    plt.title('Genetic Diversity over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{os.getcwd()}/figures/{parameter}/{args.benchmark}/Div_{args.config_plot}.png")
    plt.close()