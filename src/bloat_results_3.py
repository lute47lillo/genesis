import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from bloat_results_2 import merge_DS_with_IT

"""
    Definition
    -----------
    
        A series of functions to do an anlaysis of the intron subtree mutations vs random subtree mutations
"""

# --------- Utils ---------- #
def parse_tree_intron(data):
        # Extract only rows corresponding to tree size and success/diversity ratio
        tree_size_data = data.iloc[8:16] 
        success_div_data = data.iloc[:8]  
        return tree_size_data, success_div_data
    
    
# Function to extract numerical values from the dataset
def extract_values(data):
    tree_sizes = data.map(lambda x: float(x.split("(")[1].strip(")")) if isinstance(x, str) else None)
    intron_ratios = data.map(lambda x: float(x.split(" ")[0]) if isinstance(x, str) else None)
    return tree_sizes, intron_ratios

def parse_all(all_data):
    
    # Correctly split the dataset into two parts
    success_diversity = all_data.iloc[:8]
    intron_tree_size = all_data.iloc[8:17]

    # Assign proper function names as the index
    functions = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]
    success_diversity.index = functions
    intron_tree_size.index = functions

    # Parse Success and Diversity
    success = success_diversity.map(lambda x: int(x.split(" ")[0]) if pd.notna(x) else None)
    diversity = success_diversity.map(lambda x: float(x.split("(")[1].strip(")")) if pd.notna(x) else None)

    # Parse Intron Ratio and Tree Size
    intron_ratio = intron_tree_size.map(lambda x: float(x.split(" ")[0]) if pd.notna(x) else None)
    tree_size = intron_tree_size.map(lambda x: float(x.split("(")[1].strip(")")) if pd.notna(x) else None)

    # Combine all parsed data into a structured DataFrame
    parsed_data_final = pd.concat({
        "Success": success,
        "Diversity": diversity,
        "Intron Ratio": intron_ratio,
        "Tree Size": tree_size
    }, axis=1)
    
    return parsed_data_final
    
# ------- Main functions --------- #

def tree_size_by_intron_ratio_comparisons():

    # Paths to the uploaded files
    intron_mutation_file = f"{os.getcwd()}/saved_data/genetic_programming/intron_mutation_all_data.csv" 
    # random_subtree_file = f"{os.getcwd()}/saved_data/genetic_programming/all_data.csv" 
    random_subtree_file = f"{os.getcwd()}/saved_data/genetic_programming/half_mut_all_data.csv" 


    # Load the data
    intron_mutation_data = pd.read_csv(intron_mutation_file, index_col=0)
    random_subtree_data = pd.read_csv(random_subtree_file, index_col=0)

    # Display the head of both datasets for verification
    intron_mutation_data.head(), random_subtree_data.head()

    # Parse intron and tree size for both mutation strategies
    tree_size_intron_mutation, success_div_intron_mutation = parse_tree_intron(intron_mutation_data)
    tree_size_random_subtree, success_div_random_subtree = parse_tree_intron(random_subtree_data)

    # ------ Tree size ~ Intron ratio plot ------- #
    
    # Ensure the parsed data aligns correctly
    tree_size_intron_mutation.head(), tree_size_random_subtree.head()

    # Extract values for both datasets
    tree_sizes_intron, intron_ratios_intron = extract_values(tree_size_intron_mutation)
    tree_sizes_random, intron_ratios_random = extract_values(tree_size_random_subtree)

    # Plotting tree size vs. intron ratio for both mutation strategies
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Intron subtree mutations
    axes[0].scatter(intron_ratios_intron.stack(), tree_sizes_intron.stack(), color="blue", alpha=0.7)
    axes[0].set_title("Tree Size vs. Intron Ratio (Intron Subtree Mutations)")
    axes[0].set_xlabel("Intron Ratio")
    axes[0].set_ylabel("Tree Size")

    # Random subtree mutations
    axes[1].scatter(intron_ratios_random.stack(), tree_sizes_random.stack(), color="orange", alpha=0.7)
    axes[1].set_title("Tree Size vs. Intron Ratio (Random Subtree Mutations)")
    axes[1].set_xlabel("Intron Ratio")

    plt.tight_layout()
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Save the figure
    plot_filename = f"genetic_programming/bloat/HALF_mutation_random_vs_intron_comparison_tree_vs_ratio.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
    # ------ Success ~ Intron ratio ------- #
    
    # Ensure the parsed data aligns correctly
    tree_size_intron_mutation.head(), tree_size_random_subtree.head()
    success_div_intron_mutation.head(), success_div_random_subtree.head()

    # Extract values for both datasets for intron ratio
    _, intron_ratios_intron = extract_values(tree_size_intron_mutation)
    _, intron_ratios_random = extract_values(tree_size_random_subtree)
    
    # Extract values for both datasets for success
    success_intron, _ = extract_values(success_div_intron_mutation)
    success_random, _ = extract_values(success_div_random_subtree)

    # Plotting tree size vs. intron ratio for both mutation strategies
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Intron subtree mutations
    axes[0].scatter(intron_ratios_intron.stack(), success_intron.stack(), color="blue", alpha=0.7)
    axes[0].set_title("Total nº successesvs. Intron Ratio (Intron Subtree Mutations)")
    axes[0].set_xlabel("Intron Ratio")
    axes[0].set_ylabel("Total nº successes")

    # Random subtree mutations
    axes[1].scatter(intron_ratios_random.stack(), success_random.stack(), color="orange", alpha=0.7)
    axes[1].set_title("Total nº successes vs. Intron Ratio (Random Subtree Mutations)")
    axes[1].set_xlabel("Intron Ratio")

    plt.tight_layout()
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Save the figure
    plot_filename = f"genetic_programming/intron_mut_study/mutation_random_vs_intron_comparison_success_vs_ratio.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def intron_vs_size_with_succ_and_div():

    # ------ Load the data for intron mutation data ------ #
    
    intron_mutation_file = f"{os.getcwd()}/saved_data/introns_study/intron_mutation_merged_data.csv" 
    intron_mutation_data = pd.read_csv(intron_mutation_file, index_col=0)
    intron_mutation_data = parse_all(intron_mutation_data)
    intron_mutation_parsed = intron_mutation_data.stack().reset_index()
        
    # Renaming columns for clarity
    intron_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']

    # Converting 'Threshold' to string for categorical representation in plots
    intron_mutation_parsed['Threshold'] = intron_mutation_parsed['Threshold'].astype(str)
    
    # ------- Load the data for random mutation data ----- #
    
    random_subtree_file = f"{os.getcwd()}/saved_data/introns_study/random_mut_merged_data.csv" 
    random_subtree_data = pd.read_csv(random_subtree_file, index_col=0)
    random_subtree_data = parse_all(random_subtree_data)
    random_mutation_parsed = random_subtree_data.stack().reset_index()
        
    # Renaming columns for clarity
    random_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']

    # Converting 'Threshold' to string for categorical representation in plots
    random_mutation_parsed['Threshold'] = random_mutation_parsed['Threshold'].astype(str)
    
    # ------- Load the data for Half-n-Half mutation data ----- #

    halfs_subtree_file = f"{os.getcwd()}/saved_data/introns_study/half_mut_merged_data.csv" 
    half_subtree_data = pd.read_csv(halfs_subtree_file, index_col=0)
    half_subtree_data = parse_all(half_subtree_data)
    half_mutation_parsed = half_subtree_data.stack().reset_index()
        
    # Renaming columns for clarity
    half_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']

    # Converting 'Threshold' to string for categorical representation in plots
    half_mutation_parsed['Threshold'] = half_mutation_parsed['Threshold'].astype(str)
    
    # ------- Load the data for random plus mutation data p(0.75) ----- #
    random_plus_subtree_file = f"{os.getcwd()}/saved_data/introns_study/random_plus_merged_data.csv" 
    random_plus_subtree_data = pd.read_csv(random_plus_subtree_file, index_col=0)
    random_plus_subtree_data = parse_all(random_plus_subtree_data)
    random_plus_mutation_parsed = random_plus_subtree_data.stack().reset_index()
        
    # Renaming columns for clarity
    random_plus_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']

    # Converting 'Threshold' to string for categorical representation in plots
    random_plus_mutation_parsed['Threshold'] = random_plus_mutation_parsed['Threshold'].astype(str)
    
    # ------- Load the data for random mutation data ----- #
    intron_plus_subtree_file = f"{os.getcwd()}/saved_data/introns_study/intron_plus_merged_data.csv" 
    intron_plus_subtree_data = pd.read_csv(intron_plus_subtree_file, index_col=0)
    intron_plus_subtree_data = parse_all(intron_plus_subtree_data)
    intron_plus_mutation_parsed = intron_plus_subtree_data.stack().reset_index()
        
    # Renaming columns for clarity
    intron_plus_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']

    # Converting 'Threshold' to string for categorical representation in plots
    intron_plus_mutation_parsed['Threshold'] = intron_plus_mutation_parsed['Threshold'].astype(str)
    
    # ------ Plots ----- #
    
    # Plotting tree size vs. intron ratio for both mutation strategies
    fig, axes = plt.subplots(1, 5, figsize=(28, 10), sharey=True)
    
    # Subplot 1
    sns.scatterplot(
        data=intron_mutation_parsed,
        x='Intron Ratio',
        y='Tree Size',
        size='Success',
        hue='Diversity',
        palette='viridis',
        sizes=(20, 200),
        alpha=0.8,
        ax=axes[0]
    )
    axes[0].set_title("Tree Size vs. Intron Ratio (Intron Subtree Mutations)")
    axes[0].set_xlabel("Intron Ratio")
    axes[0].set_ylabel("Tree Size")
    
    # Subplot 2
    sns.scatterplot(
        data=random_mutation_parsed,
        x='Intron Ratio',
        y='Tree Size',
        size='Success',
        hue='Diversity',
        palette='viridis',
        sizes=(20, 200),
        alpha=0.8,
        ax=axes[1]
    )
    # Random subtree mutations
    axes[1].set_title("Tree Size vs. Intron Ratio (Random Subtree Mutations)")
    axes[1].set_xlabel("Intron Ratio")
    
    
    # Subplot 3
    sns.scatterplot(
        data=half_mutation_parsed,
        x='Intron Ratio',
        y='Tree Size',
        size='Success',
        hue='Diversity',
        palette='viridis',
        sizes=(20, 200),
        alpha=0.8,
        ax=axes[2]
    )
    # Random subtree mutations
    axes[2].set_title("Tree Size vs. Intron Ratio (Half-n-Half Subtree Mutations)")
    axes[2].set_xlabel("Intron Ratio")
    
    # Subplot 4
    sns.scatterplot(
        data=random_plus_mutation_parsed,
        x='Intron Ratio',
        y='Tree Size',
        size='Success',
        hue='Diversity',
        palette='viridis',
        sizes=(20, 200),
        alpha=0.8,
        ax=axes[3]
    )
    # Random subtree mutations
    axes[3].set_title("Tree Size vs. Intron Ratio (Random p(0.75) Subtree Mutations)")
    axes[3].set_xlabel("Intron Ratio")
    
    # Subplot 5
    sns.scatterplot(
        data=intron_plus_mutation_parsed,
        x='Intron Ratio',
        y='Tree Size',
        size='Success',
        hue='Diversity',
        palette='viridis',
        sizes=(20, 200),
        alpha=0.8,
        ax=axes[4]
    )
    # Random subtree mutations
    axes[4].set_title("Tree Size vs. Intron Ratio (Intron p(0.75) Subtree Mutations)")
    axes[4].set_xlabel("Intron Ratio")
    
    plt.tight_layout()
    
    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/intron_mut_study/ALL_comparison_intron_vs_size_with_succ_and_div.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
sns.set_style("darkgrid")
def intron_vs_size_with_succ_and_div_2():
    # ------------------------------------------------------------------------
    # 1. Load and parse all dataframes as you did
    # ------------------------------------------------------------------------
    intron_mutation_file = f"{os.getcwd()}/saved_data/introns_study/intron_mutation_merged_data_DIV_AVG.csv" 
    intron_mutation_data = pd.read_csv(intron_mutation_file, index_col=0)
    intron_mutation_data = parse_all(intron_mutation_data)
    intron_mutation_parsed = intron_mutation_data.stack(future_stack=True).reset_index()
    intron_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    intron_mutation_parsed['Threshold'] = intron_mutation_parsed['Threshold'].astype(str)

    random_subtree_file = f"{os.getcwd()}/saved_data/introns_study/random_mut_merged_data_DIV_AVG.csv" 
    random_subtree_data = pd.read_csv(random_subtree_file, index_col=0)
    random_subtree_data = parse_all(random_subtree_data)
    random_mutation_parsed = random_subtree_data.stack(future_stack=True).reset_index()
    random_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    random_mutation_parsed['Threshold'] = random_mutation_parsed['Threshold'].astype(str)

    halfs_subtree_file = f"{os.getcwd()}/saved_data/introns_study/half_mut_merged_data_DIV_AVG.csv" 
    half_subtree_data = pd.read_csv(halfs_subtree_file, index_col=0)
    half_subtree_data = parse_all(half_subtree_data)
    half_mutation_parsed = half_subtree_data.stack(future_stack=True).reset_index()
    half_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    half_mutation_parsed['Threshold'] = half_mutation_parsed['Threshold'].astype(str)

    random_plus_subtree_file = f"{os.getcwd()}/saved_data/introns_study/random_plus_merged_data_DIV_AVG.csv" 
    random_plus_subtree_data = pd.read_csv(random_plus_subtree_file, index_col=0)
    random_plus_subtree_data = parse_all(random_plus_subtree_data)
    random_plus_mutation_parsed = random_plus_subtree_data.stack(future_stack=True).reset_index()
    random_plus_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    random_plus_mutation_parsed['Threshold'] = random_plus_mutation_parsed['Threshold'].astype(str)

    intron_plus_subtree_file = f"{os.getcwd()}/saved_data/introns_study/intron_plus_merged_data_DIV_AVG.csv" 
    intron_plus_subtree_data = pd.read_csv(intron_plus_subtree_file, index_col=0)
    intron_plus_subtree_data = parse_all(intron_plus_subtree_data)
    intron_plus_mutation_parsed = intron_plus_subtree_data.stack(future_stack=True).reset_index()
    intron_plus_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    intron_plus_mutation_parsed['Threshold'] = intron_plus_mutation_parsed['Threshold'].astype(str)

    # ------------------------------------------------------------------------
    # 2. Combine everything to find global min and max for the color scale
    # ------------------------------------------------------------------------
    all_data = pd.concat([
        intron_mutation_parsed,
        random_mutation_parsed,
        half_mutation_parsed,
        random_plus_mutation_parsed,
        intron_plus_mutation_parsed
    ], ignore_index=True)

    vmin = all_data['Diversity'].min()
    vmax = all_data['Diversity'].max()

    # ------------------------------------------------------------------------
    # 3. Plot subplots with consistent hue_norm and no separate legend
    # ------------------------------------------------------------------------
    # fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharey=True)
    # # Common keyword arguments
    # scatter_kwargs = {
    #     'x': 'Intron Ratio',
    #     'y': 'Tree Size',
    #     'hue': 'Diversity',
    #     'palette': 'viridis',
    #     'hue_norm': (vmin, vmax),  # ensure same color scale
    #     'size': 'Success',
    #     'sizes': (20, 200),
    #     'alpha': 0.8,
    #     'legend': False  # we'll create one colorbar later
    # }

    # # Subplot 1
    # sns.scatterplot(data=intron_mutation_parsed, ax=axes[0][0], **scatter_kwargs)
    # axes[0][0].set_title("Intron Subtree Mutations")
    # axes[0][0].set_xlabel("Intron Ratio")
    # axes[0][0].set_ylabel("Tree Size")

    # # Subplot 2
    # sns.scatterplot(data=random_mutation_parsed, ax=axes[0][1], **scatter_kwargs)
    # axes[0][1].set_title("Random Subtree Mutations")
    # axes[0][1].set_xlabel("Intron Ratio")

    # # Subplot 3
    # sns.scatterplot(data=half_mutation_parsed, ax=axes[0][2], **scatter_kwargs)
    # axes[0][2].set_title("Half-n-Half Subtree Mutations")
    # axes[0][2].set_xlabel("Intron Ratio")

    # # Subplot 4
    # sns.scatterplot(data=random_plus_mutation_parsed, ax=axes[1][0], **scatter_kwargs)
    # axes[1][0].set_title("Random p(0.75) Subtree Mutations")
    # axes[1][0].set_xlabel("Intron Ratio")

    # # Subplot 5
    # sns.scatterplot(data=intron_plus_mutation_parsed, ax=axes[1][1], **scatter_kwargs)
    # axes[1][1].set_title("Intron p(0.75) Subtree Mutations")
    # axes[1][1].set_xlabel("Intron Ratio")

    # plt.tight_layout()

    # # ------------------------------------------------------------------------
    # # 4. Create a single colorbar for "Diversity"
    # # ------------------------------------------------------------------------
    # # Make a scalar mappable using the same colormap and norm:
    # from matplotlib.colors import Normalize
    # import matplotlib as mpl
    
    # x_min = all_data['Intron Ratio'].min()
    # x_max = all_data['Intron Ratio'].max()

    # for ax in axes[0]:
    #     ax.set_xlim([x_min, x_max])
        
    # for ax in axes[1]:
    #     ax.set_xlim([x_min, x_max])


    # norm = Normalize(vmin=vmin, vmax=vmax)
    # sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
    # sm.set_array([])  # required for older matplotlib so colorbar knows the range

    # # Now create the colorbar on the figure, spanning all subplots:
    # cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.7, label="Diversity", pad=0.02)
    # # You could also configure the size scale legend similarly, or handle it differently.
    # # -------------------------------------------
    # # 4) Create a dummy plot to get size handles
    # # -------------------------------------------
    # # We do a separate (invisible) figure+axes so we can get
    # # the auto-generated legend for "Success" (the size).
    # # Note we omit hue so we only get the size legend items.
    # dummy_fig, dummy_ax = plt.subplots()
    # dummy_scatter = sns.scatterplot(
    #     data=all_data,
    #     x='Intron Ratio',
    #     y='Tree Size',
    #     size='Success',
    #     # If you want the circles in a specific color, do color='k' or 'gray'
    #     color='black',  
    #     sizes=(20, 200),
    #     alpha=0.8,      # invisible markers
    #     legend='full', # or 'full'
    #     ax=dummy_ax
    # )
    # # Capture the handles & labels from the dummy's legend
    # handles, labels = dummy_ax.get_legend_handles_labels()

    # # We no longer need the dummy figure
    # plt.close(dummy_fig)

    # # -------------------------------------------
    # # 5) Add a single size legend to the main fig
    # # -------------------------------------------
    # # You can pick a location (e.g., 'upper right', 'center right', etc.)
    # fig.legend(
    #     handles,
    #     labels,
    #     loc='lower center',
    #     bbox_to_anchor=(0.45, -0.05),  # shift downward by 0.2
    #     ncol=len(labels),
    #     title='Success'
    # )

    # # Adjust the layout so there's room at the bottom
    # # plt.subplots_adjust(bottom=0.2)

    # # ------------------------------------------------------------------------
    # # 5. Save figure
    # # ------------------------------------------------------------------------
    # figures_dir = os.path.join(os.getcwd(), 'figures')
    # os.makedirs(figures_dir, exist_ok=True)

    # plot_filename = "genetic_programming/intron_mut_study/ALL_vertical_comparison_intron_vs_size_with_succ_and_div_DIV_AVG.png"
    # plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
    # Adjust the subplot grid to have a blank space in the second row
    fig, axes = plt.subplots(
        2, 3, figsize=(22, 12),
        sharey=True,
        gridspec_kw={"height_ratios": [1, 1]}  # Reduce height of the second row
    )

    # Common keyword arguments
    scatter_kwargs = {
        'x': 'Intron Ratio',
        'y': 'Tree Size',
        'hue': 'Diversity',
        'palette': 'viridis',
        'hue_norm': (vmin, vmax),  # ensure same color scale
        'size': 'Success',
        'sizes': (20, 200),
        'alpha': 0.8,
        'legend': False  # we'll create one colorbar later
    }

    # Subplot 1: Intron Subtree Mutations
    sns.scatterplot(data=intron_mutation_parsed, ax=axes[0][0], **scatter_kwargs)
    axes[0][0].set_title("Intron Subtree Mutations")
    axes[0][0].set_xlabel("Intron Ratio")
    axes[0][0].set_ylabel("Tree Size")

    # Subplot 2: Random Subtree Mutations
    sns.scatterplot(data=random_mutation_parsed, ax=axes[0][1], **scatter_kwargs)
    axes[0][1].set_title("Random Subtree Mutations")
    axes[0][1].set_xlabel("Intron Ratio")

    # Subplot 3: Half-n-Half Subtree Mutations
    sns.scatterplot(data=half_mutation_parsed, ax=axes[0][2], **scatter_kwargs)
    axes[0][2].set_title("Half-n-Half Subtree Mutations")
    axes[0][2].set_xlabel("Intron Ratio")

    # Subplot 4: Random p(0.75) Subtree Mutations (centered in row 2)
    sns.scatterplot(data=random_plus_mutation_parsed, ax=axes[1][0], **scatter_kwargs)
    axes[1][0].set_title("Random p(0.75) Subtree Mutations")
    axes[1][0].set_xlabel("Intron Ratio")

    # Subplot 5: Intron p(0.75) Subtree Mutations (centered in row 2)
    sns.scatterplot(data=intron_plus_mutation_parsed, ax=axes[1][1], **scatter_kwargs)
    axes[1][1].set_title("Intron p(0.75) Subtree Mutations")
    axes[1][1].set_xlabel("Intron Ratio")

    # Remove the empty subplot in the second row
    fig.delaxes(axes[1][2])

    # Adjust spacing to center the second row
    # fig.subplots_adjust(hspace=0.5, wspace=0.4)

    # Center the subplots in the second row
    axes[1][0].set_position([
        axes[0][0].get_position().x0 +0.15,  # Align x0 with the first column
        axes[1][0].get_position().y0,  # Keep the y0 position
        axes[0][0].get_position().width,  # Match width with the first column
        axes[1][0].get_position().height  # Match height
    ])

    axes[1][1].set_position([
        axes[0][1].get_position().x0+0.15,  # Align x0 with the second column
        axes[1][1].get_position().y0,  # Keep the y0 position
        axes[0][1].get_position().width,  # Match width with the second column
        axes[1][1].get_position().height  # Match height
    ])

    # Create the shared colorbar
    from matplotlib.colors import Normalize
    import matplotlib as mpl
    
    x_min = all_data['Intron Ratio'].min()
    x_max = all_data['Intron Ratio'].max()

    for ax in axes[0]:
        ax.set_xlim([x_min, x_max])
        
    for ax in axes[1]:
        ax.set_xlim([x_min, x_max])

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.7, label="Diversity", pad=0.02)

    # Add size legend for success
    dummy_fig, dummy_ax = plt.subplots()
    dummy_scatter = sns.scatterplot(
        data=all_data,
        x='Intron Ratio',
        y='Tree Size',
        size='Success',
        color='black',
        sizes=(20, 200),
        alpha=0.8,
        legend='full',
        ax=dummy_ax
    )
    handles, labels = dummy_ax.get_legend_handles_labels()
    plt.close(dummy_fig)

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.45, 0.01),
        ncol=len(labels),
        title='Success'
    )

    # Save the adjusted plot
    # plt.tight_layout()
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_filename = "genetic_programming/intron_mut_study/Adjusted_intron_vs_size_with_succ_and_div.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def div_vs_thres():

    # Paths to the uploaded files
    intron_mutation_file = f"{os.getcwd()}/saved_data/genetic_programming/intron_mutation_all_data_noneF.csv" 
    # random_subtree_file = f"{os.getcwd()}/saved_data/genetic_programming/all_data_none.csv" 
    random_subtree_file = f"{os.getcwd()}/saved_data/genetic_programming/half_mut_all_data.csv" 
    

    random_data = pd.read_csv(random_subtree_file, index_col=0, header=[0])
    intron_data = pd.read_csv(intron_mutation_file, index_col=0, header=[0])

    # Extracting diversity data for comparison
    diversity_random = random_data.iloc[:8].map(
        lambda x: float(x.split("(")[1].strip(")")) if pd.notna(x) else None
    )
    diversity_intron = intron_data.iloc[:8].map(
        lambda x: float(x.split("(")[1].strip(")")) if pd.notna(x) else None
    )

    # Adding a column for function names
    diversity_random["Function"] = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]
    diversity_intron["Function"] = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"]

    # Melting data for easier visualization
    random_melted = diversity_random.melt(id_vars=["Function"], var_name="Threshold", value_name="Diversity")
    intron_melted = diversity_intron.melt(id_vars=["Function"], var_name="Threshold", value_name="Diversity")

    # Plotting side-by-side comparison
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Random Subtree Mutations
    sns.boxplot(data=random_melted, x="Threshold", y="Diversity", ax=ax[0])
    ax[0].set_title("Diversity in Random Subtree Mutations")
    ax[0].set_xlabel("Threshold")
    ax[0].set_ylabel("Diversity")

    # Intron Subtree Mutations
    sns.boxplot(data=intron_melted, x="Threshold", y="Diversity", ax=ax[1])
    ax[1].set_title("Diversity in Intron Subtree Mutations")
    ax[1].set_xlabel("Threshold")

    plt.tight_layout()
    
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    plot_filename = "genetic_programming/intron_mut_study/ALL_vertical_comparison_intron_vs_size_with_succ_and_div_DIV_AVG.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
   
def help_count(mutation_parsed, type_run):
    # print(mutation_parsed['Success'].to_string())
    count = 0
    none_count = 0
    n_total_runs = 10 * 8 * 15 # nº thresholds * SRs * runs
    for idx, i in enumerate(mutation_parsed['Success']):
        if idx % 11 == 0:
            # print(idx, i)
            none_count += i
        else:
            count += i
    print(f"{type_run}. Total nº Successes overall: {count}. Ratio: {(count/n_total_runs)*100:.3f}% of success")
    print(f"{type_run}. Total NONE nº Successes overall: {none_count}. Ratio: {(none_count/(8*15))*100:.3f}% of success\n")


    in_rat = 0
    non_rat = 0
    total_values = len(mutation_parsed['Intron Ratio']) - 8 # Removing None
    for idx, j in enumerate(mutation_parsed['Intron Ratio']):
        if idx % 11 == 0:
            non_rat += j
        else:
            in_rat += j
       
    print(f"{type_run}. Average Intron Ratio over all SRs: {(in_rat/total_values):.3f}.")
    print(f"{type_run}. Average NONE Intron Ratio over all SRs: {(non_rat/8):.3f}.\n")
    
    
    div = 0
    non_div = 0
    total_values = len(mutation_parsed['Diversity']) - 8 # Removing None
    for idx, j in enumerate(mutation_parsed['Diversity']):
        if idx % 11 == 0:
            non_div += j
        else:
            div += j
    
    print(f"{type_run}. Average Diversity over all SRs: {(div/total_values):.3f}.")
    print(f"{type_run}. Average NONE Diversity over all SRs: {(non_div/8):.3f}.\n")

    
    
    tree_size = 0
    non_size = 0
    total_values = len(mutation_parsed['Tree Size']) - 8 # Removing None
    for idx, j in enumerate(mutation_parsed['Tree Size']):
        if idx % 11 == 0:
            non_size += j
        else:
            tree_size += j
        
    print(f"{type_run}. Average Tree Size over all SRs: {(tree_size/total_values):.3f}.")
    print(f"{type_run}. Average NONE Tree Size over all SRs: {(non_size/8):.3f}.\n")

    
# -------- Success analysis ------- #
# intron_mutation. Total nº Successes overall: 567. Ratio: 42.955% of success
# random_mutation. Total nº Successes overall: 384. Ratio: 29.091% of success
# half_mutation. Total nº Successes overall: 511. Ratio: 38.712% of success
# random_plus_mutation. Total nº Successes overall: 439. Ratio: 33.258% of success
# intron_plus_mutation. Total nº Successes overall: 548. Ratio: 41.515% of success
# --------------------------------- #

     
def success_by_threshold_all_sr():
    
    parsed_data_collection = []
    print(f"\n# -------- Success analysis ------- #\n")
    # ------------------------------------------------------------------------
    # 1. Load and parse all dataframes as you did
    # ------------------------------------------------------------------------
    intron_mutation_file = f"{os.getcwd()}/saved_data/introns_study/intron_mutation_merged_data_DIV_AVG.csv" 
    intron_mutation_data = merge_DS_with_IT(intron_mutation_file)
    intron_mutation_parsed = intron_mutation_data.stack(future_stack=True).reset_index()
    intron_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    intron_mutation_parsed['Threshold'] = intron_mutation_parsed['Threshold'].astype(str)
    parsed_data_collection.append(intron_mutation_parsed)
    help_count(intron_mutation_parsed, "intron_mutation")

    random_subtree_file = f"{os.getcwd()}/saved_data/introns_study/random_mut_merged_data_DIV_AVG.csv" 
    random_subtree_data = merge_DS_with_IT(random_subtree_file)
    random_mutation_parsed = random_subtree_data.stack(future_stack=True).reset_index()
    random_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    random_mutation_parsed['Threshold'] = random_mutation_parsed['Threshold'].astype(str)
    parsed_data_collection.append(random_mutation_parsed)
    help_count(random_mutation_parsed, "random_mutation")

    halfs_subtree_file = f"{os.getcwd()}/saved_data/introns_study/half_mut_merged_data_DIV_AVG.csv" 
    half_subtree_data = merge_DS_with_IT(halfs_subtree_file)
    half_mutation_parsed = half_subtree_data.stack(future_stack=True).reset_index()
    half_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    half_mutation_parsed['Threshold'] = half_mutation_parsed['Threshold'].astype(str)
    parsed_data_collection.append(half_mutation_parsed)
    help_count(half_mutation_parsed, "half_mutation")

    random_plus_subtree_file = f"{os.getcwd()}/saved_data/introns_study/random_plus_merged_data_DIV_AVG.csv" 
    random_plus_subtree_data = merge_DS_with_IT(random_plus_subtree_file)
    random_plus_mutation_parsed = random_plus_subtree_data.stack(future_stack=True).reset_index()
    random_plus_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    random_plus_mutation_parsed['Threshold'] = random_plus_mutation_parsed['Threshold'].astype(str)
    parsed_data_collection.append(random_plus_mutation_parsed)
    help_count(random_plus_mutation_parsed, "random_plus_mutation")

    intron_plus_subtree_file = f"{os.getcwd()}/saved_data/introns_study/intron_plus_merged_data_DIV_AVG.csv" 
    intron_plus_subtree_data = merge_DS_with_IT(intron_plus_subtree_file)
    intron_plus_mutation_parsed = intron_plus_subtree_data.stack(future_stack=True).reset_index()
    intron_plus_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']
    intron_plus_mutation_parsed['Threshold'] = intron_plus_mutation_parsed['Threshold'].astype(str)
    parsed_data_collection.append(intron_plus_mutation_parsed)
    help_count(intron_plus_mutation_parsed, "intron_plus_mutation")
    print(f"# --------------------------------- #\n")
    exit()
    
    # Line plots for each metric across thresholds
    fig, axes = plt.subplots(1, 5, figsize=(40, 8), constrained_layout=False)

    types_runs = ["Intron Mutation","Random Mutation", "Half-n-Half Mutation", "Random p(0.75) mutation", "Intron p(0.75) mutation"]
    for i, parsed_data in enumerate(parsed_data_collection):
        for func in parsed_data['Function'].unique():
            subset = parsed_data[parsed_data['Function'] == func]
            axes[i].scatter(
                subset['Diversity'],
                subset["Success"],
                marker='o',
                label=func
            )
        axes[i].set_title(f"{types_runs[i]}")
        axes[i].set_xlabel("Threshold")
        axes[i].set_ylabel("Success")
        axes[i].legend(title="Function")

    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    # plot_filename = f"genetic_programming/intron_mut_study/success_by_threshold_all_sr.png"
    plot_filename = f"genetic_programming/intron_mut_study/TEST.png"

    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    

if __name__ == "__main__":
    
    # tree_size_by_intron_ratio_comparisons()
    # success_by_threshold_all_sr()
    intron_vs_size_with_succ_and_div_2()
    # intron_vs_size_with_succ_and_div()
    
    # div_vs_thres()
   
    