import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

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
    random_subtree_file = f"{os.getcwd()}/saved_data/genetic_programming/all_data.csv" 

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
    plot_filename = f"genetic_programming/bloat/mutation_random_vs_intron_comparison_tree_vs_ratio.png"
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
    plot_filename = f"genetic_programming/bloat/mutation_random_vs_intron_comparison_success_vs_ratio.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def intron_vs_size_with_succ_and_div():
    
    # Paths to the uploaded files
    intron_mutation_file = f"{os.getcwd()}/saved_data/genetic_programming/intron_mutation_all_data.csv" 
    random_subtree_file = f"{os.getcwd()}/saved_data/genetic_programming/all_data.csv" 

    # ------ Load the data for intron mutation data ------ #
    
    intron_mutation_data = pd.read_csv(intron_mutation_file, index_col=0)
    intron_mutation_data = parse_all(intron_mutation_data)
    intron_mutation_parsed = intron_mutation_data.stack().reset_index()
        
    # Renaming columns for clarity
    intron_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']

    # Converting 'Threshold' to string for categorical representation in plots
    intron_mutation_parsed['Threshold'] = intron_mutation_parsed['Threshold'].astype(str)
    
    # ------- Load the data for random mutation data ----- #
    random_subtree_data = pd.read_csv(random_subtree_file, index_col=0)
    random_subtree_data = parse_all(random_subtree_data)
    random_mutation_parsed = random_subtree_data.stack().reset_index()
        
    # Renaming columns for clarity
    random_mutation_parsed.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']

    # Converting 'Threshold' to string for categorical representation in plots
    random_mutation_parsed['Threshold'] = random_mutation_parsed['Threshold'].astype(str)
    
    # ------ Plots ----- #
    
    # Plotting tree size vs. intron ratio for both mutation strategies
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
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
    
    plt.tight_layout()
    
    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/mutation_random_vs_intron_comparison_intron_vs_size_with_succ_and_div.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
    
def div_vs_thres():

    # Paths to the uploaded files
    intron_mutation_file = f"{os.getcwd()}/saved_data/genetic_programming/intron_mutation_all_data_noneF.csv" 
    random_subtree_file = f"{os.getcwd()}/saved_data/genetic_programming/all_data_none.csv" 

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
    
    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/mutation_random_vs_intron_comparison_test.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')



if __name__ == "__main__":
    
    # tree_size_by_intron_ratio_comparisons()
    # intron_vs_size_with_succ_and_div()
    
    div_vs_thres()
   
    