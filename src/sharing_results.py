import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Reload the combined dataset
combined_file_path = f"{os.getcwd()}/saved_data/sharing/combined_sharing_data.csv"
combined_data = pd.read_csv(combined_file_path)

# Function to summarize metrics by grouping
def summarize_metrics(data, group_by=['W', 'T']):
    """
    Summarize the dataset by calculating the mean and std for key metrics.
    """
    summary = data.groupby(group_by).agg(
        mean_successes=('n_successes', 'mean'),
        std_successes=('n_successes', 'std'),
        mean_diversity=('diversity', 'mean'),
        std_diversity=('diversity', 'std'),
        mean_gen_success=('mean_gen_success', 'mean'),
        std_gen_success=('mean_gen_success', 'std'),
        mean_composite_score=('composite_score', 'mean'),
        std_composite_score=('composite_score', 'std')
    ).reset_index()
    return summary

# Function to plot successes vs. diversity
def plot_successes_vs_diversity(data):
    """
    Scatter plot of number of successes vs diversity, colored by W.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=data, x='diversity', y='n_successes', hue='W', style='function_name',
        palette='viridis', s=100, alpha=0.8
    )
    plt.title('Number of Successes vs Diversity')
    plt.xlabel('Diversity')
    plt.ylabel('Number of Successes')
    plt.legend(title='Sigma Share Weight (W)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/sharing/TEST_successes_vs_diversity.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

# Function to plot composite score grouped by W and T
def plot_composite_score(data):
    """
    Bar plot of composite score grouped by W and T.
    
    """    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data, x='T', y='composite_score', hue='W', errorbar=None,
        palette='coolwarm'
    )
    plt.title('Composite Score Grouped by W and T')
    plt.xlabel('Inbreeding Threshold (T)')
    plt.ylabel('Composite Score')
    plt.legend(title='Sigma Share Weight (W)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/sharing/composite_score.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

# Function to plot mean generation of success by function_name and W
def plot_mean_gen_success(data):
    """
    Line plot of mean generation of success grouped by function_name and W.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=data, x='function_name', y='mean_gen_success', hue='W', marker='o', 
        palette='Set1', linewidth=2
    )
    plt.title('Mean Generation of Success by Function and Sigma Share Weight (W)')
    plt.xlabel('Symbolic Regression Function')
    plt.ylabel('Mean Generation of Success')
    plt.legend(title='Sigma Share Weight (W)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/sharing/mean_gen_success.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def plot_composite_score_by_function(data):
    """
    Bar plots of composite scores grouped by W and T for each function.
    """
    # Ensure 'T' is treated as a string
    data['T'] = data['T'].astype(str)
    
    # Create subplots
    unique_functions = data['function_name'].unique()
    n_functions = len(unique_functions)
    fig, axes = plt.subplots(n_functions, 1, figsize=(12, 6 * n_functions), sharex=True)

    for ax, func in zip(axes, unique_functions):
        subset = data[data['function_name'] == func]
        sns.barplot(
            data=subset, x='T', y='composite_score', hue='W', errorbar=None,
            palette='coolwarm', ax=ax
        )
        ax.set_title(f'Composite Score for {func}')
        ax.set_xlabel('Inbreeding Threshold (T)')
        ax.set_ylabel('Composite Score')
        ax.legend(title='Sigma Share Weight (W)', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/sharing/composite_score_by_fn.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

if __name__ == "__main__":
    # Analyze combined data
    summary = summarize_metrics(combined_data)
    
    # Ensure T is treated as a categorical variable
    combined_data['T'] = combined_data['T'].fillna('None')
    combined_data['T'] = combined_data['T'].astype(str)


    # Generate plots
    plot_composite_score_by_function(combined_data)
    plot_successes_vs_diversity(combined_data)
    plot_composite_score(combined_data)
    plot_mean_gen_success(combined_data)
