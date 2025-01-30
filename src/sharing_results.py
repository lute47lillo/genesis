"""
    Definition
    -----------
        Compute fitness sharing figures. Used in Experimental section results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
    plot_filename = f"genetic_programming/sharing/successes_vs_diversity.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    

def plot_successes_vs_diversity_indiv(data):
    """
    Scatter plots of number of successes vs diversity, arranged in a 4x4 grid with a shared legend.
    """
    sns.set_style("darkgrid")
    
    # Get unique function names
    unique_functions = data['function_name'].unique()
    n_functions = len(unique_functions)

    # Determine grid dimensions (max 4x4 layout)
    n_cols = 4
    n_rows = (n_functions + n_cols - 1) // n_cols  # Calculate rows needed for the grid

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows), sharex=True, sharey=True,
                             gridspec_kw={
                            #  "height_ratios": [1, 1],
                            "hspace": 0.07,  # Reduced from default
                            "wspace": 0.11   # Reduced from default  # Reduce height of the second row
                            })

    # Flatten axes for easier iteration (handles cases with fewer plots)
    axes = axes.flatten()
    
    # Prepare variables to extract the legend
    handles, labels = None, None

    # Plot each function
    for i, (ax, func) in enumerate(zip(axes, unique_functions)):
        subset = data[data['function_name'] == func]
        sns.scatterplot(
            data=subset,
            x='diversity',
            y='n_successes',
            style='W',
            hue="W",
            palette='viridis',
            s=170,
            alpha=0.8,
            legend='full' if i == 0 else False,
            ax=ax
        )
        ax.set_title(f'{func}', fontsize=20)
        ax.set_xlabel('Diversity', fontsize=20)
        ax.set_ylabel('Successes', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=23) 
        
        #  Extract handles and labels from the first plot only
        if i ==0:
            handles, labels = axes[0].get_legend_handles_labels()

            # Remove individual legends
            ax.legend_.remove()

    # Remove unused subplots
    for ax in axes[len(unique_functions):]:
        ax.set_visible(False)

    # Add shared legend
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.06),
        ncol=6,
        title='Sigma Share Weight (W)',
        title_fontsize=20,
        fontsize=22
    )

    plt.tight_layout()

    # Make sure directory exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Save the figure
    plot_filename = f"genetic_programming/sharing/successes_vs_diversity_indiv_600dpi.jpg"
    plt.savefig(os.path.join(figures_dir, plot_filename), dpi=600, bbox_inches='tight')



# Function to plot composite score grouped by W and T
def plot_composite_score(data):
    """
    Bar plot of composite score grouped by W and T.
    
    """    
    t_order = ['None', 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data, x='T', y='composite_score', hue='W', errorbar=None,
        palette='coolwarm', order=t_order 
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
    sns.set_style("darkgrid")
    plt.figure(figsize=(16, 12))
    sns.lineplot(
        data=data, x='function_name', y='mean_gen_success', hue='W', style='W', 
        palette='Set1', linewidth=2, marker="o"
    )
    plt.title('Mean Generation of Success by Function and Sigma Share Weight (W)', fontsize=14)
    plt.xlabel('', fontsize=14)
    plt.ylabel('Mean Generation of Success', fontsize=14)
    plt.legend(title='Sigma Share Weight (W)', ncol=2, fontsize=15, bbox_to_anchor=(0.78, 0.10), loc='lower left', frameon=True)
    
    # Adjust the tick parameters for better readability
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
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
    # Convert T column to string for consistent ordering
    data['T'] = data['T'].astype(str)
    
    t_order = ['None', 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]

    # Get unique functions
    unique_functions = data['function_name'].unique()
    n_functions = len(unique_functions)
    
    # Create subplots
    fig, axes = plt.subplots(n_functions, 1, figsize=(12, 6 * n_functions), sharex=True)
    
    # Ensure axes is iterable even if there's only one function
    if n_functions == 1:
        axes = [axes]
    
    for ax, func in zip(axes, unique_functions):
        subset = data[data['function_name'] == func]
        sns.barplot(
            data=subset, 
            x='T', 
            y='composite_score', 
            hue='W', 
            errorbar=None,
            palette='coolwarm', 
            ax=ax, 
            order=t_order  # Use consistent x-axis order
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
    
def avg_succ_rate_by_fn(data):
    plt.figure(figsize=(12, 6))
    success_rates = data.groupby(["function_name", "W"])["n_successes"].mean().reset_index()
    sns.barplot(
        data=success_rates,
        x="function_name",
        y="n_successes",
        hue="W",
        palette="coolwarm"
    )
    # plt.title("Average Success Rates Across Functions, Grouped by W", fontsize=14)
    plt.xlabel("", fontsize=15)
    plt.ylabel("Average Success Rate", fontsize=15)
    plt.legend(title="W (Fitness Sharing)", fontsize=17, loc="upper right")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # axes[0][0].tick_params(axis='both', which='major', labelsize=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/sharing/avg_succ_rate_by_fn_600dpi.jpg"
    plt.savefig(os.path.join(figures_dir, plot_filename), dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    # Analyze combined data
    summary = summarize_metrics(combined_data)
    
    # Ensure T is treated as a categorical variable
    combined_data['T'] = combined_data['T'].fillna('None')
    combined_data['T'] = combined_data['T'].astype(str)


    # # Generate plots
    plot_successes_vs_diversity_indiv(combined_data)
    avg_succ_rate_by_fn(combined_data)
