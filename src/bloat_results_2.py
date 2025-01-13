import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np
np.set_printoptions(threshold=np.inf)

# Define the diversity/success dataset
diversity_success_data = {
    "Function": ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"],
    "None": ["13 (19.79)", "10 (19.52)", "2 (18.57)", "4 (12.39)", "0 (7.49)", "0 (8.37)", "6 (10.35)", "0 (8.55)"],
    "5": ["13 (18.14)", "12 (21.69)", "6 (16.02)", "3 (13.41)", "2 (14.3)", "0 (7.25)", "4 (9.93)", "0 (5.83)"],
    "6": ["14 (17.8)", "10 (19.09)", "4 (19.82)", "4 (15.06)", "3 (13.94)", "1 (9.76)", "6 (14.89)", "0 (9.1)"],
    "7": ["13 (20.45)", "9 (19.77)", "3 (16.78)", "2 (15.47)", "1 (12.45)", "1 (8.92)", "3 (10.02)", "0 (5.24)"],
    "8": ["13 (17.74)", "9 (19.14)", "3 (17.25)", "1 (12.09)", "2 (18.17)", "2 (9.31)", "0 (10.72)", "0 (12.93)"],
    "9": ["14 (19.4)", "7 (19.56)", "5 (17.08)", "5 (16.45)", "2 (17.17)", "1 (11.93)", "2 (10.26)", "0 (8.32)"],
    "10": ["15 (22.24)", "9 (18.33)", "4 (18.26)", "5 (15.91)", "2 (18.85)", "1 (11.76)", "0 (8.01)", "0 (8.21)"],
    "11": ["14 (18.28)", "12 (20.82)", "4 (19.1)", "1 (13.59)", "1 (10.75)", "3 (13.07)", "0 (12.19)", "0 (9.97)"],
    "12": ["14 (21.57)", "12 (21.29)", "4 (17.54)", "3 (12.23)", "2 (20.66)", "1 (11.4)", "1 (13.12)", "0 (11.96)"],
    "13": ["13 (20.64)", "9 (20.42)", "2 (15.18)", "1 (13.19)", "3 (23.4)", "0 (3.4)", "0 (8.54)", "0 (10.88)"],
    "14": ["14 (20.52)", "11 (20.8)", "2 (15.77)", "6 (16.48)", "0 (11.8)", "0 (3.85)", "0 (11.33)", "0 (5.83)"],
}

def df_convert():
    
    file_path = f"{os.getcwd()}/saved_data/genetic_programming/symbolic_regression_data.csv"
    intron_tree_data = pd.read_csv(file_path)

    # Replace None with "None" as a string in the Threshold column
    intron_tree_data["Threshold"] = intron_tree_data["Threshold"].replace({"nan": "None"}).astype(str)

    # Pivot the data to create the required format
    intron_tree_pivoted = intron_tree_data.pivot(index="Function", columns="Threshold", values=["Intron Ratio", "Average Tree Size"])

    # Combine Intron Ratio and Tree Size into a single formatted string
    intron_tree_formatted = intron_tree_pivoted.apply(
        lambda row: {
            col: f"{row['Intron Ratio'][col]:.4f} ({row['Average Tree Size'][col]:.2f})"
            if not pd.isna(row['Intron Ratio'][col]) and not pd.isna(row['Average Tree Size'][col])
            else ""
            for col in intron_tree_pivoted["Intron Ratio"].columns
        },
        axis=1
    )

    # Simplify the structure to match the diversity/success format
    intron_tree_formatted_df = pd.DataFrame(intron_tree_formatted.tolist(), index=intron_tree_pivoted.index)
    
    intron_tree_formatted_df = intron_tree_formatted_df.rename(columns={"nan": "None", "5.0":"5", "6.0":"6", "7.0":"7", "8.0":"8", "9.0":"9", "10.0":"10",
                                                                        "11.0":"11", "12.0":"12", "13.0":"13", "14.0":"14"})
    
    # Define the correct column order
    column_order = ["None", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]

    # # Reorder columns to match the desired order
    intron_tree_formatted_df = intron_tree_formatted_df[column_order]

    # Reset the index to remove the "Function" label
    intron_tree_formatted_df.reset_index(drop=True, inplace=False)
    
    intron_tree_formatted_df.index.name = None
    
    print(intron_tree_formatted_df)
    
    return intron_tree_formatted_df


def merge_DS_with_IT(intron_tree_formatted_df):
    
    output_file_path = f"{os.getcwd()}/saved_data/genetic_programming/all_data.csv"
    # merged_data.to_csv(output_file_path, index=False)
    
    all_data = pd.read_csv(output_file_path, index_col=0, header=[0])

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
    
    # Display the cleaned and parsed data for verification
    # with pd.option_context(
    #     "display.max_rows", None,  # Show all rows
    #     "display.max_columns", None,  # Show all columns
    #     "display.width", None,  # Don't limit the width
    #     "display.colheader_justify", "center"  # Center column headers
    # ):
    #     print(parsed_data_final)
    
    return parsed_data_final

def thres_impact_analysis(success, diversity, intron_ratio, tree_size):
    """
        Threshold Impact Analysis for Individual Functions
    """

    # Create subplots for each metric across thresholds
    metrics = {"Success": success, "Diversity": diversity, "Intron Ratio": intron_ratio, "Tree Size": tree_size}
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 20), sharex=True)

    for ax, (metric_name, metric_data) in zip(axes, metrics.items()):
        for function in metric_data.index:
            ax.plot(thresholds, metric_data.loc[function], marker='o', label=function)
        ax.set_title(f"Threshold Impact on {metric_name}")
        ax.set_ylabel(metric_name)
        ax.grid(True)
        ax.legend(title="Function", bbox_to_anchor=(1.05, 1), loc='upper left')

    axes[-1].set_xlabel("Threshold")
    plt.tight_layout()
    
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_thres_impact_analysis_intron.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def intron_corr_impact(success, diversity, intron_ratio):
    """
        Impact of Introns on Diversity and Successes for All Functions
    """


    # Scatterplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=intron_ratio.stack(), y=diversity.stack(), ax=axes[0], hue=success.stack(), palette="viridis")
    sns.scatterplot(x=intron_ratio.stack(), y=success.stack(), ax=axes[1], hue=diversity.stack(), palette="plasma")

    axes[0].set_title("Intron Ratio vs Diversity")
    axes[0].set_xlabel("Intron Ratio")
    axes[0].set_ylabel("Diversity")

    axes[1].set_title("Intron Ratio vs Success")
    axes[1].set_xlabel("Intron Ratio")
    axes[1].set_ylabel("Success")

    plt.tight_layout()
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_intron_corr_impact.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

    # Correlation Analysis
    correlations = {
        "Intron Ratio vs Diversity": intron_ratio.stack().corr(diversity.stack()),
        "Intron Ratio vs Success": intron_ratio.stack().corr(success.stack())
    }
    
    print(correlations)
    
def intron_diversity_by_fn(diversity, intron_ratio):
    
    # Visualize Diversity vs. Intron Ratio for each function
    fig, ax = plt.subplots(figsize=(12, 8))
    for function in diversity.index:
        sns.scatterplot(x=intron_ratio.loc[function], y=diversity.loc[function], label=function, ax=ax)

    # Add labels and title
    ax.set_title("Intron Ratio vs Diversity by Function")
    ax.set_xlabel("Intron Ratio")
    ax.set_ylabel("Diversity")
    ax.grid(True)
    ax.legend(title="Function", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_ALL_intron_diversity_by_fn.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
    # ------------------------- #

    # Compute Correlation for Each Function
    intron_diversity_correlation = {
        function: intron_ratio.loc[function].corr(diversity.loc[function])
        for function in diversity.index
    }

    # Fit regression lines and visualize trends for each function
    fig, ax = plt.subplots(figsize=(12, 8))
    for function in diversity.index:
        sns.regplot(
            x=intron_ratio.loc[function], 
            y=diversity.loc[function], 
            label=function, 
            ax=ax,
            scatter_kws={"s": 50}, 
            line_kws={"alpha": 0.7}
        )

    # Add labels and title
    ax.set_title("Intron Ratio vs Diversity with Regression by Function")
    ax.set_xlabel("Intron Ratio")
    ax.set_ylabel("Diversity")
    ax.grid(True)
    ax.legend(title="Function", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_ALL_intron_diversity_by_fn_with_regression.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def intron_vs_size_with_succ_and_div(parsed_data_final):
    
    parsed_data_final_reset = parsed_data_final.stack().reset_index()
        
    # Renaming columns for clarity
    parsed_data_final_reset.columns = ['Function', 'Threshold', 'Success', 'Diversity', 'Intron Ratio', 'Tree Size']

    # Converting 'Threshold' to string for categorical representation in plots
    parsed_data_final_reset['Threshold'] = parsed_data_final_reset['Threshold'].astype(str)

    # Scatterplot: Intron Ratio vs Tree Size vs Success and Diversity
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=parsed_data_final_reset,
        x='Intron Ratio',
        y='Tree Size',
        size='Success',
        hue='Diversity',
        palette='viridis',
        sizes=(20, 200),
        alpha=0.8,
        ax=ax
    )
    ax.set_title("Intron Ratio vs Tree Size with Success and Diversity by Threshold")
    ax.set_xlabel("Intron Ratio")
    ax.set_ylabel("Tree Size")
    ax.legend(title="Diversity", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_intron_vs_size_with_succ_and_div.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
    # ------------------------------------- #
    
    # Line plots for each metric across thresholds
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), constrained_layout=True)

    metrics = ["Success", "Intron Ratio", "Tree Size"]
    for i, metric in enumerate(metrics):
        for func in parsed_data_final_reset['Function'].unique():
            subset = parsed_data_final_reset[parsed_data_final_reset['Function'] == func]
            axes[i].plot(
                subset['Threshold'],
                subset[metric],
                marker='o',
                label=func
            )
        axes[i].set_title(f"{metric} Trend by Function and Threshold")
        axes[i].set_xlabel("Threshold")
        axes[i].set_ylabel(metric)
        axes[i].legend(title="Function")

    # Make sure dir exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_tree_size_trend.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

    
    
if __name__ == "__main__":
    
    # Read the file
    intron_tree_formatted_df = df_convert()
    
    # Combine all parsed data into a structured DataFrame
    parsed_data_final = merge_DS_with_IT(intron_tree_formatted_df)
    
    # Thresholds to use as x-axis for visualizations
    thresholds = list(parsed_data_final.columns.levels[1])

    # Extract individual metrics for plotting
    success = parsed_data_final['Success']
    diversity = parsed_data_final['Diversity']
    intron_ratio = parsed_data_final['Intron Ratio']
    tree_size = parsed_data_final['Tree Size']
    
    thres_impact_analysis(success, diversity, intron_ratio, tree_size)
    intron_corr_impact(success, diversity, intron_ratio)
    intron_diversity_by_fn(diversity, intron_ratio)
    
    intron_vs_size_with_succ_and_div(parsed_data_final)


