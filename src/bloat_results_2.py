import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
np.set_printoptions(threshold=np.inf)

diversity_success_data = {
    "Function": ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6", "nguyen7", "nguyen8"],
    "None": ["13 (13.00)", "15 (15.15)", "10 (17.39)", "10 (16.53)", "0 (14.66)", "0 (9.61)", "1 (15.79)", "0 (11.28)"],
    "5": ["14 (15.20)", "15 (16.26)", "9 (18.20)", "11 (19.15)", "0 (15.68)", "0 (15.13)", "1 (16.32)", "0 (15.33)"],
    "6": ["14 (14.24)", "12 (16.74)", "9 (19.54)", "9 (19.34)", "0 (18.48)", "0 (14.52)", "1 (19.38)", "0 (14.18)"],
    "7": ["15 (14.08)", "15 (16.07)", "12 (17.12)", "8 (18.59)", "0 (21.15)", "1 (16.35)", "2 (16.53)", "0 (14.14)"],
    "8": ["15 (13.70)", "15 (15.51)", "10 (20.23)", "12 (18.48)", "0 (18.92)", "0 (18.25)", "0 (20.11)", "0 (17.09)"],
    "9": ["15 (15.14)", "15 (16.71)", "12 (20.16)", "12 (20.20)", "0 (21.46)", "1 (17.07)", "2 (18.06)", "0 (18.16)"],
    "10": ["15 (14.21)", "15 (16.65)", "12 (19.32)", "10 (21.46)", "1 (20.76)", "0 (17.23)", "0 (23.16)", "0 (14.68)"],
    "11": ["15 (14.27)", "15 (19.05)", "11 (21.13)", "10 (23.31)", "0 (20.43)", "0 (18.03)", "0 (20.27)", "0 (19.42)"],
    "12": ["15 (15.65)", "15 (18.53)", "13 (21.48)", "9 (20.45)", "0 (20.63)", "0 (6.59)", "0 (25.02)", "0 (18.48)"],
    "13": ["15 (14.47)", "15 (18.44)", "12 (21.31)", "10 (22.11)", "0 (19.09)", "0 (3.81)", "0 (22.19)", "0 (7.43)"],
    "14": ["15 (15.65)", "15 (19.32)", "11 (21.59)", "12 (23.39)", "0 (11.40)", "0 (3.15)", "0 (23.87)", "0 (5.59)"],
}

def df_convert():
    
    # file_path = f"{os.getcwd()}/saved_data/genetic_programming/symbolic_regression_data.csv" # Original
    file_path = f"{os.getcwd()}/saved_data/genetic_programming/intron_mutation_symbolic_regression_data.csv" # intron mutations
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


def merge_DS_with_IT(final_merged_data):
    
    # output_file_path = f"{os.getcwd()}/saved_data/genetic_programming/all_data.csv" # Original
    output_file_path = f"{os.getcwd()}/saved_data/genetic_programming/intron_mutation_all_data.csv" # intron mutations

    final_merged_data.to_csv(output_file_path, index=False)
    
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
    with pd.option_context(
        "display.max_rows", None,  # Show all rows
        "display.max_columns", None,  # Show all columns
        "display.width", None,  # Don't limit the width
        "display.colheader_justify", "center"  # Center column headers
    ):
        print(parsed_data_final)        
    
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
    plot_filename = f"genetic_programming/bloat/intron_mutation_Final_thres_impact_analysis_intron.png"
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
    plot_filename = f"genetic_programming/bloat/intron_mutation_Final_intron_corr_impact.png"
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
    plot_filename = f"genetic_programming/bloat/intron_mutation_Final_ALL_intron_diversity_by_fn.png"
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
    plot_filename = f"genetic_programming/bloat/intron_mutation_Final_ALL_intron_diversity_by_fn_with_regression.png"
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
    plot_filename = f"genetic_programming/bloat/intron_mutation_Final_intron_vs_size_with_succ_and_div.png"
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
    plot_filename = f"genetic_programming/bloat/intron_mutation_Final_tree_size_trend.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

    
    
if __name__ == "__main__":
    
    # Read the file
    intron_tree_formatted_df = df_convert()
    
    # ----- TODO: Trying to combine both datasets ------> creating the ..._all_Data.csv ----- #
    
    def parse_intron_tree(row):
        parsed = {}
        for col in row.index:
            match = re.match(r"([\d.]+)\s\(([\d.]+)\)", str(row[col]))
            if match:
                parsed[col] = {"Intron Ratio": float(match.group(1)), "Average Tree Size": float(match.group(2))}
            else:
                parsed[col] = {"Intron Ratio": None, "Average Tree Size": None}
        return pd.DataFrame(parsed).T
    
    def parse_diversity_success(df):
        success_data = {}
        diversity_data = {}
        for col in df.columns:
            success = []
            diversity = []
            for val in df[col]:
                match = re.match(r"(\d+)\s\(([\d.]+)\)", val)
                if match:
                    success.append(int(match.group(1)))
                    diversity.append(float(match.group(2)))
                else:
                    success.append(None)
                    diversity.append(None)
            success_data[col] = success
            diversity_data[col] = diversity
        return pd.DataFrame({"Success": success_data, "Diversity": diversity_data})

    def reorganize_intron_tree(parsed_data):
        # Group by "Function" and reorganize data
        formatted_data = (
            parsed_data.reset_index()
            .groupby("Function", group_keys=False)  # Avoid including group keys in the operation
            .apply(lambda group: {
                row["Threshold"]: f"{row['Intron Ratio']:.4f} ({row['Average Tree Size']:.2f})"
                for _, row in group.iterrows()
            }, include_groups=False)
        )
        # Convert the resulting series to a DataFrame
        return pd.DataFrame(formatted_data.tolist(), index=formatted_data.index)
    
    parsed_intron_tree = pd.concat(
        [parse_intron_tree(intron_tree_formatted_df.loc[function]) for function in intron_tree_formatted_df.index],
        keys=intron_tree_formatted_df.index,
    )

    # Parse each function's row and assign multi-level index
    parsed_intron_tree_list = []
    for function in intron_tree_formatted_df.index:
        parsed = parse_intron_tree(intron_tree_formatted_df.loc[function])
        parsed["Function"] = function  # Add function as a column for multi-level index
        parsed_intron_tree_list.append(parsed)

    # Concatenate with multi-level index
    parsed_intron_tree = pd.concat(parsed_intron_tree_list)
    parsed_intron_tree = parsed_intron_tree.reset_index().rename(columns={"index": "Threshold"})

    # Ensure no duplicate indices
    parsed_intron_tree = parsed_intron_tree.set_index(["Function", "Threshold"])
    
    # Reorganize intron/tree size data to match the diversity/success format
    reorganized_intron_tree = reorganize_intron_tree(parsed_intron_tree)
    
    # Convert to DataFrame
    diversity_success_df = pd.DataFrame.from_dict(diversity_success_data).set_index("Function")

    # Ensure alignment with diversity/success data
    reorganized_intron_tree = reorganized_intron_tree.reindex(columns=diversity_success_df.columns)

    # Concatenate the parsed datasets
    final_merged_data = pd.concat([diversity_success_df, reorganized_intron_tree], keys=["Diversity/Success", "Intron/Tree Size"])

    #  ------ WORKS ------- #
    
    # Combine all parsed data into a structured DataFrame
    parsed_data_final = merge_DS_with_IT(final_merged_data)
    
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


