import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
np.set_printoptions(threshold=np.inf)
import json

# ----- Helper functions to create Merged DATA for plotting in bloat_results_3.py ----- #

def insert_first_column_pad(file_path):

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    if 'New_Column' not in df.columns:
        # Insert a new column at the beginning with all zeros
        df.insert(0, 'New_Column', 0)

        # Save the updated DataFrame to a new CSV file
        output_path = file_path
        df.to_csv(output_path, index=False)
    
    return df
        
def parse_intron_tree(row):
    parsed = {}
    for col in row.index:
        match = re.match(r"([\d.]+)\s\(([\d.]+)\)", str(row[col]))
        if match:
            parsed[col] = {"Mean Intron Ratio": float(match.group(1)), "Mean Average Tree Size": float(match.group(2))}
        else:
            parsed[col] = {"Mean Intron Ratio": None, "Mean Average Tree Size": None}
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

# ------------------------------------------------------------------------------------- #

def reorganize_intron_tree(parsed_data):
    # Group by "Function" and reorganize data
    formatted_data = (
        parsed_data.reset_index()
        .groupby("Function", group_keys=False)  # Avoid including group keys in the operation
        .apply(lambda group: {
            row["Threshold"]: f"{row['Mean Intron Ratio']:.4f} ({row['Mean Average Tree Size']:.2f})"
            for _, row in group.iterrows()
        }, include_groups=False)
    )
    # Convert the resulting series to a DataFrame
    return pd.DataFrame(formatted_data.tolist(), index=formatted_data.index)

def df_convert(file_path):

    intron_tree_data = pd.read_csv(file_path)

    # Replace None with "None" as a string in the Threshold column
    intron_tree_data["Threshold"] = intron_tree_data["Threshold"].replace({"nan": "None"}).astype(str)

    # Pivot the data to create the required format
    intron_tree_pivoted = intron_tree_data.pivot(index="Function", columns="Threshold", values=["Mean Intron Ratio", "Mean Average Tree Size"])

    # Combine Intron Ratio and Tree Size into a single formatted string
    intron_tree_formatted = intron_tree_pivoted.apply(
        lambda row: {
            col: f"{row['Mean Intron Ratio'][col]:.4f} ({row['Mean Average Tree Size'][col]:.2f})"
            if not pd.isna(row['Mean Intron Ratio'][col]) and not pd.isna(row['Mean Average Tree Size'][col])
            else ""
            for col in intron_tree_pivoted["Mean Intron Ratio"].columns
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


def merge_DS_with_IT(output_file_path):
    
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

def thres_impact_analysis(success, diversity, intron_ratio, tree_size, type_run):
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
    plot_filename = f"genetic_programming/bloat/{type_run}/thres_impact_analysis_intron.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def intron_corr_impact(success, diversity, intron_ratio, type_run):
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
    plot_filename = f"genetic_programming/bloat/{type_run}/intron_corr_impact.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

    # Correlation Analysis
    correlations = {
        "Intron Ratio vs Diversity": intron_ratio.stack().corr(diversity.stack()),
        "Intron Ratio vs Success": intron_ratio.stack().corr(success.stack())
    }
    
    print(correlations)
    
def intron_diversity_by_fn(diversity, intron_ratio, type_run):
    
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
    plot_filename = f"genetic_programming/bloat/{type_run}/intron_diversity_by_fn.png"
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
    plot_filename = f"genetic_programming/bloat/{type_run}/intron_diversity_by_fn_with_regression.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def intron_vs_size_with_succ_and_div(parsed_data_final, type_run):
    
    parsed_data_final_reset = parsed_data_final.stack(future_stack=True).reset_index()
        
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
    plot_filename = f"genetic_programming/bloat/{type_run}/intron_vs_size_with_succ_and_div.png"
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
    plot_filename = f"genetic_programming/bloat/{type_run}/tree_size_trend.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

    
if __name__ == "__main__":
    
    types = ["random_mut", "intron_mutation", "half_mut", "intron_plus", "random_plus"]
    
    for type_run in types:
        
        # Get File name
        file_path = f"{os.getcwd()}/saved_data/introns_study/{type_run}_symbolic_regression_data_ALL.csv"  # Original
                
        # Read the file
        intron_tree_formatted_df = df_convert(file_path)
        
        # Parse
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
        
        # Get the Success (Diversity) data from the dictionary
        dict_file_path = f"{os.getcwd()}/saved_data/introns_study/{type_run}_succ_div_dict_DIV_AVG.json"

        # Read the dictionary back from the file
        with open(dict_file_path, 'r') as f:
            diversity_success_data = json.load(f)
            
        # Convert to DataFrame
        diversity_success_df = pd.DataFrame.from_dict(diversity_success_data).set_index("Function")

        # Ensure alignment with diversity/success data
        reorganized_intron_tree = reorganized_intron_tree.reindex(columns=diversity_success_df.columns)

        # Concatenate the parsed datasets
        final_merged_data = pd.concat([diversity_success_df, reorganized_intron_tree], keys=["Diversity/Success", "Intron/Tree Size"])    

        # Get file name of merged data
        output_file_path = f"{os.getcwd()}/saved_data/introns_study/{type_run}_merged_data_DIV_AVG.csv" # Original

        # Create the file with the final data
        final_merged_data.to_csv(output_file_path, index=False)
        
        # Insert initial column
        parsed_data_final = insert_first_column_pad(output_file_path)
        
        # Combine all parsed data into a structured DataFrame
        parsed_data_final = merge_DS_with_IT(output_file_path)

        # Thresholds to use as x-axis for visualizations
        thresholds = list(parsed_data_final.columns.levels[1])

        # Extract individual metrics for plotting
        success = parsed_data_final['Success']
        diversity = parsed_data_final['Diversity']
        intron_ratio = parsed_data_final['Intron Ratio']
        tree_size = parsed_data_final['Tree Size']
        
        # ------ Plots ----- #
        
        # thres_impact_analysis(success, diversity, intron_ratio, tree_size, type_run)
        # intron_corr_impact(success, diversity, intron_ratio, type_run)
        # intron_diversity_by_fn(diversity, intron_ratio, type_run)
        # intron_vs_size_with_succ_and_div(parsed_data_final, type_run)


