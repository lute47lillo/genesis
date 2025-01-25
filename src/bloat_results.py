import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# Data for each symbolic regression function and thresholds
data = {
    "nguyen1": {
        None: (13, 19.79), 5: (13, 18.14), 6: (14, 17.80), 7: (13, 20.45),
        8: (13, 17.74), 9: (14, 19.40), 10: (15, 22.24), 11: (14, 18.28),
        12: (14, 21.57), 13: (13, 20.64), 14: (14, 20.52)
    },
    "nguyen2": {
        None: (10, 19.52), 5: (12, 21.69), 6: (10, 19.09), 7: (9, 19.77),
        8: (9, 19.14), 9: (7, 19.56), 10: (9, 18.33), 11: (12, 20.82),
        12: (12, 21.29), 13: (9, 20.42), 14: (11, 20.80)
    },
    "nguyen3": {
        None: (2, 18.57), 5: (6, 16.02), 6: (4, 19.82), 7: (3, 16.78),
        8: (3, 17.25), 9: (5, 17.08), 10: (4, 18.26), 11: (4, 19.10),
        12: (4, 17.54), 13: (2, 15.18), 14: (2, 15.77)
    },
    "nguyen4": {
        None: (4, 12.39), 5: (3, 13.41), 6: (4, 15.06), 7: (2, 15.47),
        8: (1, 12.09), 9: (5, 16.45), 10: (5, 15.91), 11: (1, 13.59),
        12: (3, 12.23), 13: (1, 13.19), 14: (6, 16.48)
    },
    "nguyen5": {
        None: (0, 7.49), 5: (2, 14.30), 6: (3, 13.94), 7: (1, 12.45),
        8: (2, 18.17), 9: (2, 17.17), 10: (2, 18.85), 11: (1, 10.75),
        12: (2, 20.66), 13: (3, 23.40), 14: (0, 11.80)
    },
    "nguyen6": {
        None: (0, 8.37), 5: (0, 7.25), 6: (1, 9.76), 7: (1, 8.92),
        8: (2, 9.31), 9: (1, 11.93), 10: (1, 11.76), 11: (3, 13.07),
        12: (1, 11.40), 13: (0, 3.40), 14: (0, 3.85)
    },
    "nguyen7": {
        None: (6, 10.35), 5: (4, 9.93), 6: (6, 14.89), 7: (3, 10.02),
        8: (0, 10.72), 9: (2, 10.26), 10: (0, 8.01), 11: (0, 12.19),
        12: (1, 13.12), 13: (0, 8.54), 14: (0, 11.33)
    },
    "nguyen8": {
        None: (0, 8.55), 5: (0, 5.83), 6: (0, 9.10), 7: (0, 5.24),
        8: (0, 12.93), 9: (0, 8.32), 10: (0, 8.21), 11: (0, 9.97),
        12: (0, 11.96), 13: (0, 10.88), 14: (0, 5.83)
    }
}

def create_ds():
    
    # Create a DataFrame from the data
    thresholds = [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    df_data = {
        (threshold if threshold is not None else "None"): [
            f"{data[function].get(threshold, (0, 0))[0]} ({data[function].get(threshold, (0, 0))[1]})"
            for function in data
        ]
        for threshold in thresholds
    }

    table = pd.DataFrame(df_data, index=data.keys())
    
    return table

def plot_success_vs_thresholds():

    # Extract data for plotting
    functions = list(data.keys())
    thresholds = [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # Prepare the success data
    success_data = {
        func: [data[func].get(threshold, (0, 0))[0] for threshold in thresholds] for func in functions
    }

    # Convert None to a string for plotting purposes
    threshold_labels = [str(thresh) if thresh is not None else "None" for thresh in thresholds]

    # Plotting success vs. threshold for each function
    plt.figure(figsize=(12, 8))

    for func in functions:
        plt.plot(threshold_labels, success_data[func], marker='o', label=func)

    # Customize the plot
    plt.title("Success vs. Threshold for Symbolic Regression Functions")
    plt.xlabel("Bloat Threshold")
    plt.ylabel("Number of Successes")
    plt.xticks(rotation=45)
    plt.legend(title="Functions", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display the plot
    # Ensure the 'figures' directory exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_All_success_vs_thresholds.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def correlation_analysis():
    
    # Prepare data for correlation analysis
    success_values = []
    diversity_values = []
    
    thresholds = [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for func in data:
        for threshold in thresholds:
            success, diversity = data[func].get(threshold, (0, 0))
            success_values.append(success)
            diversity_values.append(diversity)

    # Calculate Pearson correlation coefficient
    correlation_coefficient = np.corrcoef(success_values, diversity_values)[0, 1]

    # Display the result
    print(f"\nThe Pearson correlation coefficient between the number of successes and diversity is: {correlation_coefficient}.")
    
    # ~ 0.7506 This indicates a strong positive correlation, suggesting that higher diversity is generally associated
    # with a greater number of successes across all functions and thresholds
    
    return correlation_coefficient, success_values, diversity_values
    
def plot_succ_div_correlation(corr_coeff, success_values, diversity_values):
    # Plotting the correlation between success and diversity
    plt.figure(figsize=(8, 6))

    plt.scatter(success_values, diversity_values, alpha=0.7, edgecolor='k')
    plt.title("Correlation Between Success and Diversity")
    plt.xlabel("Number of Successes")
    plt.ylabel("Final Diversity")
    plt.grid(True)

    # Add the correlation coefficient to the plot
    plt.text(
        0.95, 0.05, f"Correlation: {corr_coeff:.2f}",
        ha="right", va="bottom", transform=plt.gca().transAxes, fontsize=12, color="blue"
    )

    # Display the plot
    
    # Ensure the 'figures' directory exists
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_All_succ_div_correlation.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
    # showing the relationship between the number of successes and final diversity

def diversity_trend_funcs():
    
    functions = list(data.keys())
    thresholds = [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Prepare data for diversity trends
    diversity_data = {
        func: [data[func].get(threshold, (0, 0))[1] for threshold in thresholds] for func in functions
    }

    # Plot diversity trends for each function
    plt.figure(figsize=(12, 8))

    # Convert None to a string for plotting purposes
    threshold_labels = [str(thresh) if thresh is not None else "None" for thresh in thresholds]
    for func in functions:
        plt.plot(threshold_labels, diversity_data[func], marker='o', label=func)

    # Customize the plot
    plt.title("Diversity Trends Across Functions")
    plt.xlabel("Bloat Threshold")
    plt.ylabel("Final Diversity")
    plt.xticks(rotation=45)
    plt.legend(title="Functions", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_diversity_trend_funcs.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
    
def threshold_impact_on_success():
    
    thresholds = [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Aggregate success rates for each threshold
    threshold_success_rates = {threshold: [] for threshold in thresholds}

    for func in data:
        for threshold in thresholds:
            success, _ = data[func].get(threshold, (0, 0))
            threshold_success_rates[threshold].append(success)

    # Calculate average success rate per threshold
    avg_success_rates = {
        threshold: np.mean(threshold_success_rates[threshold]) for threshold in thresholds
    }

    # Plot average success rate per threshold
    plt.figure(figsize=(10, 6))
    threshold_labels = [str(thresh) if thresh is not None else "None" for thresh in thresholds]

    plt.plot(threshold_labels, list(avg_success_rates.values()), marker='o', label="Average Success Rate")
    plt.title("Impact of Thresholds on Success Rates")
    plt.xlabel("Bloat Threshold")
    plt.ylabel("Average Number of Successes")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.axhline(np.mean(list(avg_success_rates.values())), color='red', linestyle='--', linewidth=0.8, label="Overall Average")
    plt.legend()

    # Display the plot
    plt.tight_layout()
    figures_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plot_filename = f"genetic_programming/bloat/Final_threshold_impact_on_success.png"
    plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')

    # Highlight thresholds with maximum and minimum success rates
    max_success_threshold = max(avg_success_rates, key=avg_success_rates.get)
    min_success_threshold = min(avg_success_rates, key=avg_success_rates.get)

    print(f"\nThe Inbreeding Threshold with more nº of success overall is: {max_success_threshold} with an average success rate of: {avg_success_rates[max_success_threshold]}.")
    print(f"\nThe Inbreeding Threshold with less nº of success overall is: {min_success_threshold} with an average success rate of: {avg_success_rates[min_success_threshold]}.")

def thres_impact_by_fn(thresholds_to_explore):
    """

    """
    
    # Prepare success data for specific thresholds across functions
    functions = list(data.keys())
    
    threshold_function_success = {
        threshold: [data[func].get(threshold, (0, 0))[0] for func in functions] for threshold in thresholds_to_explore
    }
    

    # Create individual plots for each threshold
    for threshold in thresholds_to_explore:
        plt.figure(figsize=(8, 5))
        plt.bar(functions, threshold_function_success[threshold])
        plt.title(f"Success Rates by Function for Threshold {str(threshold)}")
        plt.xlabel("Function")
        plt.ylabel("Number of Successes")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        
        figures_dir = os.path.join(os.getcwd(), 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Save the figure
        plot_filename = f"genetic_programming/bloat/Final_thres:{str(threshold)}_impact_by_fn.png"
        plt.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight')
        
def best_thresh_by_fn():
    
    functions = list(data.keys())
    thresholds = [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Determine the best threshold for each function considering diversity as a tie-breaker
    best_thresholds_with_tiebreaker = {}

    for func in functions:
        max_success = max(data[func][threshold][0] for threshold in thresholds if threshold in data[func])
        # Filter thresholds with max success
        candidates = [
            (threshold, data[func][threshold][1])  # (threshold, diversity)
            for threshold in thresholds if data[func].get(threshold, (0, 0))[0] == max_success
        ]
        # Select the one with the highest diversity
        best_threshold = max(candidates, key=lambda x: x[1])[0]
        best_thresholds_with_tiebreaker[func] = (best_threshold, max_success, data[func][best_threshold][1])

    # Display the results
    for k, v in best_thresholds_with_tiebreaker.items():
        print(f"\nFunction: {k}: Threshold {v[0]} with {v[1]} successes and diversity {v[2]}.")
        
def performance_by_fn():
    
    functions = list(data.keys())
    thresholds = [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Calculate average success and diversity for each function across all thresholds
    function_performance = {
        func: {
            "Average Success": np.mean([data[func][threshold][0] for threshold in thresholds if threshold in data[func]]),
            "Average Diversity": np.mean([data[func][threshold][1] for threshold in thresholds if threshold in data[func]])
        }
        for func in functions
    }

    # Convert to a DataFrame for better visualization
    performance_df = pd.DataFrame(function_performance).T
    
    # Display the performance differences
    print()
    print(performance_df)


if __name__ == "__main__":
    
    # Display the table
    table = create_ds()
    
    print(table)
    
    plot_success_vs_thresholds()
    
    corr_coeff, success_values, diversity_values = correlation_analysis()
    
    plot_succ_div_correlation(corr_coeff, success_values, diversity_values)
    
    diversity_trend_funcs()
    
    threshold_impact_on_success()
    
    thresholds_to_explore = [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # Specific thresholds for exploration
    thres_impact_by_fn(thresholds_to_explore)
    
    best_thresh_by_fn()
    
    performance_by_fn()