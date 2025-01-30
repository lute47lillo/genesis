import pandas as pd
from scipy.stats import fisher_exact
import itertools
import numpy as np
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)   

def percentage_to_count(percentage, total_trials=15):
    """
    Convert percentage to count based on total trials.
    """
    if pd.isna(percentage):
        return np.nan
    return int(round((percentage * total_trials) / 100))

def pairwise_fisher_tests(df_percentages, total_trials=15):
    """
    Perform pairwise Fisher's Exact Tests between all methods for each function.
    """
    # Convert percentages to counts
    df_counts = df_percentages.copy()
    for col in df_counts.columns[1:]:
        df_counts[col] = df_counts[col].apply(lambda x: percentage_to_count(x, total_trials))
    
    # List of methods
    methods = df_counts['Method'].tolist()
    
    # List of functions
    functions = df_counts.columns[1:]
    
    # Initialize a list to store results
    results = []
    
    # Iterate over each function
    for func in functions:
        # Generate all possible method pairs
        method_pairs = list(itertools.combinations(methods, 2))
        
        for pair in method_pairs:
            method1, method2 = pair
            # Get counts for method1
            success1 = df_counts.loc[df_counts['Method'] == method1, func].values[0]
            failure1 = total_trials - success1
            # Get counts for method2
            success2 = df_counts.loc[df_counts['Method'] == method2, func].values[0]
            failure2 = total_trials - success2
            
            # Check for missing data
            if np.isnan(success1) or np.isnan(success2):
                p_val = np.nan
            else:
                # Create contingency table
                contingency_table = [[success1, failure1],
                                     [success2, failure2]]
                # Handle cases with both methods having zero failures/successes
                if (success1 == 0 and failure2 == 0) or (success2 == 0 and failure1 == 0):
                    p_val = 1.0
                else:
                    # Perform Fisher's Exact Test
                    _, p_val = fisher_exact(contingency_table, alternative='two-sided')
            
            # Store the result
            results.append({
                'Function': func,
                'Method1': method1,
                'Method2': method2,
                'p-value': p_val
            })
    
    # Create a DataFrame from results
    results_df = pd.DataFrame(results)
    
    return results_df

def apply_bonferroni_correction(p_values_df, alpha=0.05):
    """
    Apply Bonferroni correction to a DataFrame of p-values.
    """
    # Total number of comparisons
    n = p_values_df['p-value'].notna().sum()
    
    # Bonferroni corrected alpha
    alpha_corrected = alpha / n
    
    # Determine significance
    p_values_df['Significant'] = p_values_df['p-value'] <= alpha_corrected
    
    return p_values_df, alpha_corrected

def create_pivot_table_with_symbols(p_values_corrected_df):
    """
    Create a pivoted table with functions as rows and method pairs as columns.
    Include symbols to indicate significance.
    """
    # Create a unique identifier for method pairs
    p_values_corrected_df['Methods: '] = p_values_corrected_df.apply(
        lambda row: f"{row['Method1']} vs {row['Method2']}", axis=1
    )
    
    # Pivot the table
    pivot_df = p_values_corrected_df.pivot(index='Function', columns='Methods: ', values='p-value')
    
    # Similarly, pivot the significance
    sig_pivot_df = p_values_corrected_df.pivot(index='Function', columns='Methods: ', values='Significant')
    
    # Combine p-values and significance
    for col in pivot_df.columns:
        pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "NA")
        # Add significance marker
        pivot_df[col] = pivot_df[col].astype(str) + sig_pivot_df[col].apply(lambda x: " *" if x else "")
    
    return pivot_df

def export_to_latex(pivot_table_df, filename="comparison_pvalues.tex"):
    """
    Export the pivoted DataFrame to a LaTeX table.
    """
    with open(filename, 'w') as f:
        f.write(pivot_table_df.to_latex(escape=False, caption="Pairwise Comparison P-Values Between Methods", label="tab:pairwise_pvalues"))
    print(f"LaTeX table exported to {filename}")

# Sample DataFrame based on your table
data = {
    'Method': ['HBC', 'non-HBC', 'non-HBC (10%)', 'SAC', 'SSC', 'SAC + HBC', 'SSC + HBC'],
    'nguyen1': [100.0, 93.33, 93.33, 33.33, 60.00, 60.00, 53.33],
    'nguyen2': [86.67, 46.67, 80.00, 26.67, 53.33, 60.00, 13.33],
    'nguyen3': [46.67, 6.67, 53.33, 26.67, 26.67, 60.00, 0.00],
    'nguyen4': [33.33, 20.00, 46.67, 20.00, 33.33, 60.00, 0.00],
    'nguyen5': [26.67, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'nguyen6': [20.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'nguyen7': [20.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
}

data = {
    'Method': ['HBC', 'SAC', 'SSC'],
    'nguyen1': [100.0, 33.33, 60.00],
    'nguyen2': [86.67, 26.67, 53.33],
    'nguyen3': [46.67, 26.67, 26.67],
    'nguyen4': [33.33, 20.00, 33.33],
    'nguyen5': [26.67, 0.00, 0.00],
    'nguyen6': [20.00, 0.00, 0.00],
    'nguyen7': [20.00, 0.00, 0.00]
}

df_percentages = pd.DataFrame(data)

# Perform pairwise Fisher's Exact Tests
p_values_df = pairwise_fisher_tests(df_percentages, total_trials=15)

# Apply Bonferroni Correction
p_values_corrected_df, alpha_corrected = apply_bonferroni_correction(p_values_df, alpha=0.05)

print(f"Bonferroni Corrected Alpha: {alpha_corrected:.5f}\n")

# Create Pivoted Table with Symbols
pivot_table_df = create_pivot_table_with_symbols(p_values_corrected_df)

print(pivot_table_df)

# # Export the pivoted table to LaTeX
# export_to_latex(pivot_table_df, filename="comparison_pvalues.tex")
