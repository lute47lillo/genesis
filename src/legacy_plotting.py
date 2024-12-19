import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_scatter_InbreedUnblock(dfs, config_plot, depths, titles=None):
    """
    Plots scatter plots for each DataFrame in dfs.
    
    Parameters:
    - dfs: list of pandas DataFrames
    - titles: list of titles for each subplot
    """
    num_dfs = len(dfs)
    fig, axes = plt.subplots(1, num_dfs, figsize=(4 * num_dfs, 4), squeeze=False)

    for idx, df in enumerate(dfs):
        ax = axes[0, idx]
        
        # Select first 5 and last 5 rows
        selected_df = pd.concat([df.head(5), df.tail(5)], ignore_index=True)
        
        # Determine colors based on 'key' prefix
        colors = selected_df['key'].apply(
            lambda x: 'blue' if x.startswith('InbreedBlock') else 'red'
        )
        
        # Create scatter plot
        sns.scatterplot(
            data=selected_df,
            x='diversity',
            y='mean_gen_success',
            hue=selected_df['key'].apply(lambda x: 'InbreedBlock' if x.startswith('InbreedBlock') else 'InbreedUnblock'),
            palette={'InbreedBlock': 'blue', 'InbreedUnblock': 'red'},
            ax=ax,
            s=100
        )
        
        # Customize plot
        ax.set_title(titles[idx] if titles else f'Max Depth {depths[idx ]}')
        ax.set_xlabel('Diversity')
        ax.set_ylabel('Mean Generation Success')

    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/figures/{config_plot}.png")
