#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

#%% Function to load the master benchmark data
def load_master_benchmark():
    """
    Load the master benchmark data from the validation results.
    """
    # Path to the master benchmark file
    benchmark_file = Path("./custom_validation_results/comprehensive/master_validation_benchmarks.csv")

    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found at {benchmark_file}")
    
    # Load data
    df = pd.read_csv(benchmark_file)
    
    # Convert profit columns to numeric if they aren't already
    profit_columns = ['Optimized_Profit', 'Simulated_Profit', 'SI_Penalty', 
                      'Volume_Penalty', 'Operating_Cost']
    for col in profit_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert processing time to numeric
    df['Processing_Time_Seconds'] = pd.to_numeric(df['Processing_Time_Seconds'], errors='coerce')
    
    return df

#%% Load and prepare data
def prepare_data():
    """
    Load and prepare the data for analysis, separating the extreme date.
    """
    # Load the master benchmark data
    df = load_master_benchmark()
    
    # Separate normal dates from the extreme date
    normal_df = df[df['New_Date'] != '2024-12-12'].copy()
    extreme_df = df[df['New_Date'] == '2024-12-12'].copy()
    
    # Check if extreme date exists in data
    if extreme_df.empty:
        print("Warning: Extreme date 2024-12-12 not found in the data!")
    else:
        print(f"Found {len(extreme_df)} entries for the extreme date 2024-12-12")
    
    print(f"Found {len(normal_df)} entries for normal dates")
    
    return normal_df, extreme_df

#%% Analysis A: Distribution plots for normal dates
def analyze_normal_dates(normal_df):
    """
    Generate separate distribution density plots and variance plots
    comparing simulated profit across different hyperparameters for normal dates only.
    Also include mean and std in legends for better comparison.
    """
    print("\n--- Analyzing Normal Dates (excluding 2024-12-12) ---")
    
    # Create output directories for different plot types
    output_dir = Path("./custom_validation_results/analysis_without_extreme")
    density_dir = output_dir / "density_plots"
    variance_dir = output_dir / "variance_plots"
    
    for directory in [output_dir, density_dir, variance_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Set up figure aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. DENSITY PLOTS
    
    # 1.1 Database type (SOS2 vs GlobalLinear)
    plt.figure(figsize=(10, 6))
    for db in normal_df['Database'].unique():
        db_data = normal_df[normal_df['Database'] == db]
        profit_data = db_data['Simulated_Profit']
        time_data = db_data['Processing_Time_Seconds']
        
        profit_mean = profit_data.mean()
        profit_std = profit_data.std()
        time_mean = time_data.mean()
        time_std = time_data.std()
        
        label = f'{db}\nProfit: {profit_mean:.2f}€ ± {profit_std:.2f}€\nTime: {time_mean:.2f}s ± {time_std:.2f}s'
        sns.kdeplot(profit_data, label=label, fill=True, alpha=0.3)
    
    plt.title('Density Plot of Simulated Profit by Database Type')
    plt.xlabel('Simulated Profit (€)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(density_dir / 'density_profit_by_database.png', dpi=300)
    plt.close()
    
    # 1.2 Architecture type (LSTM vs RNN)
    plt.figure(figsize=(10, 6))
    for arch in normal_df['Architecture'].unique():
        arch_data = normal_df[normal_df['Architecture'] == arch]
        profit_data = arch_data['Simulated_Profit']
        time_data = arch_data['Processing_Time_Seconds']
        
        profit_mean = profit_data.mean()
        profit_std = profit_data.std()
        time_mean = time_data.mean()
        time_std = time_data.std()
        
        label = f'{arch}\nProfit: {profit_mean:.2f}€ ± {profit_std:.2f}€\nTime: {time_mean:.2f}s ± {time_std:.2f}s'
        sns.kdeplot(profit_data, label=label, fill=True, alpha=0.3)
    
    plt.title('Density Plot of Simulated Profit by Architecture Type')
    plt.xlabel('Simulated Profit (€)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(density_dir / 'density_profit_by_architecture.png', dpi=300)
    plt.close()
    
    # 1.3 Number of layers
    plt.figure(figsize=(10, 6))
    for layer in sorted(normal_df['Num_Layers'].unique()):
        layer_data = normal_df[normal_df['Num_Layers'] == layer]
        profit_data = layer_data['Simulated_Profit']
        time_data = layer_data['Processing_Time_Seconds']
        
        profit_mean = profit_data.mean()
        profit_std = profit_data.std()
        time_mean = time_data.mean()
        time_std = time_data.std()
        
        label = f'{layer} Layers\nProfit: {profit_mean:.2f}€ ± {profit_std:.2f}€\nTime: {time_mean:.2f}s ± {time_std:.2f}s'
        sns.kdeplot(profit_data, label=label, fill=True, alpha=0.3)
    
    plt.title('Density Plot of Simulated Profit by Number of Layers')
    plt.xlabel('Simulated Profit (€)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(density_dir / 'density_profit_by_layers.png', dpi=300)
    plt.close()
    
    # 1.4 Number of iterations
    plt.figure(figsize=(10, 6))
    for iter_val in sorted(normal_df['Max_Iterations'].unique()):
        iter_data = normal_df[normal_df['Max_Iterations'] == iter_val]
        profit_data = iter_data['Simulated_Profit']
        time_data = iter_data['Processing_Time_Seconds']
        
        profit_mean = profit_data.mean()
        profit_std = profit_data.std()
        time_mean = time_data.mean()
        time_std = time_data.std()
        
        label = f'{iter_val} Iterations\nProfit: {profit_mean:.2f}€ ± {profit_std:.2f}€\nTime: {time_mean:.2f}s ± {time_std:.2f}s'
        sns.kdeplot(profit_data, label=label, fill=True, alpha=0.3)
    
    plt.title('Density Plot of Simulated Profit by Number of Iterations')
    plt.xlabel('Simulated Profit (€)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(density_dir / 'density_profit_by_iterations.png', dpi=300)
    plt.close()

    
    # 2. VARIANCE PLOTS (BOXPLOTS)
    
    # 2.1 Database type (SOS2 vs GlobalLinear)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Database', y='Simulated_Profit', data=normal_df)
    sns.swarmplot(x='Database', y='Simulated_Profit', data=normal_df, color='black', alpha=0.5)
    
    # Add mean and std to the title
    db_stats = normal_df.groupby('Database')['Simulated_Profit'].agg(['mean', 'std']).round(2)
    title = 'Distribution of Simulated Profit by Database Type\n'
    for db, stats in db_stats.iterrows():
        title += f"{db}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}   "
    
    plt.title(title)
    plt.ylabel('Simulated Profit')
    plt.tight_layout()
    plt.savefig(variance_dir / 'variance_profit_by_database.png', dpi=300)
    plt.close()
    
    # 2.2 Architecture type (LSTM vs RNN)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Architecture', y='Simulated_Profit', data=normal_df)
    sns.swarmplot(x='Architecture', y='Simulated_Profit', data=normal_df, color='black', alpha=0.5)
    
    # Add mean and std to the title
    arch_stats = normal_df.groupby('Architecture')['Simulated_Profit'].agg(['mean', 'std']).round(2)
    title = 'Distribution of Simulated Profit by Architecture Type\n'
    for arch, stats in arch_stats.iterrows():
        title += f"{arch}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}   "
    
    plt.title(title)
    plt.ylabel('Simulated Profit')
    plt.tight_layout()
    plt.savefig(variance_dir / 'variance_profit_by_architecture.png', dpi=300)
    plt.close()
    
    # 2.3 Number of layers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Num_Layers', y='Simulated_Profit', data=normal_df)
    sns.swarmplot(x='Num_Layers', y='Simulated_Profit', data=normal_df, color='black', alpha=0.5)
    
    # Add mean and std to the title
    layers_stats = normal_df.groupby('Num_Layers')['Simulated_Profit'].agg(['mean', 'std']).round(2)
    title = 'Distribution of Simulated Profit by Number of Layers\n'
    for layer, stats in layers_stats.iterrows():
        title += f"{int(layer)} Layers: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}   "
    
    plt.title(title)
    plt.ylabel('Simulated Profit')
    plt.tight_layout()
    plt.savefig(variance_dir / 'variance_profit_by_layers.png', dpi=300)
    plt.close()
    
    # 2.4 Number of iterations
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Max_Iterations', y='Simulated_Profit', data=normal_df)
    sns.swarmplot(x='Max_Iterations', y='Simulated_Profit', data=normal_df, color='black', alpha=0.5)
    
    # Add mean and std to the title
    iter_stats = normal_df.groupby('Max_Iterations')['Simulated_Profit'].agg(['mean', 'std']).round(2)
    title = 'Distribution of Simulated Profit by Number of Iterations\n'
    for iter_val, stats in iter_stats.iterrows():
        title += f"{int(iter_val)} Iterations: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}   "
    
    plt.title(title)
    plt.ylabel('Simulated Profit')
    plt.tight_layout()
    plt.savefig(variance_dir / 'variance_profit_by_iterations.png', dpi=300)
    plt.close()
    
    # 2.5 Interaction: Database and Architecture 
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Database', y='Simulated_Profit', hue='Architecture', data=normal_df)
    
    # Add mean and std annotations
    db_arch_stats = normal_df.groupby(['Database', 'Architecture'])['Simulated_Profit'].agg(['mean', 'std']).round(2)
    title = 'Distribution of Simulated Profit by Database and Architecture\n'
    for idx, stats in db_arch_stats.iterrows():
        db, arch = idx
        title += f"{db}-{arch}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}   "
        if len(title) > 100:  # Add line break if title gets too long
            title += '\n'
    
    plt.title(title)
    plt.ylabel('Simulated Profit')
    plt.tight_layout()
    plt.savefig(variance_dir / 'variance_profit_by_db_arch.png', dpi=300)
    plt.close()
    
    # 2.6 Interaction: Layers and Iterations
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Num_Layers', y='Simulated_Profit', hue='Max_Iterations', data=normal_df)
    
    # Add legend with mean and std
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for i, iter_val in enumerate(sorted(normal_df['Max_Iterations'].unique())):
        iter_data = normal_df[normal_df['Max_Iterations'] == iter_val]['Simulated_Profit']
        mean_val = iter_data.mean()
        std_val = iter_data.std()
        new_labels.append(f'{iter_val} Iterations (Mean: {mean_val:.2f}, Std: {std_val:.2f})')
    
    plt.legend(handles, new_labels)
    plt.title('Distribution of Simulated Profit by Layers and Iterations')
    plt.ylabel('Simulated Profit')
    plt.tight_layout()
    plt.savefig(variance_dir / 'variance_profit_by_layers_iter.png', dpi=300)
    plt.close()
    
    # Add statistical summary
    db_stats = normal_df.groupby('Database')['Simulated_Profit'].agg(['mean', 'std', 'min', 'max'])
    print("\nSimulated Profit by Database:")
    print(db_stats)
    
    arch_stats = normal_df.groupby('Architecture')['Simulated_Profit'].agg(['mean', 'std', 'min', 'max'])
    print("\nSimulated Profit by Architecture:")
    print(arch_stats)
    
    layers_stats = normal_df.groupby('Num_Layers')['Simulated_Profit'].agg(['mean', 'std', 'min', 'max'])
    print("\nSimulated Profit by Number of Layers:")
    print(layers_stats)
    
    iter_stats = normal_df.groupby('Max_Iterations')['Simulated_Profit'].agg(['mean', 'std', 'min', 'max'])
    print("\nSimulated Profit by Max Iterations:")
    print(iter_stats)
    
    # Return to default style
    plt.style.use('default')
    
    return normal_df

def generate_latex_summary_table(normal_df):
    """
    Generate a LaTeX table summarizing the mean performance across different model settings.
    """
    print("\n--- Generating LaTeX Summary Table for Model Settings ---")
    
    # Create output directory if it doesn't exist
    output_dir = Path("./custom_validation_results/analysis_without_extreme")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calculate means by different configurations
    db_means = normal_df.groupby('Database')['Simulated_Profit'].mean().round(2)
    arch_means = normal_df.groupby('Architecture')['Simulated_Profit'].mean().round(2)
    layers_means = normal_df.groupby('Num_Layers')['Simulated_Profit'].mean().round(2)
    iter_means = normal_df.groupby('Max_Iterations')['Simulated_Profit'].mean().round(2)
    
    # Generate LaTeX table
    latex_table = r"""\begin{table}[h]
\centering
\caption{Mean Ex-post Profit by Model Configuration}
\label{tab:model_settings_summary}
\begin{tabular}{llr}
\toprule
Setting & Configuration & Mean Ex-post Profit \\
\midrule
"""
    
    # Add Database rows
    for db, mean in db_means.items():
        latex_table += f"Database & {db} & {mean:.2f} \\\\\n"
    
    # Add Architecture rows
    for arch, mean in arch_means.items():
        latex_table += f"Architecture & {arch} & {mean:.2f} \\\\\n"
    
    # Add Number of Layers rows
    for layers, mean in layers_means.items():
        latex_table += f"Num Layers & {int(layers)} & {mean:.2f} \\\\\n"
    
    # Add Max Iterations rows
    for iters, mean in iter_means.items():
        latex_table += f"Max Iterations & {int(iters)} & {mean:.2f} \\\\\n"
    
    # Close the table
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save the LaTeX table to a file
    with open(output_dir / 'model_settings_summary.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX summary table saved to {output_dir / 'model_settings_summary.tex'}")
    
    return latex_table

def generate_expost_profit_time_table(normal_df):
    """
    Generate a LaTeX table documenting the Mean and Std of each configuration,
    including only Time (s) and Ex-post Profit.
    """
    print("\n--- Generating Configuration Comparison Table for Ex-post Profit and Time ---")
    
    # Create output directory if it doesn't exist
    output_dir = Path("./custom_validation_results/analysis_without_extreme")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Group by configuration parameters
    grouped_df = normal_df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
        ['Processing_Time_Seconds', 'Simulated_Profit']
    ].agg(['mean', 'std']).reset_index()
    
    # Sort by mean Simulated_Profit in descending order
    grouped_df = grouped_df.sort_values(('Simulated_Profit', 'mean'), ascending=False)
    
    # Start building the LaTeX table
    latex_table = r"""\begin{table}[h]
\centering
\caption{Comparison of Model Configurations - Ex-post Profit and Processing Time}
\label{tab:config_profit_time_comparison}
\begin{tabular}{llcccccc}
\toprule
Database & Architecture & Layers & Iterations & \multicolumn{2}{c}{Ex-post Profit} & \multicolumn{2}{c}{Time (s)} \\
\cmidrule(lr){5-6} \cmidrule(lr){7-8}
 & & & & Mean & Std & Mean & Std \\
\midrule
"""
    
    # Add a row for each configuration
    for _, row in grouped_df.iterrows():
        db = row['Database']
        arch = row['Architecture']
        layers = int(row['Num_Layers'])
        iters = int(row['Max_Iterations'])
        
        profit_mean = row[('Simulated_Profit', 'mean')]
        profit_std = row[('Simulated_Profit', 'std')]
        time_mean = row[('Processing_Time_Seconds', 'mean')]
        time_std = row[('Processing_Time_Seconds', 'std')]
        
        latex_table += f"{db} & {arch} & {layers} & {iters} & {profit_mean:.2f} & {profit_std:.2f} & {time_mean:.2f} & {time_std:.2f} \\\\\n"
    
    # Close the table
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save the LaTeX table to a file
    with open(output_dir / 'config_profit_time_comparison.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_dir / 'config_profit_time_comparison.tex'}")
    
    return latex_table
#%% Generate LaTeX table for best model
def generate_latex_table(normal_df):
    """
    Generate a LaTeX table for the best performing model based on ex-post profit (Simulated_Profit).
    """
    print("\n--- Generating LaTeX Table for Best Model ---")
    
    # Find the configuration with the highest average simulated profit
    avg_by_config = normal_df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
        'Simulated_Profit'
    ].mean().reset_index()
    
    best_config = avg_by_config.loc[avg_by_config['Simulated_Profit'].idxmax()]
    
    print(f"Best Configuration: {best_config['Database']}, {best_config['Architecture']}, "
          f"{best_config['Num_Layers']} layers, {best_config['Max_Iterations']} iterations")
    
    # Filter data for the best configuration
    best_df = normal_df[
        (normal_df['Database'] == best_config['Database']) & 
        (normal_df['Architecture'] == best_config['Architecture']) & 
        (normal_df['Num_Layers'] == best_config['Num_Layers']) & 
        (normal_df['Max_Iterations'] == best_config['Max_Iterations'])
    ]
    
    # Prepare the data for the table
    selected_df = pd.DataFrame({
        'Date': best_df['New_Date'],
        'Time': best_df['Processing_Time_Seconds'],
        'Expected Profit': best_df['Optimized_Profit'],
        'SI Penalty': best_df['SI_Penalty'],
        'Vol Penalty': best_df['Volume_Penalty'],
        'Op Cost': best_df['Operating_Cost'],
        'Ex-post Profit': best_df['Simulated_Profit']
    })
    
    # Calculate mean and std
    mean_row = selected_df.mean(numeric_only=True)
    std_row = selected_df.std(numeric_only=True)
    
    # Fix the LaTeX table generation to escape dollar signs
    db = best_config['Database']
    arch = best_config['Architecture']
    layers = int(best_config['Num_Layers'])
    iterations = int(best_config['Max_Iterations'])
    
    # Generate LaTeX table - escaping the dollar signs
    latex_header = r"""\begin{table}[h]
\centering
\caption{Performance of Best Model Configuration (""" + f"${db}$, ${arch}$, ${layers}$ layers, ${iterations}$ iterations" + r""") }
\label{tab:best_model_performance}
\begin{tabular}{lrrrrrr}
\toprule
Date & Time (s) & Expected Profit & SI Penalty & Vol Penalty & Op Cost & Ex-post Profit \\
\midrule
"""
    
    latex_rows = ""
    for _, row in selected_df.iterrows():
        latex_rows += f"{row['Date']} & {row['Time']:.2f} & {row['Expected Profit']:.2f} & {row['SI Penalty']:.2f} & {row['Vol Penalty']:.2f} & {row['Op Cost']:.2f} & {row['Ex-post Profit']:.2f} \\\\\n"
    
    # Add mean and std rows
    latex_rows += r"\midrule" + "\n"
    latex_rows += f"Mean & {mean_row['Time']:.2f} & {mean_row['Expected Profit']:.2f} & {mean_row['SI Penalty']:.2f} & {mean_row['Vol Penalty']:.2f} & {mean_row['Op Cost']:.2f} & {mean_row['Ex-post Profit']:.2f} \\\\\n"
    latex_rows += f"Std & {std_row['Time']:.2f} & {std_row['Expected Profit']:.2f} & {std_row['SI Penalty']:.2f} & {std_row['Vol Penalty']:.2f} & {std_row['Op Cost']:.2f} & {std_row['Ex-post Profit']:.2f} \\\\\n"
    
    latex_footer = r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    latex_table = latex_header + latex_rows + latex_footer
    
    # Save the LaTeX table to a file
    output_dir = Path("./custom_validation_results/analysis_without_extreme")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / 'best_model_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_dir / 'best_model_table.tex'}")
    
    return latex_table

#%% Analysis B: Extreme date analysis
def analyze_extreme_date(extreme_df, normal_df):
    """
    Analyze the performance on the extreme date 2024-12-12.
    """
    print("\n--- Analyzing Extreme Date (2024-12-12) ---")
    
    if extreme_df.empty:
        print("No data available for extreme date 2024-12-12")
        return
    
    # Create output directory for extreme date analysis
    output_dir = Path("./custom_validation_results/analysis_extreme_date")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Compare simulated profit across configurations
    plt.figure(figsize=(15, 10))
    extreme_df['Config'] = extreme_df.apply(
        lambda x: f"{x['Database']}-{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", 
        axis=1
    )
    
    # Sort by simulated profit
    sorted_df = extreme_df.sort_values('Simulated_Profit', ascending=False)
    
    plt.barh(sorted_df['Config'], sorted_df['Simulated_Profit'])
    plt.title('Model Performance on Extreme Date (2024-12-12)')
    plt.xlabel('Simulated Profit')
    plt.ylabel('Configuration')
    plt.tight_layout()
    plt.savefig(output_dir / 'extreme_date_performance.png', dpi=300)
    plt.close()
    
    # 2. Find the best performing model for the extreme date
    best_config = sorted_df.iloc[0]
    print(f"Best configuration for extreme date: {best_config['Config']}")
    print(f"Simulated Profit: {best_config['Simulated_Profit']:.2f}")
    print(f"Expected Profit: {best_config['Optimized_Profit']:.2f}")
    print(f"SI Penalty: {best_config['SI_Penalty']:.2f}")
    print(f"Volume Penalty: {best_config['Volume_Penalty']:.2f}")
    print(f"Operating Cost: {best_config['Operating_Cost']:.2f}")
    
    # 3. Compare performance metric distributions for extreme date vs normal dates
    metrics = ['Optimized_Profit', 'Simulated_Profit', 'SI_Penalty', 'Volume_Penalty', 'Operating_Cost']
    
    for metric in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get data for plotting
        extreme_data = extreme_df[metric]
        normal_data = normal_df[metric]
        
        # Histogram in first subplot
        ax1.hist([normal_data, extreme_data], bins=10, alpha=0.7, 
                 label=['Normal Dates', 'Extreme Date'])
        ax1.set_title(f'Histogram of {metric} - Normal vs Extreme Date')
        ax1.set_xlabel(metric)
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Density plot in second subplot
        sns.kdeplot(normal_data, ax=ax2, label='Normal Dates', fill=True, alpha=0.3)
        sns.kdeplot(extreme_data, ax=ax2, label='Extreme Date', fill=True, alpha=0.3)
        ax2.set_title(f'Density Plot of {metric} - Normal vs Extreme Date')
        ax2.set_xlabel(metric)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f'extreme_vs_normal_{metric}.png', dpi=300)
        plt.close()
    
    # 4. Generate a detailed report for the extreme date
    with open(output_dir / 'extreme_date_analysis.txt', 'w') as f:
        f.write("Analysis of Extreme Date (2024-12-12)\n")
        f.write("===================================\n\n")
        
        f.write("Performance across all configurations:\n")
        f.write(f"Number of configurations tested: {len(extreme_df)}\n")
        f.write(f"Average Simulated Profit: {extreme_df['Simulated_Profit'].mean():.2f}\n")
        f.write(f"Std Dev Simulated Profit: {extreme_df['Simulated_Profit'].std():.2f}\n")
        f.write(f"Min Simulated Profit: {extreme_df['Simulated_Profit'].min():.2f}\n")
        f.write(f"Max Simulated Profit: {extreme_df['Simulated_Profit'].max():.2f}\n\n")
        
        f.write("Best performing configuration:\n")
        f.write(f"Database: {best_config['Database']}\n")
        f.write(f"Architecture: {best_config['Architecture']}\n")
        f.write(f"Number of Layers: {best_config['Num_Layers']}\n")
        f.write(f"Max Iterations: {best_config['Max_Iterations']}\n")
        f.write(f"Simulated Profit: {best_config['Simulated_Profit']:.2f}\n")
        f.write(f"Expected Profit: {best_config['Optimized_Profit']:.2f}\n")
        f.write(f"SI Penalty: {best_config['SI_Penalty']:.2f}\n")
        f.write(f"Volume Penalty: {best_config['Volume_Penalty']:.2f}\n")
        f.write(f"Operating Cost: {best_config['Operating_Cost']:.2f}\n\n")
        
        # Compare with the same configuration's performance on normal dates
        normal_perf = normal_df[normal_df['Config'] == best_config['Config']]
        if not normal_perf.empty:
            f.write("Performance of this configuration on normal dates:\n")
            f.write(f"Average Simulated Profit: {normal_perf['Simulated_Profit'].mean():.2f}\n")
            f.write(f"Std Dev Simulated Profit: {normal_perf['Simulated_Profit'].std():.2f}\n")
            f.write(f"Performance difference: {best_config['Simulated_Profit'] - normal_perf['Simulated_Profit'].mean():.2f}\n")
    
    print(f"Extreme date analysis saved to {output_dir}")
    
    return extreme_df

#%% Model Comparison Analysis
def load_model_comparison_data(normal_df, extreme_df):
    """
    Load data from all four different models for comparison.
    Filter out the extreme date (2024-12-12) for normal analysis.
    """
    print("\n--- Loading Data for Model Comparison ---")
    
    # Create output directory
    output_dir = Path("./custom_validation_results/model_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Load Global Linear model data
    global_linear_path = Path("./Benchmark/global_linearized_operational_data_2024.csv")
    if global_linear_path.exists():
        global_linear_df = pd.read_csv(global_linear_path)
        global_linear_df = pd.DataFrame({
            'Date': global_linear_df['Date'],
            'Model': 'Global Linear',
            'Time': global_linear_df['SolveTime'],
            'Expected Profit': global_linear_df['ExpectedProfit'],
            'SI Penalty': global_linear_df['SIPenalty'],
            'Vol Penalty': global_linear_df['VolumePenalty'],
            'Op Cost': global_linear_df['OperatingCost'],
            'Ex-post Profit': global_linear_df['SimProfit']
        })
        global_linear_df['Date'] = pd.to_datetime(global_linear_df['Date'])
        # Filter out extreme date
        global_linear_extreme_df = global_linear_df[global_linear_df['Date'].dt.strftime('%Y-%m-%d') == '2024-12-12'].copy()
        global_linear_df = global_linear_df[global_linear_df['Date'].dt.strftime('%Y-%m-%d') != '2024-12-12'].copy()
        print(f"Loaded {len(global_linear_df)} records from Global Linear model (excluding extreme date)")
        print(f"Found {len(global_linear_extreme_df)} extreme date records for Global Linear model")
    else:
        print(f"Warning: Global Linear data file not found at {global_linear_path}")
        global_linear_df = pd.DataFrame(columns=[
            'Date', 'Model', 'Time', 'Expected Profit', 'SI Penalty', 
            'Vol Penalty', 'Op Cost', 'Ex-post Profit'
        ])
        global_linear_extreme_df = global_linear_df.copy()
    
    # 2. Load Piecewise-linear SOS2 data
    sos2_path = Path("./Benchmark/piecewise_operation_data_SOS2_2024_10seg_bm.csv")
    if sos2_path.exists():
        sos2_df = pd.read_csv(sos2_path)
        sos2_df = pd.DataFrame({
            'Date': sos2_df['Date'],
            'Model': 'Piecewise SOS2',
            'Time': sos2_df['SolveTime'],
            'Expected Profit': sos2_df['ExpectedProfit'],
            'SI Penalty': sos2_df['SIPenalty'],
            'Vol Penalty': sos2_df['VolumePenalty'],
            'Op Cost': sos2_df['OperatingCost'],
            'Ex-post Profit': sos2_df['SimProfit']
        })
        sos2_df['Date'] = pd.to_datetime(sos2_df['Date'])
        # Filter out extreme date
        sos2_extreme_df = sos2_df[sos2_df['Date'].dt.strftime('%Y-%m-%d') == '2024-12-12'].copy()
        sos2_df = sos2_df[sos2_df['Date'].dt.strftime('%Y-%m-%d') != '2024-12-12'].copy()
        print(f"Loaded {len(sos2_df)} records from Piecewise SOS2 model (excluding extreme date)")
        print(f"Found {len(sos2_extreme_df)} extreme date records for Piecewise SOS2 model")
    else:
        print(f"Warning: SOS2 data file not found at {sos2_path}")
        sos2_df = pd.DataFrame(columns=[
            'Date', 'Model', 'Time', 'Expected Profit', 'SI Penalty', 
            'Vol Penalty', 'Op Cost', 'Ex-post Profit'
        ])
        sos2_extreme_df = sos2_df.copy()
    
    # 3. Load NN-informed MPC data
    nn_mpc_path = Path("./Benchmark/NN-informed-MPC_2024.csv")
    if nn_mpc_path.exists():
        nn_mpc_df = pd.read_csv(nn_mpc_path)
        nn_mpc_df = pd.DataFrame({
            'Date': nn_mpc_df['date'],
            'Model': 'NN-MPC',
            'Time': nn_mpc_df['solution_time_opt'],
            'Expected Profit': nn_mpc_df['objective'],
            'SI Penalty': nn_mpc_df['si_penalty'],
            'Vol Penalty': nn_mpc_df['vol_penalty'],
            'Op Cost': nn_mpc_df['op_cost'],
            'Ex-post Profit': nn_mpc_df['profit']
        })
        nn_mpc_df['Date'] = pd.to_datetime(nn_mpc_df['Date'])
        # Filter out extreme date
        nn_mpc_extreme_df = nn_mpc_df[nn_mpc_df['Date'].dt.strftime('%Y-%m-%d') == '2024-12-12'].copy()
        nn_mpc_df = nn_mpc_df[nn_mpc_df['Date'].dt.strftime('%Y-%m-%d') != '2024-12-12'].copy()
        print(f"Loaded {len(nn_mpc_df)} records from NN-MPC model (excluding extreme date)")
        print(f"Found {len(nn_mpc_extreme_df)} extreme date records for NN-MPC model")
    else:
        print(f"Warning: NN-MPC data file not found at {nn_mpc_path}")
        nn_mpc_df = pd.DataFrame(columns=[
            'Date', 'Model', 'Time', 'Expected Profit', 'SI Penalty', 
            'Vol Penalty', 'Op Cost', 'Ex-post Profit'
        ])
        nn_mpc_extreme_df = nn_mpc_df.copy()
    
    # 4. Load best Recursive DFL model data - correctly handling normal and extreme dates
    # First find the best model configuration based on normal data
    best_dfl_df = get_best_dfl_model_data(normal_df, False)  # Normal dates
    best_dfl_extreme_df = get_best_dfl_model_data(extreme_df, True)  # Extreme date
    
    print(f"Using {len(best_dfl_df)} records from best Recursive DFL model (excluding extreme date)")
    print(f"Found {len(best_dfl_extreme_df)} extreme date records for best Recursive DFL model")
    
    # Combine all datasets for normal dates
    all_models_df = pd.concat([global_linear_df, sos2_df, nn_mpc_df, best_dfl_df])
    all_models_df.sort_values('Date', inplace=True)
    
    # Combine all datasets for extreme date
    all_models_extreme_df = pd.concat([global_linear_extreme_df, sos2_extreme_df, nn_mpc_extreme_df, best_dfl_extreme_df])
    if not all_models_extreme_df.empty:
        all_models_extreme_df.sort_values('Date', inplace=True)
    
    # Save the combined data for reference
    all_models_df.to_csv(output_dir / 'all_models_comparison_data.csv', index=False)
    if not all_models_extreme_df.empty:
        all_models_extreme_df.to_csv(output_dir / 'all_models_extreme_date_data.csv', index=False)
    
    return all_models_df, global_linear_df, sos2_df, nn_mpc_df, best_dfl_df, all_models_extreme_df

def get_best_dfl_model_data(df, is_extreme_date=False):
    """
    Extract data for the best DFL model configuration from the provided DataFrame.
    
    Parameters:
    df (DataFrame): Either normal_df or extreme_df depending on the is_extreme_date flag
    is_extreme_date (bool): Flag to indicate if we're processing extreme date data
    
    Returns:
    DataFrame: Formatted data for the best DFL model
    """
    if df.empty:
        return pd.DataFrame(columns=[
            'Date', 'Model', 'Time', 'Expected Profit', 'SI Penalty', 
            'Vol Penalty', 'Op Cost', 'Ex-post Profit'
        ])
    
    # Add Config column if it doesn't exist
    if 'Config' not in df.columns:
        df['Config'] = df.apply(
            lambda x: f"{x['Database']}-{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", 
            axis=1
        )
    
    # Find the configuration with the highest average simulated profit
    # For normal dates, group by configuration and find the best
    # For extreme date, we want the best for that specific date
    if not is_extreme_date:
        avg_by_config = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
            'Simulated_Profit'
        ].mean().reset_index()
        
        best_config = avg_by_config.loc[avg_by_config['Simulated_Profit'].idxmax()]
        
        # Filter data for the best configuration
        best_df = df[
            (df['Database'] == best_config['Database']) & 
            (df['Architecture'] == best_config['Architecture']) & 
            (df['Num_Layers'] == best_config['Num_Layers']) & 
            (df['Max_Iterations'] == best_config['Max_Iterations'])
        ]
    else:
        # For extreme date, get the best performing model directly
        best_df = df.loc[df['Simulated_Profit'].idxmax():df['Simulated_Profit'].idxmax()]
    
    # Get configuration details for model name
    db = best_df['Database'].iloc[0]
    arch = best_df['Architecture'].iloc[0]
    
    # Create a DataFrame in the same format as the other models
    best_model_name = f"DFL-{db}-{arch}"
    best_dfl_df = pd.DataFrame({
        'Date': best_df['New_Date'],
        'Model': best_model_name,
        'Time': best_df['Processing_Time_Seconds'],
        'Expected Profit': best_df['Optimized_Profit'],
        'SI Penalty': best_df['SI_Penalty'],
        'Vol Penalty': best_df['Volume_Penalty'],
        'Op Cost': best_df['Operating_Cost'],
        'Ex-post Profit': best_df['Simulated_Profit']
    })
    
    best_dfl_df['Date'] = pd.to_datetime(best_dfl_df['Date'])
    
    return best_dfl_df

def generate_model_comparison_plots(all_models_df):
    """
    Generate comparative plots for all models, excluding the extreme date.
    """
    print("\n--- Generating Model Comparison Plots (Excluding Extreme Date) ---")
    
    # Create output directory
    output_dir = Path("./custom_validation_results/model_comparison/plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define metrics to compare
    metrics = ['Ex-post Profit', 'Expected Profit', 'SI Penalty', 'Vol Penalty', 'Op Cost', 'Time']
    models = all_models_df['Model'].unique()
    
    # Set up figure aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. DENSITY PLOTS for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        for i, model in enumerate(models):
            model_data = all_models_df[all_models_df['Model'] == model][metric]
            mean_val = model_data.mean()
            std_val = model_data.std()
            sns.kdeplot(model_data, label=f'{model} (Mean: {mean_val:.2f}, Std: {std_val:.2f})', 
                       fill=True, alpha=0.3, color=colors[i % len(colors)])
        
        plt.title(f'Density Plot of {metric} by Model (Excluding Extreme Date)')
        plt.xlabel(metric)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'density_{metric.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()
    
    # 2. Boxplots for each metric
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x='Model', y=metric, data=all_models_df, palette=colors)
        
        # Add individual points for more detail
        sns.swarmplot(x='Model', y=metric, data=all_models_df, color='black', alpha=0.5, size=4)
        
        # Add mean values as text on top of each box
        means = all_models_df.groupby('Model')[metric].mean()
        for j, model in enumerate(ax.get_xticklabels()):
            model_name = model.get_text()
            if model_name in means.index:
                mean_val = means[model_name]
                ax.text(j, mean_val, f'Mean: {mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Comparison of {metric} Across Models (Excluding Extreme Date)')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'boxplot_{metric.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()
    
    # 3. Time vs. Profit scatterplot
    plt.figure(figsize=(12, 8))
    for i, model in enumerate(models):
        model_data = all_models_df[all_models_df['Model'] == model]
        plt.scatter(
            model_data['Time'], 
            model_data['Ex-post Profit'], 
            label=model, 
            color=colors[i % len(colors)],
            alpha=0.7,
            s=100
        )
    
    plt.title('Computational Time vs. Ex-post Profit (Excluding Extreme Date)')
    plt.xlabel('Computational Time (seconds)')
    plt.ylabel('Ex-post Profit')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'time_vs_profit_scatter.png', dpi=300)
    plt.close()
    
    # 4. Model performance by date
    # First, ensure we have matching dates across models
    common_dates = set(all_models_df['Date'].dt.date)
    for model in models:
        model_dates = set(all_models_df[all_models_df['Model'] == model]['Date'].dt.date)
        common_dates = common_dates.intersection(model_dates)
    
    common_dates = sorted(list(common_dates))
    
    if common_dates:
        common_dates_df = all_models_df[all_models_df['Date'].dt.date.isin(common_dates)]
        
        plt.figure(figsize=(14, 8))
        for i, model in enumerate(models):
            model_data = common_dates_df[common_dates_df['Model'] == model]
            model_data = model_data.sort_values('Date')
            plt.plot(
                model_data['Date'], 
                model_data['Ex-post Profit'], 
                'o-',
                label=model, 
                color=colors[i % len(colors)],
                linewidth=2,
                markersize=8
            )
        
        plt.title('Ex-post Profit by Date (Excluding Extreme Date)')
        plt.xlabel('Date')
        plt.ylabel('Ex-post Profit')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'profit_by_date.png', dpi=300)
        plt.close()
    else:
        print("Warning: No common dates found across all models for time series plot")
    
    # 5. Radar chart to compare mean performance across multiple metrics
    # Compute mean values for each model and metric
    metrics_for_radar = ['Ex-post Profit', 'Expected Profit', 'Time']
    model_means = all_models_df.groupby('Model')[metrics_for_radar].mean()
    
    # Normalize the metrics to the same scale (0-1)
    normalized_means = pd.DataFrame(index=model_means.index, columns=model_means.columns)
    for metric in metrics_for_radar:
        if metric == 'Time':
            # For Time, lower is better, so invert the normalization
            min_val = model_means[metric].min()
            max_val = model_means[metric].max()
            normalized_means[metric] = 1 - ((model_means[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0)
        else:
            # For other metrics, higher is better
            min_val = model_means[metric].min()
            max_val = model_means[metric].max()
            normalized_means[metric] = (model_means[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0
    
    # Create radar chart
    plt.figure(figsize=(10, 10))
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2*np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create axis
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics_for_radar, size=12)
    
    # Draw the radar chart for each model
    for i, model in enumerate(normalized_means.index):
        values = normalized_means.loc[model].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance Comparison (Excluding Extreme Date)', size=15)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_radar_chart.png', dpi=300)
    plt.close()
    
    # 6. Bar chart showing the mean profit improvement compared to the Global Linear baseline
    if 'Global Linear' in model_means.index:
        baseline_profit = model_means.loc['Global Linear', 'Ex-post Profit']
        profit_improvement = ((model_means['Ex-post Profit'] - baseline_profit) / baseline_profit) * 100
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(profit_improvement.index, profit_improvement.values, color=colors)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height if height > 0 else -5,
                f'{height:.2f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold'
            )
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Percentage Improvement in Ex-post Profit Compared to Global Linear Model\n(Excluding Extreme Date)')
        plt.ylabel('Improvement (%)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'profit_improvement_bar.png', dpi=300)
        plt.close()
    
    # Return to default style
    plt.style.use('default')
    
    print(f"Model comparison plots saved to {output_dir}")

def analyze_extreme_date_model_comparison(all_models_extreme_df):
    """
    Analyze and compare model performance on the extreme date 2024-12-12.
    """
    print("\n--- Analyzing Model Comparison for Extreme Date (2024-12-12) ---")
    
    if all_models_extreme_df.empty:
        print("No data available for any model on extreme date 2024-12-12")
        return
    
    # Create output directory for extreme date analysis
    output_dir = Path("./custom_validation_results/model_comparison/extreme_date")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define metrics to compare
    metrics = ['Ex-post Profit', 'Expected Profit', 'SI Penalty', 'Vol Penalty', 'Op Cost', 'Time']
    models = all_models_extreme_df['Model'].unique()
    
    # Set up colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. DENSITY PLOTS for each metric on extreme date
    # Since extreme date usually has just one record per model, we'll compare with normal dates
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        for i, model in enumerate(models):
            model_data = all_models_extreme_df[all_models_extreme_df['Model'] == model][metric]
            if not model_data.empty:
                value = model_data.iloc[0]
                plt.axvline(x=value, color=colors[i % len(colors)], 
                           linestyle='--', label=f'{model} Extreme: {value:.2f}')
        
        plt.title(f'Comparison of {metric} on Extreme Date (2024-12-12)')
        plt.xlabel(metric)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'extreme_date_comparison_{metric.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()
    
    # 2. Compare performance across models for the extreme date
    plt.figure(figsize=(12, 8))
    
    # Sort models by ex-post profit
    sorted_df = all_models_extreme_df.sort_values('Ex-post Profit', ascending=False)
    
    # Create a bar chart comparing ex-post profit
    bars = plt.bar(sorted_df['Model'], sorted_df['Ex-post Profit'], color=colors[:len(sorted_df)])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            f'{height:.2f}',
            ha='center', va='bottom',
            fontweight='bold'
        )
    
    plt.title('Model Performance Comparison on Extreme Date (2024-12-12)')
    plt.ylabel('Ex-post Profit')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'extreme_date_model_comparison.png', dpi=300)
    plt.close()
    
    # 3. Create detailed comparison table for all metrics
    latex_table = r"""\begin{table}[h]
\centering
\caption{Model Performance Comparison on Extreme Date (2024-12-12)}
\label{tab:extreme_date_model_comparison}
\begin{tabular}{lcccccc}
\toprule
Model & Time (s) & Expected Profit & SI Penalty & Vol Penalty & Op Cost & Ex-post Profit \\
\midrule
"""
    
    # Add rows for each model, sorted by Ex-post Profit
    for _, row in sorted_df.iterrows():
        model = row['Model']
        time = row['Time']
        exp_profit = row['Expected Profit']
        si_penalty = row['SI Penalty']
        vol_penalty = row['Vol Penalty']
        op_cost = row['Op Cost']
        expost_profit = row['Ex-post Profit']
        
        latex_table += f"{model} & {time:.2f} & {exp_profit:.2f} & {si_penalty:.2f} & {vol_penalty:.2f} & {op_cost:.2f} & {expost_profit:.2f} \\\\\n"
    
    # Close the table
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save the LaTeX table
    with open(output_dir / 'extreme_date_model_comparison.tex', 'w') as f:
        f.write(latex_table)
    
    # 4. Create a comprehensive comparison figure with multiple metrics
    metrics_to_plot = ['Ex-post Profit', 'Expected Profit', 'SI Penalty', 'Vol Penalty', 'Time']
    
    plt.figure(figsize=(16, 12))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(3, 2, i+1)
        
        # Sort models by the current metric (ascending for penalties and time, descending for profits)
        if metric in ['SI Penalty', 'Vol Penalty', 'Time']:
            sort_ascending = True
        else:
            sort_ascending = False
        
        sorted_metric_df = all_models_extreme_df.sort_values(metric, ascending=sort_ascending)
        
        # Create bar chart
        bars = plt.bar(sorted_metric_df['Model'], sorted_metric_df[metric], 
                      color=[colors[models.tolist().index(m) % len(colors)] for m in sorted_metric_df['Model']])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1 if height > 0 else -0.1,
                f'{height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=9
            )
        
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
    
    # Add a text summary at the bottom
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    # Find the best model for the extreme date
    best_model = sorted_df.iloc[0]['Model']
    best_profit = sorted_df.iloc[0]['Ex-post Profit']
    
    summary_text = (
        f"Summary for Extreme Date (2024-12-12):\n\n"
        f"Best Performing Model: {best_model}\n"
        f"Ex-post Profit: {best_profit:.2f}\n\n"
        f"Number of Models Available: {len(sorted_df)}"
    )
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'extreme_date_metrics_comparison.png', dpi=300)
    plt.close()
    
    print(f"Extreme date model comparison analysis saved to {output_dir}")
    
    return sorted_df

def generate_dfl_comparison_table(normal_df):
    """
    Generate a LaTeX table documenting the Mean and Std of each DFL setup,
    including only Time (s) and Expected Profit.
    """
    print("\n--- Generating DFL Configuration Comparison Table ---")
    
    # Create output directory
    output_dir = Path("./validation_results/model_comparison/tables")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Group by configuration parameters
    grouped_df = normal_df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
        ['Processing_Time_Seconds', 'Optimized_Profit']
    ].agg(['mean', 'std']).reset_index()
    
    # Sort by mean Optimized_Profit in descending order
    grouped_df = grouped_df.sort_values(('Optimized_Profit', 'mean'), ascending=False)
    
    # Start building the LaTeX table
    latex_table = r"""\begin{table}[h]
\centering
\caption{Comparison of Different DFL Configurations (Excluding Extreme Date)}
\label{tab:dfl_config_comparison}
\begin{tabular}{llccccc}
\toprule
Database & Architecture & Layers & Iterations & \multicolumn{2}{c}{Time (s)} & \multicolumn{2}{c}{Expected Profit} \\
\cmidrule(lr){5-6} \cmidrule(lr){7-8}
 & & & & Mean & Std & Mean & Std \\
\midrule
"""
    
    # Add a row for each configuration
    for _, row in grouped_df.iterrows():
        db = row['Database']
        arch = row['Architecture']
        layers = int(row['Num_Layers'])
        iters = int(row['Max_Iterations'])
        
        time_mean = row[('Processing_Time_Seconds', 'mean')]
        time_std = row[('Processing_Time_Seconds', 'std')]
        profit_mean = row[('Optimized_Profit', 'mean')]
        profit_std = row[('Optimized_Profit', 'std')]
        
        latex_table += f"{db} & {arch} & {layers} & {iters} & {time_mean:.2f} & {time_std:.2f} & {profit_mean:.2f} & {profit_std:.2f} \\\\\n"
    
    # Close the table
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save the LaTeX table
    with open(output_dir / 'dfl_configuration_comparison.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"DFL configuration comparison table saved to {output_dir / 'dfl_configuration_comparison.tex'}")
    
    return latex_table
#%% Update main function to include model comparison analysis
def main():
    """
    Main function to run the analysis.
    """
    print("Starting analysis of validation results...")
    
    # Load and prepare data
    normal_df, extreme_df = prepare_data()
    
    # Add Config column for easier reference
    normal_df['Config'] = normal_df.apply(
        lambda x: f"{x['Database']}-{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", 
        axis=1
    )
    
    if not extreme_df.empty:
        extreme_df['Config'] = extreme_df.apply(
            lambda x: f"{x['Database']}-{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", 
            axis=1
        )
    
    # Analysis A: Distribution plots for normal dates
    normal_df = analyze_normal_dates(normal_df)
    
    # Generate LaTeX table for best model
    latex_table = generate_latex_table(normal_df)
    
    # Generate LaTeX summary table for model settings
    summary_latex_table = generate_latex_summary_table(normal_df)
    
    # Generate LaTeX table for DFL configurations (new)
    dfl_latex_table = generate_dfl_comparison_table(normal_df)

    # Analysis B: Extreme date analysis for DFL model
    if not extreme_df.empty:
        extreme_df = analyze_extreme_date(extreme_df, normal_df)
    
    # Analysis C: Model Comparison (new) - excluding extreme date
    # Pass normal_df and extreme_df to properly handle DFL data
    all_models_df, global_linear_df, sos2_df, nn_mpc_df, best_dfl_df, all_models_extreme_df = load_model_comparison_data(normal_df, extreme_df)
    
    # Generate comparison plots (excluding extreme date)
    generate_model_comparison_plots(all_models_df)
    
    # # Generate comparison tables (excluding extreme date)
    # generate_model_comparison_tables(all_models_df, global_linear_df, sos2_df, nn_mpc_df, best_dfl_df)
    
    # Analysis D: Extreme date model comparison analysis
    if not all_models_extreme_df.empty:
        analyze_extreme_date_model_comparison(all_models_extreme_df)
    
    print("\nAnalysis complete!")

#%% Run the script
if __name__ == "__main__":
    main()
    
# %%
