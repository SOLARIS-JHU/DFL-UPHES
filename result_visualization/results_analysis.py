#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

#%% Database file paths and configurations
DATABASE_PATHS = {
    'MIQP_Linear': r'..\MIQP\MIQP_linear\MILP_global_linear_benchmark.csv',
    'MIQP_Piecewise': r'..\MIQP\MIQP_piecewise\MIQP_piecewise_benchmark.csv', 
    'MIQP_NN': r'..\MIQP\MIQP_nn\MIQP_nn_benchmark.csv',
    'DFL_Bounded': r'..\DFL_bounded\validation_results\comprehensive\master_validation_benchmarks.csv',
    'DFL_Unbounded': r'..\DFL_unbounded\validation_results\comprehensive\master_validation_benchmarks.csv'
}

# Extreme date to exclude (same as original script)
EXTREME_DATE = '2024-12-12'

#%% Data loading and harmonization functions
def standardize_date_format(date_str):
    """Convert various date formats to YYYY-MM-DD format."""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Handle different date formats
    if '/' in date_str:
        # Convert YYYY/MM/DD or MM/DD/YYYY to YYYY-MM-DD
        parts = date_str.split('/')
        if len(parts) == 3:
            if len(parts[0]) == 4:  # YYYY/MM/DD
                return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            else:  # MM/DD/YYYY
                return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
    
    return date_str

def load_miqp_data(file_path, method_name):
    """Load MIQP benchmark data with standardized column names."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Standardize column names for MIQP data
    column_mapping = {
        'Date': 'Date',
        'Expected Profit (€)': 'Expected_Profit',
        'Ex-post Profit (€)': 'Ex_post_Profit', 
        'SI Penalty (€)': 'SI_Penalty',
        'Vol Penalty (€)': 'Volume_Penalty',
        'Op Cost (€)': 'Operating_Cost',
        'Solving Time (s)': 'Processing_Time_Seconds',
        'MIP Gap': 'MIP_Gap',
        'Binary Variables': 'Binary_Variables',
        'Continuous Variables': 'Continuous_Variables',
        'Total Constraints': 'Total_Constraints'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Add method identifier
    df['Method'] = method_name
    df['Database'] = method_name  # For consistency with DFL format
    
    # Standardize date format
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(standardize_date_format)
    
    return df

def load_dfl_data(file_path, method_name):
    """Load DFL benchmark data - find best config overall and use across all dates."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Standardize column names for DFL data
    column_mapping = {
        'New_Date': 'Date',
        'Expected_Profit': 'Expected_Profit',
        'Ex_post_Profit': 'Ex_post_Profit',
        'SI_Penalty': 'SI_Penalty',
        'Volume_Penalty': 'Volume_Penalty',
        'Operating_Cost': 'Operating_Cost',
        'Processing_Time_Seconds': 'Processing_Time_Seconds'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Standardize date format
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(standardize_date_format)
    
    # For DFL data, find the best performing configuration overall (across all dates)
    # Then use only that configuration's performance across all dates
    if 'Database' in df.columns and 'Architecture' in df.columns:
        # Create configuration identifier
        df['Config'] = df.apply(
            lambda x: f"{x.get('Database', '')}-{x.get('Architecture', '')}-{x.get('Num_Layers', '')}-{x.get('Max_Iterations', '')}", 
            axis=1
        )
        
        # Find the configuration with highest average Ex_post_Profit across all dates
        config_means = df.groupby('Config')['Ex_post_Profit'].mean()
        best_config = config_means.idxmax()
        
        print(f"Best {method_name} configuration: {best_config} (avg profit: {config_means.max():.2f})")
        
        # Keep only the best configuration's results across all dates
        df = df[df['Config'] == best_config].copy()
    
    # Add method identifier
    df['Method'] = method_name
    
    return df

def load_dfl_data_detailed(file_path, method_name):
    """Load DFL benchmark data keeping ALL configurations for detailed analysis."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Standardize column names for DFL data
    column_mapping = {
        'New_Date': 'Date',
        'Expected_Profit': 'Expected_Profit',
        'Ex_post_Profit': 'Ex_post_Profit',
        'SI_Penalty': 'SI_Penalty',
        'Volume_Penalty': 'Volume_Penalty',
        'Operating_Cost': 'Operating_Cost',
        'Processing_Time_Seconds': 'Processing_Time_Seconds'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Standardize date format
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(standardize_date_format)
    
    # Keep ALL configurations for detailed analysis
    if 'Database' in df.columns and 'Architecture' in df.columns:
        # Create configuration identifier
        df['Config'] = df.apply(
            lambda x: f"{x.get('Database', '')}-{x.get('Architecture', '')}-{x.get('Num_Layers', '')}-{x.get('Max_Iterations', '')}", 
            axis=1
        )
    
    # Add method identifier
    df['Method'] = method_name
    
    return df

def detailed_dfl_analysis(output_dir):
    """Perform detailed analysis of DFL hyperparameters."""
    print("\n" + "="*80)
    print("DETAILED DFL HYPERPARAMETER ANALYSIS")
    print("="*80)
    
    # Load detailed DFL data (all configurations)
    dfl_data = []
    dfl_methods = ['DFL_Bounded', 'DFL_Unbounded']
    
    for method in dfl_methods:
        if method in DATABASE_PATHS:
            df = load_dfl_data_detailed(DATABASE_PATHS[method], method)
            if not df.empty:
                dfl_data.append(df)
                print(f"Loaded {len(df)} detailed records from {method}")
    
    if not dfl_data:
        print("Error: No DFL data loaded for detailed analysis!")
        return
    
    # Combine DFL data
    dfl_df = pd.concat(dfl_data, ignore_index=True)
    
    # Filter out extreme date
    dfl_df = dfl_df[dfl_df['Date'] != EXTREME_DATE].copy()
    print(f"Total DFL records after filtering: {len(dfl_df)}")
    
    # Create DFL-specific output directory
    dfl_dir = output_dir / "dfl_detailed_analysis"
    dfl_plot_dir = dfl_dir / "plots"
    dfl_table_dir = dfl_dir / "tables"
    
    for dir_path in [dfl_dir, dfl_plot_dir, dfl_table_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate detailed analyses
    analyze_database_effect(dfl_df, dfl_plot_dir)
    analyze_architecture_effect(dfl_df, dfl_plot_dir)
    analyze_layers_effect(dfl_df, dfl_plot_dir)
    analyze_iterations_effect(dfl_df, dfl_plot_dir)
    analyze_hyperparameter_interactions(dfl_df, dfl_plot_dir)
    generate_dfl_latex_tables(dfl_df, dfl_table_dir)
    generate_dfl_summary(dfl_df, dfl_dir)
    
    print(f"\nDetailed DFL analysis completed! Results saved to: {dfl_dir}")

def analyze_database_effect(df, plot_dir):
    """Analyze the effect of different training databases."""
    print("\n--- Analyzing Training Database Effect ---")
    
    # 1. Database effect on Ex-post Profit
    plt.figure(figsize=(14, 8))
    
    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Database', y='Ex_post_Profit', hue='Method', data=df)
    plt.title('Ex-post Profit by Training Database')
    plt.ylabel('Ex-post Profit (€)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='DFL Method')
    plt.grid(axis='y', alpha=0.3)
    
    # Mean comparison with error bars
    plt.subplot(1, 2, 2)
    db_stats = df.groupby(['Database', 'Method']).agg({
        'Ex_post_Profit': ['mean', 'std']
    }).round(2)
    
    methods = df['Method'].unique()
    databases = df['Database'].unique()
    x = np.arange(len(databases))
    width = 0.35
    
    colors = ['#1f77b4', '#ff7f0e']
    for i, method in enumerate(methods):
        means = []
        stds = []
        for db in databases:
            if (db, method) in db_stats.index:
                means.append(db_stats.loc[(db, method), ('Ex_post_Profit', 'mean')])
                stds.append(db_stats.loc[(db, method), ('Ex_post_Profit', 'std')])
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar(x + i*width, means, width, yerr=stds, capsize=5,
               label=method, color=colors[i], alpha=0.8)
    
    plt.xlabel('Training Database')
    plt.ylabel('Mean Ex-post Profit (€)')
    plt.title('Mean Ex-post Profit by Training Database')
    plt.xticks(x + width/2, databases, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'database_effect_profit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Database effect on Processing Time
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Database', y='Processing_Time_Seconds', hue='Method', data=df)
    plt.title('Processing Time by Training Database')
    plt.ylabel('Processing Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='DFL Method')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    time_stats = df.groupby(['Database', 'Method']).agg({
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    for i, method in enumerate(methods):
        means = []
        stds = []
        for db in databases:
            if (db, method) in time_stats.index:
                means.append(time_stats.loc[(db, method), ('Processing_Time_Seconds', 'mean')])
                stds.append(time_stats.loc[(db, method), ('Processing_Time_Seconds', 'std')])
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar(x + i*width, means, width, yerr=stds, capsize=5,
               label=method, color=colors[i], alpha=0.8)
    
    plt.xlabel('Training Database')
    plt.ylabel('Mean Processing Time (seconds)')
    plt.title('Mean Processing Time by Training Database')
    plt.xticks(x + width/2, databases, rotation=45, ha='right')
    plt.legend()
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'database_effect_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_architecture_effect(df, plot_dir):
    """Analyze the effect of LSTM vs RNN architectures."""
    print("\n--- Analyzing Architecture Effect (LSTM vs RNN) ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Architecture effect on Ex-post Profit
    sns.boxplot(x='Architecture', y='Ex_post_Profit', hue='Method', data=df, ax=axes[0,0])
    axes[0,0].set_title('Ex-post Profit by Architecture')
    axes[0,0].set_ylabel('Ex-post Profit (€)')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # 2. Architecture effect on Processing Time
    sns.boxplot(x='Architecture', y='Processing_Time_Seconds', hue='Method', data=df, ax=axes[0,1])
    axes[0,1].set_title('Processing Time by Architecture')
    axes[0,1].set_ylabel('Processing Time (seconds)')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # 3. Architecture comparison by Method (Profit)
    arch_profit_stats = df.groupby(['Architecture', 'Method'])['Ex_post_Profit'].agg(['mean', 'std']).round(2)
    architectures = df['Architecture'].unique()
    methods = df['Method'].unique()
    x = np.arange(len(methods))
    width = 0.35
    
    for i, arch in enumerate(architectures):
        means = []
        stds = []
        for method in methods:
            if (arch, method) in arch_profit_stats.index:
                means.append(arch_profit_stats.loc[(arch, method), 'mean'])
                stds.append(arch_profit_stats.loc[(arch, method), 'std'])
            else:
                means.append(0)
                stds.append(0)
        
        axes[1,0].bar(x + i*width, means, width, yerr=stds, capsize=5,
                     label=arch, alpha=0.8)
    
    axes[1,0].set_xlabel('DFL Method')
    axes[1,0].set_ylabel('Mean Ex-post Profit (€)')
    axes[1,0].set_title('Mean Ex-post Profit: LSTM vs RNN')
    axes[1,0].set_xticks(x + width/2)
    axes[1,0].set_xticklabels(methods)
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 4. Architecture comparison by Method (Time)
    arch_time_stats = df.groupby(['Architecture', 'Method'])['Processing_Time_Seconds'].agg(['mean', 'std']).round(2)
    
    for i, arch in enumerate(architectures):
        means = []
        stds = []
        for method in methods:
            if (arch, method) in arch_time_stats.index:
                means.append(arch_time_stats.loc[(arch, method), 'mean'])
                stds.append(arch_time_stats.loc[(arch, method), 'std'])
            else:
                means.append(0)
                stds.append(0)
        
        axes[1,1].bar(x + i*width, means, width, yerr=stds, capsize=5,
                     label=arch, alpha=0.8)
    
    axes[1,1].set_xlabel('DFL Method')
    axes[1,1].set_ylabel('Mean Processing Time (seconds)')
    axes[1,1].set_title('Mean Processing Time: LSTM vs RNN')
    axes[1,1].set_xticks(x + width/2)
    axes[1,1].set_xticklabels(methods)
    axes[1,1].set_yscale('log')
    axes[1,1].legend()
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'architecture_effect.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_layers_effect(df, plot_dir):
    """Analyze the effect of number of layers."""
    print("\n--- Analyzing Number of Layers Effect ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Layers effect on Ex-post Profit
    sns.boxplot(x='Num_Layers', y='Ex_post_Profit', hue='Method', data=df, ax=axes[0,0])
    axes[0,0].set_title('Ex-post Profit by Number of Layers')
    axes[0,0].set_ylabel('Ex-post Profit (€)')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # 2. Layers effect on Processing Time
    sns.boxplot(x='Num_Layers', y='Processing_Time_Seconds', hue='Method', data=df, ax=axes[0,1])
    axes[0,1].set_title('Processing Time by Number of Layers')
    axes[0,1].set_ylabel('Processing Time (seconds)')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # 3. Mean profit by layers and architecture
    layers_arch_profit = df.groupby(['Num_Layers', 'Architecture'])['Ex_post_Profit'].mean().unstack()
    
    layers_arch_profit.plot(kind='bar', ax=axes[1,0], width=0.8)
    axes[1,0].set_title('Mean Ex-post Profit by Layers and Architecture')
    axes[1,0].set_xlabel('Number of Layers')
    axes[1,0].set_ylabel('Mean Ex-post Profit (€)')
    axes[1,0].legend(title='Architecture')
    axes[1,0].grid(axis='y', alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=0)
    
    # 4. Mean time by layers and architecture
    layers_arch_time = df.groupby(['Num_Layers', 'Architecture'])['Processing_Time_Seconds'].mean().unstack()
    
    layers_arch_time.plot(kind='bar', ax=axes[1,1], width=0.8)
    axes[1,1].set_title('Mean Processing Time by Layers and Architecture')
    axes[1,1].set_xlabel('Number of Layers')
    axes[1,1].set_ylabel('Mean Processing Time (seconds)')
    axes[1,1].set_yscale('log')
    axes[1,1].legend(title='Architecture')
    axes[1,1].grid(axis='y', alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'layers_effect.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_iterations_effect(df, plot_dir):
    """Analyze the effect of number of iterations."""
    print("\n--- Analyzing Number of Iterations Effect ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Iterations effect on Ex-post Profit
    sns.boxplot(x='Max_Iterations', y='Ex_post_Profit', data=df, ax=axes[0,0])
    axes[0,0].set_title('Ex-post Profit by Number of Iterations')
    axes[0,0].set_ylabel('Ex-post Profit (€)')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # 2. Iterations effect on Processing Time
    sns.boxplot(x='Max_Iterations', y='Processing_Time_Seconds', data=df, ax=axes[0,1])
    axes[0,1].set_title('Processing Time by Number of Iterations')
    axes[0,1].set_ylabel('Processing Time (seconds)')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # 3. Mean profit trend by iterations
    iter_stats = df.groupby('Max_Iterations').agg({
        'Ex_post_Profit': ['mean', 'std'],
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    iterations = sorted(df['Max_Iterations'].unique())
    profit_means = [iter_stats.loc[i, ('Ex_post_Profit', 'mean')] for i in iterations]
    profit_stds = [iter_stats.loc[i, ('Ex_post_Profit', 'std')] for i in iterations]
    
    axes[1,0].errorbar(iterations, profit_means, yerr=profit_stds, 
                      marker='o', capsize=5, capthick=2, linewidth=2)
    axes[1,0].set_title('Mean Ex-post Profit vs Number of Iterations')
    axes[1,0].set_xlabel('Number of Iterations')
    axes[1,0].set_ylabel('Mean Ex-post Profit (€)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Mean time trend by iterations
    time_means = [iter_stats.loc[i, ('Processing_Time_Seconds', 'mean')] for i in iterations]
    time_stds = [iter_stats.loc[i, ('Processing_Time_Seconds', 'std')] for i in iterations]
    
    axes[1,1].errorbar(iterations, time_means, yerr=time_stds, 
                      marker='s', capsize=5, capthick=2, linewidth=2, color='orange')
    axes[1,1].set_title('Mean Processing Time vs Number of Iterations')
    axes[1,1].set_xlabel('Number of Iterations')
    axes[1,1].set_ylabel('Mean Processing Time (seconds)')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'iterations_effect.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_hyperparameter_interactions(df, plot_dir):
    """Analyze interactions between different hyperparameters."""
    print("\n--- Analyzing Hyperparameter Interactions ---")
    
    try:
        import seaborn as sns
        
        # 1. Heatmap: Mean Ex-post Profit by Layers x Iterations
        plt.figure(figsize=(16, 6))
        
        plt.subplot(1, 2, 1)
        layers_iter_profit = df.groupby(['Num_Layers', 'Max_Iterations'])['Ex_post_Profit'].mean().unstack(fill_value=0)
        sns.heatmap(layers_iter_profit, annot=True, fmt='.1f', cmap='viridis', 
                   cbar_kws={'label': 'Mean Ex-post Profit (€)'})
        plt.title('Mean Ex-post Profit: Layers × Iterations')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Number of Layers')
        
        # 2. Heatmap: Mean Processing Time by Layers x Iterations
        plt.subplot(1, 2, 2)
        layers_iter_time = df.groupby(['Num_Layers', 'Max_Iterations'])['Processing_Time_Seconds'].mean().unstack(fill_value=0)
        sns.heatmap(layers_iter_time, annot=True, fmt='.1f', cmap='plasma', 
                   cbar_kws={'label': 'Mean Processing Time (s)'})
        plt.title('Mean Processing Time: Layers × Iterations')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Number of Layers')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'hyperparameter_interactions_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Database × Architecture interaction
        plt.figure(figsize=(14, 8))
        
        plt.subplot(1, 2, 1)
        db_arch_profit = df.groupby(['Database', 'Architecture'])['Ex_post_Profit'].mean().unstack()
        db_arch_profit.plot(kind='bar', ax=plt.gca(), width=0.8)
        plt.title('Mean Ex-post Profit: Database × Architecture')
        plt.xlabel('Training Database')
        plt.ylabel('Mean Ex-post Profit (€)')
        plt.legend(title='Architecture')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        db_arch_time = df.groupby(['Database', 'Architecture'])['Processing_Time_Seconds'].mean().unstack()
        db_arch_time.plot(kind='bar', ax=plt.gca(), width=0.8)
        plt.title('Mean Processing Time: Database × Architecture')
        plt.xlabel('Training Database')
        plt.ylabel('Mean Processing Time (seconds)')
        plt.yscale('log')
        plt.legend(title='Architecture')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'database_architecture_interaction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Seaborn not available for heatmaps, creating alternative visualizations...")
        
        # Alternative without seaborn
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Layers vs Iterations (Profit)
        for layer in df['Num_Layers'].unique():
            layer_data = df[df['Num_Layers'] == layer]
            iter_means = layer_data.groupby('Max_Iterations')['Ex_post_Profit'].mean()
            axes[0,0].plot(iter_means.index, iter_means.values, 'o-', label=f'{int(layer)} Layers', linewidth=2)
        
        axes[0,0].set_title('Ex-post Profit: Layers × Iterations Interaction')
        axes[0,0].set_xlabel('Number of Iterations')
        axes[0,0].set_ylabel('Mean Ex-post Profit (€)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Similar plots for other interactions...
        plt.tight_layout()
        plt.savefig(plot_dir / 'hyperparameter_interactions.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_dfl_latex_tables(df, table_dir):
    """Generate detailed LaTeX tables for DFL analysis."""
    print("\n--- Generating DFL LaTeX Tables ---")
    
    # 1. Best configuration table
    best_configs = df.groupby(['Method', 'Database', 'Architecture', 'Num_Layers', 'Max_Iterations']).agg({
        'Ex_post_Profit': ['mean', 'std', 'count'],
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    # Sort by mean ex-post profit
    best_configs = best_configs.sort_values(('Ex_post_Profit', 'mean'), ascending=False)
    
    latex_best_configs = r"""\begin{table}[h]
\centering
\caption{Top DFL Configurations by Mean Ex-post Profit}
\label{tab:dfl_best_configs}
\begin{tabular}{llcccccc}
\toprule
Method & Database & Arch & Layers & Iter & Ex-post Profit & Processing Time & Count \\
 & & & & & (€) & (s) & \\
\midrule
"""
    
    # Show top 10 configurations
    for i, ((method, database, arch, layers, iterations), row) in enumerate(best_configs.head(10).iterrows()):
        profit_mean = row[('Ex_post_Profit', 'mean')]
        profit_std = row[('Ex_post_Profit', 'std')]
        time_mean = row[('Processing_Time_Seconds', 'mean')]
        count = int(row[('Ex_post_Profit', 'count')])
        
        # Format for LaTeX
        method_short = method.replace('DFL_', '')
        db_short = database.split('_')[-1] if '_' in database else database
        
        latex_best_configs += f"{method_short} & {db_short} & {arch} & {int(layers)} & {int(iterations)} & "
        latex_best_configs += f"{profit_mean:.2f} $\\pm$ {profit_std:.2f} & {time_mean:.2f} & {count} \\\\\n"
    
    latex_best_configs += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(table_dir / 'dfl_best_configurations.tex', 'w') as f:
        f.write(latex_best_configs)
    
    # 2. Hyperparameter effect summary table
    latex_hyperparams = r"""\begin{table}[h]
\centering
\caption{DFL Hyperparameter Effects Summary}
\label{tab:dfl_hyperparameter_effects}
\begin{tabular}{lccc}
\toprule
Hyperparameter & Best Value & Mean Ex-post Profit & Improvement \\
 & & (€) & over Worst (\%) \\
\midrule
"""
    
    # Database effect
    db_means = df.groupby('Database')['Ex_post_Profit'].mean().sort_values(ascending=False)
    best_db = db_means.index[0]
    worst_db = db_means.index[-1]
    db_improvement = ((db_means.iloc[0] - db_means.iloc[-1]) / db_means.iloc[-1]) * 100
    
    latex_hyperparams += f"Training Database & {best_db.split('_')[-1]} & {db_means.iloc[0]:.2f} & {db_improvement:.1f} \\\\\n"
    
    # Architecture effect
    arch_means = df.groupby('Architecture')['Ex_post_Profit'].mean().sort_values(ascending=False)
    best_arch = arch_means.index[0]
    arch_improvement = ((arch_means.iloc[0] - arch_means.iloc[-1]) / arch_means.iloc[-1]) * 100
    
    latex_hyperparams += f"Architecture & {best_arch} & {arch_means.iloc[0]:.2f} & {arch_improvement:.1f} \\\\\n"
    
    # Layers effect
    layers_means = df.groupby('Num_Layers')['Ex_post_Profit'].mean().sort_values(ascending=False)
    best_layers = int(layers_means.index[0])
    layers_improvement = ((layers_means.iloc[0] - layers_means.iloc[-1]) / layers_means.iloc[-1]) * 100
    
    latex_hyperparams += f"Number of Layers & {best_layers} & {layers_means.iloc[0]:.2f} & {layers_improvement:.1f} \\\\\n"
    
    # Iterations effect
    iter_means = df.groupby('Max_Iterations')['Ex_post_Profit'].mean().sort_values(ascending=False)
    best_iter = int(iter_means.index[0])
    iter_improvement = ((iter_means.iloc[0] - iter_means.iloc[-1]) / iter_means.iloc[-1]) * 100
    
    latex_hyperparams += f"Max Iterations & {best_iter} & {iter_means.iloc[0]:.2f} & {iter_improvement:.1f} \\\\\n"
    
    latex_hyperparams += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(table_dir / 'dfl_hyperparameter_effects.tex', 'w') as f:
        f.write(latex_hyperparams)
    
    print(f"DFL LaTeX tables saved to {table_dir}")

def generate_dfl_summary(df, output_dir):
    """Generate comprehensive DFL summary."""
    print("\n--- Generating DFL Summary ---")
    
    with open(output_dir / 'dfl_detailed_summary.txt', 'w') as f:
        f.write("DETAILED DFL HYPERPARAMETER ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total DFL configurations analyzed: {len(df)}\n")
        f.write(f"DFL methods: {', '.join(df['Method'].unique())}\n")
        f.write(f"Training databases: {', '.join(df['Database'].unique())}\n")
        f.write(f"Architectures: {', '.join(df['Architecture'].unique())}\n")
        f.write(f"Layer configurations: {sorted(df['Num_Layers'].unique())}\n")
        f.write(f"Iteration ranges: {sorted(df['Max_Iterations'].unique())}\n\n")
        
        # Best overall configuration
        best_config = df.loc[df['Ex_post_Profit'].idxmax()]
        f.write("BEST OVERALL CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Method: {best_config['Method']}\n")
        f.write(f"Database: {best_config['Database']}\n")
        f.write(f"Architecture: {best_config['Architecture']}\n")
        f.write(f"Layers: {int(best_config['Num_Layers'])}\n")
        f.write(f"Iterations: {int(best_config['Max_Iterations'])}\n")
        f.write(f"Ex-post Profit: {best_config['Ex_post_Profit']:.2f}€\n")
        f.write(f"Processing Time: {best_config['Processing_Time_Seconds']:.2f}s\n\n")
        
        # Hyperparameter rankings
        f.write("HYPERPARAMETER RANKINGS:\n")
        f.write("-" * 30 + "\n")
        
        for param, param_name in [('Database', 'Training Database'), 
                                 ('Architecture', 'Architecture'),
                                 ('Num_Layers', 'Number of Layers'),
                                 ('Max_Iterations', 'Max Iterations')]:
            f.write(f"\n{param_name} Ranking (by mean ex-post profit):\n")
            param_means = df.groupby(param)['Ex_post_Profit'].mean().sort_values(ascending=False)
            for i, (value, mean) in enumerate(param_means.items(), 1):
                f.write(f"  {i}. {value}: {mean:.2f}€\n")
    
    print(f"DFL detailed summary saved to {output_dir / 'dfl_detailed_summary.txt'}")

def load_all_databases():
    """Load all database results with harmonized column names."""
    all_data = []
    
    # Load MIQP data
    miqp_methods = ['MIQP_Linear', 'MIQP_Piecewise', 'MIQP_NN']
    for method in miqp_methods:
        if method in DATABASE_PATHS:
            df = load_miqp_data(DATABASE_PATHS[method], method)
            if not df.empty:
                all_data.append(df)
                print(f"Loaded {len(df)} records from {method}")
    
    # Load DFL data
    dfl_methods = ['DFL_Bounded', 'DFL_Unbounded']
    for method in dfl_methods:
        if method in DATABASE_PATHS:
            df = load_dfl_data(DATABASE_PATHS[method], method)
            if not df.empty:
                all_data.append(df)
                print(f"Loaded {len(df)} records from {method}")
    
    if not all_data:
        print("Error: No data loaded from any database!")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter out extreme date
    combined_df = combined_df[combined_df['Date'] != EXTREME_DATE].copy()
    print(f"Excluded extreme date {EXTREME_DATE}")
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['Expected_Profit', 'Ex_post_Profit', 'SI_Penalty', 
                      'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds']
    
    for col in numeric_columns:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    print(f"Total records after combining and filtering: {len(combined_df)}")
    return combined_df

#%% Analysis and visualization functions
def create_output_directories():
    """Create output directories for results."""
    # Since script is already in result_visualization folder, create subdirectories here
    dirs_to_create = [
        Path("plots") / "density",
        Path("plots") / "boxplots", 
        Path("plots") / "performance",
        Path("tables"),
        Path("summary")
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return Path(".")  # Current directory (result_visualization)

def generate_comparative_density_plots(df, output_dir):
    """Generate density plots comparing all methods."""
    print("\n--- Generating Comparative Density Plots ---")
    
    density_dir = output_dir / "plots" / "density"
    
    # Set up figure aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    metrics = ['Ex_post_Profit', 'Expected_Profit', 'SI_Penalty', 'Volume_Penalty', 'Processing_Time_Seconds']
    metric_labels = ['Ex-post Profit (€)', 'Expected Profit (€)', 'SI Penalty (€)', 'Volume Penalty (€)', 'Processing Time (s)']
    
    for metric, label in zip(metrics, metric_labels):
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(12, 8))
        
        methods = df['Method'].unique()
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method][metric].dropna()
            if len(method_data) > 0:
                mean_val = method_data.mean()
                std_val = method_data.std()
                color = colors[i % len(colors)]
                
                sns.kdeplot(
                    method_data, 
                    label=f'{method}\n(μ={mean_val:.2f}, σ={std_val:.2f})',
                    fill=True, 
                    alpha=0.3,
                    color=color
                )
        
        plt.title(f'Density Plot of {label} by Method')
        plt.xlabel(label)
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(density_dir / f'density_{metric.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Density plots saved to {density_dir}")

def generate_comparative_boxplots(df, output_dir):
    """Generate boxplots comparing all methods."""
    print("\n--- Generating Comparative Boxplots ---")
    
    boxplot_dir = output_dir / "plots" / "boxplots"
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    metrics = ['Ex_post_Profit', 'Expected_Profit', 'SI_Penalty', 'Volume_Penalty', 'Processing_Time_Seconds']
    metric_labels = ['Ex-post Profit (€)', 'Expected Profit (€)', 'SI Penalty (€)', 'Volume Penalty (€)', 'Processing Time (s)']
    
    for metric, label in zip(metrics, metric_labels):
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Create boxplot (fix for seaborn FutureWarning)
        ax = sns.boxplot(x='Method', y=metric, data=df, hue='Method', palette=colors, legend=False)
        
        # Add swarm plot for individual points
        sns.swarmplot(x='Method', y=metric, data=df, color='black', alpha=0.5, size=3)
        
        # Add mean values as text
        means = df.groupby('Method')[metric].mean()
        for i, method in enumerate(ax.get_xticklabels()):
            method_name = method.get_text()
            if method_name in means.index:
                mean_val = means[method_name]
                ax.text(i, mean_val, f'μ={mean_val:.2f}', 
                       ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.title(f'Comparison of {label} Across Methods')
        plt.ylabel(label)
        plt.xlabel('Method')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(boxplot_dir / f'boxplot_{metric.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Boxplots saved to {boxplot_dir}")

def generate_performance_analysis(df, output_dir):
    """Generate performance analysis plots based on mean/std across all dates."""
    print("\n--- Generating Performance Analysis ---")
    
    perf_dir = output_dir / "plots" / "performance"
    
    # Calculate method statistics
    method_stats = df.groupby('Method').agg({
        'Ex_post_Profit': ['mean', 'std'],
        'Expected_Profit': ['mean', 'std'],
        'Processing_Time_Seconds': ['mean', 'std'],
        'SI_Penalty': ['mean', 'std'],
        'Volume_Penalty': ['mean', 'std']
    }).round(2)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Mean Ex-post Profit Comparison with Error Bars
    plt.figure(figsize=(12, 8))
    methods = method_stats.index
    profit_means = method_stats[('Ex_post_Profit', 'mean')]
    profit_stds = method_stats[('Ex_post_Profit', 'std')]
    
    bars = plt.bar(methods, profit_means, yerr=profit_stds, capsize=10, 
                   color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, profit_means, profit_stds)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 5,
                f'{mean_val:.1f}€\n±{std_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Mean Ex-post Profit Comparison Across 19 Dates\n(Error bars show ±1 standard deviation)')
    plt.ylabel('Ex-post Profit (€)')
    plt.xlabel('Method')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(perf_dir / 'mean_profit_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean Processing Time Comparison
    plt.figure(figsize=(12, 8))
    time_means = method_stats[('Processing_Time_Seconds', 'mean')]
    time_stds = method_stats[('Processing_Time_Seconds', 'std')]
    
    bars = plt.bar(methods, time_means, yerr=time_stds, capsize=10, 
                   color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, time_means, time_stds)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val,
                f'{mean_val:.1f}s\n±{std_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Mean Processing Time Comparison Across 19 Dates\n(Error bars show ±1 standard deviation)')
    plt.ylabel('Processing Time (seconds)')
    plt.xlabel('Method')
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')  # Use log scale for better visualization if there are large differences
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(perf_dir / 'mean_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency Scatter: Mean Time vs Mean Profit (with error bars)
    plt.figure(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        profit_mean = profit_means[method]
        profit_std = profit_stds[method]
        time_mean = time_means[method]
        time_std = time_stds[method]
        
        plt.errorbar(time_mean, profit_mean, 
                    xerr=time_std, yerr=profit_std,
                    fmt='o', markersize=10, capsize=8, capthick=2,
                    color=colors[i % len(colors)], label=method)
        
        # Add method labels
        plt.annotate(method, (time_mean, profit_mean), 
                    xytext=(5, 5), textcoords='offset points',
                    fontweight='bold', fontsize=10)
    
    plt.xlabel('Mean Processing Time (seconds)')
    plt.ylabel('Mean Ex-post Profit (€)')
    plt.title('Computational Efficiency: Mean Processing Time vs Mean Ex-post Profit\n(Error bars show ±1 standard deviation)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(perf_dir / 'efficiency_mean_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance Consistency (Coefficient of Variation)
    plt.figure(figsize=(12, 8))
    
    # Calculate coefficient of variation (std/mean) for profit
    cv_profit = (profit_stds / profit_means) * 100  # Convert to percentage
    
    bars = plt.bar(methods, cv_profit, color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, cv_val) in enumerate(zip(bars, cv_profit)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{cv_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Performance Consistency: Coefficient of Variation for Ex-post Profit\n(Lower values indicate more consistent performance)')
    plt.ylabel('Coefficient of Variation (%)')
    plt.xlabel('Method')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(perf_dir / 'consistency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance analysis plots saved to {perf_dir}")
    print("Generated plots:")
    print("  - mean_profit_comparison.png: Mean profit with error bars")
    print("  - mean_time_comparison.png: Mean processing time with error bars") 
    print("  - efficiency_mean_scatter.png: Mean time vs profit with error bars")
    print("  - consistency_comparison.png: Performance consistency analysis")

def generate_latex_tables(df, output_dir):
    """Generate LaTeX tables for publication."""
    print("\n--- Generating LaTeX Tables ---")
    
    tables_dir = output_dir / "tables"
    
    # Method Comparison Table - Mean and Std across all dates
    method_stats = df.groupby('Method').agg({
        'Ex_post_Profit': ['mean', 'std', 'min', 'max', 'count'],
        'Expected_Profit': ['mean', 'std'],
        'SI_Penalty': ['mean', 'std'],
        'Volume_Penalty': ['mean', 'std'],
        'Operating_Cost': ['mean', 'std'],
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    latex_comparison = r"""\begin{table}[h]
\centering
\caption{Method Performance Comparison - Mean ± Std Across 19 Dates (Excluding Extreme Date)}
\label{tab:method_comparison}
\begin{tabular}{lcccccc}
\toprule
Method & Ex-post Profit & Expected Profit & SI Penalty & Volume Penalty & Operating Cost & Processing Time \\
 & (€) & (€) & (€) & (€) & (€) & (s) \\
\midrule
"""
    
    for method in method_stats.index:
        expost_mean = method_stats.loc[method, ('Ex_post_Profit', 'mean')]
        expost_std = method_stats.loc[method, ('Ex_post_Profit', 'std')]
        expected_mean = method_stats.loc[method, ('Expected_Profit', 'mean')]
        expected_std = method_stats.loc[method, ('Expected_Profit', 'std')]
        si_mean = method_stats.loc[method, ('SI_Penalty', 'mean')]
        si_std = method_stats.loc[method, ('SI_Penalty', 'std')]
        vol_mean = method_stats.loc[method, ('Volume_Penalty', 'mean')]
        vol_std = method_stats.loc[method, ('Volume_Penalty', 'std')]
        op_mean = method_stats.loc[method, ('Operating_Cost', 'mean')]
        op_std = method_stats.loc[method, ('Operating_Cost', 'std')]
        time_mean = method_stats.loc[method, ('Processing_Time_Seconds', 'mean')]
        time_std = method_stats.loc[method, ('Processing_Time_Seconds', 'std')]
        
        # Format method name for LaTeX
        method_latex = method.replace('_', '\\_')
        
        latex_comparison += f"{method_latex} & "
        latex_comparison += f"{expost_mean:.2f} $\\pm$ {expost_std:.2f} & "
        latex_comparison += f"{expected_mean:.2f} $\\pm$ {expected_std:.2f} & "
        latex_comparison += f"{si_mean:.2f} $\\pm$ {si_std:.2f} & "
        latex_comparison += f"{vol_mean:.2f} $\\pm$ {vol_std:.2f} & "
        latex_comparison += f"{op_mean:.2f} $\\pm$ {op_std:.2f} & "
        latex_comparison += f"{time_mean:.2f} $\\pm$ {time_std:.2f} \\\\\n"
    
    latex_comparison += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save comparison table
    with open(tables_dir / 'method_comparison.tex', 'w') as f:
        f.write(latex_comparison)
    
    # Summary Statistics Table
    latex_summary = r"""\begin{table}[h]
\centering
\caption{Method Performance Summary Statistics}
\label{tab:method_summary}
\begin{tabular}{lccccc}
\toprule
Method & Count & Mean Ex-post & Std Dev & Min & Max \\
 & (dates) & Profit (€) & (€) & (€) & (€) \\
\midrule
"""
    
    for method in method_stats.index:
        count = int(method_stats.loc[method, ('Ex_post_Profit', 'count')])
        mean_profit = method_stats.loc[method, ('Ex_post_Profit', 'mean')]
        std_profit = method_stats.loc[method, ('Ex_post_Profit', 'std')]
        min_profit = method_stats.loc[method, ('Ex_post_Profit', 'min')]
        max_profit = method_stats.loc[method, ('Ex_post_Profit', 'max')]
        
        method_latex = method.replace('_', '\\_')
        
        latex_summary += f"{method_latex} & {count} & {mean_profit:.2f} & {std_profit:.2f} & {min_profit:.2f} & {max_profit:.2f} \\\\\n"
    
    latex_summary += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save summary table
    with open(tables_dir / 'method_summary.tex', 'w') as f:
        f.write(latex_summary)
    
    print(f"LaTeX tables saved to {tables_dir}")
    print("Generated tables:")
    print("  - method_comparison.tex: Mean ± Std comparison across all dates")
    print("  - method_summary.tex: Summary statistics by method")

def generate_summary_statistics(df, output_dir):
    """Generate comprehensive summary statistics focused on mean/std across dates."""
    print("\n--- Generating Summary Statistics ---")
    
    summary_dir = output_dir / "summary"
    
    # Overall statistics
    overall_stats = df.describe()
    overall_stats.to_csv(summary_dir / 'overall_statistics.csv')
    
    # Method-wise statistics  
    method_stats = df.groupby('Method').describe()
    method_stats.to_csv(summary_dir / 'method_statistics.csv')
    
    # Performance ranking based on mean across all dates
    method_ranking = df.groupby('Method').agg({
        'Ex_post_Profit': ['mean', 'std', 'count'],
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    # Sort by mean ex-post profit
    method_ranking = method_ranking.sort_values(('Ex_post_Profit', 'mean'), ascending=False)
    method_ranking.to_csv(summary_dir / 'method_ranking.csv')
    
    # Calculate coefficient of variation for consistency analysis
    method_consistency = df.groupby('Method').agg({
        'Ex_post_Profit': lambda x: (x.std() / x.mean()) * 100  # CV as percentage
    }).round(2)
    method_consistency.columns = ['CV_Ex_post_Profit']
    method_consistency.to_csv(summary_dir / 'method_consistency.csv')
    
    # Generate summary report
    with open(summary_dir / 'analysis_summary.txt', 'w') as f:
        f.write("COMPREHENSIVE DATABASE ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Analysis Focus: Mean ± Std across all dates\n")
        f.write(f"Total records analyzed: {len(df)}\n")
        f.write(f"Methods compared: {', '.join(df['Method'].unique())}\n")
        f.write(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")
        f.write(f"Extreme date excluded: {EXTREME_DATE}\n")
        f.write(f"Number of dates per method: ~{len(df['Date'].unique())}\n\n")
        
        f.write("PERFORMANCE RANKING (by Mean Ex-post Profit across all dates):\n")
        f.write("-" * 60 + "\n")
        for i, (method, stats) in enumerate(method_ranking.iterrows(), 1):
            mean_profit = stats[('Ex_post_Profit', 'mean')]
            std_profit = stats[('Ex_post_Profit', 'std')]
            count = int(stats[('Ex_post_Profit', 'count')])
            mean_time = stats[('Processing_Time_Seconds', 'mean')]
            cv = method_consistency.loc[method, 'CV_Ex_post_Profit']
            
            f.write(f"{i}. {method}:\n")
            f.write(f"   Ex-post Profit: {mean_profit:.2f} ± {std_profit:.2f}€ (n={count})\n")
            f.write(f"   Processing Time: {mean_time:.2f}s\n")
            f.write(f"   Consistency (CV): {cv:.1f}%\n\n")
        
        # Performance insights
        best_method = method_ranking.index[0]
        best_profit = method_ranking.iloc[0][('Ex_post_Profit', 'mean')]
        worst_method = method_ranking.index[-1] 
        worst_profit = method_ranking.iloc[-1][('Ex_post_Profit', 'mean')]
        
        f.write("PERFORMANCE INSIGHTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best performing method: {best_method} ({best_profit:.2f}€)\n")
        f.write(f"Worst performing method: {worst_method} ({worst_profit:.2f}€)\n")
        
        improvement = ((best_profit - worst_profit) / worst_profit) * 100
        f.write(f"Performance improvement (best vs worst): {improvement:.2f}%\n\n")
        
        # Most consistent method (lowest CV)
        most_consistent = method_consistency.idxmin()[0]
        consistency_cv = method_consistency.min()[0]
        
        f.write(f"Most consistent method: {most_consistent} (CV: {consistency_cv:.1f}%)\n")
        
        # Processing time analysis
        fastest_method = method_ranking.sort_values(('Processing_Time_Seconds', 'mean')).index[0]
        fastest_time = method_ranking.sort_values(('Processing_Time_Seconds', 'mean')).iloc[0][('Processing_Time_Seconds', 'mean')]
        
        f.write(f"Fastest method: {fastest_method} ({fastest_time:.2f}s)\n")
    
    print(f"Summary statistics saved to {summary_dir}")
    print("Generated files:")
    print("  - method_ranking.csv: Performance ranking by mean ex-post profit")
    print("  - method_consistency.csv: Consistency analysis (coefficient of variation)")
    print("  - analysis_summary.txt: Comprehensive text summary focusing on mean/std")

#%% Main execution function
def main():
    """Main function to run the comprehensive analysis."""
    print("Starting Comprehensive Database Analysis...")
    print("="*60)
    
    # Create output directories
    output_dir = create_output_directories()
    print(f"Output subdirectories created in current folder: {output_dir.absolute()}")
    
    # Load all databases
    df = load_all_databases()
    
    if df.empty:
        print("Error: No data loaded. Please check file paths and data formats.")
        return
    
    print(f"\nData loaded successfully:")
    print(f"- Total records: {len(df)}")
    print(f"- Methods: {', '.join(df['Method'].unique())}")
    print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"- Dates per method: ~{len(df['Date'].unique())} (for mean/std calculation)")
    print(f"- Extreme date excluded: {EXTREME_DATE}")
    print("\nAnalysis will focus on mean ± standard deviation across all dates for each method.")
    
    # Generate all analyses
    generate_comparative_density_plots(df, output_dir)
    generate_comparative_boxplots(df, output_dir) 
    generate_performance_analysis(df, output_dir)
    generate_latex_tables(df, output_dir)
    generate_summary_statistics(df, output_dir)
    
    # Generate detailed DFL analysis
    detailed_dfl_analysis(output_dir)
    
    # Return to default matplotlib style
    plt.style.use('default')
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f"All results saved in: {output_dir.absolute()}")
    print("\nGenerated analyses:")
    print("  📊 Comparative Analysis: density plots, boxplots, performance analysis")
    print("  📋 LaTeX Tables: method comparison, summary statistics")
    print("  📈 Summary Statistics: ranking, consistency analysis")
    print("  🔬 Detailed DFL Analysis: hyperparameter effects, interactions")
    print("="*60)

#%% Execute
if __name__ == "__main__":
    main()
# %%
