#%% DFL Hyperparameter Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from itertools import combinations
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

#%% Configuration
DFL_PATHS = {
    'DFL_Bounded': r'..\DFL_bounded\validation_results\comprehensive\master_validation_benchmarks.csv',
    'DFL_Unbounded': r'..\DFL_unbounded\validation_results\comprehensive\master_validation_benchmarks.csv'
}

EXTREME_DATE = '2024-12-12'

#%% Data loading functions
def standardize_date_format(date_str):
    """Convert various date formats to YYYY-MM-DD format."""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Handle different date formats
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            if len(parts[0]) == 4:  # YYYY/MM/DD
                return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            else:  # MM/DD/YYYY
                return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
    
    return date_str

def load_dfl_data():
    """Load and combine DFL validation data from both bounded and unbounded versions."""
    all_dfl_data = []
    
    for method_name, file_path in DFL_PATHS.items():
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        
        # Add method type
        df['DFL_Type'] = method_name
        
        # Standardize column names
        column_mapping = {
            'New_Date': 'Date',
            'Expected_Profit': 'Expected_Profit',
            'Ex_post_Profit': 'Ex_post_Profit', 
            'SI_Penalty': 'SI_Penalty',
            'Volume_Penalty': 'Volume_Penalty',
            'Operating_Cost': 'Operating_Cost',
            'Processing_Time_Seconds': 'Processing_Time_Seconds'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Standardize date format
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(standardize_date_format)
        
        all_dfl_data.append(df)
        print(f"Loaded {len(df)} records from {method_name}")
    
    if not all_dfl_data:
        print("Error: No DFL data loaded!")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_dfl_data, ignore_index=True)
    
    # Filter out extreme date
    combined_df = combined_df[combined_df['Date'] != EXTREME_DATE].copy()
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['Expected_Profit', 'Ex_post_Profit', 'SI_Penalty', 
                      'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
                      'Num_Layers', 'Max_Iterations']
    
    for col in numeric_columns:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    print(f"Total DFL records after filtering: {len(combined_df)}")
    return combined_df

#%% Analysis functions
def create_dfl_directories():
    """Create output directories for DFL analysis."""
    dirs_to_create = [
        Path("dfl_analysis"),
        Path("dfl_analysis") / "hyperparameter_effects",
        Path("dfl_analysis") / "interaction_analysis",
        Path("dfl_analysis") / "performance_optimization",
        Path("dfl_analysis") / "statistical_analysis",
        Path("dfl_analysis") / "tables"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return Path("dfl_analysis")

def analyze_hyperparameter_effects(df, output_dir):
    """Analyze individual hyperparameter effects on performance."""
    print("\n--- Analyzing Individual Hyperparameter Effects ---")
    
    effects_dir = output_dir / "hyperparameter_effects"
    
    # Set up color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Training Database Effect
    plt.figure(figsize=(15, 10))
    
    # Ex-post Profit by Database
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='Database', y='Ex_post_Profit', hue='Database', palette=colors, legend=False)
    plt.xticks(rotation=45)
    plt.title('Ex-post Profit by Training Database')
    plt.ylabel('Ex-post Profit (€)')
    
    # Processing Time by Database
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='Database', y='Processing_Time_Seconds', hue='Database', palette=colors, legend=False)
    plt.xticks(rotation=45)
    plt.title('Processing Time by Training Database')
    plt.ylabel('Processing Time (s)')
    plt.yscale('log')
    
    # Mean comparison table for Database
    plt.subplot(2, 2, 3)
    db_stats = df.groupby('Database').agg({
        'Ex_post_Profit': 'mean',
        'Processing_Time_Seconds': 'mean'
    }).round(2)
    
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=db_stats.values, 
                    rowLabels=[db.replace('euclidean_', '') for db in db_stats.index],
                    colLabels=['Mean Ex-post Profit', 'Mean Process Time'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title('Mean Performance by Database')
    
    # Profit vs Time scatter by Database
    plt.subplot(2, 2, 4)
    for i, db in enumerate(df['Database'].unique()):
        db_data = df[df['Database'] == db]
        plt.scatter(db_data['Processing_Time_Seconds'], db_data['Ex_post_Profit'], 
                   alpha=0.6, label=db.replace('euclidean_', ''), color=colors[i])
    plt.xlabel('Processing Time (s)')
    plt.ylabel('Ex-post Profit (€)')
    plt.xscale('log')
    plt.legend()
    plt.title('Efficiency Trade-off by Database')
    
    plt.tight_layout()
    plt.savefig(effects_dir / 'database_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Architecture Effect (LSTM vs RNN)
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    sns.boxplot(data=df, x='Architecture', y='Ex_post_Profit', hue='Architecture', palette=colors[:2], legend=False)
    plt.title('Ex-post Profit by Architecture')
    plt.ylabel('Ex-post Profit (€)')
    
    plt.subplot(2, 3, 2) 
    sns.boxplot(data=df, x='Architecture', y='Processing_Time_Seconds', hue='Architecture', palette=colors[:2], legend=False)
    plt.title('Processing Time by Architecture')
    plt.ylabel('Processing Time (s)')
    plt.yscale('log')
    
    plt.subplot(2, 3, 3)
    arch_stats = df.groupby('Architecture').agg({
        'Ex_post_Profit': ['mean', 'std'],
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    # Violin plot for distribution comparison
    plt.subplot(2, 3, 4)
    sns.violinplot(data=df, x='Architecture', y='Ex_post_Profit', hue='Architecture', palette=colors[:2], legend=False)
    plt.title('Profit Distribution by Architecture')
    
    # Statistical test
    plt.subplot(2, 3, 5)
    lstm_profit = df[df['Architecture'] == 'LSTM']['Ex_post_Profit'].dropna()
    rnn_profit = df[df['Architecture'] == 'RNN']['Ex_post_Profit'].dropna()
    
    if len(lstm_profit) > 0 and len(rnn_profit) > 0:
        t_stat, p_value = stats.ttest_ind(lstm_profit, rnn_profit)
        
        plt.bar(['LSTM', 'RNN'], [lstm_profit.mean(), rnn_profit.mean()], 
               yerr=[lstm_profit.std(), rnn_profit.std()], capsize=10, color=colors[:2])
        plt.title(f'Architecture Comparison\np-value: {p_value:.4f}')
        plt.ylabel('Mean Ex-post Profit (€)')
        
        # Add significance annotation
        if p_value < 0.05:
            plt.text(0.5, max(lstm_profit.mean(), rnn_profit.mean()) + 50, 
                    '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*',
                    ha='center', fontsize=16)
    
    # Performance by layers within each architecture
    plt.subplot(2, 3, 6)
    sns.boxplot(data=df, x='Num_Layers', y='Ex_post_Profit', hue='Architecture', palette=colors[:2])
    plt.title('Profit by Layers and Architecture')
    plt.xlabel('Number of Layers')
    
    plt.tight_layout()
    plt.savefig(effects_dir / 'architecture_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Number of Layers Effect
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='Num_Layers', y='Ex_post_Profit', hue='Num_Layers', palette=colors, legend=False)
    plt.title('Ex-post Profit by Number of Layers')
    plt.xlabel('Number of Layers')
    plt.ylabel('Ex-post Profit (€)')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='Num_Layers', y='Processing_Time_Seconds', hue='Num_Layers', palette=colors, legend=False)
    plt.title('Processing Time by Number of Layers')
    plt.xlabel('Number of Layers')
    plt.ylabel('Processing Time (s)')
    plt.yscale('log')
    
    # Layer performance trend
    plt.subplot(2, 2, 3)
    layer_means = df.groupby('Num_Layers')['Ex_post_Profit'].agg(['mean', 'std'])
    plt.errorbar(layer_means.index, layer_means['mean'], yerr=layer_means['std'], 
                marker='o', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Number of Layers')
    plt.ylabel('Mean Ex-post Profit (€)')
    plt.title('Profit Trend by Number of Layers')
    plt.grid(True, alpha=0.3)
    
    # Complexity vs Performance
    plt.subplot(2, 2, 4)
    layer_time_means = df.groupby('Num_Layers')['Processing_Time_Seconds'].mean()
    layer_profit_means = df.groupby('Num_Layers')['Ex_post_Profit'].mean()
    
    for layers in df['Num_Layers'].unique():
        plt.scatter(layer_time_means[layers], layer_profit_means[layers], 
                   s=100, label=f'{layers} layers')
        plt.annotate(f'{layers}L', (layer_time_means[layers], layer_profit_means[layers]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Mean Processing Time (s)')
    plt.ylabel('Mean Ex-post Profit (€)')
    plt.title('Complexity-Performance Trade-off')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(effects_dir / 'layers_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Max Iterations Effect  
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    sns.boxplot(data=df, x='Max_Iterations', y='Ex_post_Profit')
    plt.title('Ex-post Profit by Max Iterations')
    plt.xlabel('Max Iterations')
    plt.ylabel('Ex-post Profit (€)')
    
    plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x='Max_Iterations', y='Processing_Time_Seconds')
    plt.title('Processing Time by Max Iterations')
    plt.xlabel('Max Iterations')
    plt.ylabel('Processing Time (s)')
    plt.yscale('log')
    
    # Iteration convergence analysis
    plt.subplot(2, 3, 3)
    iter_means = df.groupby('Max_Iterations')['Ex_post_Profit'].agg(['mean', 'std'])
    plt.errorbar(iter_means.index, iter_means['mean'], yerr=iter_means['std'],
                marker='o', capsize=5, capthick=2, linewidth=2, color='red')
    plt.xlabel('Max Iterations')
    plt.ylabel('Mean Ex-post Profit (€)')
    plt.title('Convergence: Profit vs Iterations')
    plt.grid(True, alpha=0.3)
    
    # Time cost of iterations
    plt.subplot(2, 3, 4)
    iter_time_means = df.groupby('Max_Iterations')['Processing_Time_Seconds'].mean()
    plt.plot(iter_time_means.index, iter_time_means.values, 'o-', linewidth=2, color='blue')
    plt.xlabel('Max Iterations')
    plt.ylabel('Mean Processing Time (s)')
    plt.title('Time Cost vs Iterations')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Marginal improvement analysis
    plt.subplot(2, 3, 5)
    iter_profit_means = df.groupby('Max_Iterations')['Ex_post_Profit'].mean().sort_index()
    marginal_improvement = iter_profit_means.diff()
    
    plt.bar(marginal_improvement.index[1:], marginal_improvement.values[1:], color='green', alpha=0.7)
    plt.xlabel('Max Iterations')
    plt.ylabel('Marginal Profit Improvement (€)')
    plt.title('Marginal Benefit of Additional Iterations')
    plt.grid(True, alpha=0.3)
    
    # Efficiency ratio (profit improvement per second)
    plt.subplot(2, 3, 6)
    efficiency_data = []
    for iter_val in df['Max_Iterations'].unique():
        iter_data = df[df['Max_Iterations'] == iter_val]
        if len(iter_data) > 0:
            mean_profit = iter_data['Ex_post_Profit'].mean()
            mean_time = iter_data['Processing_Time_Seconds'].mean()
            efficiency = mean_profit / mean_time if mean_time > 0 else 0
            efficiency_data.append({'Iterations': iter_val, 'Efficiency': efficiency})
    
    efficiency_df = pd.DataFrame(efficiency_data)
    if not efficiency_df.empty:
        plt.bar(efficiency_df['Iterations'], efficiency_df['Efficiency'], color='purple', alpha=0.7)
        plt.xlabel('Max Iterations')
        plt.ylabel('Profit per Second (€/s)')
        plt.title('Computational Efficiency by Iterations')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(effects_dir / 'iterations_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Hyperparameter effects analysis saved to {effects_dir}")

def analyze_hyperparameter_interactions(df, output_dir):
    """Analyze interactions between different hyperparameters."""
    print("\n--- Analyzing Hyperparameter Interactions ---")
    
    interaction_dir = output_dir / "interaction_analysis"
    
    # 1. Database × Architecture Interaction
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    interaction_data = df.groupby(['Database', 'Architecture'])['Ex_post_Profit'].mean().unstack()
    sns.heatmap(interaction_data, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Mean Ex-post Profit (€)'})
    plt.title('Database × Architecture Interaction\n(Mean Ex-post Profit)')
    plt.xlabel('Architecture')
    plt.ylabel('Database')
    
    plt.subplot(2, 3, 2)
    interaction_time = df.groupby(['Database', 'Architecture'])['Processing_Time_Seconds'].mean().unstack()
    sns.heatmap(interaction_time, annot=True, fmt='.2f', cmap='plasma', cbar_kws={'label': 'Mean Processing Time (s)'})
    plt.title('Database × Architecture Interaction\n(Mean Processing Time)')
    plt.xlabel('Architecture')
    plt.ylabel('Database')
    
    # 2. Layers × Iterations Interaction
    plt.subplot(2, 3, 3)
    layer_iter_profit = df.groupby(['Num_Layers', 'Max_Iterations'])['Ex_post_Profit'].mean().unstack()
    sns.heatmap(layer_iter_profit, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Mean Ex-post Profit (€)'})
    plt.title('Layers × Iterations Interaction\n(Mean Ex-post Profit)')
    plt.xlabel('Max Iterations')
    plt.ylabel('Number of Layers')
    
    plt.subplot(2, 3, 4)
    layer_iter_time = df.groupby(['Num_Layers', 'Max_Iterations'])['Processing_Time_Seconds'].mean().unstack()
    sns.heatmap(layer_iter_time, annot=True, fmt='.2f', cmap='plasma', cbar_kws={'label': 'Mean Processing Time (s)'})
    plt.title('Layers × Iterations Interaction\n(Mean Processing Time)')
    plt.xlabel('Max Iterations')
    plt.ylabel('Number of Layers')
    
    # 3. Architecture × Iterations for different databases
    plt.subplot(2, 3, 5)
    # Focus on most common database for clarity
    main_db = df['Database'].mode()[0] if not df.empty else None
    if main_db:
        db_subset = df[df['Database'] == main_db]
        arch_iter_profit = db_subset.groupby(['Architecture', 'Max_Iterations'])['Ex_post_Profit'].mean().unstack()
        sns.heatmap(arch_iter_profit, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Mean Ex-post Profit (€)'})
        plt.title(f'Architecture × Iterations\n({main_db.replace("euclidean_", "")} Database)')
        plt.xlabel('Max Iterations')
        plt.ylabel('Architecture')
    
    # 4. 3-way interaction visualization
    plt.subplot(2, 3, 6)
    # Create a 3-way interaction plot using Database, Architecture, and Layers
    for db in df['Database'].unique():
        db_data = df[df['Database'] == db]
        for arch in db_data['Architecture'].unique():
            arch_data = db_data[db_data['Architecture'] == arch]
            layer_means = arch_data.groupby('Num_Layers')['Ex_post_Profit'].mean()
            plt.plot(layer_means.index, layer_means.values, 'o-', 
                    label=f'{db.replace("euclidean_", "")}-{arch}', linewidth=2)
    
    plt.xlabel('Number of Layers')
    plt.ylabel('Mean Ex-post Profit (€)')
    plt.title('3-way Interaction:\nDatabase × Architecture × Layers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(interaction_dir / 'hyperparameter_interactions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Advanced interaction analysis: Statistical significance
    interaction_stats = analyze_interaction_significance(df)
    
    # Save interaction statistics
    with open(interaction_dir / 'interaction_statistics.txt', 'w') as f:
        f.write("HYPERPARAMETER INTERACTION ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(interaction_stats)
    
    print(f"Interaction analysis saved to {interaction_dir}")

def analyze_interaction_significance(df):
    """Perform statistical tests for hyperparameter interactions."""
    results = []
    
    # Test Database × Architecture interaction using 2-way ANOVA
    try:
        from scipy.stats import f_oneway
        
        results.append("STATISTICAL SIGNIFICANCE TESTS\n")
        results.append("-" * 30 + "\n\n")
        
        # 1. Database effect
        db_groups = [df[df['Database'] == db]['Ex_post_Profit'].dropna() 
                    for db in df['Database'].unique()]
        db_groups = [group for group in db_groups if len(group) > 0]
        
        if len(db_groups) > 1:
            f_stat, p_val = f_oneway(*db_groups)
            results.append(f"Database Effect on Ex-post Profit:\n")
            results.append(f"  F-statistic: {f_stat:.4f}\n")
            results.append(f"  p-value: {p_val:.4f}\n")
            results.append(f"  Significant: {'Yes' if p_val < 0.05 else 'No'}\n\n")
        
        # 2. Architecture effect  
        arch_groups = [df[df['Architecture'] == arch]['Ex_post_Profit'].dropna()
                      for arch in df['Architecture'].unique()]
        arch_groups = [group for group in arch_groups if len(group) > 0]
        
        if len(arch_groups) > 1:
            f_stat, p_val = f_oneway(*arch_groups)
            results.append(f"Architecture Effect on Ex-post Profit:\n")
            results.append(f"  F-statistic: {f_stat:.4f}\n")
            results.append(f"  p-value: {p_val:.4f}\n")
            results.append(f"  Significant: {'Yes' if p_val < 0.05 else 'No'}\n\n")
        
        # 3. Layers effect
        layer_groups = [df[df['Num_Layers'] == layers]['Ex_post_Profit'].dropna()
                       for layers in df['Num_Layers'].unique()]
        layer_groups = [group for group in layer_groups if len(group) > 0]
        
        if len(layer_groups) > 1:
            f_stat, p_val = f_oneway(*layer_groups)
            results.append(f"Number of Layers Effect on Ex-post Profit:\n")
            results.append(f"  F-statistic: {f_stat:.4f}\n")
            results.append(f"  p-value: {p_val:.4f}\n")
            results.append(f"  Significant: {'Yes' if p_val < 0.05 else 'No'}\n\n")
        
        # 4. Iterations effect
        iter_groups = [df[df['Max_Iterations'] == iters]['Ex_post_Profit'].dropna()
                      for iters in df['Max_Iterations'].unique()]
        iter_groups = [group for group in iter_groups if len(group) > 0]
        
        if len(iter_groups) > 1:
            f_stat, p_val = f_oneway(*iter_groups)
            results.append(f"Max Iterations Effect on Ex-post Profit:\n")
            results.append(f"  F-statistic: {f_stat:.4f}\n")
            results.append(f"  p-value: {p_val:.4f}\n")
            results.append(f"  Significant: {'Yes' if p_val < 0.05 else 'No'}\n\n")
    
    except Exception as e:
        results.append(f"Error in statistical analysis: {e}\n")
    
    return "".join(results)

def optimize_hyperparameters(df, output_dir):
    """Find optimal hyperparameter combinations."""
    print("\n--- Optimizing Hyperparameter Combinations ---")
    
    optimization_dir = output_dir / "performance_optimization"
    
    # 1. Find best configurations overall
    best_configs = df.nlargest(20, 'Ex_post_Profit')[
        ['Database', 'Architecture', 'Num_Layers', 'Max_Iterations', 
         'Ex_post_Profit', 'Processing_Time_Seconds', 'DFL_Type']
    ].copy()
    
    # 2. Find best configuration per hyperparameter combination
    best_by_combo = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations']).agg({
        'Ex_post_Profit': ['mean', 'std', 'count'],
        'Processing_Time_Seconds': ['mean', 'std'],
        'Expected_Profit': 'mean'
    }).round(2)
    
    best_by_combo.columns = ['_'.join(col).strip() for col in best_by_combo.columns]
    best_by_combo = best_by_combo.reset_index()
    best_by_combo = best_by_combo.sort_values('Ex_post_Profit_mean', ascending=False)
    
    # 3. Pareto frontier analysis (Profit vs Time)
    plt.figure(figsize=(15, 12))
    
    # Overall best configurations
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(best_configs['Processing_Time_Seconds'], best_configs['Ex_post_Profit'],
                         c=range(len(best_configs)), cmap='viridis', s=100, alpha=0.7)
    plt.xlabel('Processing Time (s)')
    plt.ylabel('Ex-post Profit (€)')
    plt.title('Top 20 Configurations\n(Profit vs Time)')
    plt.xscale('log')
    plt.colorbar(scatter, label='Rank')
    plt.grid(True, alpha=0.3)
    
    # Best by database
    plt.subplot(2, 3, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, db in enumerate(df['Database'].unique()):
        db_best = best_by_combo[best_by_combo['Database'] == db].head(5)
        plt.scatter(db_best['Processing_Time_Seconds_mean'], db_best['Ex_post_Profit_mean'],
                   label=db.replace('euclidean_', ''), color=colors[i % len(colors)], s=100, alpha=0.7)
    
    plt.xlabel('Mean Processing Time (s)')
    plt.ylabel('Mean Ex-post Profit (€)')
    plt.title('Best Configurations by Database')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Architecture comparison of best configs
    plt.subplot(2, 3, 3)
    arch_best = df.groupby(['Architecture']).agg({
        'Ex_post_Profit': ['mean', 'max'],
        'Processing_Time_Seconds': ['mean', 'min']
    }).round(2)
    
    arch_profit_means = arch_best[('Ex_post_Profit', 'mean')]
    arch_time_means = arch_best[('Processing_Time_Seconds', 'mean')]
    
    for i, arch in enumerate(arch_profit_means.index):
        plt.scatter(arch_time_means[arch], arch_profit_means[arch], 
                   s=200, label=arch, color=colors[i])
        plt.annotate(arch, (arch_time_means[arch], arch_profit_means[arch]),
                    xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.xlabel('Mean Processing Time (s)')
    plt.ylabel('Mean Ex-post Profit (€)')
    plt.title('Architecture Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    # Efficiency frontier
    plt.subplot(2, 3, 4)
    efficiency_ratios = best_by_combo.copy()
    efficiency_ratios['Efficiency'] = efficiency_ratios['Ex_post_Profit_mean'] / efficiency_ratios['Processing_Time_Seconds_mean']
    efficiency_ratios = efficiency_ratios.sort_values('Efficiency', ascending=False)
    
    top_efficient = efficiency_ratios.head(10)
    bars = plt.barh(range(len(top_efficient)), top_efficient['Efficiency'])
    
    # Color bars by database
    db_colors = {db: colors[i] for i, db in enumerate(df['Database'].unique())}
    for i, (_, row) in enumerate(top_efficient.iterrows()):
        bars[i].set_color(db_colors.get(row['Database'], 'gray'))
    
    plt.yticks(range(len(top_efficient)), 
              [f"{row['Database'].replace('euclidean_', '')}-{row['Architecture']}-{row['Num_Layers']}L-{row['Max_Iterations']}it" 
               for _, row in top_efficient.iterrows()])
    plt.xlabel('Efficiency (Profit/Time)')
    plt.title('Top 10 Most Efficient Configurations')
    plt.grid(True, alpha=0.3)
    
    # Performance by complexity (layers × iterations)
    plt.subplot(2, 3, 5)
    df['Complexity'] = df['Num_Layers'] * df['Max_Iterations']
    complexity_perf = df.groupby('Complexity').agg({
        'Ex_post_Profit': 'mean',
        'Processing_Time_Seconds': 'mean'
    })
    
    plt.scatter(complexity_perf.index, complexity_perf['Ex_post_Profit'], 
               s=complexity_perf['Processing_Time_Seconds']/10, alpha=0.6)
    plt.xlabel('Model Complexity (Layers × Iterations)')
    plt.ylabel('Mean Ex-post Profit (€)')
    plt.title('Performance vs Complexity\n(Bubble size = Processing Time)')
    plt.grid(True, alpha=0.3)
    
    # Recommendations summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Get top recommendations
    top_overall = best_by_combo.iloc[0]
    most_efficient = efficiency_ratios.iloc[0]
    
    recommendations_text = f"""OPTIMIZATION RECOMMENDATIONS

📊 BEST OVERALL CONFIGURATION:
Database: {top_overall['Database'].replace('euclidean_', '')}
Architecture: {top_overall['Architecture']}
Layers: {int(top_overall['Num_Layers'])}
Iterations: {int(top_overall['Max_Iterations'])}
Profit: {top_overall['Ex_post_Profit_mean']:.2f}€
Time: {top_overall['Processing_Time_Seconds_mean']:.2f}s

⚡ MOST EFFICIENT CONFIGURATION:
Database: {most_efficient['Database'].replace('euclidean_', '')}
Architecture: {most_efficient['Architecture']}
Layers: {int(most_efficient['Num_Layers'])}
Iterations: {int(most_efficient['Max_Iterations'])}
Efficiency: {most_efficient['Efficiency']:.2f} €/s

🎯 KEY INSIGHTS:
• Best database: {df.groupby('Database')['Ex_post_Profit'].mean().idxmax().replace('euclidean_', '')}
• Best architecture: {df.groupby('Architecture')['Ex_post_Profit'].mean().idxmax()}
• Optimal layers: {df.groupby('Num_Layers')['Ex_post_Profit'].mean().idxmax()}
• Optimal iterations: {df.groupby('Max_Iterations')['Ex_post_Profit'].mean().idxmax()}
"""
    
    plt.text(0.05, 0.95, recommendations_text, fontsize=10, va='top', ha='left',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(optimization_dir / 'optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    best_configs.to_csv(optimization_dir / 'top_20_configurations.csv', index=False)
    best_by_combo.to_csv(optimization_dir / 'best_by_combination.csv', index=False)
    efficiency_ratios.to_csv(optimization_dir / 'efficiency_ranking.csv', index=False)
    
    print(f"Optimization analysis saved to {optimization_dir}")
    
    return best_configs, best_by_combo, efficiency_ratios

def generate_dfl_latex_tables(df, best_configs, best_by_combo, output_dir):
    """Generate comprehensive LaTeX tables for DFL analysis."""
    print("\n--- Generating DFL LaTeX Tables ---")
    
    tables_dir = output_dir / "tables"
    
    # 1. Hyperparameter Effects Summary Table
    hyperparameter_summary = []
    
    # Database effects
    db_effects = df.groupby('Database').agg({
        'Ex_post_Profit': ['mean', 'std'],
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    # Architecture effects
    arch_effects = df.groupby('Architecture').agg({
        'Ex_post_Profit': ['mean', 'std'],
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    # Layers effects
    layer_effects = df.groupby('Num_Layers').agg({
        'Ex_post_Profit': ['mean', 'std'],
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    # Iterations effects
    iter_effects = df.groupby('Max_Iterations').agg({
        'Ex_post_Profit': ['mean', 'std'],
        'Processing_Time_Seconds': ['mean', 'std']
    }).round(2)
    
    latex_hyperparameter_table = r"""\begin{table}[h]
\centering
\caption{DFL Hyperparameter Effects on Performance}
\label{tab:dfl_hyperparameter_effects}
\begin{tabular}{llcccc}
\toprule
Hyperparameter & Value & \multicolumn{2}{c}{Ex-post Profit (€)} & \multicolumn{2}{c}{Processing Time (s)} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
 & & Mean & Std & Mean & Std \\
\midrule
"""
    
    # Add database effects
    for db in db_effects.index:
        db_name = db.replace('euclidean_', '').replace('_', '\\_')
        profit_mean = db_effects.loc[db, ('Ex_post_Profit', 'mean')]
        profit_std = db_effects.loc[db, ('Ex_post_Profit', 'std')]
        time_mean = db_effects.loc[db, ('Processing_Time_Seconds', 'mean')]
        time_std = db_effects.loc[db, ('Processing_Time_Seconds', 'std')]
        
        latex_hyperparameter_table += f"Database & {db_name} & {profit_mean:.2f} & {profit_std:.2f} & {time_mean:.2f} & {time_std:.2f} \\\\\n"
    
    latex_hyperparameter_table += r"\midrule" + "\n"
    
    # Add architecture effects
    for arch in arch_effects.index:
        profit_mean = arch_effects.loc[arch, ('Ex_post_Profit', 'mean')]
        profit_std = arch_effects.loc[arch, ('Ex_post_Profit', 'std')]
        time_mean = arch_effects.loc[arch, ('Processing_Time_Seconds', 'mean')]
        time_std = arch_effects.loc[arch, ('Processing_Time_Seconds', 'std')]
        
        latex_hyperparameter_table += f"Architecture & {arch} & {profit_mean:.2f} & {profit_std:.2f} & {time_mean:.2f} & {time_std:.2f} \\\\\n"
    
    latex_hyperparameter_table += r"\midrule" + "\n"
    
    # Add layers effects (top 5)
    for layers in sorted(layer_effects.index)[:5]:
        profit_mean = layer_effects.loc[layers, ('Ex_post_Profit', 'mean')]
        profit_std = layer_effects.loc[layers, ('Ex_post_Profit', 'std')]
        time_mean = layer_effects.loc[layers, ('Processing_Time_Seconds', 'mean')]
        time_std = layer_effects.loc[layers, ('Processing_Time_Seconds', 'std')]
        
        latex_hyperparameter_table += f"Layers & {int(layers)} & {profit_mean:.2f} & {profit_std:.2f} & {time_mean:.2f} & {time_std:.2f} \\\\\n"
    
    latex_hyperparameter_table += r"\midrule" + "\n"
    
    # Add iterations effects (selected values)
    selected_iters = [1, 3, 5, 7, 10]
    for iters in selected_iters:
        if iters in iter_effects.index:
            profit_mean = iter_effects.loc[iters, ('Ex_post_Profit', 'mean')]
            profit_std = iter_effects.loc[iters, ('Ex_post_Profit', 'std')]
            time_mean = iter_effects.loc[iters, ('Processing_Time_Seconds', 'mean')]
            time_std = iter_effects.loc[iters, ('Processing_Time_Seconds', 'std')]
            
            latex_hyperparameter_table += f"Iterations & {int(iters)} & {profit_mean:.2f} & {profit_std:.2f} & {time_mean:.2f} & {time_std:.2f} \\\\\n"
    
    latex_hyperparameter_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save hyperparameter effects table
    with open(tables_dir / 'dfl_hyperparameter_effects.tex', 'w') as f:
        f.write(latex_hyperparameter_table)
    
    # 2. Best Configurations Table
    top_configs = best_by_combo.head(10)
    
    latex_best_configs = r"""\begin{table}[h]
\centering
\caption{Top 10 DFL Configurations by Ex-post Profit}
\label{tab:dfl_best_configurations}
\begin{tabular}{llcccccc}
\toprule
Rank & Database & Architecture & Layers & Iterations & Ex-post Profit (€) & Processing Time (s) & Count \\
\midrule
"""
    
    for i, (_, row) in enumerate(top_configs.iterrows(), 1):
        db_name = row['Database'].replace('euclidean_', '').replace('_', '\\_')
        arch = row['Architecture']
        layers = int(row['Num_Layers'])
        iterations = int(row['Max_Iterations'])
        profit_mean = row['Ex_post_Profit_mean']
        time_mean = row['Processing_Time_Seconds_mean']
        count = int(row['Ex_post_Profit_count'])
        
        latex_best_configs += f"{i} & {db_name} & {arch} & {layers} & {iterations} & {profit_mean:.2f} & {time_mean:.2f} & {count} \\\\\n"
    
    latex_best_configs += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save best configurations table
    with open(tables_dir / 'dfl_best_configurations.tex', 'w') as f:
        f.write(latex_best_configs)
    
    # 3. Statistical Significance Table (if available)
    try:
        significance_results = perform_anova_analysis(df)
        
        latex_significance = r"""\begin{table}[h]
\centering
\caption{Statistical Significance of DFL Hyperparameters (ANOVA)}
\label{tab:dfl_statistical_significance}
\begin{tabular}{lccc}
\toprule
Hyperparameter & F-statistic & p-value & Significant \\
\midrule
"""
        
        for param, (f_stat, p_val) in significance_results.items():
            significance = "Yes" if p_val < 0.05 else "No"
            param_name = param.replace('_', '\\_')
            latex_significance += f"{param_name} & {f_stat:.4f} & {p_val:.4f} & {significance} \\\\\n"
        
        latex_significance += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        # Save significance table
        with open(tables_dir / 'dfl_statistical_significance.tex', 'w') as f:
            f.write(latex_significance)
            
    except Exception as e:
        print(f"Warning: Could not generate statistical significance table: {e}")
    
    print(f"DFL LaTeX tables saved to {tables_dir}")

def perform_anova_analysis(df):
    """Perform ANOVA analysis for hyperparameter significance."""
    from scipy.stats import f_oneway
    
    results = {}
    
    # Database effect
    try:
        db_groups = [df[df['Database'] == db]['Ex_post_Profit'].dropna() 
                    for db in df['Database'].unique()]
        db_groups = [group for group in db_groups if len(group) > 1]
        if len(db_groups) > 1:
            f_stat, p_val = f_oneway(*db_groups)
            results['Database'] = (f_stat, p_val)
    except:
        pass
    
    # Architecture effect
    try:
        arch_groups = [df[df['Architecture'] == arch]['Ex_post_Profit'].dropna()
                      for arch in df['Architecture'].unique()]
        arch_groups = [group for group in arch_groups if len(group) > 1]
        if len(arch_groups) > 1:
            f_stat, p_val = f_oneway(*arch_groups)
            results['Architecture'] = (f_stat, p_val)
    except:
        pass
    
    # Layers effect
    try:
        layer_groups = [df[df['Num_Layers'] == layers]['Ex_post_Profit'].dropna()
                       for layers in df['Num_Layers'].unique()]
        layer_groups = [group for group in layer_groups if len(group) > 1]
        if len(layer_groups) > 1:
            f_stat, p_val = f_oneway(*layer_groups)
            results['Num_Layers'] = (f_stat, p_val)
    except:
        pass
    
    # Iterations effect
    try:
        iter_groups = [df[df['Max_Iterations'] == iters]['Ex_post_Profit'].dropna()
                      for iters in df['Max_Iterations'].unique()]
        iter_groups = [group for group in iter_groups if len(group) > 1]
        if len(iter_groups) > 1:
            f_stat, p_val = f_oneway(*iter_groups)
            results['Max_Iterations'] = (f_stat, p_val)
    except:
        pass
    
    return results

#%% Main execution
def main():
    """Main function for DFL hyperparameter analysis."""
    print("Starting DFL Hyperparameter Analysis...")
    print("="*60)
    
    # Create output directories
    output_dir = create_dfl_directories()
    print(f"Output directories created: {output_dir.absolute()}")
    
    # Load DFL data
    df = load_dfl_data()
    
    if df.empty:
        print("Error: No DFL data loaded. Please check file paths.")
        return
    
    print(f"\nDFL Data loaded successfully:")
    print(f"- Total records: {len(df)}")
    print(f"- DFL Types: {', '.join(df['DFL_Type'].unique())}")
    print(f"- Databases: {', '.join(df['Database'].unique())}")
    print(f"- Architectures: {', '.join(df['Architecture'].unique())}")
    print(f"- Layers range: {df['Num_Layers'].min()}-{df['Num_Layers'].max()}")
    print(f"- Iterations range: {df['Max_Iterations'].min()}-{df['Max_Iterations'].max()}")
    print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Perform analyses
    analyze_hyperparameter_effects(df, output_dir)
    analyze_hyperparameter_interactions(df, output_dir)
    best_configs, best_by_combo, efficiency_ratios = optimize_hyperparameters(df, output_dir)
    generate_dfl_latex_tables(df, best_configs, best_by_combo, output_dir)
    
    print("\n" + "="*60)
    print("DFL HYPERPARAMETER ANALYSIS COMPLETED!")
    print(f"All results saved in: {output_dir.absolute()}")
    print("="*60)
    
    # Print key findings
    print("\n📊 KEY FINDINGS:")
    print("-" * 20)
    
    best_overall = best_by_combo.iloc[0]
    print(f"🏆 Best Overall Configuration:")
    print(f"   Database: {best_overall['Database'].replace('euclidean_', '')}")
    print(f"   Architecture: {best_overall['Architecture']}")
    print(f"   Layers: {int(best_overall['Num_Layers'])}")
    print(f"   Iterations: {int(best_overall['Max_Iterations'])}")
    print(f"   Avg Ex-post Profit: {best_overall['Ex_post_Profit_mean']:.2f}€")
    print(f"   Avg Processing Time: {best_overall['Processing_Time_Seconds_mean']:.2f}s")
    
    # Most efficient configuration
    most_efficient = efficiency_ratios.iloc[0]
    print(f"\n⚡ Most Efficient Configuration:")
    print(f"   Database: {most_efficient['Database'].replace('euclidean_', '')}")
    print(f"   Architecture: {most_efficient['Architecture']}")  
    print(f"   Layers: {int(most_efficient['Num_Layers'])}")
    print(f"   Iterations: {int(most_efficient['Max_Iterations'])}")
    print(f"   Efficiency: {most_efficient['Efficiency']:.2f} €/s")

if __name__ == "__main__":
    main()

# %%
