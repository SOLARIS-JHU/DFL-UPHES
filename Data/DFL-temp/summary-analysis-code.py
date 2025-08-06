# %%
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def generate_comprehensive_analysis():
    """
    Generate detailed CSV tables analyzing the impact of different hyperparameters
    on model performance, focusing on comparing databases, architectures, and
    analyzing effects of layers and iterations.
    """
    print("Starting comprehensive hyperparameter analysis...")
    
    # Create output directory
    output_dir = Path("./analysis_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load the master benchmark file from validation results
    master_file = Path("./validation_results/comprehensive/master_validation_benchmarks.csv")
    
    if not master_file.exists():
        print(f"Error: Master benchmark file not found at {master_file}")
        return
    
    df = pd.read_csv(master_file)
    print(f"Loaded data with {len(df)} rows.")
    
    # Calculate profit efficiency (Simulated/Optimized ratio)
    df['Profit_Efficiency'] = df['Simulated_Profit'] / df['Optimized_Profit'] * 100
    
    # 1. Database Comparison Analysis
    analyze_databases(df, output_dir)
    
    # 2. Architecture Comparison Analysis
    analyze_architectures(df, output_dir)
    
    # 3. Best Model Analysis (Layers and Iterations)
    analyze_best_model(df, output_dir)
    
    # 4. Comprehensive Hyperparameter Analysis
    analyze_all_hyperparameters(df, output_dir)
    
    # 5. Performance Metrics Analysis
    analyze_performance_metrics(df, output_dir)
    
    # 6. Create visualizations
    create_visualizations(df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

def analyze_databases(df, output_dir):
    """Comprehensive analysis of database differences (SOS2 vs GlobalLinear)"""
    print("Analyzing database differences...")
    
    # Define performance metrics to analyze
    metrics = [
        'Simulated_Profit', 'Optimized_Profit', 'SI_Penalty',
        'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
        'Distance_Metric', 'Profit_Efficiency'
    ]
    
    # 1. Basic statistics by database
    db_stats = df.groupby('Database')[metrics].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Flatten multi-level columns
    db_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                       for col in db_stats.columns]
    
    # Save to CSV
    db_stats.to_csv(output_dir / "1_database_comparison.csv", index=False)
    
    # 2. Database performance by architecture
    db_arch_stats = df.groupby(['Database', 'Architecture'])[metrics].agg([
        'mean', 'median', 'std'
    ]).reset_index()
    
    # Flatten multi-level columns
    db_arch_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                            for col in db_arch_stats.columns]
    
    # Save to CSV
    db_arch_stats.to_csv(output_dir / "1a_database_by_architecture.csv", index=False)
    
    # 3. Database performance by number of layers
    db_layers_stats = df.groupby(['Database', 'Num_Layers'])[metrics].agg([
        'mean', 'median', 'std'
    ]).reset_index()
    
    # Flatten multi-level columns
    db_layers_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                              for col in db_layers_stats.columns]
    
    # Save to CSV
    db_layers_stats.to_csv(output_dir / "1b_database_by_layers.csv", index=False)
    
    # 4. Database performance by max iterations
    db_iter_stats = df.groupby(['Database', 'Max_Iterations'])[metrics].agg([
        'mean', 'median', 'std'
    ]).reset_index()
    
    # Flatten multi-level columns
    db_iter_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                            for col in db_iter_stats.columns]
    
    # Save to CSV
    db_iter_stats.to_csv(output_dir / "1c_database_by_iterations.csv", index=False)
    
    # 5. Count of best performances by database
    best_by_date = df.loc[df.groupby('New_Date')['Simulated_Profit'].idxmax()]
    best_counts = best_by_date['Database'].value_counts().reset_index()
    best_counts.columns = ['Database', 'Count_Best_Dates']
    
    # Calculate percentage
    total_dates = len(df['New_Date'].unique())
    best_counts['Percent_Best_Dates'] = best_counts['Count_Best_Dates'] / total_dates * 100
    
    # Save to CSV
    best_counts.to_csv(output_dir / "1d_database_best_counts.csv", index=False)
    
    # 6. Direct comparison with paired data
    # Create a dataframe to store paired comparisons
    paired_data = []
    
    # For each date, compare SOS2 vs GlobalLinear with the same architecture, layers, iterations
    for date in df['New_Date'].unique():
        date_df = df[df['New_Date'] == date]
        
        for arch in date_df['Architecture'].unique():
            for layers in date_df['Num_Layers'].unique():
                for iters in date_df['Max_Iterations'].unique():
                    # Get matching configurations
                    config_df = date_df[
                        (date_df['Architecture'] == arch) & 
                        (date_df['Num_Layers'] == layers) & 
                        (date_df['Max_Iterations'] == iters)
                    ]
                    
                    if len(config_df) == 2:  # We have both databases
                        sos2_row = config_df[config_df['Database'] == 'SOS2'].iloc[0]
                        global_row = config_df[config_df['Database'] == 'GlobalLinear'].iloc[0]
                        
                        # Calculate differences
                        sim_profit_diff = sos2_row['Simulated_Profit'] - global_row['Simulated_Profit']
                        sim_profit_pct = (sim_profit_diff / abs(global_row['Simulated_Profit'])) * 100 if global_row['Simulated_Profit'] != 0 else 0
                        
                        opt_profit_diff = sos2_row['Optimized_Profit'] - global_row['Optimized_Profit']
                        opt_profit_pct = (opt_profit_diff / abs(global_row['Optimized_Profit'])) * 100 if global_row['Optimized_Profit'] != 0 else 0
                        
                        si_penalty_diff = sos2_row['SI_Penalty'] - global_row['SI_Penalty']
                        vol_penalty_diff = sos2_row['Volume_Penalty'] - global_row['Volume_Penalty']
                        op_cost_diff = sos2_row['Operating_Cost'] - global_row['Operating_Cost']
                        
                        # Add to paired data
                        paired_data.append({
                            'Date': date,
                            'Architecture': arch,
                            'Num_Layers': layers,
                            'Max_Iterations': iters,
                            'SOS2_Simulated_Profit': sos2_row['Simulated_Profit'],
                            'GlobalLinear_Simulated_Profit': global_row['Simulated_Profit'],
                            'Simulated_Profit_Diff': sim_profit_diff,
                            'Simulated_Profit_Pct_Diff': sim_profit_pct,
                            'SOS2_Optimized_Profit': sos2_row['Optimized_Profit'],
                            'GlobalLinear_Optimized_Profit': global_row['Optimized_Profit'],
                            'Optimized_Profit_Diff': opt_profit_diff,
                            'Optimized_Profit_Pct_Diff': opt_profit_pct,
                            'SOS2_SI_Penalty': sos2_row['SI_Penalty'],
                            'GlobalLinear_SI_Penalty': global_row['SI_Penalty'],
                            'SI_Penalty_Diff': si_penalty_diff,
                            'SOS2_Volume_Penalty': sos2_row['Volume_Penalty'],
                            'GlobalLinear_Volume_Penalty': global_row['Volume_Penalty'],
                            'Volume_Penalty_Diff': vol_penalty_diff,
                            'SOS2_Operating_Cost': sos2_row['Operating_Cost'],
                            'GlobalLinear_Operating_Cost': global_row['Operating_Cost'],
                            'Operating_Cost_Diff': op_cost_diff,
                            'SOS2_Processing_Time': sos2_row['Processing_Time_Seconds'],
                            'GlobalLinear_Processing_Time': global_row['Processing_Time_Seconds'],
                            'Processing_Time_Diff': sos2_row['Processing_Time_Seconds'] - global_row['Processing_Time_Seconds']
                        })
    
    # Convert to DataFrame
    paired_df = pd.DataFrame(paired_data)
    
    if not paired_df.empty:
        # Save detailed paired data
        paired_df.to_csv(output_dir / "1e_database_paired_comparison.csv", index=False)
        
        # Create summary of paired comparisons - FIX: Use a list for multiple columns
        paired_summary = paired_df.groupby(['Architecture', 'Num_Layers', 'Max_Iterations'])[
            ['Simulated_Profit_Diff', 'Simulated_Profit_Pct_Diff',
            'Optimized_Profit_Diff', 'Optimized_Profit_Pct_Diff',
            'SI_Penalty_Diff', 'Volume_Penalty_Diff', 'Operating_Cost_Diff',
            'Processing_Time_Diff']
        ].agg(['mean', 'std', 'count']).reset_index()
        
        # Flatten multi-level columns
        paired_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                for col in paired_summary.columns]
        
        # Save summary
        paired_summary.to_csv(output_dir / "1f_database_paired_summary.csv", index=False)
    else:
        print("Warning: No paired data available for direct database comparison")

def analyze_architectures(df, output_dir):
    """Comprehensive analysis of architecture differences (LSTM vs RNN)"""
    print("Analyzing architecture differences...")
    
    # Define performance metrics to analyze
    metrics = [
        'Simulated_Profit', 'Optimized_Profit', 'SI_Penalty',
        'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
        'Distance_Metric', 'Profit_Efficiency'
    ]
    
    # 1. Basic statistics by architecture
    arch_stats = df.groupby('Architecture')[metrics].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Flatten multi-level columns
    arch_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                         for col in arch_stats.columns]
    
    # Save to CSV
    arch_stats.to_csv(output_dir / "2_architecture_comparison.csv", index=False)
    
    # 2. Architecture performance by database
    arch_db_stats = df.groupby(['Architecture', 'Database'])[metrics].agg([
        'mean', 'median', 'std'
    ]).reset_index()
    
    # Flatten multi-level columns
    arch_db_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                            for col in arch_db_stats.columns]
    
    # Save to CSV
    arch_db_stats.to_csv(output_dir / "2a_architecture_by_database.csv", index=False)
    
    # 3. Architecture performance by number of layers
    arch_layers_stats = df.groupby(['Architecture', 'Num_Layers'])[metrics].agg([
        'mean', 'median', 'std'
    ]).reset_index()
    
    # Flatten multi-level columns
    arch_layers_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                for col in arch_layers_stats.columns]
    
    # Save to CSV
    arch_layers_stats.to_csv(output_dir / "2b_architecture_by_layers.csv", index=False)
    
    # 4. Architecture performance by max iterations
    arch_iter_stats = df.groupby(['Architecture', 'Max_Iterations'])[metrics].agg([
        'mean', 'median', 'std'
    ]).reset_index()
    
    # Flatten multi-level columns
    arch_iter_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                              for col in arch_iter_stats.columns]
    
    # Save to CSV
    arch_iter_stats.to_csv(output_dir / "2c_architecture_by_iterations.csv", index=False)
    
    # 5. Count of best performances by architecture
    best_by_date = df.loc[df.groupby('New_Date')['Simulated_Profit'].idxmax()]
    best_counts = best_by_date['Architecture'].value_counts().reset_index()
    best_counts.columns = ['Architecture', 'Count_Best_Dates']
    
    # Calculate percentage
    total_dates = len(df['New_Date'].unique())
    best_counts['Percent_Best_Dates'] = best_counts['Count_Best_Dates'] / total_dates * 100
    
    # Save to CSV
    best_counts.to_csv(output_dir / "2d_architecture_best_counts.csv", index=False)
    
    # 6. Direct comparison with paired data
    # Create a dataframe to store paired comparisons
    paired_data = []
    
    # For each date, compare LSTM vs RNN with the same database, layers, iterations
    for date in df['New_Date'].unique():
        date_df = df[df['New_Date'] == date]
        
        for db in date_df['Database'].unique():
            for layers in date_df['Num_Layers'].unique():
                for iters in date_df['Max_Iterations'].unique():
                    # Get matching configurations
                    config_df = date_df[
                        (date_df['Database'] == db) & 
                        (date_df['Num_Layers'] == layers) & 
                        (date_df['Max_Iterations'] == iters)
                    ]
                    
                    if len(config_df) == 2:  # We have both architectures
                        lstm_row = config_df[config_df['Architecture'] == 'LSTM'].iloc[0]
                        rnn_row = config_df[config_df['Architecture'] == 'RNN'].iloc[0]
                        
                        # Calculate differences
                        sim_profit_diff = lstm_row['Simulated_Profit'] - rnn_row['Simulated_Profit']
                        sim_profit_pct = (sim_profit_diff / abs(rnn_row['Simulated_Profit'])) * 100 if rnn_row['Simulated_Profit'] != 0 else 0
                        
                        opt_profit_diff = lstm_row['Optimized_Profit'] - rnn_row['Optimized_Profit']
                        opt_profit_pct = (opt_profit_diff / abs(rnn_row['Optimized_Profit'])) * 100 if rnn_row['Optimized_Profit'] != 0 else 0
                        
                        si_penalty_diff = lstm_row['SI_Penalty'] - rnn_row['SI_Penalty']
                        vol_penalty_diff = lstm_row['Volume_Penalty'] - rnn_row['Volume_Penalty']
                        op_cost_diff = lstm_row['Operating_Cost'] - rnn_row['Operating_Cost']
                        
                        # Add to paired data
                        paired_data.append({
                            'Date': date,
                            'Database': db,
                            'Num_Layers': layers,
                            'Max_Iterations': iters,
                            'LSTM_Simulated_Profit': lstm_row['Simulated_Profit'],
                            'RNN_Simulated_Profit': rnn_row['Simulated_Profit'],
                            'Simulated_Profit_Diff': sim_profit_diff,
                            'Simulated_Profit_Pct_Diff': sim_profit_pct,
                            'LSTM_Optimized_Profit': lstm_row['Optimized_Profit'],
                            'RNN_Optimized_Profit': rnn_row['Optimized_Profit'],
                            'Optimized_Profit_Diff': opt_profit_diff,
                            'Optimized_Profit_Pct_Diff': opt_profit_pct,
                            'LSTM_SI_Penalty': lstm_row['SI_Penalty'],
                            'RNN_SI_Penalty': rnn_row['SI_Penalty'],
                            'SI_Penalty_Diff': si_penalty_diff,
                            'LSTM_Volume_Penalty': lstm_row['Volume_Penalty'],
                            'RNN_Volume_Penalty': rnn_row['Volume_Penalty'],
                            'Volume_Penalty_Diff': vol_penalty_diff,
                            'LSTM_Operating_Cost': lstm_row['Operating_Cost'],
                            'RNN_Operating_Cost': rnn_row['Operating_Cost'],
                            'Operating_Cost_Diff': op_cost_diff,
                            'LSTM_Processing_Time': lstm_row['Processing_Time_Seconds'],
                            'RNN_Processing_Time': rnn_row['Processing_Time_Seconds'],
                            'Processing_Time_Diff': lstm_row['Processing_Time_Seconds'] - rnn_row['Processing_Time_Seconds']
                        })
    
    # Convert to DataFrame
    paired_df = pd.DataFrame(paired_data)
    
    if not paired_df.empty:
        # Save detailed paired data
        paired_df.to_csv(output_dir / "2e_architecture_paired_comparison.csv", index=False)
        
        # Create summary of paired comparisons - FIX: Use a list for multiple columns
        paired_summary = paired_df.groupby(['Database', 'Num_Layers', 'Max_Iterations'])[
            ['Simulated_Profit_Diff', 'Simulated_Profit_Pct_Diff',
            'Optimized_Profit_Diff', 'Optimized_Profit_Pct_Diff',
            'SI_Penalty_Diff', 'Volume_Penalty_Diff', 'Operating_Cost_Diff',
            'Processing_Time_Diff']
        ].agg(['mean', 'std', 'count']).reset_index()
        
        # Flatten multi-level columns
        paired_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                for col in paired_summary.columns]
        
        # Save summary
        paired_summary.to_csv(output_dir / "2f_architecture_paired_summary.csv", index=False)
    else:
        print("Warning: No paired data available for direct architecture comparison")

def analyze_best_model(df, output_dir):
    """Analyze the impact of layers and iterations on the best model configuration"""
    print("Analyzing best model configuration...")
    
    # First, determine the best database and architecture based on average simulated profit
    best_avg = df.groupby(['Database', 'Architecture'])['Simulated_Profit'].mean().reset_index()
    best_config = best_avg.loc[best_avg['Simulated_Profit'].idxmax()]
    
    best_db = best_config['Database']
    best_arch = best_config['Architecture']
    
    print(f"Best model: {best_db} database with {best_arch} architecture")
    
    # Filter for the best database and architecture
    best_model_df = df[
        (df['Database'] == best_db) & 
        (df['Architecture'] == best_arch)
    ]
    
    # Define metrics to analyze
    metrics = [
        'Simulated_Profit', 'Optimized_Profit', 'SI_Penalty',
        'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
        'Profit_Efficiency'
    ]
    
    # 1. Impact of layers and iterations on the best model
    # First, create pivot tables for each metric
    for metric in metrics:
        pivot_data = best_model_df.pivot_table(
            values=metric,
            index='Num_Layers',
            columns='Max_Iterations',
            aggfunc='mean'
        ).reset_index()
        
        # Save to CSV
        pivot_data.to_csv(output_dir / f"3_{metric}_by_layers_iterations.csv", index=False)
    
    # 2. More detailed analysis with statistics
    layers_iter_stats = best_model_df.groupby(['Num_Layers', 'Max_Iterations'])[metrics].agg([
        'mean', 'median', 'std', 'count'
    ]).reset_index()
    
    # Flatten multi-level columns
    layers_iter_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                               for col in layers_iter_stats.columns]
    
    # Sort by Simulated_Profit_mean in descending order
    layers_iter_stats = layers_iter_stats.sort_values('Simulated_Profit_mean', ascending=False)
    
    # Save to CSV
    layers_iter_stats.to_csv(output_dir / "3a_layers_iterations_detailed.csv", index=False)
    
    # 3. Analysis of improvement from increasing number of layers
    layer_improvement = []
    
    for iters in best_model_df['Max_Iterations'].unique():
        iter_df = best_model_df[best_model_df['Max_Iterations'] == iters]
        
        # Group by layers and get mean profit
        layer_profits = iter_df.groupby('Num_Layers')['Simulated_Profit'].mean()
        
        # If we have at least two layer values
        if len(layer_profits) >= 2:
            layer_values = sorted(layer_profits.index)
            
            for i in range(1, len(layer_values)):
                prev_layer = layer_values[i-1]
                curr_layer = layer_values[i]
                
                prev_profit = layer_profits[prev_layer]
                curr_profit = layer_profits[curr_layer]
                
                # Calculate improvement
                abs_improvement = curr_profit - prev_profit
                pct_improvement = (abs_improvement / abs(prev_profit)) * 100 if prev_profit != 0 else 0
                
                layer_improvement.append({
                    'Max_Iterations': iters,
                    'From_Layers': prev_layer,
                    'To_Layers': curr_layer,
                    'Profit_Before': prev_profit,
                    'Profit_After': curr_profit,
                    'Absolute_Improvement': abs_improvement,
                    'Percent_Improvement': pct_improvement
                })
    
    # Convert to DataFrame
    layer_improvement_df = pd.DataFrame(layer_improvement)
    
    if not layer_improvement_df.empty:
        # Save to CSV
        layer_improvement_df.to_csv(output_dir / "3b_layer_improvement_analysis.csv", index=False)
    
    # 4. Analysis of improvement from increasing number of iterations
    iter_improvement = []
    
    for layers in best_model_df['Num_Layers'].unique():
        layer_df = best_model_df[best_model_df['Num_Layers'] == layers]
        
        # Group by iterations and get mean profit
        iter_profits = layer_df.groupby('Max_Iterations')['Simulated_Profit'].mean()
        
        # If we have at least two iteration values
        if len(iter_profits) >= 2:
            iter_values = sorted(iter_profits.index)
            
            for i in range(1, len(iter_values)):
                prev_iter = iter_values[i-1]
                curr_iter = iter_values[i]
                
                prev_profit = iter_profits[prev_iter]
                curr_profit = iter_profits[curr_iter]
                
                # Calculate improvement
                abs_improvement = curr_profit - prev_profit
                pct_improvement = (abs_improvement / abs(prev_profit)) * 100 if prev_profit != 0 else 0
                
                iter_improvement.append({
                    'Num_Layers': layers,
                    'From_Iterations': prev_iter,
                    'To_Iterations': curr_iter,
                    'Profit_Before': prev_profit,
                    'Profit_After': curr_profit,
                    'Absolute_Improvement': abs_improvement,
                    'Percent_Improvement': pct_improvement
                })
    
    # Convert to DataFrame
    iter_improvement_df = pd.DataFrame(iter_improvement)
    
    if not iter_improvement_df.empty:
        # Save to CSV
        iter_improvement_df.to_csv(output_dir / "3c_iteration_improvement_analysis.csv", index=False)
    
    # 5. Find optimal number of layers and iterations
    best_config_row = layers_iter_stats.iloc[0]
    optimal_layers = best_config_row['Num_Layers']
    optimal_iters = best_config_row['Max_Iterations']
    
    print(f"Optimal configuration: {optimal_layers} layers, {optimal_iters} iterations")
    
    # Save a summary of the best model
    best_model_summary = {
        'Best_Database': best_db,
        'Best_Architecture': best_arch,
        'Optimal_Layers': int(optimal_layers),
        'Optimal_Iterations': int(optimal_iters),
        'Average_Simulated_Profit': float(best_config_row['Simulated_Profit_mean']),
        'Average_Optimized_Profit': float(best_config_row['Optimized_Profit_mean']),
        'Average_SI_Penalty': float(best_config_row['SI_Penalty_mean']),
        'Average_Volume_Penalty': float(best_config_row['Volume_Penalty_mean']),
        'Average_Operating_Cost': float(best_config_row['Operating_Cost_mean']),
        'Average_Processing_Time': float(best_config_row['Processing_Time_Seconds_mean']),
        'Average_Profit_Efficiency': float(best_config_row['Profit_Efficiency_mean'])
    }
    
    # Convert to DataFrame and save
    pd.DataFrame([best_model_summary]).to_csv(output_dir / "3d_best_model_summary.csv", index=False)

def analyze_all_hyperparameters(df, output_dir):
    """Comprehensive analysis of all hyperparameter combinations"""
    print("Analyzing all hyperparameter combinations...")
    
    # Define metrics to analyze
    metrics = [
        'Simulated_Profit', 'Optimized_Profit', 'SI_Penalty',
        'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
        'Profit_Efficiency'
    ]
    
    # 1. Statistics for all hyperparameter combinations
    all_config_stats = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[metrics].agg([
        'mean', 'median', 'std', 'count'
    ]).reset_index()
    
    # Flatten multi-level columns
    all_config_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                               for col in all_config_stats.columns]
    
    # Sort by simulated profit
    all_config_stats = all_config_stats.sort_values('Simulated_Profit_mean', ascending=False)
    
    # Save to CSV
    all_config_stats.to_csv(output_dir / "4_all_hyperparameter_combinations.csv", index=False)
    
    # 2. Top 10 configurations
    top_10 = all_config_stats.head(10)
    top_10.to_csv(output_dir / "4a_top_10_configurations.csv", index=False)
    
    # 3. Impact of each individual hyperparameter
    
    # 3a. Num_Layers
    layers_impact = df.groupby('Num_Layers')[metrics].agg([
        'mean', 'median', 'std', 'count'
    ]).reset_index()
    
    # Flatten multi-level columns
    layers_impact.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                            for col in layers_impact.columns]
    
    # Save to CSV
    layers_impact.to_csv(output_dir / "4b_layers_impact.csv", index=False)
    
    # 3b. Max_Iterations
    iter_impact = df.groupby('Max_Iterations')[metrics].agg([
        'mean', 'median', 'std', 'count'
    ]).reset_index()
    
    # Flatten multi-level columns
    iter_impact.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in iter_impact.columns]
    
    # Save to CSV
    iter_impact.to_csv(output_dir / "4c_iterations_impact.csv", index=False)
    
    # 4. Best configuration for each date
    best_configs_by_date = []
    
    for date in df['New_Date'].unique():
        date_df = df[df['New_Date'] == date]
        best_idx = date_df['Simulated_Profit'].idxmax()
        best_row = date_df.loc[best_idx]
        
        best_configs_by_date.append({
            'Date': date,
            'Database': best_row['Database'],
            'Architecture': best_row['Architecture'],
            'Num_Layers': best_row['Num_Layers'],
            'Max_Iterations': best_row['Max_Iterations'],
            'Simulated_Profit': best_row['Simulated_Profit'],
            'Optimized_Profit': best_row['Optimized_Profit'],
            'SI_Penalty': best_row['SI_Penalty'],
            'Volume_Penalty': best_row['Volume_Penalty'],
            'Operating_Cost': best_row['Operating_Cost'],
            'Profit_Efficiency': best_row['Profit_Efficiency'],
            'Processing_Time': best_row['Processing_Time_Seconds']
        })
    
    # Convert to DataFrame and save
    best_configs_df = pd.DataFrame(best_configs_by_date)
    best_configs_df.to_csv(output_dir / "4d_best_config_by_date.csv", index=False)
    
    # 5. Configuration frequency among best performers
    config_counts = best_configs_df.groupby(
        ['Database', 'Architecture', 'Num_Layers', 'Max_Iterations']
    ).size().reset_index(name='Count')
    
    # Calculate percentage
    config_counts['Percentage'] = config_counts['Count'] / len(best_configs_df) * 100
    
    # Sort by count
    config_counts = config_counts.sort_values('Count', ascending=False)
    
    # Save to CSV
    config_counts.to_csv(output_dir / "4e_best_config_frequencies.csv", index=False)

def analyze_performance_metrics(df, output_dir):
    """Analyze the relationship between performance metrics"""
    print("Analyzing performance metrics relationships...")
    
    # 1. Correlation between metrics
    metrics = [
        'Simulated_Profit', 'Optimized_Profit', 'SI_Penalty',
        'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
        'Profit_Efficiency', 'Distance_Metric'
    ]
    
    correlation = df[metrics].corr()
    correlation.to_csv(output_dir / "5_metric_correlations.csv")
    
    # 2. Penalty analysis by configuration
    penalty_analysis = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
        ['SI_Penalty', 'Volume_Penalty', 'Operating_Cost']
    ].mean().reset_index()
    
    # Calculate total penalty
    penalty_analysis['Total_Penalty'] = (
        penalty_analysis['SI_Penalty'] + 
        penalty_analysis['Volume_Penalty'] + 
        penalty_analysis['Operating_Cost']
    )
    
    # Calculate penalty percentages
    for col in ['SI_Penalty', 'Volume_Penalty', 'Operating_Cost']:
        penalty_analysis[f'{col}_Percent'] = penalty_analysis[col] / penalty_analysis['Total_Penalty'] * 100
    
    # Sort by total penalty (ascending)
    penalty_analysis = penalty_analysis.sort_values('Total_Penalty')
    
    # Save to CSV
    penalty_analysis.to_csv(output_dir / "5a_penalty_breakdown.csv", index=False)
    
    # 3. Profit efficiency analysis (Simulated / Optimized)
    efficiency_analysis = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
        ['Simulated_Profit', 'Optimized_Profit', 'Profit_Efficiency']
    ].agg(['mean', 'median', 'std']).reset_index()
    
    # Flatten multi-level columns
    efficiency_analysis.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                 for col in efficiency_analysis.columns]
    
    # Sort by profit efficiency
    efficiency_analysis = efficiency_analysis.sort_values('Profit_Efficiency_mean', ascending=False)
    
    # Save to CSV
    efficiency_analysis.to_csv(output_dir / "5b_profit_efficiency.csv", index=False)
    
    # 4. Processing time analysis
    time_analysis = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
        ['Processing_Time_Seconds']
    ].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
    
    # Flatten multi-level columns
    time_analysis.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                           for col in time_analysis.columns]
    
    # Sort by mean processing time
    time_analysis = time_analysis.sort_values('Processing_Time_Seconds_mean')
    
    # Save to CSV
    time_analysis.to_csv(output_dir / "5c_processing_time.csv", index=False)
    
    # 5. Trade-off between profit and processing time
    tradeoff_analysis = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
        ['Simulated_Profit', 'Processing_Time_Seconds']
    ].mean().reset_index()
    
    # Calculate profit per second
    tradeoff_analysis['Profit_Per_Second'] = tradeoff_analysis['Simulated_Profit'] / tradeoff_analysis['Processing_Time_Seconds']
    
    # Sort by profit per second
    tradeoff_analysis = tradeoff_analysis.sort_values('Profit_Per_Second', ascending=False)
    
    # Save to CSV
    tradeoff_analysis.to_csv(output_dir / "5d_profit_time_tradeoff.csv", index=False)

def create_visualizations(df, output_dir):
    """Create visualizations to accompany the CSV tables"""
    print("Creating visualizations...")
    
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Database comparison bar chart
    plt.figure(figsize=(10, 6))
    db_means = df.groupby('Database')['Simulated_Profit'].mean()
    db_std = df.groupby('Database')['Simulated_Profit'].std()
    
    plt.bar(db_means.index, db_means.values, yerr=db_std.values, capsize=10, alpha=0.7)
    plt.title('Average Simulated Profit by Database', fontsize=14)
    plt.ylabel('Simulated Profit', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(viz_dir / "database_comparison.png", dpi=300)
    plt.close()
    
    # 2. Architecture comparison bar chart
    plt.figure(figsize=(10, 6))
    arch_means = df.groupby('Architecture')['Simulated_Profit'].mean()
    arch_std = df.groupby('Architecture')['Simulated_Profit'].std()
    
    plt.bar(arch_means.index, arch_means.values, yerr=arch_std.values, capsize=10, alpha=0.7)
    plt.title('Average Simulated Profit by Architecture', fontsize=14)
    plt.ylabel('Simulated Profit', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(viz_dir / "architecture_comparison.png", dpi=300)
    plt.close()
    
    # 3. Heatmap of layers vs iterations (for all data)
    plt.figure(figsize=(12, 8))
    layers_iter_pivot = df.pivot_table(
        values='Simulated_Profit',
        index='Num_Layers',
        columns='Max_Iterations',
        aggfunc='mean'
    )
    
    sns.heatmap(layers_iter_pivot, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Average Simulated Profit by Layers and Iterations', fontsize=14)
    plt.savefig(viz_dir / "layers_iterations_heatmap.png", dpi=300)
    plt.close()
    
    # 4. Faceted heatmaps of layers vs iterations by database and architecture
    for db in df['Database'].unique():
        for arch in df['Architecture'].unique():
            db_arch_df = df[(df['Database'] == db) & (df['Architecture'] == arch)]
            
            if len(db_arch_df) > 0:
                plt.figure(figsize=(10, 8))
                pivot = db_arch_df.pivot_table(
                    values='Simulated_Profit',
                    index='Num_Layers',
                    columns='Max_Iterations',
                    aggfunc='mean'
                )
                
                sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.2f')
                plt.title(f'Simulated Profit: {db} with {arch}', fontsize=14)
                plt.savefig(viz_dir / f"heatmap_{db}_{arch}.png", dpi=300)
                plt.close()
    
    # 5. Penalty breakdown
    penalty_pivot = df.groupby(['Database', 'Architecture'])[
        ['SI_Penalty', 'Volume_Penalty', 'Operating_Cost']
    ].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    x = range(len(penalty_pivot))
    width = 0.25
    
    # Create labels for x-axis
    labels = [f"{row['Database']}-{row['Architecture']}" for _, row in penalty_pivot.iterrows()]
    
    plt.bar([i - width for i in x], penalty_pivot['SI_Penalty'], width, label='SI Penalty', color='red', alpha=0.7)
    plt.bar(x, penalty_pivot['Volume_Penalty'], width, label='Volume Penalty', color='blue', alpha=0.7)
    plt.bar([i + width for i in x], penalty_pivot['Operating_Cost'], width, label='Operating Cost', color='green', alpha=0.7)
    
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Average Penalty', fontsize=12)
    plt.title('Penalty Breakdown by Database and Architecture', fontsize=14)
    plt.xticks(x, labels)
    plt.legend()
    plt.savefig(viz_dir / "penalty_breakdown.png", dpi=300)
    plt.close()
    
    # 6. Processing time by configuration
    time_data = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
        'Processing_Time_Seconds'
    ].mean().reset_index()
    
    # Create configuration label
    time_data['Config'] = time_data.apply(
        lambda x: f"{x['Database']}-{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter",
        axis=1
    )
    
    # Sort by processing time
    time_data = time_data.sort_values('Processing_Time_Seconds')
    # Take top 15 for readability
    time_data = time_data.head(15)
    
    plt.figure(figsize=(14, 8))
    plt.barh(time_data['Config'], time_data['Processing_Time_Seconds'], alpha=0.7)
    plt.title('Average Processing Time by Configuration (Top 15 Fastest)', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "processing_time.png", dpi=300)
    plt.close()
    
    # 7. Profit efficiency by configuration
    efficiency_data = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
        'Profit_Efficiency'
    ].mean().reset_index()
    
    efficiency_data['Config'] = efficiency_data.apply(
        lambda x: f"{x['Database']}-{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter",
        axis=1
    )
    
    # Sort by efficiency and get top 15
    efficiency_data = efficiency_data.sort_values('Profit_Efficiency', ascending=False).head(15)
    
    plt.figure(figsize=(14, 8))
    plt.barh(efficiency_data['Config'], efficiency_data['Profit_Efficiency'], alpha=0.7)
    plt.title('Profit Efficiency by Configuration (Top 15)', fontsize=14)
    plt.xlabel('Efficiency (Simulated/Optimized Profit) %', fontsize=12)
    plt.axvline(x=100, color='r', linestyle='--', label='100% Efficiency')
    plt.grid(axis='x', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / "profit_efficiency.png", dpi=300)
    plt.close()
    
    print(f"All visualizations saved to {viz_dir}")

if __name__ == "__main__":
    generate_comprehensive_analysis()

#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def create_hyperparameter_comparison_plots():
    """
    Generate distribution plots comparing the simulated profit across different hyperparameters:
    1. Database type (SOS2 vs GlobalLinear)
    2. Architecture type (LSTM vs RNN)
    3. Number of layers
    4. Number of iterations
    
    Each plot includes mean and standard deviation in the legend.
    """
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    
    # Create output directory
    output_dir = Path("./analysis_results/hyperparameter_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load master benchmark data
    master_file = Path("./validation_results/comprehensive/master_validation_benchmarks.csv")
    
    if not master_file.exists():
        print(f"Error: Master benchmark file not found at {master_file}")
        return
    
    df = pd.read_csv(master_file)
    print(f"Loaded data with {len(df)} rows.")
    
    # 1. Database Comparison (SOS2 vs GlobalLinear)
    compare_database(df, output_dir)
    
    # 2. Architecture Comparison (LSTM vs RNN)
    compare_architecture(df, output_dir)
    
    # 3. Layers Comparison
    compare_layers(df, output_dir)
    
    # 4. Iterations Comparison
    compare_iterations(df, output_dir)
    
    print(f"All comparison plots and statistics saved to {output_dir}")

def compare_database(df, output_dir):
    """Compare simulated profit between SOS2 and GlobalLinear databases."""
    print("Generating database comparison plots...")
    
    # First make sure we have both database types
    db_types = df['Database'].unique()
    if len(db_types) < 2:
        print(f"Warning: Only found {len(db_types)} database types: {db_types}")
    
    # Filter data for each database
    sos2_data = df[df['Database'] == 'SOS2']['Simulated_Profit']
    global_linear_data = df[df['Database'] == 'GlobalLinear']['Simulated_Profit']
    
    # Calculate statistics
    sos2_stats = {
        'mean': sos2_data.mean(),
        'median': sos2_data.median(),
        'std': sos2_data.std(),
        'min': sos2_data.min(),
        'max': sos2_data.max()
    }
    
    global_linear_stats = {
        'mean': global_linear_data.mean(),
        'median': global_linear_data.median(),
        'std': global_linear_data.std(),
        'min': global_linear_data.min(),
        'max': global_linear_data.max()
    }
    
    # Print statistics
    print("\nDatabase Comparison Statistics:")
    print(f"SOS2: Mean={sos2_stats['mean']:.2f}, Std={sos2_stats['std']:.2f}")
    print(f"GlobalLinear: Mean={global_linear_stats['mean']:.2f}, Std={global_linear_stats['std']:.2f}")
    
    # 1. Create density plot
    plt.figure(figsize=(12, 8))
    
    # Include mean and std in the labels
    sns.kdeplot(sos2_data, fill=True, alpha=0.5, 
               label=f'SOS2 (μ={sos2_stats["mean"]:.2f}, σ={sos2_stats["std"]:.2f})', 
               color='purple')
    sns.kdeplot(global_linear_data, fill=True, alpha=0.5, 
               label=f'GlobalLinear (μ={global_linear_stats["mean"]:.2f}, σ={global_linear_stats["std"]:.2f})', 
               color='orange')
    
    plt.xlabel('Simulated Profit (€)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Distribution of Simulated Profit by Database Type', fontsize=16)
    plt.legend(title='Database', loc='best', fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add mean lines
    plt.axvline(x=sos2_stats['mean'], color='purple', linestyle='--')
    plt.axvline(x=global_linear_stats['mean'], color='orange', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "database_profit_distribution.png", dpi=300)
    plt.savefig(output_dir / "database_profit_distribution.pdf")
    plt.close()
    
    # 2. Create box plot
    plt.figure(figsize=(10, 8))
    
    # Create a dataframe for seaborn boxplot
    plot_data = pd.DataFrame({
        'Database': ['SOS2']*len(sos2_data) + ['GlobalLinear']*len(global_linear_data),
        'Simulated Profit': pd.concat([sos2_data, global_linear_data]).values
    })
    
    ax = sns.boxplot(x='Database', y='Simulated Profit', data=plot_data, palette=['purple', 'orange'])
    sns.stripplot(x='Database', y='Simulated Profit', data=plot_data, 
                  color='black', alpha=0.5, jitter=True)
    
    # Add stats to title and labels
    plt.title('Simulated Profit by Database Type', fontsize=16)
    plt.xlabel('Database', fontsize=14)
    plt.ylabel('Simulated Profit (€)', fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistics as text annotations
    stats_text = ""
    for i, db in enumerate(['SOS2', 'GlobalLinear']):
        stats = sos2_stats if db == 'SOS2' else global_linear_stats
        plt.annotate(f'μ={stats["mean"]:.2f}, σ={stats["std"]:.2f}',
                    xy=(i, stats['median']), 
                    xytext=(i, plot_data['Simulated Profit'].min() - 100),
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "database_profit_boxplot.png", dpi=300)
    plt.savefig(output_dir / "database_profit_boxplot.pdf")
    plt.close()
    
    # 3. Save statistics to CSV
    stats_df = pd.DataFrame({
        'Database': ['SOS2', 'GlobalLinear'],
        'Mean': [sos2_stats['mean'], global_linear_stats['mean']],
        'Median': [sos2_stats['median'], global_linear_stats['median']],
        'Std': [sos2_stats['std'], global_linear_stats['std']],
        'Min': [sos2_stats['min'], global_linear_stats['min']],
        'Max': [sos2_stats['max'], global_linear_stats['max']]
    })
    
    stats_df.to_csv(output_dir / "database_profit_statistics.csv", index=False)

def compare_architecture(df, output_dir):
    """Compare simulated profit between LSTM and RNN architectures."""
    print("Generating architecture comparison plots...")
    
    # First make sure we have both architecture types
    arch_types = df['Architecture'].unique()
    if len(arch_types) < 2:
        print(f"Warning: Only found {len(arch_types)} architecture types: {arch_types}")
    
    # Filter data for each architecture
    lstm_data = df[df['Architecture'] == 'LSTM']['Simulated_Profit']
    rnn_data = df[df['Architecture'] == 'RNN']['Simulated_Profit']
    
    # Calculate statistics
    lstm_stats = {
        'mean': lstm_data.mean(),
        'median': lstm_data.median(),
        'std': lstm_data.std(),
        'min': lstm_data.min(),
        'max': lstm_data.max()
    }
    
    rnn_stats = {
        'mean': rnn_data.mean(),
        'median': rnn_data.median(),
        'std': rnn_data.std(),
        'min': rnn_data.min(),
        'max': rnn_data.max()
    }
    
    # Print statistics
    print("\nArchitecture Comparison Statistics:")
    print(f"LSTM: Mean={lstm_stats['mean']:.2f}, Std={lstm_stats['std']:.2f}")
    print(f"RNN: Mean={rnn_stats['mean']:.2f}, Std={rnn_stats['std']:.2f}")
    
    # 1. Create density plot
    plt.figure(figsize=(12, 8))
    
    # Include mean and std in the labels
    sns.kdeplot(lstm_data, fill=True, alpha=0.5, 
               label=f'LSTM (μ={lstm_stats["mean"]:.2f}, σ={lstm_stats["std"]:.2f})', 
               color='blue')
    sns.kdeplot(rnn_data, fill=True, alpha=0.5, 
               label=f'RNN (μ={rnn_stats["mean"]:.2f}, σ={rnn_stats["std"]:.2f})', 
               color='green')
    
    plt.xlabel('Simulated Profit (€)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Distribution of Simulated Profit by Architecture Type', fontsize=16)
    plt.legend(title='Architecture', loc='best', fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add mean lines
    plt.axvline(x=lstm_stats['mean'], color='blue', linestyle='--')
    plt.axvline(x=rnn_stats['mean'], color='green', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "architecture_profit_distribution.png", dpi=300)
    plt.savefig(output_dir / "architecture_profit_distribution.pdf")
    plt.close()
    
    # 2. Create box plot
    plt.figure(figsize=(10, 8))
    
    # Create a dataframe for seaborn boxplot
    plot_data = pd.DataFrame({
        'Architecture': ['LSTM']*len(lstm_data) + ['RNN']*len(rnn_data),
        'Simulated Profit': pd.concat([lstm_data, rnn_data]).values
    })
    
    ax = sns.boxplot(x='Architecture', y='Simulated Profit', data=plot_data, palette=['blue', 'green'])
    sns.stripplot(x='Architecture', y='Simulated Profit', data=plot_data, 
                  color='black', alpha=0.5, jitter=True)
    
    plt.title('Simulated Profit by Architecture Type', fontsize=16)
    plt.xlabel('Architecture', fontsize=14)
    plt.ylabel('Simulated Profit (€)', fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistics as text annotations
    for i, arch in enumerate(['LSTM', 'RNN']):
        stats = lstm_stats if arch == 'LSTM' else rnn_stats
        plt.annotate(f'μ={stats["mean"]:.2f}, σ={stats["std"]:.2f}',
                    xy=(i, stats['median']), 
                    xytext=(i, plot_data['Simulated Profit'].min() - 100),
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "architecture_profit_boxplot.png", dpi=300)
    plt.savefig(output_dir / "architecture_profit_boxplot.pdf")
    plt.close()
    
    # 3. Save statistics to CSV
    stats_df = pd.DataFrame({
        'Architecture': ['LSTM', 'RNN'],
        'Mean': [lstm_stats['mean'], rnn_stats['mean']],
        'Median': [lstm_stats['median'], rnn_stats['median']],
        'Std': [lstm_stats['std'], rnn_stats['std']],
        'Min': [lstm_stats['min'], rnn_stats['min']],
        'Max': [lstm_stats['max'], rnn_stats['max']]
    })
    
    stats_df.to_csv(output_dir / "architecture_profit_statistics.csv", index=False)

def compare_layers(df, output_dir):
    """Compare simulated profit across different numbers of layers."""
    print("Generating layers comparison plots...")
    
    # Get unique layer counts
    layer_counts = sorted(df['Num_Layers'].unique())
    print(f"Found layer counts: {layer_counts}")
    
    # Set up colors for different layers
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    colors = colors[:len(layer_counts)]  # Slice to match number of layers
    
    # 1. Create density plot
    plt.figure(figsize=(12, 8))
    
    # Dictionary to store statistics for each layer count
    layer_stats = {}
    
    # Create KDE plot for each layer count
    for i, layers in enumerate(layer_counts):
        layer_data = df[df['Num_Layers'] == layers]['Simulated_Profit']
        
        # Calculate statistics
        layer_stats[layers] = {
            'mean': layer_data.mean(),
            'median': layer_data.median(),
            'std': layer_data.std(),
            'min': layer_data.min(),
            'max': layer_data.max()
        }
        
        # Print statistics
        print(f"\nLayers={layers} Statistics:")
        print(f"Mean={layer_stats[layers]['mean']:.2f}, Std={layer_stats[layers]['std']:.2f}")
        
        # Plot KDE with mean and std in the label
        sns.kdeplot(layer_data, fill=True, alpha=0.4, 
                   label=f'{layers} Layer(s) (μ={layer_stats[layers]["mean"]:.2f}, σ={layer_stats[layers]["std"]:.2f})', 
                   color=colors[i])
        
        # Add mean line
        plt.axvline(x=layer_stats[layers]['mean'], color=colors[i], linestyle='--')
    
    plt.xlabel('Simulated Profit (€)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Distribution of Simulated Profit by Number of Layers', fontsize=16)
    plt.legend(title='Layers', loc='best', fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "layers_profit_distribution.png", dpi=300)
    plt.savefig(output_dir / "layers_profit_distribution.pdf")
    plt.close()
    
    # 2. Create box plot
    plt.figure(figsize=(12, 8))
    
    # Create a dataframe for seaborn boxplot
    plot_data_list = []
    for layers in layer_counts:
        layer_data = df[df['Num_Layers'] == layers]['Simulated_Profit']
        plot_data_list.append(pd.DataFrame({
            'Layers': [f'{layers} Layer(s)']*len(layer_data),
            'Simulated Profit': layer_data.values
        }))
    
    plot_data = pd.concat(plot_data_list)
    
    ax = sns.boxplot(x='Layers', y='Simulated Profit', data=plot_data, palette=colors[:len(layer_counts)])
    sns.stripplot(x='Layers', y='Simulated Profit', data=plot_data, 
                  color='black', alpha=0.5, jitter=True)
    
    plt.title('Simulated Profit by Number of Layers', fontsize=16)
    plt.xlabel('Number of Layers', fontsize=14)
    plt.ylabel('Simulated Profit (€)', fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistics as text annotations
    for i, layers in enumerate(layer_counts):
        stats = layer_stats[layers]
        plt.annotate(f'μ={stats["mean"]:.2f}, σ={stats["std"]:.2f}',
                    xy=(i, stats['median']), 
                    xytext=(i, plot_data['Simulated Profit'].min() - 100),
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "layers_profit_boxplot.png", dpi=300)
    plt.savefig(output_dir / "layers_profit_boxplot.pdf")
    plt.close()
    
    # 3. Save statistics to CSV
    stats_data = {
        'Layers': [f'{l} Layer(s)' for l in layer_counts],
        'Mean': [layer_stats[l]['mean'] for l in layer_counts],
        'Median': [layer_stats[l]['median'] for l in layer_counts],
        'Std': [layer_stats[l]['std'] for l in layer_counts],
        'Min': [layer_stats[l]['min'] for l in layer_counts],
        'Max': [layer_stats[l]['max'] for l in layer_counts]
    }
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_dir / "layers_profit_statistics.csv", index=False)

def compare_iterations(df, output_dir):
    """Compare simulated profit across different numbers of iterations."""
    print("Generating iterations comparison plots...")
    
    # Get unique iteration counts
    iter_counts = sorted(df['Max_Iterations'].unique())
    print(f"Found iteration counts: {iter_counts}")
    
    # Set up colors for different iterations
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    colors = colors[:len(iter_counts)]  # Slice to match number of iterations
    
    # 1. Create density plot
    plt.figure(figsize=(12, 8))
    
    # Dictionary to store statistics for each iteration count
    iter_stats = {}
    
    # Create KDE plot for each iteration count
    for i, iters in enumerate(iter_counts):
        iter_data = df[df['Max_Iterations'] == iters]['Simulated_Profit']
        
        # Calculate statistics
        iter_stats[iters] = {
            'mean': iter_data.mean(),
            'median': iter_data.median(),
            'std': iter_data.std(),
            'min': iter_data.min(),
            'max': iter_data.max()
        }
        
        # Print statistics
        print(f"\nIterations={iters} Statistics:")
        print(f"Mean={iter_stats[iters]['mean']:.2f}, Std={iter_stats[iters]['std']:.2f}")
        
        # Plot KDE with mean and std in the label
        sns.kdeplot(iter_data, fill=True, alpha=0.4, 
                   label=f'{iters} Iterations (μ={iter_stats[iters]["mean"]:.2f}, σ={iter_stats[iters]["std"]:.2f})', 
                   color=colors[i])
        
        # Add mean line
        plt.axvline(x=iter_stats[iters]['mean'], color=colors[i], linestyle='--')
    
    plt.xlabel('Simulated Profit (€)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Distribution of Simulated Profit by Number of Iterations', fontsize=16)
    plt.legend(title='Iterations', loc='best', fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "iterations_profit_distribution.png", dpi=300)
    plt.savefig(output_dir / "iterations_profit_distribution.pdf")
    plt.close()
    
    # 2. Create box plot
    plt.figure(figsize=(12, 8))
    
    # Create a dataframe for seaborn boxplot
    plot_data_list = []
    for iters in iter_counts:
        iter_data = df[df['Max_Iterations'] == iters]['Simulated_Profit']
        plot_data_list.append(pd.DataFrame({
            'Iterations': [f'{iters} Iterations']*len(iter_data),
            'Simulated Profit': iter_data.values
        }))
    
    plot_data = pd.concat(plot_data_list)
    
    ax = sns.boxplot(x='Iterations', y='Simulated Profit', data=plot_data, palette=colors[:len(iter_counts)])
    sns.stripplot(x='Iterations', y='Simulated Profit', data=plot_data, 
                  color='black', alpha=0.5, jitter=True)
    
    plt.title('Simulated Profit by Number of Iterations', fontsize=16)
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('Simulated Profit (€)', fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistics as text annotations
    for i, iters in enumerate(iter_counts):
        stats = iter_stats[iters]
        plt.annotate(f'μ={stats["mean"]:.2f}, σ={stats["std"]:.2f}',
                    xy=(i, stats['median']), 
                    xytext=(i, plot_data['Simulated Profit'].min() - 100),
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "iterations_profit_boxplot.png", dpi=300)
    plt.savefig(output_dir / "iterations_profit_boxplot.pdf")
    plt.close()
    
    # 3. Save statistics to CSV
    stats_data = {
        'Iterations': [f'{i} Iterations' for i in iter_counts],
        'Mean': [iter_stats[i]['mean'] for i in iter_counts],
        'Median': [iter_stats[i]['median'] for i in iter_counts],
        'Std': [iter_stats[i]['std'] for i in iter_counts],
        'Min': [iter_stats[i]['min'] for i in iter_counts],
        'Max': [iter_stats[i]['max'] for i in iter_counts]
    }
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_dir / "iterations_profit_statistics.csv", index=False)
    
    # Create a line plot to show profit improvement with increasing iterations
    plt.figure(figsize=(12, 8))
    
    # Aggregate data by iteration count
    iter_means = [iter_stats[i]['mean'] for i in iter_counts]
    iter_stds = [iter_stats[i]['std'] for i in iter_counts]
    
    # Plot line with error bands
    plt.errorbar(iter_counts, iter_means, yerr=iter_stds, fmt='-o', linewidth=2, markersize=10,
                 elinewidth=1, capsize=5, label='Mean with Std Dev')
    
    # Add detailed annotations
    for i, iters in enumerate(iter_counts):
        plt.annotate(f'μ={iter_stats[iters]["mean"]:.2f}\nσ={iter_stats[iters]["std"]:.2f}',
                    xy=(iters, iter_stats[iters]['mean']), 
                    xytext=(iters, iter_stats[iters]['mean'] + 50),
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
    
    plt.title('Effect of Iteration Count on Simulated Profit', fontsize=16)
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('Mean Simulated Profit (€)', fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xticks(iter_counts)
    
    plt.tight_layout()
    plt.savefig(output_dir / "iterations_profit_trend.png", dpi=300)
    plt.savefig(output_dir / "iterations_profit_trend.pdf")
    plt.close()

if __name__ == "__main__":
    create_hyperparameter_comparison_plots()

#%%
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import os

def generate_compact_latex_tables():
    """
    Generate a single LaTeX file containing all four three-line tables
    with small font and narrow margins.
    """
    print("Generating compact LaTeX three-line tables in a single file...")
    
    # Create output directory if it doesn't exist
    output_dir = Path("./analysis_results/latex_tables")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load benchmark data
    global_linear_df = load_global_linear_data()
    nn_mpc_df = load_nn_mpc_data()
    sos2_df = load_sos2_data()
    
    # Load best model data (and identify architecture)
    best_model_df, best_arch = load_best_model_data()
    
    # Start the LaTeX file with small font and narrow margins
    latex_content = [
        "\\documentclass{article}",
        "\\usepackage[margin=0.5in]{geometry}",  # Narrow margins
        "\\usepackage{booktabs}",
        "\\usepackage{siunitx}",
        "\\usepackage{caption}",
        "\\usepackage{array}",
        "",
        "\\begin{document}",
        "\\small  % Set small font size for the entire document",
        "",
        "% Table 1: Global Linear Model",
        create_table(
            global_linear_df, 
            "global_linear_table", 
            "Global Linear Model Results"
        ),
        "",
        "% Table 2: NN-informed MPC Model",
        create_table(
            nn_mpc_df, 
            "nn_mpc_table", 
            "NN-informed MPC Model Results"
        ),
        "",
        "% Table 3: Piecewise-linear SOS2 Model",
        create_table(
            sos2_df, 
            "sos2_table", 
            "Piecewise-linear SOS2 Model Results"
        ),
        "",
        f"% Table 4: Best Recursive Linearization Model ({best_arch})",
        create_table(
            best_model_df, 
            "best_model_table", 
            f"Best Recursive Linearization Model ({best_arch}) Results"
        ),
        "",
        "\\end{document}"
    ]
    
    # Write the LaTeX file
    latex_path = output_dir / "model_comparison_tables_compact.tex"
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"Compact LaTeX file generated: {latex_path}")

def load_global_linear_data():
    """Load and preprocess Global Linear model data."""
    file_path = "./Benchmark/global_linearized_operational_data_2024.csv"
    try:
        df = pd.read_csv(file_path)
        
        # Select relevant columns
        selected_df = pd.DataFrame({
            'Date': df['Date'],
            'Time': df['SolveTime'],
            'Expected Profit': df['ExpectedProfit'],
            'SI Penalty': df['SIPenalty'],
            'Vol Penalty': df['VolumePenalty'],
            'Op Cost': df['OperatingCost'],
            'Ex-post Profit': df['SimProfit']
        })
        
        # Sort by date
        selected_df['Date'] = pd.to_datetime(selected_df['Date'])
        selected_df = selected_df.sort_values('Date')
        selected_df['Date'] = selected_df['Date'].dt.strftime('%Y-%m-%d')
        
        return selected_df
    
    except Exception as e:
        print(f"Error loading Global Linear data: {e}")
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=[
            'Date', 'Time', 'Expected Profit', 'SI Penalty', 
            'Vol Penalty', 'Op Cost', 'Ex-post Profit'
        ])

def load_nn_mpc_data():
    """Load and preprocess NN-informed MPC data."""
    file_path = "./Benchmark/NN-informed-MPC_2024.csv"
    try:
        df = pd.read_csv(file_path)
        
        # Select relevant columns
        selected_df = pd.DataFrame({
            'Date': df['date'],
            'Time': df['solution_time_opt'],
            'Expected Profit': df['objective'],
            'SI Penalty': df['si_penalty'],
            'Vol Penalty': df['vol_penalty'],
            'Op Cost': df['op_cost'],
            'Ex-post Profit': df['profit']
        })
        
        # Sort by date
        selected_df['Date'] = pd.to_datetime(selected_df['Date'])
        selected_df = selected_df.sort_values('Date')
        selected_df['Date'] = selected_df['Date'].dt.strftime('%Y-%m-%d')
        
        return selected_df
    
    except Exception as e:
        print(f"Error loading NN-informed MPC data: {e}")
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=[
            'Date', 'Time', 'Expected Profit', 'SI Penalty', 
            'Vol Penalty', 'Op Cost', 'Ex-post Profit'
        ])

def load_sos2_data():
    """Load and preprocess Piecewise-linear SOS2 data."""
    file_path = "./Benchmark/piecewise_operation_data_SOS2_2024_10seg_bm.csv"
    try:
        df = pd.read_csv(file_path)
        
        # Select relevant columns
        selected_df = pd.DataFrame({
            'Date': df['Date'],
            'Time': df['SolveTime'],
            'Expected Profit': df['ExpectedProfit'],
            'SI Penalty': df['SIPenalty'],
            'Vol Penalty': df['VolumePenalty'],
            'Op Cost': df['OperatingCost'],
            'Ex-post Profit': df['SimProfit']
        })
        
        # Sort by date
        selected_df['Date'] = pd.to_datetime(selected_df['Date'])
        selected_df = selected_df.sort_values('Date')
        selected_df['Date'] = selected_df['Date'].dt.strftime('%Y-%m-%d')
        
        return selected_df
    
    except Exception as e:
        print(f"Error loading Piecewise-linear SOS2 data: {e}")
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=[
            'Date', 'Time', 'Expected Profit', 'SI Penalty', 
            'Vol Penalty', 'Op Cost', 'Ex-post Profit'
        ])

def load_best_model_data():
    """Load best model data for fixed configuration SOS2-RNN with 1 layer and 3 iterations."""
    best_db     = "SOS2"
    best_arch   = "RNN"
    best_layers = 1
    best_iters  = 3
    result_file = Path(f"./validation_results/{best_db}/{best_arch}_{best_layers}layer_{best_iters}iter/scheduling_benchmarks.csv")
    
    if not result_file.exists():
        print(f"Error: Best model result file not found at {result_file}")
        # Return empty DataFrame with expected columns
        cols = ['Date', 'Time', 'Expected Profit', 'SI Penalty', 'Vol Penalty', 'Op Cost', 'Ex-post Profit']
        return pd.DataFrame(columns=cols), f"{best_arch}-{best_layers}L-{best_iters}iter"
    
    df = pd.read_csv(result_file)
    selected_df = pd.DataFrame({
        'Date': df['New_Date'],
        'Time': df['Processing_Time_Seconds'],
        'Expected Profit': df['Optimized_Profit'],
        'SI Penalty': df['SI_Penalty'],
        'Vol Penalty': df['Volume_Penalty'],
        'Op Cost': df['Operating_Cost'],
        'Ex-post Profit': df['Simulated_Profit']
    })
    # Format dates
    selected_df['Date'] = pd.to_datetime(selected_df['Date']).dt.strftime('%Y-%m-%d')
    
    return selected_df, f"{best_arch}-{best_layers}L-{best_iters}iter"

def create_table(df, table_id, caption):
    """Create a LaTeX three-line table from the dataframe."""
    if df.empty:
        return "\\begin{table}\n\\centering\n\\caption{No data available}\n\\end{table}"
    
    # Calculate mean and standard deviation for each numeric column
    numeric_cols = df.columns[1:]  # Skip 'Date' column
    mean_row = df[numeric_cols].mean()
    std_row = df[numeric_cols].std()
    
    # Start building the table
    table_lines = [
        "\\begin{table}[ht]",
        "\\footnotesize  % Use footnotesize for table contents",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{table_id}}}",
        "\\begin{tabular}{l S[table-format=3.2] S[table-format=5.2] S[table-format=3.2] S[table-format=3.2] S[table-format=4.2] S[table-format=5.2]}",
        "\\toprule",
        "Date & {Time (s)} & {Expected Profit (€)} & {SI Penalty (€)} & {Vol Penalty (€)} & {Op Cost (€)} & {Ex-post Profit (€)} \\\\",
        "\\midrule"
    ]
    
    # Add data rows
    for _, row in df.iterrows():
        date_str = row['Date']
        time_val = format_number(row['Time'])
        exp_profit = format_number(row['Expected Profit'])
        si_penalty = format_number(row['SI Penalty'])
        vol_penalty = format_number(row['Vol Penalty'])
        op_cost = format_number(row['Op Cost'])
        expost_profit = format_number(row['Ex-post Profit'])
        
        table_lines.append(f"{date_str} & {time_val} & {exp_profit} & {si_penalty} & {vol_penalty} & {op_cost} & {expost_profit} \\\\")
    
    # Add mean row
    table_lines.append("\\midrule")
    mean_values = [format_number(mean_row[col]) for col in numeric_cols]
    table_lines.append(f"\\textbf{{Mean}} & {' & '.join(mean_values)} \\\\")
    
    # Add std row
    std_values = [format_number(std_row[col]) for col in numeric_cols]
    table_lines.append(f"\\textbf{{Std}} & {' & '.join(std_values)} \\\\")
    
    # Close the table
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{table}")
    
    # Join all lines into a single string
    return '\n'.join(table_lines)

def format_number(num):
    """Format a number with 2 decimal places."""
    try:
        return f"{num:.2f}"
    except:
        return "N/A"

if __name__ == "__main__":
    generate_compact_latex_tables()