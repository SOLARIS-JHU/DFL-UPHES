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
    Split the analysis into regular data (excluding 2024-12-12) and anomaly data (only 2024-12-12).
    """
    print("Starting comprehensive hyperparameter analysis...")
    
    # Create output directories
    output_dir_normal = Path("./analysis_results/normal")
    output_dir_normal.mkdir(exist_ok=True, parents=True)
    
    output_dir_anomaly = Path("./analysis_results/anomaly")
    output_dir_anomaly.mkdir(exist_ok=True, parents=True)
    
    # Load the master benchmark file from validation results
    master_file = Path("./validation_results/comprehensive/master_validation_benchmarks.csv")
    
    if not master_file.exists():
        print(f"Error: Master benchmark file not found at {master_file}")
        return
    
    df = pd.read_csv(master_file)
    print(f"Loaded data with {len(df)} rows.")
    
    # Calculate profit efficiency (Simulated/Optimized ratio)
    df['Profit_Efficiency'] = df['Simulated_Profit'] / df['Optimized_Profit'] * 100
    
    # Split the data into normal and anomaly datasets
    anomaly_date = "2024-12-12"
    df_anomaly = df[df['New_Date'] == anomaly_date]
    df_normal = df[df['New_Date'] != anomaly_date]
    
    print(f"Regular dataset: {len(df_normal)} rows")
    print(f"Anomaly dataset (2024-12-12): {len(df_anomaly)} rows")
    
    # Analyze normal data (excluding 2024-12-12)
    print("\n=== ANALYZING REGULAR DATA (EXCLUDING 2024-12-12) ===")
    analyze_databases(df_normal, output_dir_normal)
    analyze_architectures(df_normal, output_dir_normal)
    analyze_best_model(df_normal, output_dir_normal)
    analyze_all_hyperparameters(df_normal, output_dir_normal)
    analyze_performance_metrics(df_normal, output_dir_normal)
    create_visualizations(df_normal, output_dir_normal, "normal")
    
    # Analyze anomaly data (only 2024-12-12)
    print("\n=== ANALYZING ANOMALY DATA (ONLY 2024-12-12) ===")
    if len(df_anomaly) > 0:
        analyze_databases(df_anomaly, output_dir_anomaly)
        analyze_architectures(df_anomaly, output_dir_anomaly)
        analyze_best_model(df_anomaly, output_dir_anomaly)
        analyze_all_hyperparameters(df_anomaly, output_dir_anomaly)
        analyze_performance_metrics(df_anomaly, output_dir_anomaly)
        create_visualizations(df_anomaly, output_dir_anomaly, "anomaly")
    else:
        print("No data found for the anomaly date 2024-12-12")
    
    print(f"Analysis complete. Results saved to {output_dir_normal} and {output_dir_anomaly}")

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

def create_visualizations(df, output_dir, dataset_type=""):
    """Create visualizations to accompany the CSV tables"""
    print(f"Creating visualizations for {dataset_type if dataset_type else 'all'} data...")
    
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create a title suffix to indicate the dataset
    title_suffix = f" - {dataset_type.capitalize()}" if dataset_type else ""
    filename_suffix = f"_{dataset_type}" if dataset_type else ""
    
    # 1. Database comparison bar chart
    plt.figure(figsize=(10, 6))
    db_means = df.groupby('Database')['Simulated_Profit'].mean()
    db_std = df.groupby('Database')['Simulated_Profit'].std()
    
    plt.bar(db_means.index, db_means.values, yerr=db_std.values, capsize=10, alpha=0.7)
    plt.title(f'Average Simulated Profit by Database{title_suffix}', fontsize=14)
    plt.ylabel('Simulated Profit', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(viz_dir / f"database_comparison{filename_suffix}.png", dpi=300)
    plt.close()
    
    # 2. Architecture comparison bar chart
    plt.figure(figsize=(10, 6))
    arch_means = df.groupby('Architecture')['Simulated_Profit'].mean()
    arch_std = df.groupby('Architecture')['Simulated_Profit'].std()
    
    plt.bar(arch_means.index, arch_means.values, yerr=arch_std.values, capsize=10, alpha=0.7)
    plt.title(f'Average Simulated Profit by Architecture{title_suffix}', fontsize=14)
    plt.ylabel('Simulated Profit', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(viz_dir / f"architecture_comparison{filename_suffix}.png", dpi=300)
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
    plt.title(f'Average Simulated Profit by Layers and Iterations{title_suffix}', fontsize=14)
    plt.savefig(viz_dir / f"layers_iterations_heatmap{filename_suffix}.png", dpi=300)
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
                plt.title(f'Simulated Profit: {db} with {arch}{title_suffix}', fontsize=14)
                plt.savefig(viz_dir / f"heatmap_{db}_{arch}{filename_suffix}.png", dpi=300)
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
    plt.title(f'Penalty Breakdown by Database and Architecture{title_suffix}', fontsize=14)
    plt.xticks(x, labels)
    plt.legend()
    plt.savefig(viz_dir / f"penalty_breakdown{filename_suffix}.png", dpi=300)
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
    plt.title(f'Average Processing Time by Configuration (Top 15 Fastest){title_suffix}', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / f"processing_time{filename_suffix}.png", dpi=300)
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
    plt.title(f'Profit Efficiency by Configuration (Top 15){title_suffix}', fontsize=14)
    plt.xlabel('Efficiency (Simulated/Optimized Profit) %', fontsize=12)
    plt.axvline(x=100, color='r', linestyle='--', label='100% Efficiency')
    plt.grid(axis='x', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / f"profit_efficiency{filename_suffix}.png", dpi=300)
    plt.close()
    
    # 8. Comparison of dates (only for normal dataset, not needed for anomaly)
    if dataset_type == "normal":
        # Create a plot showing performance by date
        date_performance = df.groupby('New_Date')['Simulated_Profit'].mean().reset_index()
        date_performance = date_performance.sort_values('New_Date')
        
        plt.figure(figsize=(14, 6))
        plt.plot(date_performance['New_Date'], date_performance['Simulated_Profit'], marker='o')
        plt.title('Average Simulated Profit by Date (Excluding 2024-12-12)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Simulated Profit', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / "profit_by_date.png", dpi=300)
        plt.close()
    
    print(f"All visualizations saved to {viz_dir}")

if __name__ == "__main__":
    generate_comprehensive_analysis()