# %% Import libraries and setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from joblib import Parallel, delayed
import multiprocessing
import os
import csv
import time
from datetime import datetime
from pathlib import Path
import json
import traceback
import itertools

device = torch.device("cpu")

# load portfolio data
sys.path.append('../Library')
from V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('../preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = torch.tensor(77.0, device=device)  # Initial head value
print(f"Initial head: {head_init.item()}")
v_low_init = torch.tensor(h_to_v_low_fitted(head_init), device=device)  # Initial lower reservoir volume
print(f"Initial head: {head_init.item()}, Initial v_low: {v_low_init.item()}")

def hourly_to_quarterly(tensor_data):
    return tensor_data.repeat_interleave(4)

# Import classes and functions from DFL_pretraining
from DFL_pretraining import (
    HydroParameters, TaylorRegressionLayer, OptiLayer, SimulationLayer,
    BoundedLogWeightPredictor, RecursiveLinearizationPipeline
)

# %% Validation Functions

def load_new_price_data(file_path="../Data/price_data_2024.csv"):
    """
    Load new price data for scheduling validation.
    
    Args:
        file_path: Path to the CSV file with new price data
        
    Returns:
        dict: Dictionary with date strings as keys and price tensors as values
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check column names from the first line
        if 'date' not in df.columns or 'cluster_index' not in df.columns or 'prices_hourly' not in df.columns:
            # Try to handle the case where column headers might be different
            if len(df.columns) >= 3:
                # Assume first column is date, third column has hourly prices
                df.columns = ['date', 'cluster_index', 'prices_hourly']
            else:
                raise ValueError(f"Expected columns 'date', 'cluster_index', 'prices_hourly' but got {df.columns}")
        
        # Dictionary to store price data by date
        price_data = {}
        
        # Process each row
        for _, row in df.iterrows():
            date_str = row['date']
            prices_str = row['prices_hourly']
            
            # Parse the prices (attempting different delimiter formats)
            try:
                # First try splitting by comma
                prices = [float(p) for p in prices_str.split(',')]
            except:
                try:
                    # If that fails, try splitting by semicolon
                    prices = [float(p) for p in prices_str.split(';')]
                except:
                    # If that fails too, try to interpret as a list-like string
                    prices_str = prices_str.strip('[]')
                    prices = [float(p) for p in prices_str.split()]
            
            # Ensure we have 24 hours of data
            if len(prices) != 24:
                print(f"Warning: Date {date_str} has {len(prices)} price values instead of 24")
                # Pad or truncate as needed
                if len(prices) < 24:
                    prices.extend([prices[-1]] * (24 - len(prices)))  # Pad with last value
                else:
                    prices = prices[:24]  # Truncate
            
            # Convert to tensor
            price_tensor = torch.tensor(prices, dtype=torch.float32, device=device)
            
            # Add to dictionary
            price_data[date_str] = price_tensor
        
        print(f"Successfully loaded price data for {len(price_data)} days.")
        return price_data
    
    except Exception as e:
        print(f"Error loading new price data: {e}")
        return None

def load_data_for_validation(file_path, source_name):
    """
    Load historical data for finding similar price profiles.
    
    Args:
        file_path: Path to the data file
        source_name: Name of the source (for logging purposes)
        
    Returns:
        dict: Dictionary with data grouped by date
    """
    try:
        # Read the file
        df = pd.read_csv(file_path, sep=',', header=0)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        print(f"Loading validation data from {source_name}: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
        print(f"First few rows:\n{df.head(3)}")
        
        # Check for required columns
        required_columns = ['date', 'hour', 'power', 'head', 'flow']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check if price column already exists in the file
        if 'price' in df.columns:
            print(f"Using price data from {source_name} file.")
            # Check for missing price values
            if df['price'].isna().any():
                print("Warning: Some price values are missing. Filling with synthetic prices.")
                # Fill missing prices with synthetic data
                missing_mask = df['price'].isna()
                df.loc[missing_mask, 'price'] = 50 + 20 * np.sin(2 * np.pi * df.loc[missing_mask, 'hour'] / 24) + 5 * np.random.randn(missing_mask.sum())
        else:
            print(f"No price column found in {source_name}. Trying to load from original MIQP file...")
            
            # Load original MIQP data to get price information
            original_miqp_file = "../MIQP/MIQP_linear/MILP_global_linear_results.csv"
            if os.path.exists(original_miqp_file):
                print(f"Loading price data from {original_miqp_file}...")
                price_df = pd.read_csv(original_miqp_file)
                price_df.columns = price_df.columns.str.strip()
                
                # Convert date formats to match for merging
                # Handle different date formats
                try:
                    df['date_normalized'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    price_df['date_normalized'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')
                    
                    # Merge price data with validation data on normalized date and hour
                    df = df.merge(price_df[['date_normalized', 'hour', 'price']], 
                                 left_on=['date_normalized', 'hour'], 
                                 right_on=['date_normalized', 'hour'], 
                                 how='left')
                    df.drop('date_normalized', axis=1, inplace=True)
                    
                except Exception as e:
                    print(f"Date format conversion failed: {e}")
                    df['price'] = None
                
                if 'price' not in df.columns or df['price'].isna().all():
                    print("Warning: Price merge failed. Using synthetic prices.")
                    df['price'] = 50 + 20 * np.sin(2 * np.pi * df['hour'] / 24) + 5 * np.random.randn(len(df))
            else:
                print(f"Warning: Original MIQP file {original_miqp_file} not found. Using synthetic prices.")
                # Generate synthetic price data
                df['price'] = 50 + 20 * np.sin(2 * np.pi * df['hour'] / 24) + 5 * np.random.randn(len(df))
        
        # Convert date column - handle different formats
        try:
            df['Date'] = pd.to_datetime(df['date'])
        except:
            # Try different date formats
            try:
                df['Date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            except:
                df['Date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        
        df['Time'] = df['hour']
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'power': 'Power',
            'head': 'Head', 
            'flow': 'Flow',
            'price': 'Price'
        })
        
        # Add 'Mode' column if not present
        if 'Mode' not in df.columns:
            # Determine mode based on power and flow values
            conditions = [
                (abs(df['Power']) < 0.01),  # Idle mode (power close to zero)
                (df['Power'] > 0),          # Turbine mode (positive power)
                (df['Power'] < 0)           # Pump mode (negative power)
            ]
            choices = ['Idle', 'Turbine', 'Pump']
            df['Mode'] = np.select(conditions, choices, default='Unknown')
        
        # Verify price data
        print(f"Price data statistics:")
        print(f"  Min: {df['Price'].min():.2f}")
        print(f"  Max: {df['Price'].max():.2f}")
        print(f"  Mean: {df['Price'].mean():.2f}")
        print(f"  Missing values: {df['Price'].isna().sum()}")
        
        # Group data by date
        data_by_date = {}
        for date, group in df.groupby('Date'):
            # Sort by Time to ensure correct order
            group = group.sort_values('Time')
            
            # Ensure we have 24 hours of data
            if len(group) != 24:
                print(f"Warning: Date {date.strftime('%Y-%m-%d')} has {len(group)} hours instead of 24. Skipping.")
                continue
            
            # Convert date to string format
            date_str = date.strftime('%Y-%m-%d')
            
            # Create dictionary for this date
            date_data = {
                'power': torch.tensor(group['Power'].values, dtype=torch.float32, device=device),
                'head': torch.tensor(group['Head'].values, dtype=torch.float32, device=device),
                'flow': torch.tensor(group['Flow'].values, dtype=torch.float32, device=device),
                'price': torch.tensor(group['Price'].values, dtype=torch.float32, device=device),
                'mode': group['Mode'].values
            }
            
            data_by_date[date_str] = date_data
        
        print(f"Successfully loaded {source_name} data for {len(data_by_date)} days.")
        return data_by_date
    
    except Exception as e:
        print(f"Error loading {source_name} data: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_closest_date(new_price, historical_data):
    """
    Find the date in historical data with the most similar price signal.
    
    Args:
        new_price: Tensor of shape [24] with hourly prices
        historical_data: Dictionary of historical data
        
    Returns:
        str: Date string of the closest match
        float: Distance metric value
    """
    closest_date = None
    min_distance = float('inf')
    
    for date_str, date_data in historical_data.items():
        historical_price = date_data['price'][:24]  # Ensure we use only 24 hours
        
        # Calculate Euclidean distance between price profiles
        distance = torch.norm(new_price - historical_price).item()
        
        if distance < min_distance:
            min_distance = distance
            closest_date = date_str
    
    return closest_date, min_distance

def comprehensive_validation():
    """
    Perform validation across all model configurations from pretraining.
    
    Tests all combinations of:
    - Noise levels: 0.1, 0.2, 0.3, 0.4, 0.5
    - Random samples database
    - Architectures: LSTM
    - Number of layers: 3
    - Max iterations: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    """
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting comprehensive validation at {start_timestamp}...")
    
    # Define noise levels and generate corresponding database sources
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    database_sources = {}
    
    for noise_level in noise_levels:
        source_name = f"MIQP_piecewise_results_relative_noise_{int(noise_level*100)}pct"
        file_path = f"MIQP_piecewise_results_relative_noise_{int(noise_level*100)}pct.csv"
        database_sources[source_name] = {
            'file_path': file_path,
            'data_type': 'relative_noise',
            'noise_level': noise_level
        }
    
    # Add random samples database
    random_samples_source = "MIQP_piecewise_results_random_samples"
    random_samples_file = "MIQP_piecewise_results_random_samples.csv"
    database_sources[random_samples_source] = {
        'file_path': random_samples_file,
        'data_type': 'random_samples',
        'noise_level': None
    }
    
    # Define grid search parameters - simplified as requested
    architectures = ['LSTM']
    num_layers_list = [3]
    max_iterations_list = list(range(1, 11))  # 1 to 10
    
    # Load new price data (common to all validations)
    new_price_data = load_new_price_data()
    if not new_price_data:
        print("Error: Could not load new price data")
        return
    
    # Initialize parameters object (common to all validations)
    params = HydroParameters()
    
    # Create master benchmark file for all configurations
    master_dir = Path("./validation_results/comprehensive")
    master_dir.mkdir(exist_ok=True, parents=True)
    
    master_benchmark_file = master_dir / "master_validation_benchmarks.csv"
    with open(master_benchmark_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Database', 'Data_Type', 'Noise_Level', 'Architecture', 'Num_Layers', 'Max_Iterations',
            'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
            'Expected_Profit', 'Ex_post_Profit', 'SI_Penalty',
            'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
            'Timestamp'
        ])
    
    # Create a tracking dict for best configurations per date
    best_configs = {}
    
    # Total configurations to test
    total_configs = len(database_sources) * len(architectures) * len(num_layers_list) * len(max_iterations_list)
    config_counter = 0
    
    # Iterate through all configurations
    for db_name, arch, num_layers, max_iter in itertools.product(
            database_sources.keys(), architectures, num_layers_list, max_iterations_list):
        
        config_counter += 1
        config_name = f"{arch}_{num_layers}layer_{max_iter}iter"
        
        # Get database info
        db_info = database_sources[db_name]
        data_type = db_info['data_type']
        noise_level = db_info['noise_level']
        
        print(f"\n{'='*80}")
        if data_type == 'random_samples':
            print(f"[{config_counter}/{total_configs}] Validating with configuration: {db_name} (Random Samples)/{config_name}")
        else:
            print(f"[{config_counter}/{total_configs}] Validating with configuration: {db_name} (Noise {int(noise_level*100)}%)/{config_name}")
        print(f"{'='*80}")
        
        # Create output directory for this configuration
        config_dir = Path(f"./validation_results/{db_name}/{config_name}")
        config_dir.mkdir(exist_ok=True, parents=True)
        
        # Create benchmark CSV file for this configuration
        benchmark_file = config_dir / "scheduling_benchmarks.csv"
        with open(benchmark_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
                'Expected_Profit', 'Ex_post_Profit', 'SI_Penalty',
                'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
                'Timestamp'
            ])
        
        try:
            # Load historical data for this database
            historical_data = load_data_for_validation(db_info['file_path'], db_name)
            if not historical_data:
                print(f"Error: Could not load historical data for {db_name}")
                continue
            
            # Initialize layers
            regression_layer = TaylorRegressionLayer(params)
            optimizer_layer = OptiLayer(params)
            
            # Process each new date
            for date_idx, (new_date, new_price) in enumerate(new_price_data.items()):
                print(f"\n[{date_idx+1}/{len(new_price_data)}] Processing date: {new_date} with {db_name}/{config_name}")
                
                # Create directory for this date (replace forward slashes with dashes for filesystem safety)
                safe_date = new_date.replace('/', '-')
                date_dir = config_dir / safe_date
                date_dir.mkdir(exist_ok=True, parents=True)
                
                try:
                    # Start timing
                    start_time = time.time()
                    
                    # 1. Find the closest historical date
                    closest_date, distance = find_closest_date(new_price, historical_data)
                    print(f"Closest historical date: {closest_date} (distance: {distance:.2f})")
                    
                    # 2. Look for the pretrained model at this path
                    model_path = Path(f"./trained_models/{db_name}/{config_name}/{closest_date}/best_model.pt")
                    
                    if not model_path.exists():
                        # Try model.pt if best_model.pt doesn't exist
                        model_path = Path(f"./trained_models/{db_name}/{config_name}/{closest_date}/model.pt")
                        
                        if not model_path.exists():
                            print(f"Warning: No model found at {model_path}. Skipping this date.")
                            continue
                    
                    # 3. Initialize weight network with the configuration
                    weight_network = BoundedLogWeightPredictor(
                        input_size=4,
                        hidden_size=128,  # Fixed hidden size to match training
                        num_layers=num_layers,
                        dropout=0.2,
                        time_horizon=params.time_horizon,
                        archetype=arch,
                        init_w_p=0.6,
                        init_w_q=0.02,
                        init_w_h=0.1,
                        w_p_min=0.1,  
                        w_p_max=3.0,   
                        w_q_min=0.001,
                        w_q_max=0.2,
                        w_h_min=0.01,
                        w_h_max=5.0     
                    ).to(device)
                    
                    # 4. Load the pretrained weights
                    weight_network.load_state_dict(torch.load(model_path, map_location=device))
                    weight_network.eval()
                    
                    # 5. Get the power, head, and flow from the closest date
                    closest_data = historical_data[closest_date]
                    power_init = closest_data['power'][:24].clone()
                    head_init = closest_data['head'][:24].clone()
                    flow_init = predict_q_poly(power_init, head_init)
                    
                    # 6. Create pipeline for recursive linearization
                    pipeline = RecursiveLinearizationPipeline(
                        weight_network, params, optimizer_layer, regression_layer, 
                        {closest_date: closest_data}, max_iterations=max_iter, 
                        penalty_growth_rate=1.5
                    )
                    
                    # 7. Prepare input for weight prediction
                    x = torch.stack([new_price, power_init, flow_init, head_init], dim=1)
                    
                    # 8. Predict weights
                    with torch.no_grad():
                        log_w_p, log_w_q, log_w_h = weight_network(x)
                        w_p = torch.exp(log_w_p)
                        w_q = torch.exp(log_w_q)
                        w_h = torch.exp(log_w_h)
                    
                    # 9. Initialize and run the recursive linearization
                    p_current = power_init.clone().detach()
                    h_current = head_init.clone().detach()
                    flow_current = flow_init.clone().detach()
                    
                    # Track iteration results
                    iter_results = []
                    
                    for iteration in range(max_iter):
                        # Apply growth to weights
                        growth_factor = pipeline.penalty_growth_rate ** iteration
                        w_p_iter = w_p * growth_factor
                        w_q_iter = w_q * growth_factor
                        w_h_iter = w_h * growth_factor
                        
                        # Compute linearization coefficients
                        c, d, e, a, b = regression_layer.run_regression(p_current, h_current, flow_current)
                        
                        # Initialize OptiLayer
                        optimizer_layer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())
                        
                        # Run optimization
                        p_opt, q_opt, h_opt, v_opt, expected_profit, optimized_objective = optimizer_layer.forward(
                            new_price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                            p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                            w_p_iter.cpu(), w_h_iter.cpu(), w_q_iter.cpu()
                        )
                        
                        # Store iteration results
                        iter_results.append({
                            'iteration': iteration,
                            'p_opt': p_opt.detach().cpu().numpy(),
                            'q_opt': q_opt.detach().cpu().numpy(),
                            'h_opt': h_opt.detach().cpu().numpy(),
                            'expected_profit': expected_profit.item()
                        })
                        
                        # Update for next iteration
                        if iteration < max_iter - 1:
                            p_current = p_opt.clone().detach().to(device=power_init.device) 
                            h_current = h_opt.clone().detach().to(device=head_init.device)
                            flow_current = q_opt.clone().detach().to(device=flow_init.device)
                    
                    # 10. Run simulation
                    simulator = SimulationLayer(params)
                    p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(
                        p_opt.to(device), q_opt.to(device), h_opt.to(device)
                    )
                    
                    # 11. Calculate ex-post profit (was simulated profit)
                    ex_post_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
                        p_sim, p_opt.to(device), v_low_sim, new_price.to(device)
                    )
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # 12. Save results
                    results = {
                        'p_opt': p_opt.detach().cpu().numpy(),
                        'q_opt': q_opt.detach().cpu().numpy(),
                        'h_opt': h_opt.detach().cpu().numpy(),
                        'v_opt': v_opt.detach().cpu().numpy(),
                        'p_sim': p_sim.detach().cpu().numpy(),
                        'q_sim': q_sim.detach().cpu().numpy(),
                        'h_sim': h_sim.detach().cpu().numpy(),
                        'v_low_sim': v_low_sim.detach().cpu().numpy(),
                        'new_price': new_price.detach().cpu().numpy(),
                        'closest_price': closest_data['price'][:24].detach().cpu().numpy(),
                        'closest_power': closest_data['power'][:24].detach().cpu().numpy(),
                        'expected_profit': expected_profit.item(),
                        'ex_post_profit': ex_post_profit.item(),
                        'SI_penalty': SI_penalty.item(),
                        'volume_penalty': volume_penalty.item(),
                        'operating_cost': operating_cost.item(),
                        'iter_results': iter_results,
                        'database': db_name,
                        'data_type': data_type,
                        'noise_level': noise_level,
                        'architecture': arch,
                        'num_layers': num_layers,
                        'max_iterations': max_iter
                    }
                    
                    # Save as numpy arrays
                    np.save(date_dir / "results.npy", results)
                    
                    # 13. Generate plots
                    plt.figure(figsize=(18, 12))
                    
                    # Plot price comparison
                    plt.subplot(3, 2, 1)
                    plt.plot(range(24), results['new_price'], 'b-', label='New Price')
                    plt.plot(range(24), results['closest_price'], 'r--', label=f'Closest ({closest_date})')
                    plt.title('Price Comparison')
                    plt.xlabel('Hour')
                    plt.ylabel('Price (EUR/MWh)')
                    plt.legend()
                    plt.grid(True)
                    
                    # Plot power comparison
                    plt.subplot(3, 2, 2)
                    plt.plot(range(24), results['p_opt'], 'g-', label='Optimized Power')
                    plt.plot(range(24), results['p_sim'], 'b-', label='Simulated Power')
                    plt.plot(range(24), results['closest_power'], 'r--', label=f'Historical ({closest_date})')
                    plt.title('Power Schedule')
                    plt.xlabel('Hour')
                    plt.ylabel('Power (MW)')
                    plt.legend()
                    plt.grid(True)
                    
                    # Plot flow
                    plt.subplot(3, 2, 3)
                    plt.plot(range(24), results['q_opt'], 'b-')
                    plt.title('Optimized Flow')
                    plt.xlabel('Hour')
                    plt.ylabel('Flow (m³/s)')
                    plt.grid(True)
                    
                    # Plot head
                    plt.subplot(3, 2, 4)
                    plt.plot(range(24), results['h_opt'], 'g-')
                    plt.title('Optimized Head')
                    plt.xlabel('Hour')
                    plt.ylabel('Head (m)')
                    plt.grid(True)
                    
                    # Plot iteration power evolution
                    plt.subplot(3, 2, 5)
                    sample_hours = [0, 6, 12, 18, 23]
                    for hour in sample_hours:
                        hour_values = [iter_result['p_opt'][hour] for iter_result in iter_results]
                        plt.plot(range(len(hour_values)), hour_values, marker='o', label=f'Hour {hour}')
                    
                    plt.title('Power Evolution Across Iterations')
                    plt.xlabel('Iteration')
                    plt.ylabel('Power (MW)')
                    plt.legend()
                    plt.grid(True)
                    
                    # Add result statistics as text
                    plt.subplot(3, 2, 6)
                    plt.axis('off')
                    
                    # Create configuration description
                    if data_type == 'random_samples':
                        config_desc = f"Configuration: {db_name}\n  Data Type: Random Samples\n  Architecture: {arch}, {num_layers} layers, {max_iter} iterations\n\n"
                    else:
                        config_desc = f"Configuration: {db_name}\n  Noise Level: {int(noise_level*100)}%\n  Architecture: {arch}, {num_layers} layers, {max_iter} iterations\n\n"
                    
                    stats_text = (
                        f"Date: {new_date}\n"
                        f"Closest historical date: {closest_date}\n"
                        f"Distance metric: {distance:.2f}\n\n"
                        + config_desc +
                        f"Expected profit: {expected_profit.item():.2f}\n"
                        f"Ex-post profit: {ex_post_profit.item():.2f}\n"
                        f"SI penalty: {SI_penalty.item():.2f}\n"
                        f"Volume penalty: {volume_penalty.item():.2f}\n"
                        f"Operating cost: {operating_cost.item():.2f}\n\n"
                        f"Processing time: {processing_time:.2f} seconds"
                    )
                    plt.text(0.1, 0.5, stats_text, fontsize=10, va='center')
                    
                    plt.suptitle(f"Validation Results for {new_date} using {db_name}/{config_name}", fontsize=16)
                    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
                    plt.savefig(date_dir / "validation_results.png")
                    plt.close()
                    
                    # 14. Append to configuration benchmark CSV
                    with open(benchmark_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            safe_date, closest_date, f"{distance:.2f}",
                            f"{expected_profit.item():.2f}", f"{ex_post_profit.item():.2f}",
                            f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                            f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                            start_timestamp
                        ])
                    
                    # 15. Append to master benchmark CSV
                    noise_val = f"{noise_level}" if noise_level is not None else 'N/A'
                    with open(master_benchmark_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            db_name, data_type, noise_val, arch, num_layers, max_iter,
                            safe_date, closest_date, f"{distance:.2f}",
                            f"{expected_profit.item():.2f}", f"{ex_post_profit.item():.2f}",
                            f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                            f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                            start_timestamp
                        ])
                    
                    # 16. Track the best configuration for this date
                    config_key = (db_name, arch, num_layers, max_iter)
                    if safe_date not in best_configs:
                        best_configs[safe_date] = {'config': config_key, 'profit': ex_post_profit.item()}
                    elif ex_post_profit.item() > best_configs[safe_date]['profit']:
                        best_configs[safe_date] = {'config': config_key, 'profit': ex_post_profit.item()}
                    
                    print(f"Validation for {new_date} completed:")
                    print(f"  Configuration: {db_name}/{config_name}")
                    print(f"  Processing time: {processing_time:.2f} seconds")
                    print(f"  Expected profit: {expected_profit.item():.2f}")
                    print(f"  Ex-post profit: {ex_post_profit.item():.2f}")
                    print(f"  Results saved to: {date_dir}")
                    
                except Exception as e:
                    print(f"Error processing date {new_date} with {db_name}/{config_name}: {e}")
                    print(traceback.format_exc())
                    
                    # Log the error
                    with open(config_dir / "error_log.txt", 'a') as f:
                        f.write(f"\n[{datetime.now()}] Error processing {new_date}:\n")
                        f.write(traceback.format_exc())
                        f.write("\n" + "-"*50 + "\n")
        
        except Exception as e:
            print(f"Error with configuration {db_name}/{config_name}: {e}")
            print(traceback.format_exc())
            
            # Log the error
            with open(master_dir / "error_log.txt", 'a') as f:
                f.write(f"\n[{datetime.now()}] Error with configuration {db_name}/{config_name}:\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-"*50 + "\n")
    
    # Save best configuration for each date
    with open(master_dir / "best_configurations.json", 'w') as f:
        best_configs_serializable = {}
        for date, data in best_configs.items():
            config = data['config']
            db_info = database_sources[config[0]]
            
            best_configs_serializable[date] = {
                'database': config[0],
                'data_type': db_info['data_type'],
                'noise_level': db_info['noise_level'],
                'architecture': config[1],
                'num_layers': int(config[2]),
                'max_iterations': int(config[3]),
                'profit': float(data['profit'])
            }
        json.dump(best_configs_serializable, f, indent=4)
    
    # Generate comprehensive summary analysis
    generate_comprehensive_summary(master_benchmark_file)
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_duration = datetime.strptime(end_timestamp, "%Y%m%d_%H%M%S") - datetime.strptime(start_timestamp, "%Y%m%d_%H%M%S")
    
    print(f"\nComprehensive validation completed!")
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Total duration: {total_duration}")
    print(f"Master benchmark saved to: {master_benchmark_file}")
    print(f"Best configurations saved to: {master_dir / 'best_configurations.json'}")

def generate_comprehensive_summary(master_benchmark_file):
    """Generate comprehensive analysis of all validation results."""
    try:
        # Import seaborn for better visualization
        try:
            import seaborn as sns
        except ImportError:
            print("Warning: seaborn not found. Some visualizations may be limited.")
            sns = None
        
        # Read master benchmark data
        df = pd.read_csv(master_benchmark_file)
        
        # Create output directory
        summary_dir = Path("./validation_results/comprehensive/summary")
        summary_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Compute average performance by configuration
        avg_by_config = df.groupby(['Database', 'Noise_Level', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
            ['Expected_Profit', 'Ex_post_Profit', 'SI_Penalty', 
             'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds']
        ].mean().reset_index()
        
        # Add a Config column for easier plotting
        avg_by_config['Config'] = avg_by_config.apply(
            lambda x: f"Noise{int(x['Noise_Level']*100)}%-{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", 
            axis=1
        )
        
        # Find the best configuration based on average ex-post profit
        best_config_row = avg_by_config.loc[avg_by_config['Ex_post_Profit'].idxmax()]
        best_config = best_config_row['Config']
        
        # 2. Plot average ex-post profit by configuration
        plt.figure(figsize=(15, 8))
        avg_by_config = avg_by_config.sort_values('Ex_post_Profit', ascending=False)
        bar_colors = ['green' if config == best_config else 'skyblue' for config in avg_by_config['Config']]
        
        plt.bar(avg_by_config['Config'], avg_by_config['Ex_post_Profit'], color=bar_colors)
        plt.title('Average Ex-post Profit by Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Average Ex-post Profit')
        plt.xticks(rotation=90)
        plt.axhline(y=avg_by_config['Ex_post_Profit'].mean(), color='r', linestyle='--', 
                   label=f'Mean: {avg_by_config["Ex_post_Profit"].mean():.2f}')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(summary_dir / "avg_profit_by_config.png")
        plt.close()
        
        # 3. Plot processing time by configuration
        plt.figure(figsize=(15, 8))
        avg_by_config_time = avg_by_config.sort_values('Processing_Time_Seconds')
        
        plt.bar(avg_by_config_time['Config'], avg_by_config_time['Processing_Time_Seconds'], color='orange')
        plt.title('Average Processing Time by Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Average Processing Time (seconds)')
        plt.xticks(rotation=90)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(summary_dir / "avg_processing_time.png")
        plt.close()
        
        # 4. Analysis by noise level
        noise_analysis = df.groupby('Noise_Level')['Ex_post_Profit'].agg(['mean', 'max', 'min', 'std']).reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(noise_analysis['Noise_Level']*100, noise_analysis['mean'], yerr=noise_analysis['std'], 
                capsize=10, color=['blue', 'green', 'orange', 'red', 'purple'])
        plt.title('Average Ex-post Profit by Noise Level')
        plt.xlabel('Noise Level (%)')
        plt.ylabel('Average Ex-post Profit')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(summary_dir / "profit_by_noise_level.png")
        plt.close()
        
        # 5. Effect of max iterations by noise level
        iter_noise_analysis = df.groupby(['Noise_Level', 'Max_Iterations'])['Ex_post_Profit'].mean().reset_index()
        
        # Convert to pivot for heatmap
        pivot_iter_noise = iter_noise_analysis.pivot(
            index='Noise_Level', columns='Max_Iterations', values='Ex_post_Profit'
        )
        
        plt.figure(figsize=(12, 8))
        if sns is not None:
            sns.heatmap(pivot_iter_noise, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        else:
            # Fallback without seaborn
            im = plt.imshow(pivot_iter_noise.values, cmap='viridis', aspect='auto')
            plt.colorbar(im)
            plt.xticks(range(len(pivot_iter_noise.columns)), pivot_iter_noise.columns)
            plt.yticks(range(len(pivot_iter_noise.index)), [f'{x*100:.0f}%' for x in pivot_iter_noise.index])
            plt.xlabel('Max Iterations')
            plt.ylabel('Noise Level')
            
            # Add text annotations
            for i in range(len(pivot_iter_noise.index)):
                for j in range(len(pivot_iter_noise.columns)):
                    plt.text(j, i, f'{pivot_iter_noise.iloc[i, j]:.2f}', 
                            ha='center', va='center', color='white')
        
        plt.title('Average Ex-post Profit by Noise Level and Max Iterations')
        plt.tight_layout()
        plt.savefig(summary_dir / "noise_iterations_heatmap.png")
        plt.close()
        
        # 6. Save summary statistics
        summary_stats = {
            'best_overall_config': {
                'database': best_config_row['Database'],
                'noise_level': float(best_config_row['Noise_Level']),
                'architecture': best_config_row['Architecture'],
                'num_layers': int(best_config_row['Num_Layers']),
                'max_iterations': int(best_config_row['Max_Iterations']),
                'avg_ex_post_profit': float(best_config_row['Ex_post_Profit']),
                'avg_expected_profit': float(best_config_row['Expected_Profit']),
                'avg_processing_time': float(best_config_row['Processing_Time_Seconds'])
            },
            'best_by_noise_level': {},
            'overall_stats': {
                'total_configurations': len(avg_by_config),
                'total_dates_processed': len(df['New_Date'].unique()),
                'avg_ex_post_profit_all': float(df['Ex_post_Profit'].mean()),
                'avg_processing_time_all': float(df['Processing_Time_Seconds'].mean())
            }
        }
        
        # Add best by noise level
        for noise_level in df['Noise_Level'].unique():
            noise_df = df[df['Noise_Level'] == noise_level]
            noise_avg = noise_df.groupby(['Max_Iterations'])['Ex_post_Profit'].mean()
            best_idx = noise_avg.idxmax()
            
            summary_stats['best_by_noise_level'][float(noise_level)] = {
                'max_iterations': int(best_idx),
                'avg_ex_post_profit': float(noise_avg.max())
            }
        
        # Save as JSON
        with open(summary_dir / "comprehensive_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=4)
        
        # 7. Create a text summary report
        with open(summary_dir / "comprehensive_summary.txt", 'w') as f:
            f.write("Comprehensive Validation Summary\n")
            f.write("===============================\n\n")
            
            f.write(f"Total configurations tested: {len(avg_by_config)}\n")
            f.write(f"Total dates processed: {len(df['New_Date'].unique())}\n\n")
            
            f.write("Best Overall Configuration:\n")
            f.write(f"  {best_config}\n")
            f.write(f"  Database: {best_config_row['Database']}\n")
            f.write(f"  Noise Level: {int(best_config_row['Noise_Level']*100)}%\n")
            f.write(f"  Architecture: {best_config_row['Architecture']}\n")
            f.write(f"  Number of Layers: {best_config_row['Num_Layers']}\n")
            f.write(f"  Max Iterations: {best_config_row['Max_Iterations']}\n")
            f.write(f"  Average Ex-post Profit: {best_config_row['Ex_post_Profit']:.2f}\n")
            f.write(f"  Average Processing Time: {best_config_row['Processing_Time_Seconds']:.2f} seconds\n\n")
            
            f.write("Performance by Noise Level:\n")
            for _, row in noise_analysis.iterrows():
                f.write(f"  Noise {int(row['Noise_Level']*100)}%:\n")
                f.write(f"    Average Profit: {row['mean']:.2f}\n")
                f.write(f"    Max Profit: {row['max']:.2f}\n")
                f.write(f"    Min Profit: {row['min']:.2f}\n")
                f.write(f"    Standard Deviation: {row['std']:.2f}\n")
                
                best_noise_config = summary_stats['best_by_noise_level'][row['Noise_Level']]
                f.write(f"    Best Max Iterations: {best_noise_config['max_iterations']} "
                        f"(Profit: {best_noise_config['avg_ex_post_profit']:.2f})\n\n")
        
        print(f"Comprehensive summary generated in {summary_dir}")
        
    except Exception as e:
        print(f"Error generating comprehensive summary: {e}")
        print(traceback.format_exc())

# Run the comprehensive validation if this file is executed directly
if __name__ == "__main__":
    # Make sure seaborn is imported for heatmaps
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn not found. Some visualizations may be limited.")
    
    comprehensive_validation()
# %%