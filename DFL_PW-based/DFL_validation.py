# %% Import libraries and setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import sys
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
            original_miqp_file = "../MIQP/MIQP_piecewise/MIQP_piecewise_results.csv"
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
    max_iterations_list = [1, 5]
    
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
                    
                    # 13. Visualization skipped per requirements
                    
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
    
    # Skip summary generation to avoid saving unused visualization files
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_duration = datetime.strptime(end_timestamp, "%Y%m%d_%H%M%S") - datetime.strptime(start_timestamp, "%Y%m%d_%H%M%S")
    
    print(f"\nComprehensive validation completed!")
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Total duration: {total_duration}")
    print(f"Master benchmark saved to: {master_benchmark_file}")
    print(f"Best configurations saved to: {master_dir / 'best_configurations.json'}")

if __name__ == "__main__":
    comprehensive_validation()
# %%