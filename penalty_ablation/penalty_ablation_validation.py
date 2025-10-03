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

# Import classes from DFL_pretraining
from DFL_pretraining import (
    HydroParameters, TaylorRegressionLayer, OptiLayer, SimulationLayer,
    BoundedLogWeightPredictor, RecursiveLinearizationPipeline
)

def load_new_price_data(file_path="../Data/price_data_2024.csv"):
    """Load new price data for validation."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check column names from the first line
        if 'date' not in df.columns or 'prices_hourly' not in df.columns:
            # Try to handle different column formats
            if len(df.columns) >= 3:
                # Assume first column is date, third column has hourly prices
                df.columns = ['date', 'cluster_index', 'prices_hourly']
            else:
                raise ValueError(f"Expected columns 'date', 'prices_hourly' but got {df.columns}")
        
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
    """Load historical data for finding similar price profiles."""
    try:
        # Read the file
        df = pd.read_csv(file_path, sep=',', header=0)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        print(f"Loading validation data from {source_name}: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['date', 'hour', 'power', 'head', 'flow', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['date'])
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
            conditions = [
                (abs(df['Power']) < 0.01),  # Idle mode
                (df['Power'] > 0),          # Turbine mode
                (df['Power'] < 0)           # Pump mode
            ]
            choices = ['Idle', 'Turbine', 'Pump']
            df['Mode'] = np.select(conditions, choices, default='Unknown')
        
        # Group data by date
        data_by_date = {}
        for date, group in df.groupby('Date'):
            # Sort by Time to ensure correct order
            group = group.sort_values('Time')
            
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
        return None

def find_closest_date(new_price, historical_data):
    """Find the date in historical data with the most similar price signal."""
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

def validate_penalty_experiments():
    """
    Validate all trained penalty experiment models on new price data.
    """
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting penalty ablation validation at {start_timestamp}...")
    
    # Fixed configuration parameters
    DATABASE = 'euclidean_piecewise'
    ARCHITECTURE = 'RNN'
    NUM_LAYERS = 3
    MAX_ITERATIONS = 10
    
    # Load new price data
    new_price_data = load_new_price_data()
    if not new_price_data:
        print("Error: Could not load new price data")
        return
    
    # Load historical data
    file_path = '../MIQP/historical_operation_solver/euclidean_piecewise/detailed_results.csv'
    historical_data = load_data_for_validation(file_path, DATABASE)
    if not historical_data:
        print("Error: Could not load historical data")
        return
    
    # Initialize parameters
    params = HydroParameters()
    regression_layer = TaylorRegressionLayer(params)
    optimizer_layer = OptiLayer(params)
    
    # Create validation results directory
    validation_dir = Path("./penalty_ablation_validation_results")
    validation_dir.mkdir(exist_ok=True, parents=True)
    
    # Define experiment methods (same as in pretraining)
    experiment_methods = []
    experiment_methods.append('Baseline')
    for a in np.arange(0.1, 1.1, 0.1):
        experiment_methods.append(f'SI_{a:.1f}')
    for b in np.arange(0.1, 1.1, 0.1):  
        experiment_methods.append(f'Vol_{b:.1f}')
    
    # Create master validation benchmark file
    master_benchmark_file = validation_dir / "validation_benchmarks.csv"
    with open(master_benchmark_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Method', 'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
            'Expected_Profit', 'Ex_post_Profit', 'SI_Penalty',
            'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
            'Timestamp'
        ])
    
    # Results for LaTeX table (aggregated by method)
    latex_results = {}
    
    # Process each experiment method
    for method_idx, method_name in enumerate(experiment_methods):
        print(f"\n{'='*60}")
        print(f"[{method_idx+1}/{len(experiment_methods)}] Validating method: {method_name}")
        print(f"{'='*60}")
        
        # Check if trained model exists
        model_path = Path(f"./penalty_ablation_results/{method_name}/best_model.pt")
        if not model_path.exists():
            print(f"Warning: No trained model found at {model_path}. Skipping...")
            continue
        
        # Create method validation directory
        method_dir = validation_dir / method_name
        method_dir.mkdir(exist_ok=True)
        
        # Initialize results for this method
        method_results = {
            'expected_profits': [],
            'ex_post_profits': [],
            'si_penalties': [],
            'volume_penalties': [],
            'operating_costs': [],
            'processing_times': []
        }
        
        try:
            # Initialize weight network with the same configuration
            weight_network = BoundedLogWeightPredictor(
                input_size=4,
                hidden_size=128,
                num_layers=NUM_LAYERS,
                dropout=0.2,
                time_horizon=params.time_horizon,
                archetype=ARCHITECTURE,
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
            
            # Load the trained weights
            weight_network.load_state_dict(torch.load(model_path, map_location=device))
            weight_network.eval()
            
            # Process each new date
            for date_idx, (new_date, new_price) in enumerate(new_price_data.items()):
                print(f"  [{date_idx+1}/{len(new_price_data)}] Processing date: {new_date}")
                
                try:
                    # Start timing
                    start_time = time.time()
                    
                    # Find the closest historical date
                    closest_date, distance = find_closest_date(new_price, historical_data)
                    
                    # Get the historical data for the closest date
                    closest_data = historical_data[closest_date]
                    power_init = closest_data['power'][:24].clone()
                    head_init = closest_data['head'][:24].clone()
                    flow_init = predict_q_poly(power_init, head_init)
                    
                    # Create pipeline for recursive linearization
                    pipeline = RecursiveLinearizationPipeline(
                        weight_network, params, optimizer_layer, regression_layer, 
                        {closest_date: closest_data}, max_iterations=MAX_ITERATIONS, 
                        penalty_growth_rate=1.5
                    )
                    
                    # Prepare input for weight prediction
                    x = torch.stack([new_price, power_init, flow_init, head_init], dim=1)
                    
                    # Predict weights
                    with torch.no_grad():
                        log_w_p, log_w_q, log_w_h = weight_network(x)
                        w_p = torch.exp(log_w_p)
                        w_q = torch.exp(log_w_q)
                        w_h = torch.exp(log_w_h)
                    
                    # Run recursive linearization
                    p_current = power_init.clone().detach()
                    h_current = head_init.clone().detach()
                    flow_current = flow_init.clone().detach()
                    
                    for iteration in range(MAX_ITERATIONS):
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
                        
                        # Update for next iteration
                        if iteration < MAX_ITERATIONS - 1:
                            p_current = p_opt.clone().detach().to(device=power_init.device) 
                            h_current = h_opt.clone().detach().to(device=head_init.device)
                            flow_current = q_opt.clone().detach().to(device=flow_init.device)
                    
                    # Run simulation
                    simulator = SimulationLayer(params)
                    p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(
                        p_opt.to(device), q_opt.to(device), h_opt.to(device)
                    )
                    
                    # Calculate ex-post profit
                    ex_post_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
                        p_sim, p_opt.to(device), v_low_sim, new_price.to(device)
                    )
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Store results for this method
                    method_results['expected_profits'].append(expected_profit.item())
                    method_results['ex_post_profits'].append(ex_post_profit.item())
                    method_results['si_penalties'].append(SI_penalty.item())
                    method_results['volume_penalties'].append(volume_penalty.item())
                    method_results['operating_costs'].append(operating_cost.item())
                    method_results['processing_times'].append(processing_time)
                    
                    # Save individual results
                    safe_date = new_date.replace('/', '-')
                    date_dir = method_dir / safe_date
                    date_dir.mkdir(exist_ok=True)
                    
                    results = {
                        'method': method_name,
                        'new_date': new_date,
                        'closest_date': closest_date,
                        'distance': distance,
                        'expected_profit': expected_profit.item(),
                        'ex_post_profit': ex_post_profit.item(),
                        'SI_penalty': SI_penalty.item(),
                        'volume_penalty': volume_penalty.item(),
                        'operating_cost': operating_cost.item(),
                        'processing_time': processing_time,
                        'p_opt': p_opt.detach().cpu().numpy(),
                        'q_opt': q_opt.detach().cpu().numpy(),
                        'h_opt': h_opt.detach().cpu().numpy(),
                        'p_sim': p_sim.detach().cpu().numpy(),
                        'q_sim': q_sim.detach().cpu().numpy(),
                        'h_sim': h_sim.detach().cpu().numpy(),
                        'new_price': new_price.detach().cpu().numpy(),
                        'closest_price': closest_data['price'][:24].detach().cpu().numpy()
                    }
                    
                    np.save(date_dir / "results.npy", results)
                    
                    # Append to master benchmark file
                    with open(master_benchmark_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            method_name, safe_date, closest_date, f"{distance:.2f}",
                            f"{expected_profit.item():.2f}", f"{ex_post_profit.item():.2f}",
                            f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                            f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                            start_timestamp
                        ])
                    
                except Exception as e:
                    print(f"    Error processing date {new_date}: {e}")
                    continue
            
            # Calculate average results for this method
            if method_results['ex_post_profits']:  # Check if we have any results
                latex_results[method_name] = {
                    'expected_profit': np.mean(method_results['expected_profits']),
                    'ex_post_profit': np.mean(method_results['ex_post_profits']),
                    'si_penalty': np.mean(method_results['si_penalties']),
                    'volume_penalty': np.mean(method_results['volume_penalties']),
                    'operating_cost': np.mean(method_results['operating_costs']),
                    'processing_time': np.mean(method_results['processing_times'])
                }
                
                print(f"Method {method_name} completed:")
                print(f"  Average Ex-post Profit: {latex_results[method_name]['ex_post_profit']:.2f}")
                print(f"  Average SI Penalty: {latex_results[method_name]['si_penalty']:.2f}")
                print(f"  Average Volume Penalty: {latex_results[method_name]['volume_penalty']:.2f}")
            
        except Exception as e:
            print(f"Error with method {method_name}: {e}")
            print(traceback.format_exc())
            continue
    
    # Generate LaTeX table
    generate_latex_table(latex_results, validation_dir)
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_duration = datetime.strptime(end_timestamp, "%Y%m%d_%H%M%S") - datetime.strptime(start_timestamp, "%Y%m%d_%H%M%S")
    
    print(f"\nPenalty ablation validation completed!")
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Total duration: {total_duration}")
    print(f"Results saved in: {validation_dir}")

def generate_latex_table(results, output_dir):
    """Generate LaTeX table with validation results."""
    print("\nGenerating LaTeX table...")
    
    # Sort methods: Baseline first, then SI methods, then Volume methods
    method_order = ['Baseline']
    method_order.extend([f'SI_{a:.1f}' for a in np.arange(0.1, 1.1, 0.1)])
    method_order.extend([f'Vol_{b:.1f}' for b in np.arange(0.1, 1.1, 0.1)])
    
    latex_file = output_dir / "penalty_ablation_results.tex"
    
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Penalty Ablation Study Results}\n")
        f.write("\\label{tab:penalty_ablation}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & Ex-post Profit & Expected Profit & SI Penalty & Vol Penalty & Op Cost & Time (s) \\\\\n")
        f.write("\\midrule\n")
        
        for method in method_order:
            if method in results:
                r = results[method]
                f.write(f"{method:12s} & {r['ex_post_profit']:8.2f} & {r['expected_profit']:8.2f} & "
                       f"{r['si_penalty']:7.2f} & {r['volume_penalty']:7.2f} & {r['operating_cost']:7.2f} & "
                       f"{r['processing_time']:6.2f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {latex_file}")
    
    # Also generate a summary analysis
    summary_file = output_dir / "penalty_ablation_analysis.txt"
    with open(summary_file, 'w') as f:
        f.write("Penalty Ablation Study Analysis\n")
        f.write("===============================\n\n")
        
        if 'Baseline' in results:
            baseline = results['Baseline']
            f.write(f"Baseline Results:\n")
            f.write(f"  Ex-post Profit: {baseline['ex_post_profit']:.2f}\n")
            f.write(f"  Expected Profit: {baseline['expected_profit']:.2f}\n")
            f.write(f"  SI Penalty: {baseline['si_penalty']:.2f}\n")
            f.write(f"  Volume Penalty: {baseline['volume_penalty']:.2f}\n")
            f.write(f"  Operating Cost: {baseline['operating_cost']:.2f}\n\n")
        
        # Find best performing methods
        si_methods = {k: v for k, v in results.items() if k.startswith('SI_')}
        vol_methods = {k: v for k, v in results.items() if k.startswith('Vol_')}
        
        if si_methods:
            best_si = max(si_methods.items(), key=lambda x: x[1]['ex_post_profit'])
            f.write(f"Best SI Penalty Method: {best_si[0]}\n")
            f.write(f"  Ex-post Profit: {best_si[1]['ex_post_profit']:.2f}\n")
            f.write(f"  SI Penalty: {best_si[1]['si_penalty']:.2f}\n\n")
        
        if vol_methods:
            best_vol = max(vol_methods.items(), key=lambda x: x[1]['ex_post_profit'])
            f.write(f"Best Volume Penalty Method: {best_vol[0]}\n")
            f.write(f"  Ex-post Profit: {best_vol[1]['ex_post_profit']:.2f}\n")
            f.write(f"  Volume Penalty: {best_vol[1]['volume_penalty']:.2f}\n\n")
        
        # Overall best method
        if results:
            overall_best = max(results.items(), key=lambda x: x[1]['ex_post_profit'])
            f.write(f"Overall Best Method: {overall_best[0]}\n")
            f.write(f"  Ex-post Profit: {overall_best[1]['ex_post_profit']:.2f}\n")
            f.write(f"  Expected Profit: {overall_best[1]['expected_profit']:.2f}\n")
            f.write(f"  SI Penalty: {overall_best[1]['si_penalty']:.2f}\n")
            f.write(f"  Volume Penalty: {overall_best[1]['volume_penalty']:.2f}\n")
            f.write(f"  Operating Cost: {overall_best[1]['operating_cost']:.2f}\n")
    
    print(f"Analysis summary saved to: {summary_file}")

if __name__ == "__main__":
    validate_penalty_experiments()
    print("Penalty ablation validation completed.")
