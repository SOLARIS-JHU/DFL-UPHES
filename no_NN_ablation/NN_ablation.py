# %% Ablation Study - Validation without LSTM (Modified Version)
# This script runs the same validation as DFL_validation.py but WITHOUT the LSTM network
# It uses fixed penalty weights instead of learned weights to establish a baseline
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import sys
import numpy as np
from pathlib import Path
import json
import csv
import time
import os
from datetime import datetime
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

head_init = torch.tensor(77.0, device=device)
print(f"Initial head: {head_init.item()}")
v_low_init = torch.tensor(h_to_v_low_fitted(head_init), device=device)
print(f"Initial head: {head_init.item()}, Initial v_low: {v_low_init.item()}")

# Import classes from DFL_pretraining (reuse existing infrastructure)
from DFL_pretraining import (
    HydroParameters, TaylorRegressionLayer, OptiLayer, SimulationLayer
)

# %% Fixed Weight Configuration
class FixedWeightConfig:
    """Configuration for fixed penalty weights (no LSTM)"""
    def __init__(self, w_p=0.6, w_q=0.02, w_h=0.1, time_horizon=24):
        self.w_p_base = w_p
        self.w_q_base = w_q
        self.w_h_base = w_h
        self.time_horizon = time_horizon
    
    def get_weights(self):
        """Return fixed weights as tensors"""
        w_p = torch.full((self.time_horizon,), self.w_p_base, dtype=torch.float32, device=device)
        w_q = torch.full((self.time_horizon,), self.w_q_base, dtype=torch.float32, device=device)
        w_h = torch.full((self.time_horizon,), self.w_h_base, dtype=torch.float32, device=device)
        return w_p, w_q, w_h

class BaselineRecursiveLinearization:
    """
    Baseline pipeline using fixed weights instead of LSTM predictions.
    Applies the same recursive linearization and penalty growth.
    """
    def __init__(self, weight_config, params, optimizer, regression, 
                 max_iterations=3, penalty_growth_rate=1.5):
        self.weight_config = weight_config
        self.params = params
        self.optimizer = optimizer
        self.regression = regression
        self.max_iterations = max_iterations
        self.penalty_growth_rate = penalty_growth_rate
        self.simulator = SimulationLayer(params)
    
    def forward(self, price, power_init, head_init, flow_init):
        """
        Run recursive linearization with fixed weights.
        
        Returns same outputs as LSTM version for fair comparison.
        """
        # Get fixed weights
        w_p_base, w_q_base, w_h_base = self.weight_config.get_weights()
        
        # Initialize for first iteration
        p_current = power_init.clone().detach()
        h_current = head_init.clone().detach()
        flow_current = flow_init.clone().detach()
        
        # Store iteration results
        iter_results = []
        
        # Recursive linearization loop
        for iteration in range(self.max_iterations):
            # Apply growth to penalty weights
            growth_factor = self.penalty_growth_rate ** iteration
            w_p = w_p_base * growth_factor
            w_q = w_q_base * growth_factor
            w_h = w_h_base * growth_factor
            
            # Compute linearization coefficients
            c, d, e, a, b = self.regression.run_regression(p_current, h_current, flow_current)
            
            # Initialize OptiLayer
            self.optimizer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())
            
            # Run optimization
            p_opt, q_opt, h_opt, v_opt, expected_profit, optimized_objective = self.optimizer.forward(
                price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                w_p.cpu(), w_h.cpu(), w_q.cpu()
            )
            
            # Store iteration results
            iter_results.append({
                'iteration': iteration,
                'expected_profit': expected_profit.item(),
                'optimized_objective': optimized_objective.item(),
                'p_opt': p_opt.detach().cpu().numpy(),
                'q_opt': q_opt.detach().cpu().numpy(),
                'h_opt': h_opt.detach().cpu().numpy(),
                'growth_factor': growth_factor,
                'w_p_mean': w_p.mean().item(),
                'w_q_mean': w_q.mean().item(),
                'w_h_mean': w_h.mean().item()
            })
            
            # Update for next iteration
            if iteration < self.max_iterations - 1:
                p_current = p_opt.clone().detach().to(device=power_init.device)
                h_current = h_opt.clone().detach().to(device=head_init.device)
                flow_current = q_opt.clone().detach().to(device=flow_init.device)
        
        # Run simulation with final optimized schedule
        p_sim, q_sim, h_sim, v_low_sim = self.simulator.simulate_operation(
            p_opt.to(device), q_opt.to(device), h_opt.to(device)
        )
        
        # Calculate ex-post profit
        ex_post_profit, SI_penalty, volume_penalty, operating_cost = self.simulator.calc_profit(
            p_sim, p_opt.to(device), v_low_sim, price.to(device)
        )
        
        return (ex_post_profit, expected_profit, p_opt, q_opt, h_opt, v_opt,
                p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty, 
                operating_cost, iter_results)

# %% Utility Functions (reuse from validation script)
def load_new_price_data(file_path="../Data/price_data_2024.csv"):
    """Load new price data for validation"""
    try:
        df = pd.read_csv(file_path)
        
        if 'date' not in df.columns or 'cluster_index' not in df.columns or 'prices_hourly' not in df.columns:
            if len(df.columns) >= 3:
                df.columns = ['date', 'cluster_index', 'prices_hourly']
            else:
                raise ValueError(f"Expected columns 'date', 'cluster_index', 'prices_hourly' but got {df.columns}")
        
        price_data = {}
        
        for _, row in df.iterrows():
            date_str = row['date']
            prices_str = row['prices_hourly']
            
            try:
                prices = [float(p) for p in prices_str.split(',')]
            except:
                try:
                    prices = [float(p) for p in prices_str.split(';')]
                except:
                    prices_str = prices_str.strip('[]')
                    prices = [float(p) for p in prices_str.split()]
            
            if len(prices) != 24:
                print(f"Warning: Date {date_str} has {len(prices)} price values instead of 24")
                if len(prices) < 24:
                    prices.extend([prices[-1]] * (24 - len(prices)))
                else:
                    prices = prices[:24]
            
            price_tensor = torch.tensor(prices, dtype=torch.float32, device=device)
            price_data[date_str] = price_tensor
        
        print(f"Successfully loaded price data for {len(price_data)} days.")
        return price_data
    
    except Exception as e:
        print(f"Error loading new price data: {e}")
        return None

def load_data_for_validation(file_path, source_name):
    """Load historical data for finding similar price profiles"""
    try:
        df = pd.read_csv(file_path, sep=',', header=0)
        df.columns = df.columns.str.strip()
        
        print(f"Loading validation data from {source_name}: {list(df.columns)}")
        
        required_columns = ['date', 'hour', 'power', 'head', 'flow']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if 'price' in df.columns:
            print(f"Using price data from {source_name} file.")
        else:
            print(f"No price column found. Loading from original MIQP file...")
            original_miqp_file = "../MIQP/MIQP_piecewise/MIQP_piecewise_results.csv"
            if os.path.exists(original_miqp_file):
                price_df = pd.read_csv(original_miqp_file)
                price_df.columns = price_df.columns.str.strip()
                
                try:
                    df['date_normalized'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    price_df['date_normalized'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')
                    
                    df = df.merge(price_df[['date_normalized', 'hour', 'price']], 
                                 left_on=['date_normalized', 'hour'], 
                                 right_on=['date_normalized', 'hour'], 
                                 how='left')
                    df.drop('date_normalized', axis=1, inplace=True)
                except Exception as e:
                    print(f"Date format conversion failed: {e}")
        
        try:
            df['Date'] = pd.to_datetime(df['date'])
        except:
            try:
                df['Date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            except:
                df['Date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        
        df['Time'] = df['hour']
        df = df.rename(columns={
            'power': 'Power',
            'head': 'Head', 
            'flow': 'Flow',
            'price': 'Price'
        })
        
        # Group data by date
        data_by_date = {}
        for date, group in df.groupby('Date'):
            group = group.sort_values('Time')
            
            if len(group) != 24:
                print(f"Warning: Date {date.strftime('%Y-%m-%d')} has {len(group)} hours. Skipping.")
                continue
            
            date_str = date.strftime('%Y-%m-%d')
            date_data = {
                'power': torch.tensor(group['Power'].values, dtype=torch.float32, device=device),
                'head': torch.tensor(group['Head'].values, dtype=torch.float32, device=device),
                'flow': torch.tensor(group['Flow'].values, dtype=torch.float32, device=device),
                'price': torch.tensor(group['Price'].values, dtype=torch.float32, device=device),
            }
            data_by_date[date_str] = date_data
        
        print(f"Successfully loaded {source_name} data for {len(data_by_date)} days.")
        return data_by_date
    
    except Exception as e:
        print(f"Error loading {source_name} data: {e}")
        traceback.print_exc()
        return None

def find_closest_date(new_price, historical_data):
    """Find the date with most similar price signal"""
    closest_date = None
    min_distance = float('inf')
    
    for date_str, date_data in historical_data.items():
        historical_price = date_data['price'][:24]
        distance = torch.norm(new_price - historical_price).item()
        
        if distance < min_distance:
            min_distance = distance
            closest_date = date_str
    
    return closest_date, min_distance

# %% Main Ablation Validation
def ablation_validation():
    """
    Perform ablation study validation WITHOUT LSTM network.
    Uses fixed penalty weights with the same recursive linearization process.
    """
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting ablation study validation (no LSTM) at {start_timestamp}...")
    
    # Define databases to test
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
    
    # Add random samples
    random_samples_source = "MIQP_piecewise_results_random_samples"
    random_samples_file = "MIQP_piecewise_results_random_samples.csv"
    database_sources[random_samples_source] = {
        'file_path': random_samples_file,
        'data_type': 'random_samples',
        'noise_level': None
    }
    
    # Test parameters (reduced range for ablation study)
    max_iterations_list = list(range(1, 6))  # 1 to 5
    
    # Fixed weight configurations to test
    weight_configs = [
        {'w_p': 0.1, 'w_q': 0.01, 'w_h': 0.05},
    ]
    
    # Load new price data
    new_price_data = load_new_price_data()
    if not new_price_data:
        print("Error: Could not load new price data")
        return
    
    # Initialize parameters
    params = HydroParameters()
    
    # Create master output directory
    master_dir = Path("./validation_results/ablation_study")
    master_dir.mkdir(exist_ok=True, parents=True)
    
    # Create master benchmark file
    master_benchmark_file = master_dir / "ablation_benchmarks.csv"
    with open(master_benchmark_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Database', 'Data_Type', 'Noise_Level', 'Weight_Config', 
            'W_P_Base', 'W_Q_Base', 'W_H_Base', 'Max_Iterations',
            'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
            'Expected_Profit', 'Ex_post_Profit', 'SI_Penalty',
            'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
            'Timestamp'
        ])
    
    # Counter for progress
    total_configs = len(database_sources) * len(weight_configs) * len(max_iterations_list)
    config_counter = 0
    
    # Iterate through all configurations
    for db_name, weight_cfg, max_iter in itertools.product(
            database_sources.keys(), weight_configs, max_iterations_list):
        
        config_counter += 1
        db_info = database_sources[db_name]
        data_type = db_info['data_type']
        noise_level = db_info['noise_level']
        
        config_name = f"baseline_low_w{max_iter}iter"
        
        print(f"\n{'='*80}")
        if data_type == 'random_samples':
            print(f"[{config_counter}/{total_configs}] Ablation: {db_name}/{config_name}")
        else:
            print(f"[{config_counter}/{total_configs}] Ablation: {db_name} (Noise {int(noise_level*100)}%)/{config_name}")
        print(f"{'='*80}")
        
        # Create output directory
        config_dir = Path(f"./validation_results/ablation_study/{db_name}/{config_name}")
        config_dir.mkdir(exist_ok=True, parents=True)
        
        # Create config-specific benchmark
        config_benchmark_file = config_dir / "benchmarks.csv"
        with open(config_benchmark_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
                'Expected_Profit', 'Ex_post_Profit', 'SI_Penalty',
                'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
                'Timestamp'
            ])
        
        try:
            # Load historical data
            historical_data = load_data_for_validation(db_info['file_path'], db_name)
            if not historical_data:
                print(f"Error: Could not load historical data for {db_name}")
                continue
            
            # Initialize fixed weight configuration
            weight_config = FixedWeightConfig(
                w_p=weight_cfg['w_p'],
                w_q=weight_cfg['w_q'],
                w_h=weight_cfg['w_h'],
                time_horizon=params.time_horizon
            )
            
            # Initialize layers
            regression_layer = TaylorRegressionLayer(params)
            optimizer_layer = OptiLayer(params)
            
            # Create baseline pipeline
            pipeline = BaselineRecursiveLinearization(
                weight_config=weight_config,
                params=params,
                optimizer=optimizer_layer,
                regression=regression_layer,
                max_iterations=max_iter,
                penalty_growth_rate=1.5
            )
            
            # Process each new date
            for date_idx, (new_date, new_price) in enumerate(new_price_data.items()):
                print(f"\n[{date_idx+1}/{len(new_price_data)}] Processing {new_date}")
                
                safe_date = new_date.replace('/', '-')
                date_dir = config_dir / safe_date
                date_dir.mkdir(exist_ok=True, parents=True)
                
                try:
                    start_time = time.time()
                    
                    # Find closest historical date
                    closest_date, distance = find_closest_date(new_price, historical_data)
                    print(f"Closest historical date: {closest_date} (distance: {distance:.2f})")
                    
                    # Get initialization from closest date
                    closest_data = historical_data[closest_date]
                    power_init = closest_data['power'][:24].clone()
                    head_init = closest_data['head'][:24].clone()
                    flow_init = predict_q_poly(power_init, head_init)
                    
                    # Run baseline recursive linearization
                    (ex_post_profit, expected_profit, p_opt, q_opt, h_opt, v_opt,
                     p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty,
                     operating_cost, iter_results) = pipeline.forward(
                        new_price, power_init, head_init, flow_init
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Save results
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
                        'expected_profit': expected_profit.item(),
                        'ex_post_profit': ex_post_profit.item(),
                        'SI_penalty': SI_penalty.item(),
                        'volume_penalty': volume_penalty.item(),
                        'operating_cost': operating_cost.item(),
                        'iter_results': iter_results,
                        'weight_config': weight_cfg
                    }
                    
                    np.save(date_dir / "results.npy", results)
                    
                    # Append to config benchmark
                    with open(config_benchmark_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            safe_date, closest_date, f"{distance:.2f}",
                            f"{expected_profit.item():.2f}", f"{ex_post_profit.item():.2f}",
                            f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                            f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                            start_timestamp
                        ])
                    
                    # Append to master benchmark
                    noise_val = f"{noise_level}" if noise_level is not None else 'N/A'
                    with open(master_benchmark_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            db_name, data_type, noise_val, 'low',
                            weight_cfg['w_p'], weight_cfg['w_q'], weight_cfg['w_h'], max_iter,
                            safe_date, closest_date, f"{distance:.2f}",
                            f"{expected_profit.item():.2f}", f"{ex_post_profit.item():.2f}",
                            f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                            f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                            start_timestamp
                        ])
                    
                    print(f"  Expected profit: {expected_profit.item():.2f}")
                    print(f"  Ex-post profit: {ex_post_profit.item():.2f}")
                    print(f"  Processing time: {processing_time:.2f}s")
                    
                except Exception as e:
                    print(f"Error processing {new_date}: {e}")
                    traceback.print_exc()
        
        except Exception as e:
            print(f"Error with configuration {db_name}/{config_name}: {e}")
            traceback.print_exc()
    
    # Generate summary analysis
    generate_ablation_summary(master_benchmark_file)
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nAblation study completed!")
    print(f"Results saved to: {master_dir}")

def generate_ablation_summary(master_benchmark_file):
    """Generate summary analysis of ablation study results"""
    try:
        df = pd.read_csv(master_benchmark_file)
        
        summary_dir = Path("./validation_results/ablation_study/summary")
        summary_dir.mkdir(exist_ok=True, parents=True)
        
        # Average performance by weight config and iterations
        avg_by_config = df.groupby(['Weight_Config', 'Max_Iterations'])[
            ['Expected_Profit', 'Ex_post_Profit', 'SI_Penalty', 
             'Volume_Penalty', 'Processing_Time_Seconds']
        ].mean().reset_index()
        
        # Find best configuration
        best_row = avg_by_config.loc[avg_by_config['Ex_post_Profit'].idxmax()]
        
        # Save summary
        with open(summary_dir / "ablation_summary.txt", 'w') as f:
            f.write("Ablation Study Summary (No LSTM - Fixed Weights)\n")
            f.write("================================================\n\n")
            f.write(f"Total configurations: {len(avg_by_config)}\n")
            f.write(f"Weight configs tested: {df['Weight_Config'].unique().tolist()}\n")
            f.write(f"Iteration range: {df['Max_Iterations'].min()}-{df['Max_Iterations'].max()}\n\n")
            f.write(f"Best Configuration:\n")
            f.write(f"  Weight Config: {best_row['Weight_Config']}\n")
            f.write(f"  Max Iterations: {best_row['Max_Iterations']}\n")
            f.write(f"  Average Ex-post Profit: {best_row['Ex_post_Profit']:.2f}\n")
        
        print(f"Ablation summary generated in {summary_dir}")
        
    except Exception as e:
        print(f"Error generating ablation summary: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    ablation_validation()