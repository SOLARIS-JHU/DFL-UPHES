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

def load_data_for_pretraining(file_path, source_name):
    """Load and process historical data for pretraining."""
    try:
        # Force comma separator
        df = pd.read_csv(file_path, sep=',', header=0)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        print(f"Actual columns in {source_name}: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
        print(f"First few rows:\n{df.head(3)}")
        
        # Check for required columns
        required_columns = ['date', 'hour', 'power', 'head', 'flow', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['date'])
        df['Time'] = df['hour']
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'power': 'Power',
            'head': 'Head', 
            'flow': 'Flow',
            'price': 'Price'
        })
        
        # Add 'Mode' column based on power values
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
            # Sort by hour to ensure correct order
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
        import traceback
        traceback.print_exc()
        return None

def train_recursive_linearization_with_penalties(weight_network, params, optimizer_layer, regression_layer, 
                                               historical_data, num_epochs=100, learning_rate=0.001, 
                                               patience=10, max_iterations=3, penalty_growth_rate=1.5,
                                               si_penalty_factor=0.0, volume_penalty_factor=0.0):
    """
    Modified training function that includes penalty factors in the loss function.
    
    Args:
        si_penalty_factor: Factor A for SI penalty in loss function
        volume_penalty_factor: Factor B for volume penalty in loss function
    """
    # Move network to the appropriate device
    device = next(weight_network.parameters()).device
    weight_network.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(weight_network.parameters(), lr=learning_rate)
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5
    )

    # Create the pipeline with growth rate parameter
    pipeline = RecursiveLinearizationPipeline(
        weight_network, params, optimizer_layer, regression_layer, historical_data, 
        max_iterations=max_iterations, penalty_growth_rate=penalty_growth_rate
    )
    
    # Select a single date for training
    train_date = list(historical_data.keys())[0]
    print(f"Training on date: {train_date}")
    print(f"SI penalty factor: {si_penalty_factor}, Volume penalty factor: {volume_penalty_factor}")
    
    # Get original data
    date_data = historical_data[train_date]
    power_orig = date_data['power']
    head_orig = date_data['head']
    flow_orig = predict_q_poly(power_orig, head_orig)
    
    # Initialize history tracking
    history = {
        'epoch': [],
        'loss': [],
        'profit': [],
        'simulated_profit': [],
        'SI_penalty': [],
        'volume_penalty': [],
        'operating_cost': [],
        'log_w_p': [],
        'log_w_q': [],
        'log_w_h': [],
        'w_p': [],
        'w_q': [],
        'w_h': [],
        'p_opt': [],
        'h_opt': [],
        'q_opt': [],
        'p_sim': [],
        'q_sim': [],
        'h_sim': [],
        'v_sim': [],
        'p_orig': power_orig.cpu().numpy(),
        'h_orig': head_orig.cpu().numpy(),
        'q_orig': flow_orig.cpu().numpy(),
        'iterations': [],
        'si_penalty_factor': si_penalty_factor,
        'volume_penalty_factor': volume_penalty_factor
    }
    
    # Initialize early stopping
    best_profit = float('-inf')
    best_weights = None
    patience_counter = 0
    
    print(f"Starting training with penalty factors (SI: {si_penalty_factor}, Volume: {volume_penalty_factor})...")
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with recursive linearization and simulation
        simulated_profit, optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
        p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty, operating_cost, \
        (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results = pipeline.forward(train_date)
        
        # Record iteration details
        history['iterations'].append(iter_results)
        
        # Modified loss function with penalty factors
        loss = -simulated_profit + si_penalty_factor * SI_penalty + volume_penalty_factor * volume_penalty
        
        # Backward pass and optimization
        loss.backward()
        
        # Optional: Gradient clipping
        torch.nn.utils.clip_grad_norm_(weight_network.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update learning rate scheduler using simulated profit
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(simulated_profit)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Adjust learning rate if it has changed
        if new_lr != old_lr:
            print(f"Learning rate adjusted from {old_lr:.6f} to {new_lr:.6f}")
        
        # Record history
        history['epoch'].append(epoch)
        history['loss'].append(loss.item())
        history['profit'].append(optimized_profit.item())
        history['simulated_profit'].append(simulated_profit.item())
        history['SI_penalty'].append(SI_penalty.item())
        history['volume_penalty'].append(volume_penalty.item())
        history['operating_cost'].append(operating_cost.item())
        history['log_w_p'].append(log_w_p.detach().cpu().numpy())
        history['log_w_q'].append(log_w_q.detach().cpu().numpy())
        history['log_w_h'].append(log_w_h.detach().cpu().numpy())
        history['w_p'].append(w_p.detach().cpu().numpy())
        history['w_q'].append(w_q.detach().cpu().numpy())
        history['w_h'].append(w_h.detach().cpu().numpy())
        history['p_opt'].append(p_opt.detach().numpy())
        history['h_opt'].append(h_opt.detach().numpy())
        history['q_opt'].append(q_opt.detach().numpy())
        history['p_sim'].append(p_sim.detach().cpu().numpy())
        history['q_sim'].append(q_sim.detach().cpu().numpy())
        history['h_sim'].append(h_sim.detach().cpu().numpy())
        history['v_sim'].append(v_low_sim.detach().cpu().numpy())
        
        # Print results every 50 epochs or for first 5 epochs
        if epoch < 5 or epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Simulated Profit = {simulated_profit.item():.4f}")
            print(f"  SI Penalty: {SI_penalty.item():.4f}, Volume Penalty: {volume_penalty.item():.4f}")
        
        # Early stopping check based on simulated profit
        if simulated_profit.item() > best_profit:
            best_profit = simulated_profit.item()
            best_weights = weight_network.state_dict().copy()
            patience_counter = 0
            if epoch < 5 or epoch % 50 == 0:
                print(f"  New best simulated profit: {best_profit:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    # Load best weights
    if best_weights is not None:
        weight_network.load_state_dict(best_weights)
    
    # Print final results
    print(f"Training Complete - Best Simulated Profit: {best_profit:.4f}")
    
    return weight_network, history

def train_single_penalty_experiment(method_name, si_factor, vol_factor, file_path, train_date, start_timestamp):
    """Train a single penalty experiment configuration."""
    try:
        # Re-import and reload everything in this worker process
        import torch
        import torch.nn as nn
        import dill as pickle
        import pandas as pd
        import sys
        import numpy as np
        from pathlib import Path
        import time
        import json
        
        device = torch.device("cpu")
        
        # Reload portfolio data and preprocessing in worker process
        sys.path.append('../Library')
        from V_H_relations import load_portfolio_data, gross_head, get_v_low
        load_portfolio_data()
        from V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

        # Reload preprocessed functions & data in worker process
        with open('../preprocess.pkl', 'rb') as f:
            v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

        head_init = torch.tensor(77.0, device=device)
        v_low_init = torch.tensor(h_to_v_low_fitted(head_init), device=device)
        
        # Import classes from DFL_pretraining
        from DFL_pretraining import (
            HydroParameters, TaylorRegressionLayer, OptiLayer, SimulationLayer,
            BoundedLogWeightPredictor, RecursiveLinearizationPipeline
        )
        
        # Load training data in worker process
        train_data = load_data_for_pretraining(file_path, 'euclidean_piecewise')
        if not train_data or train_date not in train_data:
            raise ValueError(f"Could not load training data for date {train_date}")
        
        # Get only the specific date we need
        train_data_single = {train_date: train_data[train_date]}
        
        # Fixed configuration
        DATABASE = 'euclidean_piecewise'
        ARCHITECTURE = 'RNN'
        NUM_LAYERS = 3
        MAX_ITERATIONS = 10
        
        # Initialize parameters (each process needs its own)
        params = HydroParameters()
        regression_layer = TaylorRegressionLayer(params)
        optimizer_layer = OptiLayer(params)
        
        # Initialize network
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
        
        # Train with penalty factors
        start_time = time.time()
        trained_network, history = train_recursive_linearization_with_penalties(
            weight_network=weight_network,
            params=params,
            optimizer_layer=optimizer_layer,
            regression_layer=regression_layer,
            historical_data=train_data_single,
            num_epochs=500,
            learning_rate=1e-3,
            patience=20,
            max_iterations=MAX_ITERATIONS,
            penalty_growth_rate=1.5,
            si_penalty_factor=si_factor,
            volume_penalty_factor=vol_factor
        )
        training_time = time.time() - start_time
        
        # Save trained model
        exp_dir = Path("./penalty_ablation_results") / method_name
        exp_dir.mkdir(exist_ok=True, parents=True)
        
        torch.save(trained_network.state_dict(), exp_dir / "model.pt")
        torch.save(trained_network.state_dict(), exp_dir / "best_model.pt")
        
        # Save training history
        simplified_history = {
            'epoch': history['epoch'],
            'loss': [float(x) for x in history['loss']],
            'profit': [float(x) for x in history['profit']],
            'simulated_profit': [float(x) for x in history['simulated_profit']],
            'SI_penalty': [float(x) if hasattr(x, 'item') else x for x in history['SI_penalty']],
            'volume_penalty': [float(x) if hasattr(x, 'item') else x for x in history['volume_penalty']],
            'operating_cost': [float(x) if hasattr(x, 'item') else x for x in history['operating_cost']],
            'si_penalty_factor': si_factor,
            'volume_penalty_factor': vol_factor,
            'method': method_name
        }
        
        with open(exp_dir / "training_history.json", 'w') as f:
            json.dump(simplified_history, f, indent=4)
        
        # Get final metrics
        last_idx = len(history['epoch']) - 1
        best_epoch_idx = np.argmax(history['simulated_profit'])
        best_epoch = history['epoch'][best_epoch_idx]
        
        final_expected_profit = float(history['profit'][last_idx])
        final_simulated_profit = float(history['simulated_profit'][last_idx])
        final_si_penalty = float(history['SI_penalty'][last_idx])
        final_volume_penalty = float(history['volume_penalty'][last_idx])
        final_operating_cost = float(history['operating_cost'][last_idx])
        
        # Save experiment summary
        with open(exp_dir / "experiment_summary.txt", 'w') as f:
            f.write(f"Penalty Ablation Experiment: {method_name}\n")
            f.write(f"=====================================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Database: euclidean_piecewise\n")
            f.write(f"  Architecture: {ARCHITECTURE}\n")
            f.write(f"  Number of Layers: {NUM_LAYERS}\n")
            f.write(f"  Max Iterations: {MAX_ITERATIONS}\n")
            f.write(f"  SI Penalty Factor: {si_factor}\n")
            f.write(f"  Volume Penalty Factor: {vol_factor}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Training Time: {training_time:.2f} seconds\n")
            f.write(f"  Epochs Trained: {last_idx+1}\n")
            f.write(f"  Best Epoch: {best_epoch}\n")
            f.write(f"  Final Expected Profit: {final_expected_profit:.2f}\n")
            f.write(f"  Final Simulated Profit: {final_simulated_profit:.2f}\n")
            f.write(f"  Final SI Penalty: {final_si_penalty:.2f}\n")
            f.write(f"  Final Volume Penalty: {final_volume_penalty:.2f}\n")
            f.write(f"  Final Operating Cost: {final_operating_cost:.2f}\n")
        
        print(f"Worker: Completed {method_name} - Simulated Profit: {final_simulated_profit:.2f}")
        
        # Return results for benchmark
        return {
            'method_name': method_name,
            'si_factor': si_factor,
            'vol_factor': vol_factor,
            'train_date': train_date,
            'training_time': training_time,
            'epochs_trained': last_idx + 1,
            'best_epoch': best_epoch,
            'expected_profit': final_expected_profit,
            'simulated_profit': final_simulated_profit,
            'SI_penalty': final_si_penalty,
            'volume_penalty': final_volume_penalty,
            'operating_cost': final_operating_cost,
            'timestamp': start_timestamp,
            'success': True
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error training {method_name} (SI: {si_factor}, Vol: {vol_factor}): {e}")
        print(f"Full traceback: {error_trace}")
        return {
            'method_name': method_name,
            'si_factor': si_factor,
            'vol_factor': vol_factor,
            'error': str(e),
            'traceback': error_trace,
            'success': False
        }

def penalty_ablation_study():
    """
    Conduct ablation study on penalty factors with parallel processing.
    Fixed model configuration:
    - 3 layers RNN
    - 10 iterations
    - euclidean_piecewise database
    """
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting penalty ablation study at {start_timestamp}...")
    
    # Load data just to get the first date for training
    file_path = '../MIQP/historical_operation_solver/euclidean_piecewise/detailed_results.csv'
    historical_data = load_data_for_pretraining(file_path, 'euclidean_piecewise')
    
    if not historical_data:
        print("Error: Could not load historical data")
        return
    
    # Take first date for training
    train_date = list(historical_data.keys())[0]
    print(f"Using training date: {train_date}")
    
    # Create output directories
    root_dir = Path("./penalty_ablation_results")
    root_dir.mkdir(exist_ok=True, parents=True)
    
    # Define penalty factor experiments
    penalty_experiments = []
    
    # Baseline (no penalties)
    penalty_experiments.append(('Baseline', 0.0, 0.0))
    
    # SI penalty factors (A from 0.1 to 1.0)
    for a in np.arange(0.1, 1.1, 0.1):
        penalty_experiments.append((f'SI_{a:.1f}', round(a, 1), 0.0))
    
    # Volume penalty factors (B from 0.1 to 1.0)  
    for b in np.arange(0.1, 1.1, 0.1):
        penalty_experiments.append((f'Vol_{b:.1f}', 0.0, round(b, 1)))
    
    print(f"Total experiments to run: {len(penalty_experiments)}")
    print(f"Using {min(21, multiprocessing.cpu_count())} parallel processes")
    
    # Prepare all training jobs - pass file path instead of loaded data
    all_jobs = []
    for method_name, si_factor, vol_factor in penalty_experiments:
        all_jobs.append((method_name, si_factor, vol_factor, file_path, train_date, start_timestamp))
    
    # Run in parallel (use up to 21 cores since we have 21 experiments)
    results = Parallel(n_jobs=min(21, multiprocessing.cpu_count()), verbose=1)(
        delayed(train_single_penalty_experiment)(*job) for job in all_jobs
    )
    
    # Create benchmark CSV file and write results
    benchmark_file = root_dir / "penalty_ablation_benchmarks.csv"
    with open(benchmark_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Method', 'SI_Factor', 'Volume_Factor', 'Date',
            'Training_Time_Seconds', 'Epochs_Trained', 'Best_Epoch',
            'Expected_Profit', 'Simulated_Profit', 'SI_Penalty', 
            'Volume_Penalty', 'Operating_Cost', 'Timestamp'
        ])
        
        # Write successful results
        successful_results = 0
        failed_results = 0
        
        for result in results:
            if result['success']:
                writer.writerow([
                    result['method_name'], result['si_factor'], result['vol_factor'], 
                    result['train_date'], f"{result['training_time']:.2f}", 
                    result['epochs_trained'], result['best_epoch'],
                    f"{result['expected_profit']:.2f}", f"{result['simulated_profit']:.2f}",
                    f"{result['SI_penalty']:.2f}", f"{result['volume_penalty']:.2f}",
                    f"{result['operating_cost']:.2f}", result['timestamp']
                ])
                successful_results += 1
                
                print(f"✅ {result['method_name']}: Simulated Profit = {result['simulated_profit']:.2f}, "
                      f"SI Penalty = {result['SI_penalty']:.2f}, Vol Penalty = {result['volume_penalty']:.2f}")
            else:
                failed_results += 1
                print(f"❌ Failed: {result['method_name']} - {result.get('error', 'Unknown error')}")
                
                # Log the error
                with open(root_dir / "error_log.txt", 'a') as f_err:
                    f_err.write(f"\n[{datetime.now()}] Error in experiment {result['method_name']}:\n")
                    f_err.write(f"SI factor: {result['si_factor']}, Volume factor: {result['vol_factor']}\n")
                    f_err.write(f"Error: {result.get('error', 'Unknown error')}\n")
                    f_err.write("\n" + "-"*50 + "\n")
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_duration = datetime.strptime(end_timestamp, "%Y%m%d_%H%M%S") - datetime.strptime(start_timestamp, "%Y%m%d_%H%M%S")
    
    print(f"\nPenalty ablation study completed!")
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Total duration: {total_duration}")
    print(f"Successful experiments: {successful_results}/{len(penalty_experiments)}")
    print(f"Failed experiments: {failed_results}")
    print(f"Results saved in: {root_dir}")

if __name__ == "__main__":
    penalty_ablation_study()
    print("Penalty ablation pretraining completed.")