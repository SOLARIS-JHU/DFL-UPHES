# %% Import libraries
import torch
import numpy as np
import cvxpy as cp
import dill as pickle
import pandas as pd
import sys
import gurobipy as gp
from gurobipy import GRB

device = torch.device("cpu")

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = 77.0  # Initial head value
v_low_init = h_to_v_low_fitted(head_init)  # Initial lower reservoir volume

# Load day-ahead prices
def load_prices():
    """Load day-ahead prices from Belgium historical database."""
    # Load Belgium historical data
    belgium_file = "./Data/Belgium_historical_data.csv"
    
    if os.path.exists(belgium_file):
        print("Loading Belgium historical data...")
        df = pd.read_csv(belgium_file)
        
        # Process each day's prices
        price_data = {}
        for _, row in df.iterrows():
            date = row['date']  # This is now the Belgium historical date
            prices = [float(p) for p in row['prices_hourly'].split(',')]
            price_data[date] = prices
        
        print(f"Loaded Belgium historical data for {len(price_data)} days")
        return price_data
    
    else:
        raise FileError("Belgium historical data file not found. Please run price_matcher.py first.")

# %% Enhanced Error Analysis with Comprehensive Metrics
import pandas as pd
import numpy as np
import torch
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import glob

# Assuming the same imports and functions from your original code
# Load the existing datasets
print("Loading existing datasets...")
piecewise_data = pd.read_csv('./Data/piecewise_operation_data_SOS2_2024_10seg.csv')
global_data = pd.read_csv('./Data/database_no_piecewise_2024.csv')

# Check if volume column exists in global data
if 'Volume' not in global_data.columns:
    print("Warning: Volume column not found in global_data. Please run the volume addition script first.")
    print("Available columns:", global_data.columns.tolist())

print("Piecewise data columns:", piecewise_data.columns.tolist())
print("Global data columns:", global_data.columns.tolist())
print(f"Piecewise data shape: {piecewise_data.shape}")
print(f"Global data shape: {global_data.shape}")

# Load Neural Network MPC data
def load_nn_mpc_data():
    """Load all Neural Network MPC results from the NN-MPC directory"""
    nn_files = glob.glob('./Data/NN-MPC/*_detailed.csv')
    
    if not nn_files:
        print("No NN-MPC files found in ./Data/NN-MPC/")
        return pd.DataFrame()
    
    print(f"Found {len(nn_files)} NN-MPC files")
    
    all_nn_data = []
    for file in nn_files:
        try:
            df = pd.read_csv(file)
            # Add date from filename if not present
            if 'date' not in df.columns:
                date_str = os.path.basename(file).split('_')[0]  # Extract date from filename
                df['date'] = date_str
            all_nn_data.append(df)
            print(f"Loaded {file}: {df.shape[0]} rows, columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if all_nn_data:
        combined_data = pd.concat(all_nn_data, ignore_index=True)
        print(f"Combined NN data shape: {combined_data.shape}")
        print("NN data columns:", combined_data.columns.tolist())
        return combined_data
    else:
        return pd.DataFrame()

# Load NN data
print("\nLoading Neural Network MPC data...")
nn_data = load_nn_mpc_data()

# Clean the data: set Power and Flow values close to zero to exactly zero
def clean_near_zero_values(data, threshold=0.2):
    """Set Power and Flow values between -threshold and +threshold to 0"""
    data_clean = data.copy()
    
    # Map column names for NN data
    power_col = 'p_total' if 'p_total' in data_clean.columns else 'Power'
    flow_col = 'q' if 'q' in data_clean.columns else 'Flow'
    
    if power_col in data_clean.columns and flow_col in data_clean.columns:
        # Count values that will be changed
        power_near_zero = ((data_clean[power_col] >= -threshold) & (data_clean[power_col] <= threshold)).sum()
        flow_near_zero = ((data_clean[flow_col] >= -threshold) & (data_clean[flow_col] <= threshold)).sum()
        
        print(f"Setting {power_near_zero} {power_col} values and {flow_near_zero} {flow_col} values near zero to exactly 0")
        
        # Set near-zero values to exactly zero
        data_clean.loc[(data_clean[power_col] >= -threshold) & (data_clean[power_col] <= threshold), power_col] = 0
        data_clean.loc[(data_clean[flow_col] >= -threshold) & (data_clean[flow_col] <= threshold), flow_col] = 0
    
    return data_clean

print("\nCleaning piecewise data...")
piecewise_data = clean_near_zero_values(piecewise_data)

print("\nCleaning global data...")
global_data = clean_near_zero_values(global_data)

if not nn_data.empty:
    print("\nCleaning NN data...")
    nn_data = clean_near_zero_values(nn_data)

# Comprehensive error calculation functions
def calculate_comprehensive_errors(actual, predicted):
    """Calculate comprehensive error metrics"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {
            'MSE': np.nan,
            'MAE': np.nan,
            'MeanError': np.nan,
            'MaxError': np.nan,
            'MAPE': np.nan,
            'R2': np.nan
        }
    
    errors = predicted - actual
    
    # Mean Squared Error
    mse = mean_squared_error(actual, predicted)
    
    # Mean Absolute Error
    mae = mean_absolute_error(actual, predicted)
    
    # Mean Error (can be positive or negative)
    mean_error = np.mean(errors)
    
    # Maximum Absolute Error
    max_error = np.max(np.abs(errors))
    
    # Mean Absolute Percentage Error (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs(errors) / np.abs(actual) * 100
        # Remove infinite values (when actual is 0)
        percentage_errors = percentage_errors[np.isfinite(percentage_errors)]
        mape = np.mean(percentage_errors) if len(percentage_errors) > 0 else np.nan
    
    # R-squared
    r2 = r2_score(actual, predicted)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'MeanError': mean_error,
        'MaxError': max_error,
        'MAPE': mape,
        'R2': r2
    }

def calculate_upc_errors(data, method_name, power_col=None, flow_col=None, head_col=None):
    """Calculate comprehensive errors for UPC relationship q = predict_q_poly(p, h)"""
    if data.empty:
        print(f"{method_name}: No data available")
        return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    # Smart column mapping based on available columns
    if power_col is None:
        if 'p_total' in data.columns:
            power_col = 'p_total'
        elif 'Power' in data.columns:
            power_col = 'Power'
        else:
            print(f"{method_name}: No power column found")
            return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    if flow_col is None:
        if 'q' in data.columns:
            flow_col = 'q'
        elif 'Flow' in data.columns:
            flow_col = 'Flow'
        else:
            print(f"{method_name}: No flow column found")
            return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    if head_col is None:
        if 'h' in data.columns:
            head_col = 'h'
        elif 'Head' in data.columns:
            head_col = 'Head'
        else:
            print(f"{method_name}: No head column found")
            return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    required_cols = [power_col, flow_col, head_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"{method_name}: Missing columns {missing_cols}")
        return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    q_actual_list = []
    q_predicted_list = []
    
    for _, row in data.iterrows():
        p = row[power_col]
        h = row[head_col]
        q_actual = row[flow_col]
        
        try:
            # Predict flow using the nonlinear UPC function
            q_predicted = predict_q_poly(p, h).item()
            
            q_actual_list.append(q_actual)
            q_predicted_list.append(q_predicted)
        except Exception as e:
            print(f"Error in UPC prediction for {method_name}: {e}")
            continue
    
    if not q_actual_list:
        return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    errors = calculate_comprehensive_errors(q_actual_list, q_predicted_list)
    
    print(f"{method_name} UPC Errors:")
    print(f"  MSE: {errors['MSE']:.8f}, MAE: {errors['MAE']:.8f}")
    print(f"  Mean Error: {errors['MeanError']:.8f}, Max Error: {errors['MaxError']:.8f}")
    print(f"  MAPE: {errors['MAPE']:.4f}%, R²: {errors['R2']:.6f}")
    
    return errors

def calculate_vh_errors_from_dynamics(data, method_name):
    """Calculate volume from flow dynamics and then compute comprehensive errors"""
    if data.empty:
        print(f"{method_name}: No data available for dynamics calculation")
        return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    # Map column names for different datasets
    head_col = 'h' if 'h' in data.columns else 'Head'
    flow_col = 'q' if 'q' in data.columns else 'Flow'
    date_col = 'date' if 'date' in data.columns else 'Date'
    time_col = 'hour' if 'hour' in data.columns else 'Time'
    
    required_cols = [head_col, flow_col, date_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"{method_name}: Missing columns for dynamics calculation: {missing_cols}")
        return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    # Group by date to calculate volume dynamics for each day
    v_actual_list = []
    v_predicted_list = []
    
    for date in data[date_col].unique():
        date_data = data[data[date_col] == date].copy()
        
        # Sort by time if time column exists
        if time_col in date_data.columns:
            date_data = date_data.sort_values(time_col).reset_index(drop=True)
        
        # Calculate actual volumes from flow dynamics
        v_actual = [v_low_init]  # Start with initial volume
        for i, row in date_data.iterrows():
            if i > 0:
                v_prev = v_actual[-1]
                q = row[flow_col]
                v_curr = v_prev + 3600 * q  # Volume dynamics
                v_actual.append(v_curr)
        
        # Calculate predicted volumes and errors for this date
        for i, row in date_data.iterrows():
            h = row[head_col]
            try:
                v_pred = h_to_v_low_fitted(torch.tensor(h)).item()
                
                if i < len(v_actual):
                    v_actual_list.append(v_actual[i])
                    v_predicted_list.append(v_pred)
            except Exception as e:
                print(f"Error in VH dynamics calculation for {method_name}: {e}")
                continue
    
    if not v_actual_list:
        return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    errors = calculate_comprehensive_errors(v_actual_list, v_predicted_list)
    
    print(f"{method_name} Volume-Head Errors (from dynamics):")
    print(f"  MSE: {errors['MSE']:.8f}, MAE: {errors['MAE']:.8f}")
    print(f"  Mean Error: {errors['MeanError']:.8f}, Max Error: {errors['MaxError']:.8f}")
    print(f"  MAPE: {errors['MAPE']:.4f}%, R²: {errors['R2']:.6f}")
    
    return errors

def calculate_vh_errors(data, method_name, head_col=None, volume_col=None):
    """Calculate comprehensive errors for volume-head relationship v_low = h_to_v_low_fitted(h)"""
    if data.empty:
        print(f"{method_name}: No data available")
        return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    # Smart column mapping
    if head_col is None:
        if 'h' in data.columns:
            head_col = 'h'
        elif 'Head' in data.columns:
            head_col = 'Head'
        else:
            print(f"{method_name}: No head column found")
            return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    if volume_col is None:
        if 'v_low' in data.columns:
            volume_col = 'v_low'
        elif 'Volume' in data.columns:
            volume_col = 'Volume'
        else:
            print(f"{method_name}: Volume column not found, calculating from dynamics...")
            return calculate_vh_errors_from_dynamics(data, method_name)
    
    required_cols = [head_col, volume_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"{method_name}: Missing columns {missing_cols}")
        return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    v_actual_list = []
    v_predicted_list = []
    
    for _, row in data.iterrows():
        h = row[head_col]
        v_actual = row[volume_col]
        
        try:
            # Predict volume using the nonlinear volume-head function
            v_predicted = h_to_v_low_fitted(torch.tensor(h)).item()
            
            v_actual_list.append(v_actual)
            v_predicted_list.append(v_predicted)
        except Exception as e:
            print(f"Error in VH prediction for {method_name}: {e}")
            continue
    
    if not v_actual_list:
        return {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    
    errors = calculate_comprehensive_errors(v_actual_list, v_predicted_list)
    
    print(f"{method_name} Volume-Head Errors:")
    print(f"  MSE: {errors['MSE']:.8f}, MAE: {errors['MAE']:.8f}")
    print(f"  Mean Error: {errors['MeanError']:.8f}, Max Error: {errors['MaxError']:.8f}")
    print(f"  MAPE: {errors['MAPE']:.4f}%, R²: {errors['R2']:.6f}")
    
    return errors

# Calculate comprehensive errors for all methods
print("\nCalculating comprehensive errors for all methods...")

# Piecewise Linearization
piecewise_upc_errors = calculate_upc_errors(piecewise_data, "Piecewise Linearization")
piecewise_vh_errors = calculate_vh_errors(piecewise_data, "Piecewise Linearization")

# Global Linearization
global_upc_errors = calculate_upc_errors(global_data, "Global Linearization")
global_vh_errors = calculate_vh_errors(global_data, "Global Linearization")

# Neural Network MPC
if not nn_data.empty:
    nn_upc_errors = calculate_upc_errors(nn_data, "Neural Network MPC")
    nn_vh_errors = calculate_vh_errors(nn_data, "Neural Network MPC")
else:
    print("Warning: No Neural Network MPC data available")
    nn_upc_errors = {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}
    nn_vh_errors = {k: np.nan for k in ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']}

# Format metric values for LaTeX table
def format_metric(value, metric_type='regular'):
    """Format metric value, handling NaN appropriately"""
    if pd.isna(value) or np.isnan(value):
        return "N/A"
    elif value == 0:
        return "0.0000"
    elif metric_type == 'percentage':
        return f"{value:.2f}\\%"
    elif abs(value) >= 1e-2:
        return f"{value:.4f}"
    else:
        return f"{value:.2e}"

# Create comprehensive LaTeX table
latex_table = f"""\\begin{{table}}[h]
\\centering
\\tiny
\\begin{{tabular}}{{|l|c|c|c|c|c|c|c|c|c|c|c|c|}}
\\hline
\\multirow{{2}}{{*}}{{Method}} & \\multicolumn{{6}}{{c|}}{{UPC Relationship}} & \\multicolumn{{6}}{{c|}}{{Volume-Head Relationship}} \\\\
\\cline{{2-13}}
 & MSE & MAE & Mean Err & Max Err & MAPE & R² & MSE & MAE & Mean Err & Max Err & MAPE & R² \\\\
\\hline
Piecewise & {format_metric(piecewise_upc_errors['MSE'])} & {format_metric(piecewise_upc_errors['MAE'])} & {format_metric(piecewise_upc_errors['MeanError'])} & {format_metric(piecewise_upc_errors['MaxError'])} & {format_metric(piecewise_upc_errors['MAPE'], 'percentage')} & {format_metric(piecewise_upc_errors['R2'])} & {format_metric(piecewise_vh_errors['MSE'])} & {format_metric(piecewise_vh_errors['MAE'])} & {format_metric(piecewise_vh_errors['MeanError'])} & {format_metric(piecewise_vh_errors['MaxError'])} & {format_metric(piecewise_vh_errors['MAPE'], 'percentage')} & {format_metric(piecewise_vh_errors['R2'])} \\\\
Global & {format_metric(global_upc_errors['MSE'])} & {format_metric(global_upc_errors['MAE'])} & {format_metric(global_upc_errors['MeanError'])} & {format_metric(global_upc_errors['MaxError'])} & {format_metric(global_upc_errors['MAPE'], 'percentage')} & {format_metric(global_upc_errors['R2'])} & {format_metric(global_vh_errors['MSE'])} & {format_metric(global_vh_errors['MAE'])} & {format_metric(global_vh_errors['MeanError'])} & {format_metric(global_vh_errors['MaxError'])} & {format_metric(global_vh_errors['MAPE'], 'percentage')} & {format_metric(global_vh_errors['R2'])} \\\\
Neural Net & {format_metric(nn_upc_errors['MSE'])} & {format_metric(nn_upc_errors['MAE'])} & {format_metric(nn_upc_errors['MeanError'])} & {format_metric(nn_upc_errors['MaxError'])} & {format_metric(nn_upc_errors['MAPE'], 'percentage')} & {format_metric(nn_upc_errors['R2'])} & {format_metric(nn_vh_errors['MSE'])} & {format_metric(nn_vh_errors['MAE'])} & {format_metric(nn_vh_errors['MeanError'])} & {format_metric(nn_vh_errors['MaxError'])} & {format_metric(nn_vh_errors['MAPE'], 'percentage')} & {format_metric(nn_vh_errors['R2'])} \\\\
\\hline
\\end{{tabular}}
\\caption{{Comprehensive Error Analysis for Nonlinear Function Approximations}}
\\label{{tab:comprehensive_error_analysis}}
\\end{{table}}"""

# Create separate tables for UPC and Volume-Head relationships
upc_table = f"""\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|c|c|}}
\\hline
Method & MSE & MAE & Mean Error & Max Error & MAPE (\\%) & R² \\\\
\\hline
Piecewise Linearization & {format_metric(piecewise_upc_errors['MSE'])} & {format_metric(piecewise_upc_errors['MAE'])} & {format_metric(piecewise_upc_errors['MeanError'])} & {format_metric(piecewise_upc_errors['MaxError'])} & {format_metric(piecewise_upc_errors['MAPE'])} & {format_metric(piecewise_upc_errors['R2'])} \\\\
Global Linearization & {format_metric(global_upc_errors['MSE'])} & {format_metric(global_upc_errors['MAE'])} & {format_metric(global_upc_errors['MeanError'])} & {format_metric(global_upc_errors['MaxError'])} & {format_metric(global_upc_errors['MAPE'])} & {format_metric(global_upc_errors['R2'])} \\\\
Neural Network MPC & {format_metric(nn_upc_errors['MSE'])} & {format_metric(nn_upc_errors['MAE'])} & {format_metric(nn_upc_errors['MeanError'])} & {format_metric(nn_upc_errors['MaxError'])} & {format_metric(nn_upc_errors['MAPE'])} & {format_metric(nn_upc_errors['R2'])} \\\\
\\hline
\\end{{tabular}}
\\caption{{UPC Relationship Error Analysis}}
\\label{{tab:upc_error_analysis}}
\\end{{table}}"""

vh_table = f"""\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|c|c|}}
\\hline
Method & MSE & MAE & Mean Error & Max Error & MAPE (\\%) & R² \\\\
\\hline
Piecewise Linearization & {format_metric(piecewise_vh_errors['MSE'])} & {format_metric(piecewise_vh_errors['MAE'])} & {format_metric(piecewise_vh_errors['MeanError'])} & {format_metric(piecewise_vh_errors['MaxError'])} & {format_metric(piecewise_vh_errors['MAPE'])} & {format_metric(piecewise_vh_errors['R2'])} \\\\
Global Linearization & {format_metric(global_vh_errors['MSE'])} & {format_metric(global_vh_errors['MAE'])} & {format_metric(global_vh_errors['MeanError'])} & {format_metric(global_vh_errors['MaxError'])} & {format_metric(global_vh_errors['MAPE'])} & {format_metric(global_vh_errors['R2'])} \\\\
Neural Network MPC & {format_metric(nn_vh_errors['MSE'])} & {format_metric(nn_vh_errors['MAE'])} & {format_metric(nn_vh_errors['MeanError'])} & {format_metric(nn_vh_errors['MaxError'])} & {format_metric(nn_vh_errors['MAPE'])} & {format_metric(nn_vh_errors['R2'])} \\\\
\\hline
\\end{{tabular}}
\\caption{{Volume-Head Relationship Error Analysis}}
\\label{{tab:vh_error_analysis}}
\\end{{table}}"""

print("\n" + "="*120)
print("COMPREHENSIVE LaTeX TABLE:")
print("="*120)
print(latex_table)

print("\n" + "="*80)
print("UPC RELATIONSHIP ERROR TABLE:")
print("="*80)
print(upc_table)

print("\n" + "="*80)
print("VOLUME-HEAD RELATIONSHIP ERROR TABLE:")
print("="*80)
print(vh_table)

# Print detailed summary statistics
print("\n" + "="*120)
print("DETAILED SUMMARY:")
print("="*120)

methods = ["Piecewise Linearization", "Global Linearization", "Neural Network MPC"]
upc_errors_list = [piecewise_upc_errors, global_upc_errors, nn_upc_errors]
vh_errors_list = [piecewise_vh_errors, global_vh_errors, nn_vh_errors]

for i, method in enumerate(methods):
    print(f"\n{method}:")
    print("  UPC Relationship:")
    upc_err = upc_errors_list[i]
    for metric, value in upc_err.items():
        if metric == 'MAPE':
            print(f"    {metric}: {value:.4f}%" if not pd.isna(value) else f"    {metric}: N/A")
        else:
            print(f"    {metric}: {value:.8f}" if not pd.isna(value) else f"    {metric}: N/A")
    
    print("  Volume-Head Relationship:")
    vh_err = vh_errors_list[i]
    for metric, value in vh_err.items():
        if metric == 'MAPE':
            print(f"    {metric}: {value:.4f}%" if not pd.isna(value) else f"    {metric}: N/A")
        else:
            print(f"    {metric}: {value:.8f}" if not pd.isna(value) else f"    {metric}: N/A")

# Find best performing method for each metric
print("\n" + "="*120)
print("BEST PERFORMING METHODS BY METRIC:")
print("="*120)

metrics_to_compare = ['MSE', 'MAE', 'MeanError', 'MaxError', 'MAPE', 'R2']

for metric in metrics_to_compare:
    print(f"\n{metric}:")
    
    # UPC relationship
    upc_values = [err[metric] for err in upc_errors_list]
    valid_upc = [(i, val) for i, val in enumerate(upc_values) if not (pd.isna(val) or np.isnan(val))]
    
    if valid_upc:
        if metric == 'R2':  # Higher is better for R²
            best_upc_idx = max(valid_upc, key=lambda x: x[1])[0]
        else:  # Lower is better for error metrics
            best_upc_idx = min(valid_upc, key=lambda x: abs(x[1]))[0]
        
        print(f"  UPC Best: {methods[best_upc_idx]} ({upc_values[best_upc_idx]:.6f})")
    else:
        print(f"  UPC Best: No valid data")
    
    # Volume-Head relationship
    vh_values = [err[metric] for err in vh_errors_list]
    valid_vh = [(i, val) for i, val in enumerate(vh_values) if not (pd.isna(val) or np.isnan(val))]
    
    if valid_vh:
        if metric == 'R2':  # Higher is better for R²
            best_vh_idx = max(valid_vh, key=lambda x: x[1])[0]
        else:  # Lower is better for error metrics
            best_vh_idx = min(valid_vh, key=lambda x: abs(x[1]))[0]
        
        print(f"  V-H Best: {methods[best_vh_idx]} ({vh_values[best_vh_idx]:.6f})")
    else:
        print(f"  V-H Best: No valid data")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)
# %%
