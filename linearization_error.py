"""
LINEARIZATION ERROR ANALYSIS SCRIPT

This script evaluates the accuracy of three linearization approaches for UPHES optimization:
1. Piecewise Linearization (SOS2-based segmentation)
2. Global Linearization (single linear approximation)  
3. Neural Network (feedforward neural network)

The analysis examines two relationships:
- UPC Relationship: Flow rate prediction q = f(power, head) in m³/s
- Volume-Head Relationship: Volume prediction v = f(head) in m³

Key metrics calculated:
- MAE: Mean Absolute Error (average magnitude of errors)
- Mean Error: Systematic bias (over/under-estimation tendency)
- Max Error: Worst-case approximation error
- MAPE: Mean Absolute Percentage Error (relative accuracy)
- R²: Coefficient of determination (goodness of fit)
"""
# %% Import libraries
import torch
import numpy as np
import dill as pickle
import pandas as pd
import sys

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

# %% Error Analysis
import os
from sklearn.metrics import r2_score, mean_absolute_error
import glob

# Load the existing datasets
print("Loading existing datasets...")
piecewise_data = pd.read_csv('./Data/piecewise_operation_data_SOS2_2024_10seg.csv')
global_data = pd.read_csv('./Data/database_no_piecewise_2024.csv')

print(f"Piecewise data shape: {piecewise_data.shape}")
print(f"Global data shape: {global_data.shape}")

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
            if 'date' not in df.columns:
                date_str = os.path.basename(file).split('_')[0]
                df['date'] = date_str
            all_nn_data.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return pd.concat(all_nn_data, ignore_index=True) if all_nn_data else pd.DataFrame()

# Load NN data
print("\nLoading Neural Network MPC data...")
nn_data = load_nn_mpc_data()

def clean_near_zero_values(data, threshold=0.2):
    """Set Power and Flow values between -threshold and +threshold to 0"""
    data_clean = data.copy()
    
    power_col = 'p_total' if 'p_total' in data_clean.columns else 'Power'
    flow_col = 'q' if 'q' in data_clean.columns else 'Flow'
    
    if power_col in data_clean.columns and flow_col in data_clean.columns:
        power_near_zero = ((data_clean[power_col] >= -threshold) & (data_clean[power_col] <= threshold)).sum()
        flow_near_zero = ((data_clean[flow_col] >= -threshold) & (data_clean[flow_col] <= threshold)).sum()
        
        print(f"Cleaning {power_near_zero} power and {flow_near_zero} flow values near zero")
        
        data_clean.loc[(data_clean[power_col] >= -threshold) & (data_clean[power_col] <= threshold), power_col] = 0
        data_clean.loc[(data_clean[flow_col] >= -threshold) & (data_clean[flow_col] <= threshold), flow_col] = 0
    
    return data_clean

# Clean all datasets
print("\nCleaning datasets...")
piecewise_data = clean_near_zero_values(piecewise_data)
global_data = clean_near_zero_values(global_data)
if not nn_data.empty:
    nn_data = clean_near_zero_values(nn_data)

def calculate_essential_errors(actual, predicted, units=""):
    """Calculate essential error metrics as relative errors with % units"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {'MAE': np.nan, 'Mean_Error': np.nan, 'Max_Error': np.nan, 'MAPE': np.nan, 'R2': np.nan}
    
    # Calculate relative errors (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = (predicted - actual) / actual
        relative_errors = relative_errors[np.isfinite(relative_errors)]
    
    if len(relative_errors) == 0:
        return {'MAE': np.nan, 'Mean_Error': np.nan, 'Max_Error': np.nan, 'MAPE': np.nan, 'R2': np.nan}
    
    # Convert to percentages
    relative_errors_pct = relative_errors * 100
    
    # Relative Mean Absolute Error (%)
    mae = np.mean(np.abs(relative_errors_pct))
    
    # Relative Mean Error (bias) (%)
    mean_error = np.mean(relative_errors_pct)
    
    # Relative Maximum Absolute Error (%)
    max_error = np.max(np.abs(relative_errors_pct))
    
    # Mean Absolute Percentage Error (already calculated correctly)
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs(predicted - actual) / np.abs(actual) * 100
        percentage_errors = percentage_errors[np.isfinite(percentage_errors)]
        mape = np.mean(percentage_errors) if len(percentage_errors) > 0 else np.nan
    
    # R-squared (unchanged)
    r2 = r2_score(actual, predicted)
    
    return {
        'MAE': mae,
        'Mean_Error': mean_error,
        'Max_Error': max_error,
        'MAPE': mape,
        'R2': r2
    }

def calculate_upc_errors(data, method_name):
    """Calculate UPC relationship errors: q = f(p, h)"""
    if data.empty:
        print(f"{method_name}: No data available")
        return {k: np.nan for k in ['MAE', 'Mean_Error', 'Max_Error', 'MAPE', 'R2']}
    
    # Smart column mapping
    power_col = 'p_total' if 'p_total' in data.columns else 'Power'
    flow_col = 'q' if 'q' in data.columns else 'Flow'
    head_col = 'h' if 'h' in data.columns else 'Head'
    
    required_cols = [power_col, flow_col, head_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"{method_name}: Missing columns {missing_cols}")
        return {k: np.nan for k in ['MAE', 'Mean_Error', 'Max_Error', 'MAPE', 'R2']}
    
    q_actual_list = []
    q_predicted_list = []
    
    for _, row in data.iterrows():
        p = row[power_col]  # MW
        h = row[head_col]   # m
        q_actual = row[flow_col]  # m³/s
        
        try:
            q_predicted = predict_q_poly(p, h).item()  # m³/s
            q_actual_list.append(q_actual)
            q_predicted_list.append(q_predicted)
        except Exception as e:
            continue
    
    if not q_actual_list:
        return {k: np.nan for k in ['MAE', 'Mean_Error', 'Max_Error', 'MAPE', 'R2']}
    
    errors = calculate_essential_errors(q_actual_list, q_predicted_list, "m³/s")
    
    print(f"\n{method_name} - UPC Relationship Errors:")
    print(f"  MAE: {errors['MAE']:.2f}%")
    print(f"  Mean Error (bias): {errors['Mean_Error']:.2f}%") 
    print(f"  Max Error: {errors['Max_Error']:.2f}%")
    print(f"  MAPE: {errors['MAPE']:.2f}%")
    print(f"  R²: {errors['R2']:.6f}")
    
    return errors

def calculate_vh_errors_from_dynamics(data, method_name):
    """Calculate volume from flow dynamics for VH relationship analysis"""
    if data.empty:
        return {k: np.nan for k in ['MAE', 'Mean_Error', 'Max_Error', 'MAPE', 'R2']}
    
    head_col = 'h' if 'h' in data.columns else 'Head'
    flow_col = 'q' if 'q' in data.columns else 'Flow'
    date_col = 'date' if 'date' in data.columns else 'Date'
    time_col = 'hour' if 'hour' in data.columns else 'Time'
    
    v_actual_list = []
    v_predicted_list = []
    
    for date in data[date_col].unique():
        date_data = data[data[date_col] == date].copy()
        
        if time_col in date_data.columns:
            date_data = date_data.sort_values(time_col).reset_index(drop=True)
        
        # Calculate actual volumes from flow dynamics
        v_actual = [v_low_init]  # m³
        for i, row in date_data.iterrows():
            if i > 0:
                v_prev = v_actual[-1]
                q = row[flow_col]  # m³/s
                v_curr = v_prev + 3600 * q  # m³ (hourly dynamics)
                v_actual.append(v_curr)
        
        # Calculate predicted volumes
        for i, row in date_data.iterrows():
            h = row[head_col]  # m
            try:
                v_pred = h_to_v_low_fitted(torch.tensor(h)).item()  # m³
                
                if i < len(v_actual):
                    v_actual_list.append(v_actual[i])
                    v_predicted_list.append(v_pred)
            except Exception as e:
                continue
    
    if not v_actual_list:
        return {k: np.nan for k in ['MAE', 'Mean_Error', 'Max_Error', 'MAPE', 'R2']}
    
    errors = calculate_essential_errors(v_actual_list, v_predicted_list, "m³")
    
    print(f"\n{method_name} - Volume-Head Relationship Errors:")
    print(f"  MAE: {errors['MAE']:.1f} m³")
    print(f"  Mean Error (bias): {errors['Mean_Error']:.1f} m³")
    print(f"  Max Error: {errors['Max_Error']:.1f} m³")
    print(f"  MAPE: {errors['MAPE']:.2f}%")
    print(f"  R²: {errors['R2']:.6f}")
    
    return errors

def calculate_vh_errors(data, method_name):
    """Calculate VH relationship errors: v = f(h)"""
    if data.empty:
        return {k: np.nan for k in ['MAE', 'Mean_Error', 'Max_Error', 'MAPE', 'R2']}
    
    head_col = 'h' if 'h' in data.columns else 'Head'
    volume_col = 'v_low' if 'v_low' in data.columns else 'Volume'
    
    if volume_col not in data.columns:
        print(f"{method_name}: Volume column not found, calculating from dynamics...")
        return calculate_vh_errors_from_dynamics(data, method_name)
    
    v_actual_list = []
    v_predicted_list = []
    
    for _, row in data.iterrows():
        h = row[head_col]  # m
        v_actual = row[volume_col]  # m³
        
        try:
            v_predicted = h_to_v_low_fitted(torch.tensor(h)).item()  # m³
            v_actual_list.append(v_actual)
            v_predicted_list.append(v_predicted)
        except Exception as e:
            continue
    
    if not v_actual_list:
        return {k: np.nan for k in ['MAE', 'Mean_Error', 'Max_Error', 'MAPE', 'R2']}
    
    errors = calculate_essential_errors(v_actual_list, v_predicted_list, "m³")
    
    print(f"\n{method_name} - Volume-Head Relationship Errors:")
    print(f"  MAE: {errors['MAE']:.2f}%")
    print(f"  Mean Error (bias): {errors['Mean_Error']:.2f}%")
    print(f"  Max Error: {errors['Max_Error']:.2f}%")
    print(f"  MAPE: {errors['MAPE']:.2f}%")
    print(f"  R²: {errors['R2']:.6f}")
    
    return errors

# Calculate errors for all methods
print("\n" + "="*80)
print("LINEARIZATION ERROR ANALYSIS")
print("="*80)

# Calculate errors
piecewise_upc = calculate_upc_errors(piecewise_data, "Piecewise Linearization")
piecewise_vh = calculate_vh_errors(piecewise_data, "Piecewise Linearization")

global_upc = calculate_upc_errors(global_data, "Global Linearization")
global_vh = calculate_vh_errors(global_data, "Global Linearization")

if not nn_data.empty:
    nn_upc = calculate_upc_errors(nn_data, "Neural Network MPC")
    nn_vh = calculate_vh_errors(nn_data, "Neural Network MPC")
else:
    print("\nWarning: No Neural Network MPC data available")
    nn_upc = {k: np.nan for k in ['MAE', 'Mean_Error', 'Max_Error', 'MAPE', 'R2']}
    nn_vh = {k: np.nan for k in ['MAE', 'Mean_Error', 'Max_Error', 'MAPE', 'R2']}

def format_metric(value, metric_type='regular', units=''):
    """Format metric value with appropriate precision and units"""
    if pd.isna(value) or np.isnan(value):
        return "N/A"
    elif value == 0:
        return f"0.0000{units}"
    elif metric_type == 'percentage':
        return f"{value:.2f}\\%"
    elif metric_type == 'r2':
        return f"{value:.4f}"
    elif abs(value) >= 1e-2:
        return f"{value:.4f}{units}"
    else:
        return f"{value:.2e}{units}"

# Create LaTeX tables with relative error units
upc_table = f"""\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|c|}}
\\hline
Method & MAE (\\%) & Mean Error (\\%) & Max Error (\\%) & MAPE (\\%) & R² \\\\
\\hline
Piecewise Linearization & {format_metric(piecewise_upc['MAE'], 'percentage')} & {format_metric(piecewise_upc['Mean_Error'], 'percentage')} & {format_metric(piecewise_upc['Max_Error'], 'percentage')} & {format_metric(piecewise_upc['MAPE'], 'percentage')} & {format_metric(piecewise_upc['R2'], 'r2')} \\\\
Global Linearization & {format_metric(global_upc['MAE'], 'percentage')} & {format_metric(global_upc['Mean_Error'], 'percentage')} & {format_metric(global_upc['Max_Error'], 'percentage')} & {format_metric(global_upc['MAPE'], 'percentage')} & {format_metric(global_upc['R2'], 'r2')} \\\\
Neural Network MPC & {format_metric(nn_upc['MAE'], 'percentage')} & {format_metric(nn_upc['Mean_Error'], 'percentage')} & {format_metric(nn_upc['Max_Error'], 'percentage')} & {format_metric(nn_upc['MAPE'], 'percentage')} & {format_metric(nn_upc['R2'], 'r2')} \\\\
\\hline
\\end{{tabular}}
\\caption{{UPC Relationship Error Analysis: Flow Rate Prediction Accuracy (Relative Errors)}}
\\label{{tab:upc_error_analysis}}
\\end{{table}}"""

vh_table = f"""\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|c|}}
\\hline
Method & MAE (\\%) & Mean Error (\\%) & Max Error (\\%) & MAPE (\\%) & R² \\\\
\\hline
Piecewise Linearization & {format_metric(piecewise_vh['MAE'], 'percentage')} & {format_metric(piecewise_vh['Mean_Error'], 'percentage')} & {format_metric(piecewise_vh['Max_Error'], 'percentage')} & {format_metric(piecewise_vh['MAPE'], 'percentage')} & {format_metric(piecewise_vh['R2'], 'r2')} \\\\
Global Linearization & {format_metric(global_vh['MAE'], 'percentage')} & {format_metric(global_vh['Mean_Error'], 'percentage')} & {format_metric(global_vh['Max_Error'], 'percentage')} & {format_metric(global_vh['MAPE'], 'percentage')} & {format_metric(global_vh['R2'], 'r2')} \\\\
Neural Network MPC & {format_metric(nn_vh['MAE'], 'percentage')} & {format_metric(nn_vh['Mean_Error'], 'percentage')} & {format_metric(nn_vh['Max_Error'], 'percentage')} & {format_metric(nn_vh['MAPE'], 'percentage')} & {format_metric(nn_vh['R2'], 'r2')} \\\\
\\hline
\\end{{tabular}}
\\caption{{Volume-Head Relationship Error Analysis: Volume Prediction Accuracy (Relative Errors)}}
\\label{{tab:vh_error_analysis}}
\\end{{table}}"""

print("\n" + "="*80)
print("LaTeX TABLES WITH UNITS:")
print("="*80)
print("\nUPC RELATIONSHIP ERROR TABLE:")
print(upc_table)
print("\nVOLUME-HEAD RELATIONSHIP ERROR TABLE:")
print(vh_table)

# Summary analysis
print("\n" + "="*80)
print("PERFORMANCE SUMMARY:")
print("="*80)

methods = ["Piecewise Linearization", "Global Linearization", "Neural Network MPC"]
upc_errors = [piecewise_upc, global_upc, nn_upc]
vh_errors = [piecewise_vh, global_vh, nn_vh]

# Find best performing methods
print("\nBest performing methods:")
for metric in ['MAE', 'MAPE', 'R2']:
    print(f"\n{metric}:")
    
    # UPC relationship
    upc_values = [err[metric] for err in upc_errors]
    valid_upc = [(i, val) for i, val in enumerate(upc_values) if not (pd.isna(val) or np.isnan(val))]
    
    if valid_upc:
        if metric == 'R2':
            best_upc_idx = max(valid_upc, key=lambda x: x[1])[0]
        else:
            best_upc_idx = min(valid_upc, key=lambda x: x[1])[0]
        print(f"  UPC Best: {methods[best_upc_idx]}")
    
    # Volume-Head relationship
    vh_values = [err[metric] for err in vh_errors]
    valid_vh = [(i, val) for i, val in enumerate(vh_values) if not (pd.isna(val) or np.isnan(val))]
    
    if valid_vh:
        if metric == 'R2':
            best_vh_idx = max(valid_vh, key=lambda x: x[1])[0]
        else:
            best_vh_idx = min(valid_vh, key=lambda x: x[1])[0]
        print(f"  V-H Best: {methods[best_vh_idx]}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
# %%

