"""
Script for generating noisy variants of hydro power plant optimization results.

This script takes baseline MIQP (Mixed Integer Quadratic Programming) optimization results
and generates multiple datasets with different types of noise/variations:

1. RELATIVE NOISE: Adds relative noise (10%-50%) to baseline power schedules based on 
    capacity range at each head level. Uses retry mechanism to ensure feasibility.

2. RANDOM SAMPLING: Generates random power schedules while preserving operational modes
    (turbine/pump/idle) from original data. Also uses retry for volume constraint compliance.

Output: Multiple CSV files with noisy/random variants of the original optimization results
for robust analysis of hydro power plant operations under uncertainty.
"""
# %%  Imports
import torch
import numpy as np
import pandas as pd
import dill as pickle
import sys
import os
from copy import deepcopy

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")

# Load portfolio data
sys.path.append('../Library')
from V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# Load preprocessed functions & data
with open('../preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = 77.0
v_low_init = h_to_v_low_fitted(head_init)

# %% noise insertion 
class HydroParameters:
    def __init__(
        self,
        time_horizon=24,
        operational_cost=0.4,
        rho=1000,
        g=9.81,
        mu=0.9,
        head_init=head_init,
        v_low_init=v_low_init,
        target_head=target_head,
        target_vol_low=target_vol_low,
        max_vol_up=max_vol_up,
        min_vol_low=min_vol_low,
        neg_min=neg_min,
        neg_max=neg_max,
        pos_min=pos_min,
        pos_max=pos_max,
        predict_q_poly=predict_q_poly,
        h_to_v_low_fitted=h_to_v_low_fitted,
        v_low_to_h_fitted=v_low_to_h_fitted,
    ):
        self.time_horizon = time_horizon
        self.operational_cost = operational_cost
        self.rho = torch.tensor(rho, dtype=torch.float32, device=device)
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)
        self.head_init = torch.tensor(head_init, dtype=torch.float32, device=device)
        self.v_low_init = torch.tensor(v_low_init, dtype=torch.float32, device=device)
        self.target_head = torch.tensor(target_head, dtype=torch.float32, device=device)
        self.target_vol_low = torch.tensor(target_vol_low, dtype=torch.float32, device=device)
        self.max_vol_up = torch.tensor(max_vol_up, dtype=torch.float32, device=device)
        self.min_vol_low = torch.tensor(min_vol_low, dtype=torch.float32, device=device)
        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.predict_q_poly = predict_q_poly
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.v_low_to_h_fitted = v_low_to_h_fitted

class BaselineSimulator:
    """Baseline simulator without noise for generating reference results."""
    def __init__(self, params):
        self.params = params

    def simulate_daily_operation(self, p_schedule, initial_head, track_changes=False):
        """Simulate daily operation without noise, optionally tracking mode changes and clamping."""
        TH = len(p_schedule)
        p_list = []
        q_list = []
        h_list = []
        v_list = []
        
        # Tracking variables
        mode_changes = []
        power_clamps = []
        
        # Start with initial volume corresponding to initial head
        v_current = self.params.h_to_v_low_fitted(initial_head)
        
        for i in range(TH):
            p_current = p_schedule[i]
            original_power = p_current.item()
            
            # For hour 0, use initial head; for other hours, calculate from volume
            if i == 0:
                h_current = initial_head
            else:
                h_current = self.params.v_low_to_h_fitted(v_current)
            
            q_candidate = torch.zeros_like(p_current)
            p_clamped = p_current
            power_was_clamped = False
            original_mode = None
            final_mode = None

            # Determine original mode
            if original_power > 0.5:
                original_mode = "Turbine"
            elif original_power < -0.5:
                original_mode = "Pump"
            else:
                original_mode = "Idle"

            if p_current > 0.5:  # Turbine mode
                p_min_turb = self.params.pos_min(h_current)
                p_max_turb = self.params.pos_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                if abs(p_clamped.item() - p_current.item()) > 1e-6:
                    power_was_clamped = True
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            elif p_current < -0.5:  # Pump mode
                p_min_pump = self.params.neg_min(h_current)
                p_max_pump = self.params.neg_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_pump, max=p_max_pump)
                if abs(p_clamped.item() - p_current.item()) > 1e-6:
                    power_was_clamped = True
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)

            v_next = v_current + q_candidate * 3600
            out_of_bounds = (v_next > self.params.max_vol_up) | (v_next < self.params.min_vol_low)

            if out_of_bounds:
                p_final = torch.zeros_like(p_current)
                q_final = torch.zeros_like(q_candidate)
                v_next = v_current
                h_final = h_current
                final_mode = "Idle"
                if original_mode != "Idle":
                    mode_changes.append({
                        'hour': i,
                        'original_mode': original_mode,
                        'final_mode': final_mode,
                        'reason': 'Volume out of bounds',
                        'original_power': original_power,
                        'final_power': 0.0
                    })
            else:
                p_final = p_clamped if abs(p_current) > 0.5 else torch.zeros_like(p_current)
                q_final = q_candidate
                h_final = self.params.v_low_to_h_fitted(v_next)
                
                # Determine final mode
                final_power = p_final.item()
                if final_power > 0.5:
                    final_mode = "Turbine"
                elif final_power < -0.5:
                    final_mode = "Pump"
                else:
                    final_mode = "Idle"

            # Track power clamping
            if track_changes and power_was_clamped:
                power_clamps.append({
                    'hour': i,
                    'original_power': original_power,
                    'clamped_power': p_clamped.item(),
                    'mode': original_mode,
                    'head': h_current.item()
                })

            # Track mode changes (only if mode actually changed and not due to out of bounds)
            if track_changes and not out_of_bounds and original_mode != final_mode:
                mode_changes.append({
                    'hour': i,
                    'original_mode': original_mode,
                    'final_mode': final_mode,
                    'reason': 'Mode transition',
                    'original_power': original_power,
                    'final_power': p_final.item()
                })

            p_list.append(p_final)
            q_list.append(q_final)
            h_list.append(h_final)
            v_list.append(v_next)
            v_current = v_next

        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list)
        v_low_sim = torch.stack(v_list)
        
        if track_changes:
            return p_sim, q_sim, h_sim, v_low_sim, mode_changes, power_clamps
        else:
            return p_sim, q_sim, h_sim, v_low_sim

class NoiseSimulator:
    """Simulator with relative noise addition and retry mechanism."""
    def __init__(self, params, relative_noise_level=0.15):
        self.params = params
        self.relative_noise_level = relative_noise_level
        
    def get_capacity_range(self, head):
        """Get capacity range for turbine and pump modes at given head."""
        pos_min_val = self.params.pos_min(head).item()
        pos_max_val = self.params.pos_max(head).item()
        neg_min_val = self.params.neg_min(head).item()
        neg_max_val = self.params.neg_max(head).item()
        
        return pos_min_val, pos_max_val, neg_min_val, neg_max_val
    
    def generate_noisy_power(self, baseline_power, head, debug=False):
        """
        Generate noisy power in overlapping region between noise region and feasible region.
        
        Args:
            baseline_power: Original power value
            head: Current head level
            debug: Print debug information
            
        Returns:
            Tuple of (noisy_power, relative_error, debug_info) or (None, None, debug_info) if invalid
        """
        pos_min_val, pos_max_val, neg_min_val, neg_max_val = self.get_capacity_range(head)
        
        debug_info = {
            'head': head.item(),
            'baseline_power': baseline_power,
            'pos_range': (pos_min_val, pos_max_val),
            'neg_range': (neg_min_val, neg_max_val)
        }
        
        # Determine mode and capacity range
        if baseline_power > 0.5:  # Turbine mode
            capacity_range = pos_max_val - pos_min_val
            feasible_min, feasible_max = pos_min_val, pos_max_val
            mode = "turbine"
        elif baseline_power < -0.5:  # Pump mode
            capacity_range = neg_max_val - neg_min_val
            feasible_min, feasible_max = neg_min_val, neg_max_val
            mode = "pump"
        else:  # Idle mode
            debug_info['mode'] = "idle"
            return 0.0, 0.0, debug_info  # No noise for idle mode
        
        # Calculate noise range based on capacity
        noise_magnitude = self.relative_noise_level * capacity_range
        noise_min = baseline_power - noise_magnitude
        noise_max = baseline_power + noise_magnitude
        
        # Find overlapping region with feasible region
        overlap_min = max(noise_min, feasible_min)
        overlap_max = min(noise_max, feasible_max)
        
        debug_info.update({
            'mode': mode,
            'capacity_range': capacity_range,
            'noise_magnitude': noise_magnitude,
            'noise_range': (noise_min, noise_max),
            'feasible_range': (feasible_min, feasible_max),
            'overlap_range': (overlap_min, overlap_max)
        })
        
        # Check if there's a valid overlap
        if overlap_min >= overlap_max:
            debug_info['error'] = f"No overlap: noise range [{noise_min:.2f}, {noise_max:.2f}] vs feasible [{feasible_min:.2f}, {feasible_max:.2f}]"
            return None, None, debug_info
        
        # Generate random power in overlapping region
        noisy_power = np.random.uniform(overlap_min, overlap_max)
        relative_error = abs(noisy_power - baseline_power) / capacity_range
        
        debug_info.update({
            'noisy_power': noisy_power,
            'relative_error': relative_error
        })
        
        return noisy_power, relative_error, debug_info
    
    def simulate_daily_operation_with_noise(self, baseline_schedule, initial_head, max_retries=100, debug=False):
        """
        Simulate daily operation with noise, retrying if constraints are violated.
        """
        TH = len(baseline_schedule)
        
        for retry in range(max_retries):
            p_list = []
            q_list = []
            h_list = []
            v_list = []
            relative_errors = []
            
            # Start with initial volume corresponding to initial head
            v_current = self.params.h_to_v_low_fitted(initial_head)
            failed = False
            failure_reason = ""
            
            for i in range(TH):
                baseline_power = baseline_schedule[i]
                
                # For hour 0, use initial head; for other hours, calculate from volume
                if i == 0:
                    h_current = initial_head
                else:
                    h_current = self.params.v_low_to_h_fitted(v_current)
                
                # Generate noisy power
                noisy_power, relative_error, debug_info = self.generate_noisy_power(baseline_power, h_current, debug)
                
                if noisy_power is None:
                    # No overlap - retry whole day
                    failure_reason = f"Hour {i}: {debug_info.get('error', 'Unknown error')}"
                    failed = True
                    break
                
                relative_errors.append(relative_error)
                p_current = torch.tensor(noisy_power, dtype=torch.float32, device=device)
                
                # Simulate power and flow (same as baseline simulator)
                q_candidate = torch.zeros_like(p_current)
                p_clamped = p_current

                if p_current > 0.5:  # Turbine mode
                    p_min_turb = self.params.pos_min(h_current)
                    p_max_turb = self.params.pos_max(h_current)
                    p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                    q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
                elif p_current < -0.5:  # Pump mode
                    p_min_pump = self.params.neg_min(h_current)
                    p_max_pump = self.params.neg_max(h_current)
                    p_clamped = torch.clamp(p_current, min=p_min_pump, max=p_max_pump)
                    q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)

                v_next = v_current + q_candidate * 3600
                out_of_bounds = (v_next > self.params.max_vol_up) | (v_next < self.params.min_vol_low)

                if out_of_bounds:
                    # Volume out of bounds - retry whole day
                    failure_reason = f"Hour {i}: Volume out of bounds {v_next:.0f} (limits: [{self.params.min_vol_low:.0f}, {self.params.max_vol_up:.0f}])"
                    failed = True
                    break

                p_final = p_clamped if abs(p_current) > 0.5 else torch.zeros_like(p_current)
                q_final = q_candidate
                h_final = self.params.v_low_to_h_fitted(v_next)

                p_list.append(p_final)
                q_list.append(q_final)
                h_list.append(h_final)
                v_list.append(v_next)
                v_current = v_next
            
            if not failed:
                # Success! Return results
                p_sim = torch.stack(p_list)
                q_sim = torch.stack(q_list)
                h_sim = torch.stack(h_list)
                v_low_sim = torch.stack(v_list)
                
                return p_sim, q_sim, h_sim, v_low_sim, True, retry, relative_errors
        
        # All retries failed
        return None, None, None, None, False, max_retries, []

def process_daily_schedule(day_data, noise_simulator, baseline_simulator):
    """
    Process one day's schedule by adding relative noise to baseline results and re-simulating.
    
    Workflow:
    1. Original data → Baseline simulation (handles clamping/mode changes)
    2. Baseline results → Add relative noise → Noisy simulation
    
    Args:
        day_data: DataFrame containing one day's hourly data
        noise_simulator: NoiseSimulator instance
        baseline_simulator: BaselineSimulator instance
    
    Returns:
        Tuple of (modified_data, actual_relative_errors) or (None, None) if failed
    """
    # Extract original power schedule
    original_powers = torch.tensor(day_data['power'].values, dtype=torch.float32, device=device)
    initial_head = torch.tensor(head_init, dtype=torch.float32, device=device)
    
    # STEP 1: Run baseline simulation on original data to get the baseline schedule
    p_baseline, q_baseline, h_baseline, v_baseline = baseline_simulator.simulate_daily_operation(original_powers, initial_head)
    
    # STEP 2: Apply relative noise to baseline results and re-simulate
    p_sim, q_sim, h_sim, v_low_sim, success, retry_count, relative_errors = \
        noise_simulator.simulate_daily_operation_with_noise(p_baseline.cpu().numpy(), initial_head, max_retries=50)
    
    if not success:
        return None, None
    
    # Create modified data
    modified_data = day_data.copy()
    modified_data['power'] = p_sim.cpu().numpy()
    modified_data['flow'] = q_sim.cpu().numpy()
    modified_data['head'] = h_sim.cpu().numpy()
    modified_data['volume'] = v_low_sim.cpu().numpy()
    
    return modified_data, relative_errors

def process_noise_level(df, noise_level, params):
    """
    Process the entire dataset for a specific relative noise level.
    
    Args:
        df: Original DataFrame
        noise_level: Relative noise level to apply (e.g., 0.1 for 10%)
        params: HydroParameters instance
    
    Returns:
        Tuple of (DataFrame with noise applied, actual_relative_errors_stats)
    """
    print(f"\nProcessing relative noise level {noise_level*100:.0f}% (applied to baseline results)...")
    
    # Set random seed for each noise level to ensure reproducibility
    np.random.seed(42 + int(noise_level * 1000))  # Different seed for each noise level
    
    # Initialize simulators
    baseline_simulator = BaselineSimulator(params)
    noise_simulator = NoiseSimulator(params, relative_noise_level=noise_level)
    
    modified_results = []
    all_relative_errors = []  # Collect all relative errors for analysis
    unique_dates = df['date'].unique()
    total_dates = len(unique_dates)
    failed_dates = []
    
    for idx, date in enumerate(unique_dates, 1):
        if idx % 50 == 0:  # Print progress every 50 days
            print(f"  Processing {date} ({idx}/{total_dates})...")
        
        # Get data for this day
        day_data = df[df['date'] == date].copy().reset_index(drop=True)
        
        if len(day_data) != 24:
            print(f"  Warning: Day {date} has {len(day_data)} hours instead of 24. Skipping...")
            continue
        
        try:
            # Process this day's schedule
            modified_day_data, daily_relative_errors = process_daily_schedule(day_data, noise_simulator, baseline_simulator)
            
            if modified_day_data is not None:
                modified_results.append(modified_day_data)
                # Collect relative errors (excluding idle mode hours which have 0.0 error)
                non_idle_errors = [err for err in daily_relative_errors if err > 0.0]
                all_relative_errors.extend(non_idle_errors)
            else:
                failed_dates.append(date)
                
        except Exception as e:
            print(f"  Error processing {date}: {e}")
            failed_dates.append(date)
            continue
    
    # Calculate actual relative error statistics
    actual_error_stats = None
    if all_relative_errors:
        actual_error_stats = {
            'target_noise_level': noise_level,
            'count': len(all_relative_errors),
            'mean': np.mean(all_relative_errors),
            'std': np.std(all_relative_errors),
            'min': np.min(all_relative_errors),
            'max': np.max(all_relative_errors),
            'median': np.median(all_relative_errors),
            'percentile_25': np.percentile(all_relative_errors, 25),
            'percentile_75': np.percentile(all_relative_errors, 75)
        }
    
    # Report results
    if failed_dates:
        print(f"  Failed to process {len(failed_dates)} dates: {failed_dates[:10]}...")  # Show first 10
    
    # Combine all modified results
    if modified_results:
        final_df = pd.concat(modified_results, ignore_index=True)
        return final_df, actual_error_stats
    else:
        return None, None

def analyze_baseline_changes(df, params):
    """
    Analyze mode changes and power clamping in baseline simulation for sample dates.
    """
    print(f"\n=== BASELINE SIMULATION ANALYSIS ===")
    
    baseline_simulator = BaselineSimulator(params)
    unique_dates = df['date'].unique()
    sample_dates = unique_dates[:5]  # Analyze first 5 dates as sample
    
    total_mode_changes = 0
    total_power_clamps = 0
    
    for date in sample_dates:
        day_data = df[df['date'] == date].copy().reset_index(drop=True)
        
        if len(day_data) != 24:
            continue
            
        original_powers = torch.tensor(day_data['power'].values, dtype=torch.float32, device=device)
        initial_head = torch.tensor(head_init, dtype=torch.float32, device=device)
        
        # Run baseline simulation with tracking
        p_sim, q_sim, h_sim, v_low_sim, mode_changes, power_clamps = \
            baseline_simulator.simulate_daily_operation(original_powers, initial_head, track_changes=True)
        
        if mode_changes or power_clamps:
            print(f"\nDate: {date}")
            
        if mode_changes:
            print(f"  Mode changes ({len(mode_changes)}):")
            for change in mode_changes:
                print(f"    Hour {change['hour']}: {change['original_mode']} → {change['final_mode']} "
                      f"({change['original_power']:.2f} → {change['final_power']:.2f} MW) - {change['reason']}")
            total_mode_changes += len(mode_changes)
        
        if power_clamps:
            print(f"  Power clamps ({len(power_clamps)}):")
            for clamp in power_clamps:
                print(f"    Hour {clamp['hour']}: {clamp['original_power']:.2f} → {clamp['clamped_power']:.2f} MW "
                      f"({clamp['mode']} mode, head={clamp['head']:.1f}m)")
            total_power_clamps += len(power_clamps)
    
    print(f"\nTotal mode changes in sample: {total_mode_changes}")
    print(f"Total power clamps in sample: {total_power_clamps}")

def main_noise_insertion():
    """Main function to process MIQP linear results with multiple relative noise levels."""
    
    # Define relative noise levels to generate (0% to 80%)
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Load original results from MIQP directory
    original_file = "../MIQP/MIQP_linear/MILP_global_linear_results.csv"
    
    if not os.path.exists(original_file):
        print(f"Error: {original_file} not found!")
        print("Please make sure the file exists in the MIQP/MIQP_linear directory.")
        return
    
    print(f"Loading {original_file}...")
    df = pd.read_csv(original_file)
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique dates: {len(df['date'].unique())}")
    
    # Initialize parameters
    head_init_val = torch.tensor(head_init, dtype=torch.float32, device=device)
    v_low_init_val = torch.tensor(v_low_init, dtype=torch.float32, device=device)
    
    params = HydroParameters(
        head_init=head_init_val,
        v_low_init=v_low_init_val,
        neg_min=neg_min, neg_max=neg_max,
        pos_min=pos_min, pos_max=pos_max,
        predict_q_poly=predict_q_poly,
        h_to_v_low_fitted=h_to_v_low_fitted,
        v_low_to_h_fitted=v_low_to_h_fitted
    )
    
    # Analyze baseline simulation changes
    analyze_baseline_changes(df, params)
    
    # Create 0% noise file (direct copy of original)
    print(f"\n=== CREATING 0% NOISE FILE ===")
    output_0pct = "MIQP_linear_results_relative_noise_00pct.csv"
    df.to_csv(output_0pct, index=False)
    print(f"✓ {output_0pct} created (direct copy of original)")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique dates: {len(df['date'].unique())}")
    
    print(f"\n=== NOISE APPLICATION WORKFLOW ===")
    print("1. Original MIQP data → Baseline simulation (handles any clamping/mode changes)")
    print("2. Baseline results → Add relative noise (% of capacity range) → Noisy simulation") 
    print("3. Generate CSV files and analyze actual relative errors achieved")
    print("Note: Noise is applied to baseline results, not original data")
    
    # Process each noise level (skip 0.0 as it's already created)
    for noise_level in noise_levels:
        if noise_level == 0.0:
            continue  # Already created above
        try:
            # Process the dataset for this noise level
            final_df, actual_error_stats = process_noise_level(df, noise_level, params)
            
            if final_df is not None:
                # Save to CSV file in current directory
                output_filename = f"MIQP_linear_results_relative_noise_{noise_level*100:.0f}pct.csv"
                final_df.to_csv(output_filename, index=False)
                
                print(f"  Results saved to {output_filename}")
                print(f"  Total rows: {len(final_df)}")
                print(f"  Success rate: {len(final_df)/len(df)*100:.1f}%")
                
                # Report actual relative error statistics
                if actual_error_stats:
                    print(f"  \n=== ACTUAL RELATIVE ERROR ANALYSIS ===")
                    print(f"  Target noise level: {actual_error_stats['target_noise_level']*100:.0f}%")
                    print(f"  Non-idle hours analyzed: {actual_error_stats['count']:,}")
                    print(f"  Actual mean relative error: {actual_error_stats['mean']*100:.2f}%")
                    print(f"  Actual median relative error: {actual_error_stats['median']*100:.2f}%")
                    print(f"  Actual std deviation: {actual_error_stats['std']*100:.2f}%")
                    print(f"  Actual range: [{actual_error_stats['min']*100:.2f}%, {actual_error_stats['max']*100:.2f}%]")
                    print(f"  25th-75th percentile: [{actual_error_stats['percentile_25']*100:.2f}%, {actual_error_stats['percentile_75']*100:.2f}%]")
                    
                    # Compare target vs actual (avoid division by zero)
                    if noise_level > 0:
                        deviation_from_target = abs(actual_error_stats['mean'] - noise_level) / noise_level * 100
                        print(f"  Deviation from target: {deviation_from_target:.1f}%")
                
                # Show summary statistics
                print(f"  \n=== PHYSICAL VARIABLE RANGES ===")
                print(f"  Power range: [{final_df['power'].min():.2f}, {final_df['power'].max():.2f}] MW")
                print(f"  Head range: [{final_df['head'].min():.2f}, {final_df['head'].max():.2f}] m")
                print(f"  Flow range: [{final_df['flow'].min():.2f}, {final_df['flow'].max():.2f}] m³/s")
                print(f"  Volume range: [{final_df['volume'].min():.0f}, {final_df['volume'].max():.0f}] m³")
                
                # Show mode distribution
                idle_count = (abs(final_df['power']) < 0.5).sum()
                turbine_count = (final_df['power'] > 0.5).sum()
                pump_count = (final_df['power'] < -0.5).sum()
                
                print(f"  \n=== MODE DISTRIBUTION ===")
                print(f"  Idle: {idle_count:,} hours ({100*idle_count/len(final_df):.1f}%)")
                print(f"  Turbine: {turbine_count:,} hours ({100*turbine_count/len(final_df):.1f}%)")
                print(f"  Pump: {pump_count:,} hours ({100*pump_count/len(final_df):.1f}%)")
                
            else:
                print(f"  Failed to process relative noise level {noise_level*100:.0f}%")
                
        except Exception as e:
            print(f"  Error processing noise level {noise_level*100:.0f}%: {e}")
            continue
    
    print(f"\nAll relative noise levels processed!")
    print(f"Generated files:")
    for noise_level in noise_levels:
        filename = f"MIQP_linear_results_relative_noise_{noise_level*100:.0f}pct.csv"
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (failed)")

    # Print a chart of actual vs target relative errors
    print(f"\n=== SUMMARY OF ACTUAL VS TARGET RELATIVE ERRORS ===")
    print(f"{'Noise Level (%)':<20}{'Actual Mean (%)':<20}")
    for noise_level in noise_levels:
        filename = f"MIQP_linear_results_relative_noise_{noise_level*100:.0f}pct.csv"
        if os.path.exists(filename):
            df_noise = pd.read_csv(filename)
            non_idle_hours = df_noise[abs(df_noise['power']) >= 0.5]
            if not non_idle_hours.empty:
                actual_errors = []
                baseline_simulator = BaselineSimulator(params)
                noise_simulator = NoiseSimulator(params, relative_noise_level=noise_level)
                
                unique_dates = non_idle_hours['date'].unique()
                for date in unique_dates:
                    day_data = non_idle_hours[non_idle_hours['date'] == date].copy().reset_index(drop=True)
                    if len(day_data) != 24:
                        continue
                    original_powers = torch.tensor(day_data['power'].values, dtype=torch.float32, device=device)
                    initial_head = torch.tensor(head_init, dtype=torch.float32, device=device)
                    p_baseline, _, _, _ = baseline_simulator.simulate_daily_operation(original_powers, initial_head)
                    for i in range(24):
                        if abs(original_powers[i].item()) >= 0.5:
                            noisy_power = day_data.loc[i, 'power']
                            baseline_power = p_baseline[i].item()
                            head = day_data.loc[i, 'head']
                            _, rel_error, _ = noise_simulator.generate_noisy_power(baseline_power, head)
                            if rel_error is not None and rel_error > 0.0:
                                actual_errors.append(rel_error)
                
                if actual_errors:
                    mean_actual_error = np.mean(actual_errors) * 100
                    print(f"{noise_level*100:<20.0f}{mean_actual_error:<20.2f}")
                else:
                    print(f"{noise_level*100:<20.0f}{'N/A':<20}")
            else:
                print(f"{noise_level*100:<20.0f}{'N/A':<20}")
        else:
            print(f"{noise_level*100:<20.0f}{'File not found':<20}")

if __name__ == "__main__":
    main_noise_insertion()

# %% Random Sampling with Mode Preservation
"""
Code for random sampling of power schedules within power boundaries
while retaining operational modes with retry mechanism for boundary violations.
"""

class RandomModeSampler:
    """Random sampler that preserves operational modes with retry for boundary violations."""
    def __init__(self, params):
        self.params = params
    
    def sample_power_for_mode(self, mode, head):
        """Randomly sample power within feasible range for given mode and head."""
        if mode == "turbine":
            p_min = self.params.pos_min(head).item()
            p_max = self.params.pos_max(head).item()
            return np.random.uniform(p_min, p_max)
        elif mode == "pump":
            p_min = self.params.neg_min(head).item()
            p_max = self.params.neg_max(head).item()
            return np.random.uniform(p_min, p_max)
        else:  # idle
            return 0.0
    
    def get_mode(self, power):
        """Determine mode from power value."""
        if power > 0.5:
            return "turbine"
        elif power < -0.5:
            return "pump"
        else:
            return "idle"
    
    def generate_and_simulate_random_schedule(self, original_schedule, initial_head, max_retries=100):
        """
        Generate random power schedule preserving modes with retry for boundary violations.
        Retries entire day if any hour goes out of bounds.
        
        Returns:
            Tuple of (random_schedule, p_sim, q_sim, h_sim, v_low_sim, success, retry_count)
        """
        if isinstance(original_schedule, torch.Tensor):
            original_schedule = original_schedule.cpu().numpy()
        
        TH = len(original_schedule)
        original_modes = [self.get_mode(p) for p in original_schedule]
        
        for retry in range(max_retries):
            random_powers = []
            p_list, q_list, h_list, v_list = [], [], [], []
            v_current = self.params.h_to_v_low_fitted(initial_head)
            failed = False
            
            for i in range(TH):
                h_current = initial_head if i == 0 else self.params.v_low_to_h_fitted(v_current)
                
                # Sample power for original mode
                mode = original_modes[i]
                random_power = self.sample_power_for_mode(mode, h_current)
                random_powers.append(random_power)
                
                # Simulate this hour
                p_current = torch.tensor(random_power, dtype=torch.float32, device=device)
                q_candidate = torch.zeros_like(p_current)
                p_clamped = p_current
                
                if p_current > 0.5:  # Turbine
                    p_min_turb = self.params.pos_min(h_current)
                    p_max_turb = self.params.pos_max(h_current)
                    p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                    q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
                elif p_current < -0.5:  # Pump
                    p_min_pump = self.params.neg_min(h_current)
                    p_max_pump = self.params.neg_max(h_current)
                    p_clamped = torch.clamp(p_current, min=p_min_pump, max=p_max_pump)
                    q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
                
                # Check volume constraints
                v_next = v_current + q_candidate * 3600
                if (v_next > self.params.max_vol_up) | (v_next < self.params.min_vol_low):
                    failed = True
                    break
                
                p_final = p_clamped if abs(p_current) > 0.5 else torch.zeros_like(p_current)
                p_list.append(p_final)
                q_list.append(q_candidate)
                h_list.append(self.params.v_low_to_h_fitted(v_next))
                v_list.append(v_next)
                v_current = v_next
            
            if not failed:
                return (np.array(random_powers), 
                        torch.stack(p_list), torch.stack(q_list), 
                        torch.stack(h_list), torch.stack(v_list), 
                        True, retry)
        
        return None, None, None, None, None, False, max_retries

def compare_modes(original_powers, simulated_powers):
    """Compare modes between original and simulated schedules."""
    def get_mode(p):
        return "turbine" if p > 0.5 else "pump" if p < -0.5 else "idle"
    
    if isinstance(original_powers, torch.Tensor):
        original_powers = original_powers.cpu().numpy()
    if isinstance(simulated_powers, torch.Tensor):
        simulated_powers = simulated_powers.cpu().numpy()
    
    mode_matches = sum(1 for i in range(len(original_powers)) 
                       if get_mode(original_powers[i]) == get_mode(simulated_powers[i]))
    
    return mode_matches, len(original_powers)

def process_random_sampling(df, params, num_samples_per_day=1):
    """
    Process dataset with random sampling while preserving operational modes.
    
    Args:
        df: Original DataFrame
        params: HydroParameters instance
        num_samples_per_day: Number of random samples per day (default: 1)
        
    Returns:
        DataFrame with results
    """
    print(f"\n=== RANDOM MODE-PRESERVING SAMPLING ===")
    print(f"Generating {num_samples_per_day} sample(s) per day with retry mechanism...\n")
    
    np.random.seed(42)
    random_sampler = RandomModeSampler(params)
    
    unique_dates = df['date'].unique()
    all_results = []
    total_retries = 0
    total_matches = 0
    total_hours = 0
    failed_count = 0
    
    for idx, date in enumerate(unique_dates, 1):
        if idx % 10 == 0:
            print(f"  Processing {date} ({idx}/{len(unique_dates)})...")
        
        day_data = df[df['date'] == date].copy().reset_index(drop=True)
        if len(day_data) != 24:
            continue
        
        original_powers = torch.tensor(day_data['power'].values, dtype=torch.float32, device=device)
        initial_head = torch.tensor(head_init, dtype=torch.float32, device=device)
        
        for sample_idx in range(num_samples_per_day):
            random_schedule, p_sim, q_sim, h_sim, v_low_sim, success, retry_count = \
                random_sampler.generate_and_simulate_random_schedule(original_powers, initial_head)
            
            if not success:
                print(f"  WARNING: Failed for {date}, sample {sample_idx+1} after {retry_count} retries")
                failed_count += 1
                continue
            
            total_retries += retry_count
            matches, hours = compare_modes(original_powers, p_sim)
            total_matches += matches
            total_hours += hours
            
            # Store results
            sample_data = day_data.copy()
            if num_samples_per_day > 1:
                sample_data['date'] = f"{date}_sample_{sample_idx+1}"
            sample_data['power'] = p_sim.cpu().numpy()
            sample_data['flow'] = q_sim.cpu().numpy()
            sample_data['head'] = h_sim.cpu().numpy()
            sample_data['volume'] = v_low_sim.cpu().numpy()
            
            all_results.append(sample_data)
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Print statistics
        print(f"\n=== RESULTS ===")
        print(f"Successfully generated: {len(all_results)} samples")
        print(f"Failed samples: {failed_count}")
        print(f"Mode retention rate: {100*total_matches/total_hours:.2f}%")
        print(f"Average retries per sample: {total_retries/len(all_results):.2f}")
        print(f"Samples succeeding on first try: {sum(1 for _ in range(len(all_results))) - (total_retries > 0)} ({100*(len(all_results)-min(total_retries, len(all_results)))/len(all_results):.1f}%)")
        
        return final_df
    else:
        return None

def main_random_sampling(num_samples_per_day=1):
    """
    Main function to generate random samples with mode preservation.
    
    Args:
        num_samples_per_day: Number of samples per day (default: 1)
    """
    
    # Load original results
    original_file = "../MIQP/MIQP_linear/MILP_global_linear_results.csv"
    
    if not os.path.exists(original_file):
        print(f"Error: {original_file} not found!")
        return
    
    print(f"Loading {original_file}...")
    df = pd.read_csv(original_file)
    print(f"Loaded {len(df)} rows ({len(df['date'].unique())} unique dates)")
    
    # Initialize parameters
    params = HydroParameters(
        head_init=torch.tensor(head_init, dtype=torch.float32, device=device),
        v_low_init=torch.tensor(v_low_init, dtype=torch.float32, device=device),
        neg_min=neg_min, neg_max=neg_max,
        pos_min=pos_min, pos_max=pos_max,
        predict_q_poly=predict_q_poly,
        h_to_v_low_fitted=h_to_v_low_fitted,
        v_low_to_h_fitted=v_low_to_h_fitted
    )
    
    # Process with random sampling
    final_df = process_random_sampling(df, params, num_samples_per_day=num_samples_per_day)
    
    if final_df is not None:
        # Save results
        if num_samples_per_day == 1:
            output_filename = "MIQP_linear_results_random_samples.csv"
        else:
            output_filename = f"MIQP_linear_results_random_samples_{num_samples_per_day}x.csv"
        
        final_df.to_csv(output_filename, index=False)
        print(f"\nResults saved to {output_filename}")
        
        # Summary statistics
        idle_count = (abs(final_df['power']) < 0.5).sum()
        turbine_count = (final_df['power'] > 0.5).sum()
        pump_count = (final_df['power'] < -0.5).sum()
        
        print(f"\n=== SUMMARY ===")
        print(f"Total rows: {len(final_df)}")
        print(f"Mode distribution:")
        print(f"  Idle: {idle_count:,} ({100*idle_count/len(final_df):.1f}%)")
        print(f"  Turbine: {turbine_count:,} ({100*turbine_count/len(final_df):.1f}%)")
        print(f"  Pump: {pump_count:,} ({100*pump_count/len(final_df):.1f}%)")
        print(f"Physical ranges:")
        print(f"  Power: [{final_df['power'].min():.2f}, {final_df['power'].max():.2f}] MW")
        print(f"  Head: [{final_df['head'].min():.2f}, {final_df['head'].max():.2f}] m")
        print(f"  Volume: [{final_df['volume'].min():.0f}, {final_df['volume'].max():.0f}] m³")
    else:
        print("\nFailed to generate random samples.")

# Run with default (1 sample per day)
if __name__ == "__main__":
    main_random_sampling(num_samples_per_day=1)  # Change to 5 for multiple samples

#%%
"""
Calculate and compare relative errors for all noise injection and random sampling methods.
"""

def calculate_relative_errors_for_dataset(df_method, df_original, params, method_name):
    """
    Calculate relative errors by comparing method results against baseline simulation.
    
    Args:
        df_method: DataFrame with noisy/random results
        df_original: Original MIQP DataFrame
        params: HydroParameters instance
        method_name: Name of the method for reporting
    
    Returns:
        Dictionary with error statistics
    """
    print(f"\nCalculating relative errors for {method_name}...")
    
    baseline_simulator = BaselineSimulator(params)
    relative_errors = []
    
    unique_dates = df_method['date'].unique()
    
    for date in unique_dates:
        # Handle multi-sample dates (e.g., "2024-01-01_sample_1")
        original_date = date.split('_sample_')[0] if '_sample_' in str(date) else date
        
        # Get original and method data for this day
        day_original = df_original[df_original['date'] == original_date].copy().reset_index(drop=True)
        day_method = df_method[df_method['date'] == date].copy().reset_index(drop=True)
        
        if len(day_original) != 24 or len(day_method) != 24:
            continue
        
        # Run baseline simulation on original data
        original_powers = torch.tensor(day_original['power'].values, dtype=torch.float32, device=device)
        initial_head = torch.tensor(head_init, dtype=torch.float32, device=device)
        p_baseline, _, _, _ = baseline_simulator.simulate_daily_operation(original_powers, initial_head)
        
        # Calculate relative errors for each hour
        for i in range(24):
            baseline_power = p_baseline[i].item()
            method_power = day_method.loc[i, 'power']
            head = day_method.loc[i, 'head']
            
            # Skip idle hours
            if abs(baseline_power) < 0.5:
                continue
            
            # Get capacity range at this head
            if baseline_power > 0.5:  # Turbine mode
                p_min = params.pos_min(torch.tensor(head, device=device)).item()
                p_max = params.pos_max(torch.tensor(head, device=device)).item()
                capacity_range = p_max - p_min
            else:  # Pump mode
                p_min = params.neg_min(torch.tensor(head, device=device)).item()
                p_max = params.neg_max(torch.tensor(head, device=device)).item()
                capacity_range = p_max - p_min
            
            # Calculate relative error
            if capacity_range > 0:
                rel_error = abs(method_power - baseline_power) / capacity_range
                relative_errors.append(rel_error)
    
    # Calculate statistics
    if relative_errors:
        stats = {
            'method': method_name,
            'count': len(relative_errors),
            'mean': np.mean(relative_errors),
            'median': np.median(relative_errors),
            'std': np.std(relative_errors),
            'min': np.min(relative_errors),
            'max': np.max(relative_errors),
            'p25': np.percentile(relative_errors, 25),
            'p75': np.percentile(relative_errors, 75)
        }
        return stats
    else:
        return None

def compare_all_methods():
    """
    Compare relative errors across all noise injection levels and random sampling.
    """
    print("\n" + "="*80)
    print("RELATIVE ERROR ANALYSIS FOR ALL METHODS")
    print("="*80)
    
    # Load original MIQP results
    original_file = "../MIQP/MIQP_linear/MILP_global_linear_results.csv"
    if not os.path.exists(original_file):
        print(f"Error: {original_file} not found!")
        return
    
    df_original = pd.read_csv(original_file)
    print(f"\nLoaded original MIQP results: {len(df_original)} rows, {len(df_original['date'].unique())} days")
    
    # Initialize parameters
    params = HydroParameters(
        head_init=torch.tensor(head_init, dtype=torch.float32, device=device),
        v_low_init=torch.tensor(v_low_init, dtype=torch.float32, device=device),
        neg_min=neg_min, neg_max=neg_max,
        pos_min=pos_min, pos_max=pos_max,
        predict_q_poly=predict_q_poly,
        h_to_v_low_fitted=h_to_v_low_fitted,
        v_low_to_h_fitted=v_low_to_h_fitted
    )
    
    # Collect all results
    all_results = []
    
    # Process noise injection methods
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for noise_level in noise_levels:
        filename = f"MIQP_linear_results_relative_noise_{noise_level*100:.0f}pct.csv"
        if os.path.exists(filename):
            df_noise = pd.read_csv(filename)
            method_name = f"Noise {noise_level*100:.0f}%"
            stats = calculate_relative_errors_for_dataset(df_noise, df_original, params, method_name)
            if stats:
                all_results.append(stats)
        else:
            print(f"Warning: {filename} not found, skipping...")
    
    # Process random sampling
    random_filename = "MIQP_linear_results_random_samples.csv"
    if os.path.exists(random_filename):
        df_random = pd.read_csv(random_filename)
        method_name = "Random Sampling"
        stats = calculate_relative_errors_for_dataset(df_random, df_original, params, method_name)
        if stats:
            all_results.append(stats)
    else:
        print(f"Warning: {random_filename} not found, skipping...")
    
    # Print results table
    if all_results:
        print("\n" + "="*80)
        print("RELATIVE ERROR COMPARISON TABLE")
        print("="*80)
        print(f"\n{'Method':<20} {'Count':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'P25-P75':<15}")
        print("-"*120)
        
        for stats in all_results:
            print(f"{stats['method']:<20} "
                  f"{stats['count']:<10,} "
                  f"{stats['mean']*100:<10.2f} "
                  f"{stats['median']*100:<10.2f} "
                  f"{stats['std']*100:<10.2f} "
                  f"{stats['min']*100:<10.2f} "
                  f"{stats['max']*100:<10.2f} "
                  f"[{stats['p25']*100:.2f}, {stats['p75']*100:.2f}]")
        
        print("\nNote: All values except Count are in percentage (%).")
        print("P25-P75: 25th and 75th percentile range")
        
        # Additional insights
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        # Find method with lowest and highest mean error
        min_method = min(all_results, key=lambda x: x['mean'])
        max_method = max(all_results, key=lambda x: x['mean'])
        
        print(f"\nLowest mean relative error: {min_method['method']} ({min_method['mean']*100:.2f}%)")
        print(f"Highest mean relative error: {max_method['method']} ({max_method['mean']*100:.2f}%)")
        
        # Compare noise levels progression
        noise_methods = [r for r in all_results if r['method'].startswith('Noise')]
        if len(noise_methods) > 1:
            print(f"\nNoise level progression:")
            for stats in noise_methods:
                deviation_from_target = None
                try:
                    target = float(stats['method'].split()[1].rstrip('%')) / 100
                    deviation = abs(stats['mean'] - target) / target * 100
                    print(f"  {stats['method']}: Mean error {stats['mean']*100:.2f}% "
                          f"(target {target*100:.0f}%, deviation {deviation:.1f}%)")
                except:
                    print(f"  {stats['method']}: Mean error {stats['mean']*100:.2f}%")
        
        # Compare random sampling to noise methods
        random_methods = [r for r in all_results if 'Random' in r['method']]
        if random_methods and noise_methods:
            random_mean = random_methods[0]['mean']
            print(f"\nRandom Sampling comparison:")
            print(f"  Random sampling mean error: {random_mean*100:.2f}%")
            for noise_stat in noise_methods:
                diff = (random_mean - noise_stat['mean']) * 100
                if diff > 0:
                    print(f"  {diff:.2f}% higher than {noise_stat['method']}")
                else:
                    print(f"  {abs(diff):.2f}% lower than {noise_stat['method']}")
    
    else:
        print("\nNo results found to compare!")

if __name__ == "__main__":
    compare_all_methods()

# %% Profit Calculation
"""
Calculate ex-post profit for all perturbation methods using SimulationLayer.
"""

class SimulationLayer:
    def __init__(self, params):
        """
        A simplified class for hourly simulation of the operation,
        using the same parameters object as the other modules.
        """
        self.params = params

    def simulate_operation(self, p, q, h):
        """
        Simulate hourly operation with physical constraints.
        
        Args:
            p (torch.Tensor): Hourly power schedule [time_horizon]
            q (torch.Tensor): Hourly flow schedule [time_horizon] (not directly used, recalculated)
            h (torch.Tensor): Hourly head schedule [time_horizon] (from optimization, for reference)
        
        Returns:
            tuple: Calibrated hourly (p, q, h, v_low) schedules.
        """
        TH = self.params.time_horizon
        
        # Initialize lists for each state
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        # Start states - use initial conditions
        v_current = self.params.v_low_init  # Initial reservoir volume
        h_current = self.params.head_init   # Initial head value
        
        v_list.append(v_current)
        h_list.append(h_current)  # Store initial head

        for i in range(TH):
            p_current = p[i]
            
            # a) Base: idle => q=0
            q_candidate = torch.zeros_like(p_current)
            p_clamped = p_current

            # b) For turbine mode (p_current>0), clamp p between pos_min(h) and pos_max(h)
            #    then get q via polynomial using CURRENT head (not optimized head)
            if p_current > 0.5:  # Turbine mode
                p_min_turb = self.params.pos_min(h_current)  # Use current head
                p_max_turb = self.params.pos_max(h_current)  # Use current head
                p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            
            # c) For pump mode (p_current<0), clamp p between neg_min(h) and neg_max(h)
            elif p_current < -0.5:  # Pump mode
                p_min_pump = self.params.neg_min(h_current)  # Use current head
                p_max_pump = self.params.neg_max(h_current)  # Use current head
                p_clamped = torch.clamp(p_current, min=p_min_pump, max=p_max_pump)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            
            # Update volume: v_next = v_current + q * 3600 (seconds in an hour)
            v_next = v_current + q_candidate * 3600
            
            # Check if volume is within bounds
            out_of_bounds = (v_next > self.params.max_vol_up) | (v_next < self.params.min_vol_low)
            
            # If out of bounds, set to idle mode
            if out_of_bounds:
                p_final = torch.zeros_like(p_current)
                q_final = torch.zeros_like(q_candidate)
                v_next = v_current  # No change to volume
                h_next = h_current  # No change to head
            else:
                p_final = p_clamped if p_current != 0 else torch.zeros_like(p_current)
                q_final = q_candidate
                # Update head based on new volume
                h_next = self.params.v_low_to_h_fitted(v_next)
            
            # Append states for this hour
            p_list.append(p_final)
            q_list.append(q_final)
            
            # Update current states for next iteration
            v_current = v_next
            h_current = h_next  # Important: update h_current for next iteration
            
            v_list.append(v_current.item())
            h_list.append(h_current)
        
        # Convert lists to tensors
        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list[:-1])  # Remove the extra head value (we have TH+1 heads)
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32, device=device)  # Remove extra volume
        
        return p_sim, q_sim, h_sim, v_low_sim

    def calc_profit(self, p_sim, p_opt, v_low_sim, DA_price):
        """
        Calculate the daily profit from the hourly simulation.
        """
        # Calculate energy per hour (MWh)
        e_sim = p_sim  # Already in MW, and we're using hourly intervals

        # Calculate revenue
        revenue = torch.sum(DA_price * e_sim)

        # Determine the System Imbalance (SI) price
        surplus_penalty_multiplier = -0.5
        shortage_penalty_multiplier = -2.0

        SI_price = torch.where(
            e_sim < p_opt,  # Shortage in simulation
            shortage_penalty_multiplier * DA_price,  # Lower output penalty
            surplus_penalty_multiplier * DA_price  # Higher output penalty
        )
        
        # Calculate imbalance penalty
        imbalance = e_sim - p_opt
        penalty = imbalance * SI_price
        SI_penalty = penalty.sum()

        # Volume penalty - if final volume exceeds target
        volume_deficit = max(0, v_low_sim[-1] - self.params.target_vol_low)
        energy_loss = self.params.rho * volume_deficit * self.params.g * self.params.target_head * self.params.mu / 3.6e9  # Convert J to MWh
        volume_penalty = energy_loss * torch.median(DA_price)

        # Operating cost
        operating_cost = self.params.operational_cost * torch.sum(p_sim**2)

        # Total profit
        total_profit = revenue - operating_cost - SI_penalty - volume_penalty
        
        return total_profit, SI_penalty, volume_penalty, operating_cost


def get_price_for_date(date_str, price_data_hour):
    """
    Get hourly prices for a given date, handling different formats and data structures.
    
    Args:
        date_str: Date string in various formats
        price_data_hour: Price data (could be DataFrame, dict, or other structure)
    
    Returns:
        torch.Tensor with 24 hourly prices, or None if not found
    """
    # Normalize date string to YYYY-MM-DD format
    try:
        # Handle both '2024/04/09' and '2024-04-09' formats
        date_str = str(date_str).replace('/', '-')
        date_obj = pd.to_datetime(date_str)
        
        # Try different access methods
        if isinstance(price_data_hour, pd.DataFrame):
            if date_obj in price_data_hour.index:
                return torch.tensor(price_data_hour.loc[date_obj].values, dtype=torch.float32, device=device)
        elif isinstance(price_data_hour, pd.Series):
            if date_obj in price_data_hour.index:
                return torch.tensor([price_data_hour.loc[date_obj]], dtype=torch.float32, device=device)
        elif isinstance(price_data_hour, dict):
            if date_obj in price_data_hour:
                return torch.tensor(price_data_hour[date_obj], dtype=torch.float32, device=device)
            if date_str in price_data_hour:
                return torch.tensor(price_data_hour[date_str], dtype=torch.float32, device=device)
        
        return None
    except Exception as e:
        return None

def calculate_expost_profit_for_dataset(df, params, price_data_hour, method_name, exclude_date='2024-12-12'):
    """
    Calculate ex-post profit for all dates in a dataset.
    
    Args:
        df: DataFrame with power schedules
        params: HydroParameters instance
        price_data_hour: Hourly price data
        method_name: Name of method for reporting
        exclude_date: Date to exclude from calculation (default: '2024-12-12')
    
    Returns:
        Dictionary with profit statistics
    """
    print(f"\nCalculating ex-post profit for {method_name}...")
    
    simulator = SimulationLayer(params)
    daily_profits = []
    
    unique_dates = df['date'].unique()
    excluded_count = 0
    skipped_count = 0
    
    for date in unique_dates:
        # Handle multi-sample dates (e.g., "2024-01-01_sample_1")
        original_date = date.split('_sample_')[0] if '_sample_' in str(date) else date
        
        # Normalize date format for comparison
        normalized_date = str(original_date).replace('/', '-')
        normalized_exclude = str(exclude_date).replace('/', '-')
        
        # Skip excluded date
        if normalized_date == normalized_exclude:
            excluded_count += 1
            continue
        
        day_data = df[df['date'] == date].copy().reset_index(drop=True)
        
        if len(day_data) != 24:
            continue
        
        # Extract schedules
        p_schedule = torch.tensor(day_data['power'].values, dtype=torch.float32, device=device)
        q_schedule = torch.tensor(day_data['flow'].values, dtype=torch.float32, device=device)
        h_schedule = torch.tensor(day_data['head'].values, dtype=torch.float32, device=device)
        
        # Get prices - try to extract from day_data first
        if 'price' in day_data.columns:
            prices = torch.tensor(day_data['price'].values, dtype=torch.float32, device=device)
        else:
            # Try to get from price_data_hour
            prices = get_price_for_date(original_date, price_data_hour)
            if prices is None:
                skipped_count += 1
                continue
        
        # Ensure we have 24 prices
        if len(prices) != 24:
            skipped_count += 1
            continue
        
        # Simulate operation
        p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_schedule, q_schedule, h_schedule)
        
        # Calculate profit (p_opt = p_schedule since we're evaluating the schedule as-is)
        total_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
            p_sim, p_schedule, v_low_sim, prices
        )
        
        daily_profits.append(total_profit.item())
    
    # Calculate statistics
    if daily_profits:
        stats = {
            'method': method_name,
            'count': len(daily_profits),
            'excluded_count': excluded_count,
            'skipped_count': skipped_count,
            'mean_profit': np.mean(daily_profits),
            'std_profit': np.std(daily_profits),
            'min_profit': np.min(daily_profits),
            'max_profit': np.max(daily_profits),
            'median_profit': np.median(daily_profits),
            'total_profit': np.sum(daily_profits)
        }
        return stats
    else:
        return None

def compare_expost_profits():
    """
    Compare ex-post profits across all methods.
    """
    print("\n" + "="*80)
    print("EX-POST PROFIT ANALYSIS FOR ALL METHODS")
    print("="*80)
    print(f"Note: Excluding date 2024-12-12 from all calculations")
    
    # Load original MIQP results
    original_file = "../MIQP/MIQP_linear/MILP_global_linear_results.csv"
    if not os.path.exists(original_file):
        print(f"Error: {original_file} not found!")
        return
    
    df_original = pd.read_csv(original_file)
    
    # Check if price column exists in the data
    print(f"\nColumns in original data: {df_original.columns.tolist()}")
    
    # Initialize parameters
    params = HydroParameters(
        head_init=torch.tensor(head_init, dtype=torch.float32, device=device),
        v_low_init=torch.tensor(v_low_init, dtype=torch.float32, device=device),
        neg_min=neg_min, neg_max=neg_max,
        pos_min=pos_min, pos_max=pos_max,
        predict_q_poly=predict_q_poly,
        h_to_v_low_fitted=h_to_v_low_fitted,
        v_low_to_h_fitted=v_low_to_h_fitted
    )
    
    # Use DA_price_hour from the loaded pickle
    price_data = DA_price_hour
    
    # Collect all results
    all_results = []
    
    # Process original MIQP results first
    print("\nProcessing original MIQP results...")
    stats_original = calculate_expost_profit_for_dataset(
        df_original, params, price_data, "Original MIQP"
    )
    if stats_original:
        all_results.append(stats_original)
    
    # Process noise injection methods
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for noise_level in noise_levels:
        filename = f"MIQP_linear_results_relative_noise_{noise_level*100:.0f}pct.csv"
        if os.path.exists(filename):
            df_noise = pd.read_csv(filename)
            method_name = f"Noise {noise_level*100:.0f}%"
            stats = calculate_expost_profit_for_dataset(df_noise, params, price_data, method_name)
            if stats:
                all_results.append(stats)
        else:
            print(f"Warning: {filename} not found, skipping...")
    
    # Process random sampling
    random_filename = "MIQP_linear_results_random_samples.csv"
    if os.path.exists(random_filename):
        df_random = pd.read_csv(random_filename)
        method_name = "Random Sampling"
        stats = calculate_expost_profit_for_dataset(df_random, params, price_data, method_name)
        if stats:
            all_results.append(stats)
    else:
        print(f"Warning: {random_filename} not found, skipping...")
    
    # Print results table
    if all_results:
        print("\n" + "="*80)
        print("EX-POST PROFIT COMPARISON TABLE")
        print("="*80)
        print(f"\n{'Method':<20} {'Days':<8} {'Mean Profit':<15} {'Std Profit':<15} {'Min Profit':<15} {'Max Profit':<15}")
        print("-"*95)
        
        for stats in all_results:
            print(f"{stats['method']:<20} "
                  f"{stats['count']:<8} "
                  f"€{stats['mean_profit']:<14,.2f} "
                  f"€{stats['std_profit']:<14,.2f} "
                  f"€{stats['min_profit']:<14,.2f} "
                  f"€{stats['max_profit']:<14,.2f}")
        
        print("\nNote: All profit values are in Euros (€).")
        if all_results[0]['excluded_count'] > 0:
            print(f"Excluded {all_results[0]['excluded_count']} instance(s) of date 2024-12-12 from each method.")
        if all_results[0]['skipped_count'] > 0:
            print(f"Skipped {all_results[0]['skipped_count']} date(s) due to missing price data or incomplete hours.")
        
        # Calculate profit degradation compared to original
        if all_results[0]['method'] == "Original MIQP":
            original_profit = all_results[0]['mean_profit']
            
            print("\n" + "="*80)
            print("PROFIT DEGRADATION ANALYSIS (vs Original MIQP)")
            print("="*80)
            print(f"\n{'Method':<20} {'Mean Profit':<15} {'Degradation':<15} {'% Change':<12}")
            print("-"*70)
            
            for stats in all_results:
                degradation = stats['mean_profit'] - original_profit
                pct_change = (degradation / original_profit) * 100 if original_profit != 0 else 0
                
                print(f"{stats['method']:<20} "
                      f"€{stats['mean_profit']:<14,.2f} "
                      f"€{degradation:<14,.2f} "
                      f"{pct_change:>10.2f}%")
            
            # Additional insights
            print("\n" + "="*80)
            print("KEY INSIGHTS")
            print("="*80)
            
            perturbed_methods = [r for r in all_results if r['method'] != "Original MIQP"]
            if perturbed_methods:
                best_method = max(perturbed_methods, key=lambda x: x['mean_profit'])
                worst_method = min(perturbed_methods, key=lambda x: x['mean_profit'])
                
                best_degradation = (best_method['mean_profit'] - original_profit) / original_profit * 100 if original_profit != 0 else 0
                worst_degradation = (worst_method['mean_profit'] - original_profit) / original_profit * 100 if original_profit != 0 else 0
                
                print(f"\nBest performing perturbation: {best_method['method']}")
                print(f"  Mean profit: €{best_method['mean_profit']:,.2f} ({best_degradation:+.2f}% vs original)")
                
                print(f"\nWorst performing perturbation: {worst_method['method']}")
                print(f"  Mean profit: €{worst_method['mean_profit']:,.2f} ({worst_degradation:+.2f}% vs original)")
                
                # Analyze noise progression
                noise_methods = [r for r in perturbed_methods if r['method'].startswith('Noise')]
                if len(noise_methods) > 1:
                    print(f"\nNoise level impact on profit:")
                    for stats in noise_methods:
                        degradation_pct = (stats['mean_profit'] - original_profit) / original_profit * 100 if original_profit != 0 else 0
                        print(f"  {stats['method']}: €{stats['mean_profit']:,.2f} ({degradation_pct:+.2f}%)")
    
    else:
        print("\nNo results found to compare!")

if __name__ == "__main__":
    compare_expost_profits()
# %%