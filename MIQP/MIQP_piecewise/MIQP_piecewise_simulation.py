"""
MIQP Piecewise Simulation Script

This script reads the MIQP_piecewise_results.csv file and runs the SimulationLayer
to calculate ex-post profits based on the optimized schedules.

Input: MIQP_piecewise_results.csv (detailed hourly results from optimization)
Output: MIQP_piecewise_benchmark_simulated.csv (daily performance metrics from simulation)
"""

# %% Import libraries
import torch
import pandas as pd
import sys
import dill as pickle
import numpy as np
import time
from pathlib import Path

device = torch.device("cpu")

# Load portfolio data
sys.path.append('../../Library')
from V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# Load preprocessed functions & data
with open('../../preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = 77.0  # Initial head value
v_low_init = h_to_v_low_fitted(head_init)  # Initial lower reservoir volume

# %% HydroParameters class
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

# %% SimulationLayer class
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
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32)  # Remove extra volume
        
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

# %% Load and process results
def load_miqp_piecewise_results(file_path="MIQP_piecewise_results.csv"):
    """Load the detailed MIQP piecewise results."""
    try:
        # Try different encodings to handle potential encoding issues
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp1252')
        
        print(f"Loaded {len(df)} detailed results from {file_path}")
        return df
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def simulate_and_benchmark():
    """Run simulation on MIQP piecewise results and create benchmark."""
    
    # Load detailed results
    df = load_miqp_piecewise_results()
    if df is None:
        return
    
    # Initialize simulation parameters
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
    
    simulator = SimulationLayer(params)
    
    # Group by date
    grouped = df.groupby('date')
    
    benchmark_results = []
    
    print(f"Processing {len(grouped)} unique dates...")
    
    for date_idx, (date, group) in enumerate(grouped, 1):
        print(f"Processing {date} ({date_idx}/{len(grouped)})...")
        
        try:
            start_time = time.time()
            
            # Sort by hour to ensure correct order
            group = group.sort_values('hour')
            
            # Ensure we have 24 hours
            if len(group) != 24:
                print(f"Warning: Date {date} has {len(group)} hours instead of 24. Skipping.")
                continue
            
            # Extract optimization results
            p_opt = torch.tensor(group['power'].values, dtype=torch.float32, device=device)
            q_opt = torch.tensor(group['flow'].values, dtype=torch.float32, device=device)
            h_opt = torch.tensor(group['head'].values, dtype=torch.float32, device=device)
            prices = torch.tensor(group['price'].values, dtype=torch.float32, device=device)
            
            # Calculate expected profit from optimization (same as optimization objective)
            expected_profit = torch.sum(p_opt * prices - 0.4 * p_opt**2)
            
            # Run simulation
            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_opt, q_opt, h_opt)
            
            # Calculate ex-post profit from simulation
            ex_post_profit, si_penalty, vol_penalty, op_cost = simulator.calc_profit(
                p_sim, p_opt, v_low_sim, prices
            )
            
            # Calculate simulation time
            simulation_time = time.time() - start_time
            
            # Store benchmark results (compatible with results_analysis.py format)
            benchmark_results.append({
                'Date': date,
                'Expected Profit (€)': expected_profit.item(),
                'Ex-post Profit (€)': ex_post_profit.item(),
                'SI Penalty (€)': si_penalty.item(),
                'Vol Penalty (€)': vol_penalty.item(),
                'Op Cost (€)': op_cost.item(),
                'Solving Time (s)': simulation_time,
                'MIP Gap': 0.0,  # Not applicable for simulation, set to 0
                'Binary Variables': 0,  # Not applicable for simulation
                'Continuous Variables': 0,  # Not applicable for simulation
                'Total Constraints': 0  # Not applicable for simulation
            })
            
            print(f"Expected: {expected_profit.item():.2f}€, Ex-post: {ex_post_profit.item():.2f}€")
            
        except Exception as e:
            print(f"Error processing {date}: {e}")
            continue
    
    # Create benchmark DataFrame
    benchmark_df = pd.DataFrame(benchmark_results)
    
    # Sort by date
    try:
        benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
        benchmark_df = benchmark_df.sort_values('Date')
        benchmark_df['Date'] = benchmark_df['Date'].dt.strftime('%Y/%m/%d')
    except:
        print("Warning: Could not sort by date. Keeping original order.")
    
    # Save results (using filename compatible with results_analysis.py)
    output_file = "MIQP_piecewise_benchmark.csv"
    benchmark_df.to_csv(output_file, index=False)
    
    print(f"\nSimulation complete!")
    print(f"Benchmark results saved to {output_file} ({len(benchmark_results)} rows)")
    print(f"Output format is compatible with results_analysis.py")
    
    # Print summary statistics
    if len(benchmark_results) > 0:
        print(f"\nSummary Statistics:")
        print(f"Average Expected Profit: {benchmark_df['Expected Profit (€)'].mean():.2f}€")
        print(f"Average Ex-post Profit: {benchmark_df['Ex-post Profit (€)'].mean():.2f}€")
        print(f"Average SI Penalty: {benchmark_df['SI Penalty (€)'].mean():.2f}€")
        print(f"Average Vol Penalty: {benchmark_df['Vol Penalty (€)'].mean():.2f}€")
        print(f"Average Op Cost: {benchmark_df['Op Cost (€)'].mean():.2f}€")
        print(f"Average Solving Time: {benchmark_df['Solving Time (s)'].mean():.3f}s")
        
        # Calculate performance gap
        profit_gap = benchmark_df['Expected Profit (€)'].mean() - benchmark_df['Ex-post Profit (€)'].mean()
        print(f"Average Profit Gap (Expected - Ex-post): {profit_gap:.2f}€")
        
        print(f"\nCompatibility Info:")
        print(f"- Column names match results_analysis.py expectations")
        print(f"- MIP Gap, Binary Variables, Continuous Variables, Total Constraints set to 0 (not applicable for simulation)")
        print(f"- Use this file directly with DATABASE_PATHS in results_analysis.py")
    
    return benchmark_df

# %% Execute simulation
if __name__ == "__main__":
    # Check if input file exists
    input_file = Path("MIQP_piecewise_results.csv")
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found!")
        print("Please make sure MIQP_piecewise_results.csv is in the current directory.")
    else:
        simulate_and_benchmark()