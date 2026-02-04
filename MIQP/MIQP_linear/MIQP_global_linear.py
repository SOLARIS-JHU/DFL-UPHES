"""MIQP Global Linear Optimization

MILP for pumped hydro using global linearization of nonlinear relationships.
Input: 2024 price data (Data/price_data_2024.csv)
Output: Results saved to script directory
"""
import torch
import dill as pickle
import pandas as pd
import sys
import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

device = torch.device("cpu")

# Setup paths to work from repo root or script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '../..'))
os.chdir(script_dir)  # Always output to script directory

# Add Library to path
sys.path.insert(0, os.path.join(repo_root, 'Library'))
from V_H_relations import load_portfolio_data
load_portfolio_data()
from V_H_relations import head_max, head_min, max_vol_up, min_vol_low, target_vol_low, target_head

# Load preprocessed data
preprocess_path = os.path.join(repo_root, 'preprocess.pkl')
with open(preprocess_path, 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = 77.0 # Initial head value
v_low_init = h_to_v_low_fitted(head_init) # Initial lower reservoir volume
# convert v_low_init to float
v_low_init = float(v_low_init)
# convert h_vlow_coeff_lin to python array
h_vlow_coeff_lin = h_vlow_coeff_lin.detach().numpy()

def read_price_data(file_path=None):
    """Read price data from CSV format."""
    if file_path is None:
        file_path = os.path.join(repo_root, 'Data/price_data_2024.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Price data file not found: {file_path}")
    
    df = pd.read_csv(file_path, dtype={'prices_hourly': str})
    price_data = {}
    
    for _, row in df.iterrows():
        date = row['date']
        prices_str = row['prices_hourly']
        
        # Handle potential NaN or float values
        if pd.isna(prices_str) or isinstance(prices_str, float):
            print(f"Skipping {date} - invalid price data")
            continue
            
        try:
            prices = [float(x.strip()) for x in prices_str.split(',')]
            if len(prices) != 24:
                print(f"Skipping {date} - expected 24 prices, got {len(prices)}")
                continue
            price_data[date] = prices
        except Exception as e:
            print(f"Error parsing prices for {date}: {e}")
            continue

    return price_data


class MILPOptimizer:
    def __init__(self, T, DA_prices, C_op=0.4, M_p=10000, h_init=head_init, h_min=head_min, h_max=head_max, v_low_init=v_low_init, v_low_target=target_vol_low, h_target=target_head):
        """Mixed Integer Linear Programming optimizer with mode-based constraints."""
        self.T = T
        self.DA_prices = DA_prices
        self.C_op = C_op
        self.M_p = M_p
        self.h_min = h_min
        self.h_max = h_max
        self.v_low_init = v_low_init
        self.v_low_target = v_low_target
        self.h_target = h_target
        self.h_init = h_init

        # Create a new Gurobi model
        self.model = gp.Model("PipelineMILP")
        self._build_model()
    
    def _build_model(self):
        T = self.T
        M_p = self.M_p

        # Decision variables
        self.p_T = self.model.addVars(T, lb=0, name="p_T")
        self.p_P = self.model.addVars(T, lb=-GRB.INFINITY, ub=0, name="p_P")
        self.q = self.model.addVars(T, lb=-GRB.INFINITY, name="q")
        self.h = self.model.addVars(T, lb=self.h_min, ub=self.h_max, name="h")
        self.v_low = self.model.addVars(T, name="v_low")

        # Mode selection: Idle, Turbine, or Pump
        self.z_I = self.model.addVars(T, vtype=GRB.BINARY, name="z_I")
        self.z_T = self.model.addVars(T, vtype=GRB.BINARY, name="z_T")
        self.z_P = self.model.addVars(T, vtype=GRB.BINARY, name="z_P")

        for t in range(T):
            self.model.addConstr(self.z_I[t] + self.z_T[t] + self.z_P[t] == 1)

        # Idle mode: zero flow
        for t in range(T):
            self.model.addConstr(self.q[t] <= M_p * (1 - self.z_I[t]))

        # Turbine mode: power bounds and linearized UPC
        for t in range(T):
            self.model.addConstr(self.p_T[t] >= pos_min_fit @ [self.h[t], 1.0] * self.z_T[t])
            self.model.addConstr(self.p_T[t] <= pos_max_fit @ [self.h[t], 1.0] * self.z_T[t])

        # Pump mode: power bounds and linearized UPC
        for t in range(T):
            self.model.addConstr(self.p_P[t] >= neg_min_fit @ [self.h[t], 1.0] * self.z_P[t])
            self.model.addConstr(self.p_P[t] <= neg_max_fit @ [self.h[t], 1.0] * self.z_P[t])

        # Flow relationships
        for t in range(T):
            q_tur = coefs_tur_lin @ [self.p_T[t], self.h[t]] + intercept_tur_lin
            q_pump = coefs_pump_lin @ [self.p_P[t], self.h[t]] + intercept_pump_lin
            self.model.addConstr(self.q[t] == q_tur * self.z_T[t] + q_pump * self.z_P[t])

        # Volume-head relationship and dynamics
        for t in range(T):
            self.model.addConstr(self.v_low[t] == h_vlow_coeff_lin @ [self.h[t], 1])
            if t == 0:
                self.model.addConstr(self.v_low[t] == self.v_low_init + 3600 * self.q[t])
            else:
                self.model.addConstr(self.v_low[t] == self.v_low[t-1] + 3600 * self.q[t])

        # Terminal constraints
        self.model.addConstr(self.v_low[T-1] <= self.v_low_target)
        self.model.addConstr(self.h[T-1] >= self.h_target)

        # Objective: maximize revenue - cost
        objective = gp.quicksum(
            (self.p_T[t] + self.p_P[t]) * self.DA_prices[t] -
            self.C_op * (self.p_T[t] + self.p_P[t]) ** 2
            for t in range(T)
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)
        self.model.Params.OutputFlag = 1

    def solve(self):
        """Solve MILP and return solution and metrics."""
        self.model.optimize()

        metrics = {
            'Status': self.model.status,
            'SolveTime': self.model.Runtime,
            'NumVars': self.model.NumVars,
            'NumConstrs': self.model.NumConstrs,
            'NumBinVars': sum(1 for v in self.model.getVars() if v.VType == GRB.BINARY),
            'ObjectiveValue': None,
            'ObjectiveBound': None,
            'MIPGap': None,
            'ExpectedProfit': None
        }

        if self.model.status == GRB.OPTIMAL:
            metrics['ObjectiveValue'] = self.model.objVal
            metrics['ObjectiveBound'] = self.model.objBound
            metrics['MIPGap'] = self.model.MIPGap
            metrics['ExpectedProfit'] = self.model.objVal

            results = {
                'p_t_T': [self.p_T[t].X for t in range(self.T)],
                'p_t_P': [self.p_P[t].X for t in range(self.T)],
                'q_t': [self.q[t].X for t in range(self.T)],
                'h_t': [self.h[t].X for t in range(self.T)],
                'v_low': [self.v_low[t].X for t in range(self.T)],
                'z_I': [self.z_I[t].X for t in range(self.T)],
                'z_T': [self.z_T[t].X for t in range(self.T)],
                'z_P': [self.z_P[t].X for t in range(self.T)]
            }
            return results, metrics
        else:
            print("No optimal solution found!")
            return None, metrics


# Simplified HydroParameters for simulation
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


class SimulationLayer:
    """Simulate hourly operation with physical constraints."""

    def __init__(self, params):
        self.params = params

    def simulate_operation(self, p, q, h):
        """Simulate operation respecting physical bounds."""
        TH = self.params.time_horizon
        p_list, q_list, h_list, v_list = [], [], [], []

        v_current = self.params.v_low_init
        v_list.append(v_current)

        for i in range(TH):
            p_current = p[i]
            q_candidate = torch.zeros_like(p_current)
            p_clamped = p_current

            if p_current > 0.5:  # Turbine
                p_clamped = torch.clamp(
                    p_current,
                    min=self.params.pos_min(h[i]),
                    max=self.params.pos_max(h[i])
                )
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h[i].unsqueeze(0)).squeeze(0)
            elif p_current < -0.5:  # Pump
                p_clamped = torch.clamp(
                    p_current,
                    min=self.params.neg_min(h[i]),
                    max=self.params.neg_max(h[i])
                )
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h[i].unsqueeze(0)).squeeze(0)

            v_next = v_current + q_candidate * 3600
            out_of_bounds = (v_next > self.params.max_vol_up) | (v_next < self.params.min_vol_low)

            if out_of_bounds:
                p_final = torch.zeros_like(p_current)
                q_final = torch.zeros_like(q_candidate)
                h_next = h[i]
            else:
                p_final = p_clamped if p_current != 0 else torch.zeros_like(p_current)
                q_final = q_candidate
                h_next = self.params.v_low_to_h_fitted(v_next)

            p_list.append(p_final)
            q_list.append(q_final)
            h_list.append(h_next)
            v_list.append(v_next.item())
            v_current = v_next

        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list)
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32)

        return p_sim, q_sim, h_sim, v_low_sim

    def calc_profit(self, p_sim, p_opt, v_low_sim, DA_price):
        """Calculate profit accounting for imbalances and volume constraints."""
        revenue = torch.sum(DA_price * p_sim)

        # System imbalance penalties
        SI_price = torch.where(
            p_sim < p_opt,
            -2.0 * DA_price,  # Shortage
            -0.5 * DA_price   # Surplus
        )
        SI_penalty = torch.sum((p_sim - p_opt) * SI_price)

        # Volume penalty
        vol_deficit = max(0, v_low_sim[-1] - self.params.target_vol_low)
        energy_loss = (self.params.rho * vol_deficit * self.params.g *
                       self.params.target_head * self.params.mu / 3.6e9)
        volume_penalty = energy_loss * torch.median(DA_price)

        operating_cost = self.params.operational_cost * torch.sum(p_sim ** 2)
        total_profit = revenue - operating_cost - SI_penalty - volume_penalty

        return total_profit, SI_penalty, volume_penalty, operating_cost


def run_milp_optimization():
    """Run MILP optimization for all days in price database."""
    print("Loading price data...")
    price_data = read_price_data()
    
    detailed_results = []
    benchmark_results = []

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
    total_dates = len(price_data)

    # Process each day
    for idx, (date_str, prices_24h) in enumerate(price_data.items(), start=1):
        print(f"Processing {date_str} ({idx}/{total_dates})...", end=" ")

        try:
            start_time = time.time()

            optimizer = MILPOptimizer(24, prices_24h)
            results, metrics = optimizer.solve()
            solution_time = time.time() - start_time

            if results is None:
                print("No solution")
                continue

            # Correct idle mode values to 0
            for t in range(24):
                if results['z_I'][t] > 0.5:
                    results['p_t_T'][t] = 0.0
                    results['p_t_P'][t] = 0.0
                    results['q_t'][t] = 0.0

            # Calculate power and volume
            power_values = [p_t + p_p for p_t, p_p in zip(results['p_t_T'], results['p_t_P'])]
            volume_values = [h_vlow_coeff_lin[0] * h + h_vlow_coeff_lin[1] for h in results['h_t']]

            # Run simulation
            p_tensor = torch.tensor(power_values, dtype=torch.float32, device=device)
            q_tensor = torch.tensor(results['q_t'], dtype=torch.float32, device=device)
            h_tensor = torch.tensor(results['h_t'], dtype=torch.float32, device=device)

            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_tensor, q_tensor, h_tensor)

            # Calculate profit
            da_prices_tensor = torch.tensor(prices_24h, dtype=torch.float32, device=device)
            profit, si_penalty, vol_penalty, op_cost = simulator.calc_profit(
                p_sim, p_tensor[:len(p_sim)], v_low_sim, da_prices_tensor[:len(p_sim)]
            )

            # Store results
            for hour in range(24):
                detailed_results.append({
                    'date': date_str,
                    'hour': hour,
                    'power': power_values[hour],
                    'head': results['h_t'][hour],
                    'volume': volume_values[hour],
                    'flow': results['q_t'][hour],
                    'price': prices_24h[hour]
                })

            benchmark_results.append({
                'Date': date_str,
                'Solving Time (s)': solution_time,
                'MIP Gap': metrics['MIPGap'],
                'Binary Variables': metrics['NumBinVars'],
                'Continuous Variables': metrics['NumVars'] - metrics['NumBinVars'],
                'Total Constraints': metrics['NumConstrs'],
                'Expected Profit (€)': metrics['ExpectedProfit'],
                'SI Penalty (€)': si_penalty.item(),
                'Vol Penalty (€)': vol_penalty.item(),
                'Op Cost (€)': op_cost.item(),
                'Ex-post Profit (€)': profit.item()
            })

            print(f"Profit: {profit.item():.2f} €")

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Save results
    pd.DataFrame(detailed_results).to_csv("MILP_global_linear_results.csv", index=False)
    pd.DataFrame(benchmark_results).to_csv("MILP_global_linear_benchmark.csv", index=False)

    print(f"\nDone! {len(detailed_results)} hourly records saved")


if __name__ == "__main__":
    run_milp_optimization()