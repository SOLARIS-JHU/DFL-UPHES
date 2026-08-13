"""MIQP Piecewise Bilinear Optimization

MILP for pumped hydro using piecewise linearization with SOS2 constraints.
Input: 2024 price data (Data/price_data_2024.csv)
Output: Results saved to script directory
"""
import torch
import numpy as np
import dill as pickle
import pandas as pd
import sys
import os
import gurobipy as gp
from gurobipy import GRB
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
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

# Import SimulationLayer from DFL
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from DFL.core.layers import SimulationLayer

head_init = 77.0  # Initial head value
v_low_init = h_to_v_low_fitted(head_init)  # Initial lower reservoir volume

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


class PiecewiseMILPOptimizerSOS2:
    """MILP optimizer with piecewise linearization and SOS2 constraints."""

    def __init__(self, T, DA_prices, num_segments_h=10, num_segments_p_pump=10, num_segments_p_turbine=10,
                 C_op=0.4, M_p=10000, h_init=head_init, h_min=head_min, h_max=head_max,
                 v_low_init=v_low_init, v_low_target=target_vol_low):
        self.T = T
        self.DA_prices = DA_prices
        self.num_segments_h = num_segments_h
        self.num_segments_p_pump = num_segments_p_pump
        self.num_segments_p_turbine = num_segments_p_turbine
        self.C_op = C_op
        self.M_p = M_p
        self.h_min = h_min
        self.h_max = h_max
        self.v_low_init = v_low_init
        self.v_low_target = v_low_target
        self.h_init = h_init
        
        # Sample the nonlinear functions
        self._sample_functions()
        
        # Create model
        self.model = gp.Model("PipelinePiecewiseMILP")
        
        # Build the model
        self._build_model()

    def _sample_functions(self):
        """Sample nonlinear functions at grid points for piecewise approximation."""
        self.h_samples = np.linspace(self.h_min, self.h_max, self.num_segments_h + 1)

        # Volume-head samples
        self.v_low_samples = [h_to_v_low_fitted(torch.tensor(h)).item() for h in self.h_samples]

        # Pump UPC samples
        self.pump_grid = {}
        for i, h in enumerate(self.h_samples):
            p_min = neg_min(h).item()
            p_max = neg_max(h).item()
            p_values = np.linspace(p_min, p_max, self.num_segments_p_pump + 1)
            q_values = [predict_q_poly(p, h).item() for p in p_values]
            self.pump_grid[i] = {'p': p_values, 'q': q_values}

        # Turbine UPC samples
        self.turbine_grid = {}
        for i, h in enumerate(self.h_samples):
            p_min = pos_min(h).item()
            p_max = pos_max(h).item()
            p_values = np.linspace(p_min, p_max, self.num_segments_p_turbine + 1)
            q_values = [predict_q_poly(p, h).item() for p in p_values]
            self.turbine_grid[i] = {'p': p_values, 'q': q_values}
    
    def _build_model(self):
        """Build MILP model with piecewise linearization."""
        T = self.T
        M_p = self.M_p

        # Decision variables
        self.z_I = self.model.addVars(T, vtype=GRB.BINARY, name="z_I")
        self.z_T = self.model.addVars(T, vtype=GRB.BINARY, name="z_T")
        self.z_P = self.model.addVars(T, vtype=GRB.BINARY, name="z_P")

        self.p = self.model.addVars(T, lb=-GRB.INFINITY, name="p")
        self.h = self.model.addVars(T, lb=self.h_min, ub=self.h_max, name="h")
        self.q = self.model.addVars(T, lb=-GRB.INFINITY, name="q")
        self.v_low = self.model.addVars(T, name="v_low")

        # Convex combination weights for volume-head
        self.lambda_vh = {
            (t, i): self.model.addVar(lb=0, ub=1, name=f"lambda_vh_{t}_{i}")
            for t in range(T) for i in range(self.num_segments_h + 1)
        }

        # Convex combination weights for UPC
        self.lambda_pump = {
            (t, i, j): self.model.addVar(lb=0, ub=1, name=f"lambda_pump_{t}_{i}_{j}")
            for t in range(T) for i in range(self.num_segments_h + 1)
            for j in range(self.num_segments_p_pump + 1)
        }

        self.lambda_turbine = {
            (t, i, j): self.model.addVar(lb=0, ub=1, name=f"lambda_turbine_{t}_{i}_{j}")
            for t in range(T) for i in range(self.num_segments_h + 1)
            for j in range(self.num_segments_p_turbine + 1)
        }

        # Mode selection: exactly one per time step
        for t in range(T):
            self.model.addConstr(self.z_I[t] + self.z_T[t] + self.z_P[t] == 1)

        # Volume-head piecewise linear constraints
        for t in range(T):
            self.model.addConstr(gp.quicksum(self.lambda_vh[t, i] for i in range(self.num_segments_h + 1)) == 1)
            self.model.addConstr(self.h[t] == gp.quicksum(
                self.lambda_vh[t, i] * self.h_samples[i] for i in range(self.num_segments_h + 1)))
            self.model.addConstr(self.v_low[t] == gp.quicksum(
                self.lambda_vh[t, i] * self.v_low_samples[i] for i in range(self.num_segments_h + 1)))
            self.model.addSOS(GRB.SOS_TYPE2, [self.lambda_vh[t, i] for i in range(self.num_segments_h + 1)])

        # UPC piecewise linear constraints
        for t in range(T):
            # Idle mode bounds
            self.model.addConstr(self.p[t] <= M_p * (1 - self.z_I[t]))
            self.model.addConstr(self.p[t] >= -M_p * (1 - self.z_I[t]))
            self.model.addConstr(self.q[t] <= M_p * (1 - self.z_I[t]))
            self.model.addConstr(self.q[t] >= -M_p * (1 - self.z_I[t]))

            # Pump mode: link to head selection
            for i in range(self.num_segments_h + 1):
                self.model.addConstr(
                    gp.quicksum(self.lambda_pump[t, i, j] for j in range(self.num_segments_p_pump + 1)) ==
                    self.z_P[t] * self.lambda_vh[t, i])

            self.model.addConstr(gp.quicksum(self.lambda_pump[t, i, j]
                for i in range(self.num_segments_h + 1) for j in range(self.num_segments_p_pump + 1)) == self.z_P[t])

            for i in range(self.num_segments_h + 1):
                self.model.addSOS(GRB.SOS_TYPE2, [self.lambda_pump[t, i, j]
                    for j in range(self.num_segments_p_pump + 1)])

            # Turbine mode: link to head selection
            for i in range(self.num_segments_h + 1):
                self.model.addConstr(
                    gp.quicksum(self.lambda_turbine[t, i, j] for j in range(self.num_segments_p_turbine + 1)) ==
                    self.z_T[t] * self.lambda_vh[t, i])

            self.model.addConstr(gp.quicksum(self.lambda_turbine[t, i, j]
                for i in range(self.num_segments_h + 1) for j in range(self.num_segments_p_turbine + 1)) == self.z_T[t])

            for i in range(self.num_segments_h + 1):
                self.model.addSOS(GRB.SOS_TYPE2, [self.lambda_turbine[t, i, j]
                    for j in range(self.num_segments_p_turbine + 1)])

            # Pump mode interpolation
            pump_p_expr = gp.quicksum(
                self.lambda_pump[t, i, j] * self.pump_grid[i]['p'][j]
                for i in range(self.num_segments_h + 1) for j in range(self.num_segments_p_pump + 1))
            pump_q_expr = gp.quicksum(
                self.lambda_pump[t, i, j] * self.pump_grid[i]['q'][j]
                for i in range(self.num_segments_h + 1) for j in range(self.num_segments_p_pump + 1))

            # Turbine mode interpolation
            turbine_p_expr = gp.quicksum(
                self.lambda_turbine[t, i, j] * self.turbine_grid[i]['p'][j]
                for i in range(self.num_segments_h + 1) for j in range(self.num_segments_p_turbine + 1))
            turbine_q_expr = gp.quicksum(
                self.lambda_turbine[t, i, j] * self.turbine_grid[i]['q'][j]
                for i in range(self.num_segments_h + 1) for j in range(self.num_segments_p_turbine + 1))

            # Combine pump and turbine
            self.model.addConstr(self.p[t] == pump_p_expr + turbine_p_expr)
            self.model.addConstr(self.q[t] == pump_q_expr + turbine_q_expr)

        # Volume balance
        for t in range(T):
            if t == 0:
                self.model.addConstr(self.v_low[t] == self.v_low_init + 3600 * self.q[t])
            else:
                self.model.addConstr(self.v_low[t] == self.v_low[t-1] + 3600 * self.q[t])

        # Terminal constraints
        self.model.addConstr(self.v_low[T-1] <= self.v_low_target)
        self.model.addConstr(self.h[T-1] >= target_head)

        # Objective: maximize revenue - cost
        objective = gp.quicksum(
            self.p[t] * self.DA_prices[t] - self.C_op * self.p[t] ** 2
            for t in range(T)
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)

    def solve(self):
        """Solve MILP and return solution and metrics."""
        self.model.Params.MIPGap = 0.01
        self.model.Params.TimeLimit = 3600

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

        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            if self.model.status == GRB.TIME_LIMIT:
                print(f"Time limit reached, gap: {self.model.MIPGap:.2%}")

            metrics['ObjectiveValue'] = self.model.objVal
            metrics['ObjectiveBound'] = self.model.objBound
            metrics['MIPGap'] = self.model.MIPGap
            metrics['ExpectedProfit'] = self.model.objVal

            results = {
                'p': [self.p[t].X for t in range(self.T)],
                'q': [self.q[t].X for t in range(self.T)],
                'h': [self.h[t].X for t in range(self.T)],
                'v_low': [self.v_low[t].X for t in range(self.T)],
                'z_I': [self.z_I[t].X for t in range(self.T)],
                'z_T': [self.z_T[t].X for t in range(self.T)],
                'z_P': [self.z_P[t].X for t in range(self.T)]
            }
            return results, metrics
        else:
            print(f"Optimization failed with status {self.model.status}")
            if self.model.status == GRB.INFEASIBLE:
                print("Computing IIS...")
                self.model.computeIIS()
            return None, metrics


# Simplified HydroParameters for simulation
class HydroParameters:
    """Parameter container for hydropower system simulation."""

    def __init__(self, time_horizon=24, operational_cost=0.4, rho=1000, g=9.81, mu=0.9,
                 head_init=head_init, v_low_init=v_low_init, target_head=target_head,
                 target_vol_low=target_vol_low, max_vol_up=max_vol_up, min_vol_low=min_vol_low,
                 neg_min=neg_min, neg_max=neg_max, pos_min=pos_min, pos_max=pos_max,
                 predict_q_poly=predict_q_poly, h_to_v_low_fitted=h_to_v_low_fitted,
                 v_low_to_h_fitted=v_low_to_h_fitted):
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


def run_piecewise_optimization(price_file=None, output_suffix=""):
    """Run piecewise MILP optimization for all days in price database."""
    print("Loading price data...")
    price_data = read_price_data(price_file)

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

            optimizer = PiecewiseMILPOptimizerSOS2(
                T=24, DA_prices=prices_24h,
                num_segments_h=10, num_segments_p_pump=10, num_segments_p_turbine=10
            )
            results, metrics = optimizer.solve()
            solution_time = time.time() - start_time

            if results is None:
                print("No solution")
                continue

            # Run simulation
            p_tensor = torch.tensor(results['p'], dtype=torch.float32, device=device)
            q_tensor = torch.tensor(results['q'], dtype=torch.float32, device=device)
            h_tensor = torch.tensor(results['h'], dtype=torch.float32, device=device)

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
                    'power': results['p'][hour],
                    'head': results['h'][hour],
                    'volume': results['v_low'][hour],
                    'flow': results['q'][hour],
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
    pd.DataFrame(detailed_results).to_csv(f"MIQP_piecewise_results{output_suffix}.csv", index=False)
    pd.DataFrame(benchmark_results).to_csv(f"MIQP_piecewise_benchmark{output_suffix}.csv", index=False)

    print(f"\nDone! {len(detailed_results)} hourly records saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MIQP Piecewise SOS2 Optimization')
    parser.add_argument('--price-file', type=str, default=None,
                        help='Path to price data CSV file')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix for output file names (e.g., _oos)')
    args = parser.parse_args()
    run_piecewise_optimization(price_file=args.price_file, output_suffix=args.output_suffix)
