"""
MIQP Piecewise Bilinear Optimization Script

Mixed-Integer Quadratic Programming approach for pumped hydro energy storage optimization 
using piecewise bilinearization with SOS2 constraints for nonlinear function approximation.

Input: 2024 price data from ../../Data/price_data_2024.csv
Output:
- MIQP_piecewise_results.csv (detailed hourly results)  
- MIQP_piecewise_benchmark.csv (daily performance metrics)

Python interactive is recommended for running this script.
"""
# %% Import libraries
import torch
import numpy as np
import cvxpy as cp
import dill as pickle
import pandas as pd
import sys
import gurobipy as gp
from gurobipy import GRB
import os
import time

device = torch.device("cpu")

# load portfolio data
sys.path.append('../../Library')
from V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('../../preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = 77.0  # Initial head value
v_low_init = h_to_v_low_fitted(head_init)  # Initial lower reservoir volume

# %% Load price data function (same as MIQP_nn.py)
def read_price_data(file_path="../../Data/price_data_2024.csv"):
    """Read price data from the new CSV format."""
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

# %% Piecewise MILP Optimizer with SOS2 constraints
class PiecewiseMILPOptimizerSOS2:
    def __init__(self, T, DA_prices, num_segments_h=5, num_segments_p_pump=5, num_segments_p_turbine=5, 
                 C_op=0.4, M_p=10000, h_init=head_init, h_min=head_min, h_max=head_max, 
                 v_low_init=v_low_init, v_low_target=target_vol_low):
        """
        MILP optimizer with piecewise linearization for nonlinear functions.
        
        Parameters:
            T (int): Number of time periods.
            DA_prices (list): Day-ahead prices for each period.
            num_segments_h (int): Number of segments for head discretization.
            num_segments_p_pump (int): Number of segments for pump power discretization.
            num_segments_p_turbine (int): Number of segments for turbine power discretization.
            C_op (float): Operational cost coefficient.
            M_p (float): Big-M constant.
            h_init (float): Initial head.
            h_min (float): Minimum head.
            h_max (float): Maximum head.
            v_low_init (float): Initial lower reservoir volume.
            v_low_target (float): Target lower reservoir volume.
        """
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
        """Sample the nonlinear functions at grid points."""
        # Sample head values
        self.h_samples = np.linspace(self.h_min, self.h_max, self.num_segments_h + 1)
        
        # Sample volume-head relationship
        self.v_low_samples = []
        for h in self.h_samples:
            v_low = h_to_v_low_fitted(torch.tensor(h)).item()
            self.v_low_samples.append(v_low)
        
        # Sample UPC for pump mode (p < 0)
        self.pump_grid = {}
        for i, h in enumerate(self.h_samples):
            # Get power bounds for this head value
            p_min = neg_min(h).item()
            p_max = neg_max(h).item()
            # Sample power values within these bounds
            p_values = np.linspace(p_min, p_max, self.num_segments_p_pump + 1)
            q_values = []
            for p in p_values:
                q = predict_q_poly(p, h).item()
                q_values.append(q)
            self.pump_grid[i] = {'p': p_values, 'q': q_values}
        
        # Sample UPC for turbine mode (p > 0)
        self.turbine_grid = {}
        for i, h in enumerate(self.h_samples):
            # Get power bounds for this head value
            p_min = pos_min(h).item()
            p_max = pos_max(h).item()
            # Sample power values within these bounds
            p_values = np.linspace(p_min, p_max, self.num_segments_p_turbine + 1)
            q_values = []
            for p in p_values:
                q = predict_q_poly(p, h).item()
                q_values.append(q)
            self.turbine_grid[i] = {'p': p_values, 'q': q_values}
    
    def _build_model(self):
        """Build the MILP model with piecewise linearization."""
        T = self.T
        M_p = self.M_p  # Big-M constant
        
        # Decision Variables
        # Mode selection variables
        self.z_I = self.model.addVars(T, vtype=GRB.BINARY, name="z_I")  # Idle
        self.z_T = self.model.addVars(T, vtype=GRB.BINARY, name="z_T")  # Turbine
        self.z_P = self.model.addVars(T, vtype=GRB.BINARY, name="z_P")  # Pump
        
        # Physical variables
        self.p = self.model.addVars(T, lb=-GRB.INFINITY, name="p")  # Net power
        self.h = self.model.addVars(T, lb=self.h_min, ub=self.h_max, name="h")  # Head
        self.q = self.model.addVars(T, lb=-GRB.INFINITY, name="q")  # Net flow
        self.v_low = self.model.addVars(T, name="v_low")  # Lower reservoir volume
        
        # Variables for volume-head piecewise linearization - THIS WILL BE THE ONLY HEAD DISCRETIZATION
        self.lambda_vh = {}
        for t in range(T):
            for i in range(self.num_segments_h + 1):
                self.lambda_vh[t, i] = self.model.addVar(lb=0, ub=1, name=f"lambda_vh_{t}_{i}")
        
        # Variables for UPC piecewise linearization (power dimension only)
        self.lambda_pump = {}
        self.lambda_turbine = {}
        for t in range(T):
            # For pump mode
            for i in range(self.num_segments_h + 1):
                for j in range(self.num_segments_p_pump + 1):
                    self.lambda_pump[t, i, j] = self.model.addVar(lb=0, ub=1, name=f"lambda_pump_{t}_{i}_{j}")
            
            # For turbine mode
            for i in range(self.num_segments_h + 1):
                for j in range(self.num_segments_p_turbine + 1):
                    self.lambda_turbine[t, i, j] = self.model.addVar(lb=0, ub=1, name=f"lambda_turbine_{t}_{i}_{j}")
        
        # Mode selection constraints
        for t in range(T):
            self.model.addConstr(self.z_I[t] + self.z_T[t] + self.z_P[t] == 1, name=f"mode_sel_{t}")
        
        # Volume-head relationship constraints (1D piecewise linear)
        for t in range(T):
            # Convex combination constraint
            self.model.addConstr(gp.quicksum(self.lambda_vh[t, i] for i in range(self.num_segments_h + 1)) == 1, name=f"vh_lambda_sum_{t}")
            
            # Interpolation for h and v_low
            self.model.addConstr(self.h[t] == gp.quicksum(self.lambda_vh[t, i] * self.h_samples[i] for i in range(self.num_segments_h + 1)), name=f"h_interp_{t}")
            self.model.addConstr(self.v_low[t] == gp.quicksum(self.lambda_vh[t, i] * self.v_low_samples[i] for i in range(self.num_segments_h + 1)), name=f"v_low_interp_{t}")
            
            # Special ordered set type 2 (SOS2) for piecewise linear interpolation - SINGLE HEAD SOS2 CONSTRAINT
            self.model.addSOS(GRB.SOS_TYPE2, [self.lambda_vh[t, i] for i in range(self.num_segments_h + 1)])
        
        # UPC constraints with SOS2
        for t in range(T):
            # Idle mode constraints: p = 0, q = 0
            self.model.addConstr(self.p[t] <= M_p * (1 - self.z_I[t]), name=f"idle_p_upper_{t}")
            self.model.addConstr(self.p[t] >= -M_p * (1 - self.z_I[t]), name=f"idle_p_lower_{t}")
            self.model.addConstr(self.q[t] <= M_p * (1 - self.z_I[t]), name=f"idle_q_upper_{t}")
            self.model.addConstr(self.q[t] >= -M_p * (1 - self.z_I[t]), name=f"idle_q_lower_{t}")
            
            # Pump mode constraints
            # Link the pump lambdas to use EXACTLY the same head selection as lambda_vh
            for i in range(self.num_segments_h + 1):
                # Sum of lambda_pump over all power levels must exactly equal lambda_vh * z_P
                self.model.addConstr(
                    gp.quicksum(self.lambda_pump[t, i, j] for j in range(self.num_segments_p_pump + 1)) == 
                    self.z_P[t] * self.lambda_vh[t, i], 
                    name=f"pump_head_equal_{t}_{i}"
                )
            
            # Convex combination constraint for pump
            self.model.addConstr(gp.quicksum(self.lambda_pump[t, i, j] 
                                        for i in range(self.num_segments_h + 1) 
                                        for j in range(self.num_segments_p_pump + 1)) == self.z_P[t],
                            name=f"pump_lambda_sum_{t}")
            
            # SOS2 constraints along the power dimension only for pump
            for i in range(self.num_segments_h + 1):
                self.model.addSOS(GRB.SOS_TYPE2, [self.lambda_pump[t, i, j] for j in range(self.num_segments_p_pump + 1)])
            
            # Link the turbine lambdas to use EXACTLY the same head selection as lambda_vh
            for i in range(self.num_segments_h + 1):
                # Sum of lambda_turbine over all power levels must exactly equal lambda_vh * z_T
                self.model.addConstr(
                    gp.quicksum(self.lambda_turbine[t, i, j] for j in range(self.num_segments_p_turbine + 1)) == 
                    self.z_T[t] * self.lambda_vh[t, i], 
                    name=f"turbine_head_equal_{t}_{i}"
                )
            
            # Convex combination constraint for turbine
            self.model.addConstr(gp.quicksum(self.lambda_turbine[t, i, j] 
                                        for i in range(self.num_segments_h + 1) 
                                        for j in range(self.num_segments_p_turbine + 1)) == self.z_T[t],
                            name=f"turbine_lambda_sum_{t}")
            
            # SOS2 constraints along the power dimension only for turbine
            for i in range(self.num_segments_h + 1):
                self.model.addSOS(GRB.SOS_TYPE2, [self.lambda_turbine[t, i, j] for j in range(self.num_segments_p_turbine + 1)])
            
            # Interpolation expressions for pump mode
            pump_p_expr = gp.LinExpr()
            pump_q_expr = gp.LinExpr()
            
            for i in range(self.num_segments_h + 1):
                for j in range(self.num_segments_p_pump + 1):
                    # Power and flow interpolation
                    pump_p_expr.add(self.lambda_pump[t, i, j] * self.pump_grid[i]['p'][j])
                    pump_q_expr.add(self.lambda_pump[t, i, j] * self.pump_grid[i]['q'][j])

            # Interpolation expressions for turbine mode
            turbine_p_expr = gp.LinExpr()
            turbine_q_expr = gp.LinExpr()
            
            for i in range(self.num_segments_h + 1):
                for j in range(self.num_segments_p_turbine + 1):
                    # Power and flow interpolation
                    turbine_p_expr.add(self.lambda_turbine[t, i, j] * self.turbine_grid[i]['p'][j])
                    turbine_q_expr.add(self.lambda_turbine[t, i, j] * self.turbine_grid[i]['q'][j])
            
            # Combine pump and turbine expressions for final p, q values
            self.model.addConstr(self.p[t] == pump_p_expr + turbine_p_expr, name=f"p_combined_{t}")
            self.model.addConstr(self.q[t] == pump_q_expr + turbine_q_expr, name=f"q_combined_{t}")
        
        # Volume dynamics
        for t in range(T):
            if t == 0:
                self.model.addConstr(self.v_low[t] == self.v_low_init + 3600 * self.q[t], name=f"vol_dyn_{t}")
            else:
                self.model.addConstr(self.v_low[t] == self.v_low[t-1] + 3600 * self.q[t], name=f"vol_dyn_{t}")
        
        # Target volume constraint
        self.model.addConstr(self.v_low[T-1] <= self.v_low_target, name="vol_target")
        
        # Target head constraint
        self.model.addConstr(self.h[T-1] >= target_head, name="head_target")

        # Objective function: maximize profit
        objective = gp.quicksum(
            self.p[t] * self.DA_prices[t] - self.C_op * self.p[t] * self.p[t]
            for t in range(T)
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)
    
    def solve(self):
        """Optimize the MILP and return the decision variables."""
        # Set some solver parameters for better performance
        self.model.Params.MIPGap = 0.01  # 1% optimality gap
        self.model.Params.TimeLimit = 3600  # 60 minute time limit

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
                print(f"Optimization reached time limit with MIP gap: {self.model.MIPGap:.2%}")
            
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
            # Try to identify infeasibility causes
            if self.model.status == GRB.INFEASIBLE:
                print("Model is infeasible. Computing IIS...")
                self.model.computeIIS()
                print("\nConstraints in the IIS:")
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print(f"{c.ConstrName}: {c}")
            return None, metrics

# %% Piecewise MILP Optimizer with Big-M quadratic constraints
class PiecewiseMILPOptimizer:
    def __init__(self, T, DA_prices, num_segments_h=5, num_segments_p_pump=5, num_segments_p_turbine=5, 
                 C_op=0.4, M_p=10000, h_init=head_init, h_min=head_min, h_max=head_max, 
                 v_low_init=v_low_init, v_low_target=target_vol_low):
        """
        MILP optimizer with piecewise linearization for nonlinear functions.
        
        Parameters:
            T (int): Number of time periods.
            DA_prices (list): Day-ahead prices for each period.
            num_segments_h (int): Number of segments for head discretization.
            num_segments_p_pump (int): Number of segments for pump power discretization.
            num_segments_p_turbine (int): Number of segments for turbine power discretization.
            C_op (float): Operational cost coefficient.
            M_p (float): Big-M constant.
            h_init (float): Initial head.
            h_min (float): Minimum head.
            h_max (float): Maximum head.
            v_low_init (float): Initial lower reservoir volume.
            v_low_target (float): Target lower reservoir volume.
        """
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
        """Sample the nonlinear functions at grid points."""
        # Sample head values
        self.h_samples = np.linspace(self.h_min, self.h_max, self.num_segments_h + 1)
        
        # Sample volume-head relationship
        self.v_low_samples = []
        for h in self.h_samples:
            v_low = h_to_v_low_fitted(torch.tensor(h)).item()
            self.v_low_samples.append(v_low)
        
        # Sample UPC for pump mode (p < 0)
        self.pump_grid = {}
        for i, h in enumerate(self.h_samples):
            # Get power bounds for this head value
            p_min = neg_min(h).item()
            p_max = neg_max(h).item()
            # Sample power values within these bounds
            p_values = np.linspace(p_min, p_max, self.num_segments_p_pump + 1)
            q_values = []
            for p in p_values:
                q = predict_q_poly(p, h).item()
                q_values.append(q)
            self.pump_grid[i] = {'p': p_values, 'q': q_values}
        
        # Sample UPC for turbine mode (p > 0)
        self.turbine_grid = {}
        for i, h in enumerate(self.h_samples):
            # Get power bounds for this head value
            p_min = pos_min(h).item()
            p_max = pos_max(h).item()
            # Sample power values within these bounds
            p_values = np.linspace(p_min, p_max, self.num_segments_p_turbine + 1)
            q_values = []
            for p in p_values:
                q = predict_q_poly(p, h).item()
                q_values.append(q)
            self.turbine_grid[i] = {'p': p_values, 'q': q_values}
    
    def _build_model(self):
        """Build the MILP model with piecewise linearization."""
        T = self.T
        M_p = self.M_p  # Big-M constant
        
        # Decision Variables
        # Mode selection variables
        self.z_I = self.model.addVars(T, vtype=GRB.BINARY, name="z_I")  # Idle
        self.z_T = self.model.addVars(T, vtype=GRB.BINARY, name="z_T")  # Turbine
        self.z_P = self.model.addVars(T, vtype=GRB.BINARY, name="z_P")  # Pump
        
        # Physical variables
        self.p = self.model.addVars(T, lb=-GRB.INFINITY, name="p")  # Net power
        self.h = self.model.addVars(T, lb=self.h_min, ub=self.h_max, name="h")  # Head
        self.q = self.model.addVars(T, lb=-GRB.INFINITY, name="q")  # Net flow
        self.v_low = self.model.addVars(T, name="v_low")  # Lower reservoir volume
        
        # Variables for volume-head piecewise linearization
        self.lambda_vh = {}
        for t in range(T):
            for i in range(self.num_segments_h + 1):
                self.lambda_vh[t, i] = self.model.addVar(lb=0, ub=1, name=f"lambda_vh_{t}_{i}")
        
        # Binary variables for UPC segment selection
        self.delta_pump = {}
        self.delta_turbine = {}
        for t in range(T):
            for i in range(self.num_segments_h):
                for j in range(self.num_segments_p_pump):
                    self.delta_pump[t, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"delta_pump_{t}_{i}_{j}")
            for i in range(self.num_segments_h):
                for j in range(self.num_segments_p_turbine):
                    self.delta_turbine[t, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"delta_turbine_{t}_{i}_{j}")
        
        # Weights for bilinear interpolation within quadrilaterals
        self.lambda_pump = {}
        self.lambda_turbine = {}
        for t in range(T):
            for i in range(self.num_segments_h):
                for j in range(self.num_segments_p_pump):
                    for c in range(4):  # 4 corners
                        self.lambda_pump[t, i, j, c] = self.model.addVar(lb=0, ub=1, name=f"lambda_pump_{t}_{i}_{j}_{c}")
            for i in range(self.num_segments_h):
                for j in range(self.num_segments_p_turbine):
                    for c in range(4):  # 4 corners
                        self.lambda_turbine[t, i, j, c] = self.model.addVar(lb=0, ub=1, name=f"lambda_turbine_{t}_{i}_{j}_{c}")
        
        # Mode selection constraints
        for t in range(T):
            self.model.addConstr(self.z_I[t] + self.z_T[t] + self.z_P[t] == 1, name=f"mode_sel_{t}")
        
        # Volume-head relationship constraints (1D piecewise linear)
        for t in range(T):
            # Convex combination constraint
            self.model.addConstr(gp.quicksum(self.lambda_vh[t, i] for i in range(self.num_segments_h + 1)) == 1, name=f"vh_lambda_sum_{t}")
            
            # Interpolation for h and v_low
            self.model.addConstr(self.h[t] == gp.quicksum(self.lambda_vh[t, i] * self.h_samples[i] for i in range(self.num_segments_h + 1)), name=f"h_interp_{t}")
            self.model.addConstr(self.v_low[t] == gp.quicksum(self.lambda_vh[t, i] * self.v_low_samples[i] for i in range(self.num_segments_h + 1)), name=f"v_low_interp_{t}")
            
            # Special ordered set type 2 (SOS2) for piecewise linear interpolation
            self.model.addSOS(GRB.SOS_TYPE2, [self.lambda_vh[t, i] for i in range(self.num_segments_h + 1)])
        
        # UPC constraints
        for t in range(T):
            # Idle mode constraints: p = 0, q = 0
            self.model.addConstr(self.p[t] <= M_p * (1 - self.z_I[t]), name=f"idle_p_upper_{t}")
            self.model.addConstr(self.p[t] >= -M_p * (1 - self.z_I[t]), name=f"idle_p_lower_{t}")
            self.model.addConstr(self.q[t] <= M_p * (1 - self.z_I[t]), name=f"idle_q_upper_{t}")
            self.model.addConstr(self.q[t] >= -M_p * (1 - self.z_I[t]), name=f"idle_q_lower_{t}")
            
            # Pump mode constraints
            # Only one quadrilateral can be active in pump mode
            self.model.addConstr(gp.quicksum(self.delta_pump[t, i, j] for i in range(self.num_segments_h) 
                                          for j in range(self.num_segments_p_pump)) == self.z_P[t], 
                              name=f"pump_segment_sel_{t}")
            
            # Interpolation within quadrilateral in pump mode
            pump_p_expr = gp.LinExpr()
            pump_q_expr = gp.LinExpr()
            
            for i in range(self.num_segments_h):
                for j in range(self.num_segments_p_pump):
                    # Define the 4 corners of the quadrilateral
                    h_lower = self.h_samples[i]
                    h_upper = self.h_samples[i+1]
                    p_lower = self.pump_grid[i]['p'][j]
                    p_upper = self.pump_grid[i]['p'][j+1]
                    
                    # Corner points (h, p, q) in counter-clockwise order
                    corners_h = [h_lower, h_lower, h_upper, h_upper]
                    corners_p = [p_lower, p_upper, p_upper, p_lower]
                    corners_q = [
                        self.pump_grid[i]['q'][j],      # bottom-left
                        self.pump_grid[i]['q'][j+1],    # bottom-right
                        self.pump_grid[i+1]['q'][j+1],  # top-right
                        self.pump_grid[i+1]['q'][j]     # top-left
                    ]
                    
                    # Convex combination constraint for the active quadrilateral
                    self.model.addConstr(gp.quicksum(self.lambda_pump[t, i, j, c] for c in range(4)) == self.delta_pump[t, i, j], 
                                      name=f"pump_lambda_sum_{t}_{i}_{j}")
                    
                    # Contribute to total expressions
                    for c in range(4):
                        pump_p_expr.add(self.lambda_pump[t, i, j, c] * corners_p[c])
                        pump_q_expr.add(self.lambda_pump[t, i, j, c] * corners_q[c])
                    
                    # Big-M constraints to enforce that p and h are within quadrilateral boundaries when active
                    # Lower bounds
                    self.model.addConstr(self.h[t] >= h_lower - M_p * (1 - self.delta_pump[t, i, j]), 
                                      name=f"pump_h_lower_{t}_{i}_{j}")
                    self.model.addConstr(self.p[t] >= p_lower - M_p * (1 - self.delta_pump[t, i, j]), 
                                      name=f"pump_p_lower_{t}_{i}_{j}")
                    # Upper bounds
                    self.model.addConstr(self.h[t] <= h_upper + M_p * (1 - self.delta_pump[t, i, j]), 
                                      name=f"pump_h_upper_{t}_{i}_{j}")
                    self.model.addConstr(self.p[t] <= p_upper + M_p * (1 - self.delta_pump[t, i, j]), 
                                      name=f"pump_p_upper_{t}_{i}_{j}")
            
            # Turbine mode constraints
            # Only one quadrilateral can be active in turbine mode
            self.model.addConstr(gp.quicksum(self.delta_turbine[t, i, j] for i in range(self.num_segments_h) 
                                          for j in range(self.num_segments_p_turbine)) == self.z_T[t], 
                              name=f"turbine_segment_sel_{t}")
            
            # Interpolation within quadrilateral in turbine mode
            turbine_p_expr = gp.LinExpr()
            turbine_q_expr = gp.LinExpr()
            
            for i in range(self.num_segments_h):
                for j in range(self.num_segments_p_turbine):
                    # Define the 4 corners of the quadrilateral
                    h_lower = self.h_samples[i]
                    h_upper = self.h_samples[i+1]
                    p_lower = self.turbine_grid[i]['p'][j]
                    p_upper = self.turbine_grid[i]['p'][j+1]
                    
                    # Corner points (h, p, q) in counter-clockwise order
                    corners_h = [h_lower, h_lower, h_upper, h_upper]
                    corners_p = [p_lower, p_upper, p_upper, p_lower]
                    corners_q = [
                        self.turbine_grid[i]['q'][j],      # bottom-left
                        self.turbine_grid[i]['q'][j+1],    # bottom-right
                        self.turbine_grid[i+1]['q'][j+1],  # top-right
                        self.turbine_grid[i+1]['q'][j]     # top-left
                    ]
                    
                    # Convex combination constraint for the active quadrilateral
                    self.model.addConstr(gp.quicksum(self.lambda_turbine[t, i, j, c] for c in range(4)) == self.delta_turbine[t, i, j], 
                                      name=f"turbine_lambda_sum_{t}_{i}_{j}")
                    
                    # Contribute to total expressions
                    for c in range(4):
                        turbine_p_expr.add(self.lambda_turbine[t, i, j, c] * corners_p[c])
                        turbine_q_expr.add(self.lambda_turbine[t, i, j, c] * corners_q[c])
                    
                    # Big-M constraints to enforce that p and h are within quadrilateral boundaries when active
                    # Lower bounds
                    self.model.addConstr(self.h[t] >= h_lower - M_p * (1 - self.delta_turbine[t, i, j]), 
                                      name=f"turbine_h_lower_{t}_{i}_{j}")
                    self.model.addConstr(self.p[t] >= p_lower - M_p * (1 - self.delta_turbine[t, i, j]), 
                                      name=f"turbine_p_lower_{t}_{i}_{j}")
                    # Upper bounds
                    self.model.addConstr(self.h[t] <= h_upper + M_p * (1 - self.delta_turbine[t, i, j]), 
                                      name=f"turbine_h_upper_{t}_{i}_{j}")
                    self.model.addConstr(self.p[t] <= p_upper + M_p * (1 - self.delta_turbine[t, i, j]), 
                                      name=f"turbine_p_upper_{t}_{i}_{j}")
            
            # Combine pump and turbine expressions
            self.model.addConstr(self.p[t] == pump_p_expr + turbine_p_expr, name=f"p_combined_{t}")
            self.model.addConstr(self.q[t] == pump_q_expr + turbine_q_expr, name=f"q_combined_{t}")
        
        # Volume dynamics
        for t in range(T):
            if t == 0:
                self.model.addConstr(self.v_low[t] == self.v_low_init + 3600 * self.q[t], name=f"vol_dyn_{t}")
            else:
                self.model.addConstr(self.v_low[t] == self.v_low[t-1] + 3600 * self.q[t], name=f"vol_dyn_{t}")
        
        # Target volume constraint
        self.model.addConstr(self.v_low[T-1] <= self.v_low_target, name="vol_target")
        
        # Objective function: maximize profit
        objective = gp.quicksum(
            self.p[t] * self.DA_prices[t] - self.C_op * self.p[t] * self.p[t]
            for t in range(T)
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)
    
    def solve(self):
        """Optimize the MILP and return the decision variables."""
        # Set some solver parameters for better performance
        self.model.Params.MIPGap = 0.01  # 1% optimality gap
        self.model.Params.TimeLimit = 3600  # 60 minute time limit

        
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            if self.model.status == GRB.TIME_LIMIT:
                print(f"Optimization reached time limit with MIP gap: {self.model.MIPGap:.2%}")
                
            results = {
                'p': [self.p[t].X for t in range(self.T)],
                'q': [self.q[t].X for t in range(self.T)],
                'h': [self.h[t].X for t in range(self.T)],
                'v_low': [self.v_low[t].X for t in range(self.T)],
                'z_I': [self.z_I[t].X for t in range(self.T)],
                'z_T': [self.z_T[t].X for t in range(self.T)],
                'z_P': [self.z_P[t].X for t in range(self.T)]
            }
            return results
        else:
            print(f"Optimization failed with status {self.model.status}")
            # Try to identify infeasibility causes
            if self.model.status == GRB.INFEASIBLE:
                print("Model is infeasible. Computing IIS...")
                self.model.computeIIS()
                print("\nConstraints in the IIS:")
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print(f"{c.ConstrName}: {c}")
            return None

# %% Simulation Layer Classes (same as MIQP_nn.py)
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
    def __init__(self, params):
        self.params = params

    def simulate_operation(self, p, q, h):
        """Simulate hourly operation with physical constraints."""
        TH = self.params.time_horizon
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        v_current = self.params.v_low_init
        v_list.append(v_current)

        for i in range(TH):
            h_current = h[i]
            p_current = p[i]
            q_candidate = torch.zeros_like(p_current)

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
                p_final = torch.zeros_like(p_current)
                q_final = torch.zeros_like(q_candidate)
                v_next = v_current
                h_next = h_current
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
        h_sim = torch.stack(h_list[:-1])
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32)
        
        return p_sim, q_sim, h_sim, v_low_sim

    def calc_profit(self, p_sim, p_opt, v_low_sim, DA_price):
        """Calculate the daily profit from the hourly simulation."""
        e_sim = p_sim
        revenue = torch.sum(DA_price * e_sim)

        surplus_penalty_multiplier = -0.5
        shortage_penalty_multiplier = -2.0

        SI_price = torch.where(
            e_sim < p_opt,
            shortage_penalty_multiplier * DA_price,
            surplus_penalty_multiplier * DA_price
        )
        
        imbalance = e_sim - p_opt
        penalty = imbalance * SI_price
        SI_penalty = penalty.sum()

        volume_deficit = max(0, v_low_sim[-1] - self.params.target_vol_low)
        energy_loss = self.params.rho * volume_deficit * self.params.g * self.params.target_head * self.params.mu / 3.6e9
        volume_penalty = energy_loss * torch.median(DA_price)

        operating_cost = self.params.operational_cost * torch.sum(p_sim**2)
        total_profit = revenue - operating_cost - SI_penalty - volume_penalty
        
        return total_profit, SI_penalty, volume_penalty, operating_cost

# %% Main execution function
def run_piecewise_optimization():
    """Run piecewise MILP optimization for all days in price database (same format as MIQP_nn.py)."""
    print("Loading price data...")
    price_data = read_price_data()
    
    # Initialize result lists
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
    
    # Get total number of dates
    total_dates = len(price_data)
    
    # Process each day
    for idx, (date_str, prices_24h) in enumerate(price_data.items(), start=1):
        print(f"Processing {date_str} ({idx}/{total_dates})...")
        
        try:
            start_time = time.time()
            
            # Create and solve optimizer (using SOS2 version with 10 segments)
            T = 24
            optimizer = PiecewiseMILPOptimizerSOS2(
                T=T, 
                DA_prices=prices_24h,
                num_segments_h=10,
                num_segments_p_pump=10,
                num_segments_p_turbine=10
            )
            
            results, metrics = optimizer.solve()
            
            solution_time = time.time() - start_time
            
            if results is None:
                print(f"No optimal solution found for {date_str}!")
                continue
            
            # Run simulation
            p_tensor = torch.tensor(results['p'], dtype=torch.float32, device=device)
            q_tensor = torch.tensor(results['q'], dtype=torch.float32, device=device)
            h_tensor = torch.tensor(results['h'], dtype=torch.float32, device=device)
            
            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_tensor, q_tensor, h_tensor)
            
            # Calculate simulation profit
            da_prices_tensor = torch.tensor(prices_24h, dtype=torch.float32, device=device)
            profit, si_penalty, vol_penalty, op_cost = simulator.calc_profit(
                p_sim, p_tensor[:len(p_sim)], v_low_sim, da_prices_tensor[:len(p_sim)]
            )
            
            # Store detailed results (same format as MIQP_nn.py)
            for hour in range(T):
                detailed_results.append({
                    'date': date_str,
                    'hour': hour,
                    'power': results['p'][hour],
                    'head': results['h'][hour],
                    'volume': results['v_low'][hour],
                    'flow': results['q'][hour],
                    'price': prices_24h[hour]
                })
            
            # Store benchmark results (same format as MIQP_nn.py)
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
            
            print(f"Expected profit: {metrics['ExpectedProfit']:.2f} €, Ex-post profit: {profit.item():.2f} €")
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            continue
    
    # Save results (same format as MIQP_nn.py)
    detailed_df = pd.DataFrame(detailed_results)
    benchmark_df = pd.DataFrame(benchmark_results)
    
    detailed_df.to_csv("MIQP_piecewise_results.csv", index=False)
    benchmark_df.to_csv("MIQP_piecewise_benchmark.csv", index=False)
    
    print(f"\nProcessing complete!")
    print(f"Detailed results saved to MIQP_piecewise_results.csv ({len(detailed_results)} rows)")
    print(f"Benchmark results saved to MIQP_piecewise_benchmark.csv ({len(benchmark_results)} rows)")

# %% Execute
if __name__ == "__main__":
    run_piecewise_optimization()

# %% Rerun Feb 14 (failed) and append to existing CSV files
# Load price data and find Feb 14
price_data = read_price_data()
target_date = '2024/2/14'  # Based on your CSV format
if target_date not in price_data:
    print(f"Date {target_date} not found in price data!")
    print(f"Available dates: {list(price_data.keys())}")
else:
    print(f"Processing {target_date}...")
    
    # Get prices
    prices_24h = price_data[target_date]
    
    # Initialize simulation parameters
    head_init_val = torch.tensor(head_init, dtype=torch.float32, device=device)
    v_low_init_val = torch.tensor(v_low_init, dtype=torch.float32, device=device)
    
    params = HydroParameters(
        head_init=head_init_val, v_low_init=v_low_init_val,
        neg_min=neg_min, neg_max=neg_max, pos_min=pos_min, pos_max=pos_max,
        predict_q_poly=predict_q_poly, h_to_v_low_fitted=h_to_v_low_fitted,
        v_low_to_h_fitted=v_low_to_h_fitted
    )
    simulator = SimulationLayer(params)
    
    # Optimize
    start_time = time.time()
    optimizer = PiecewiseMILPOptimizerSOS2(
        T=24, DA_prices=prices_24h,
        num_segments_h=10, num_segments_p_pump=10, num_segments_p_turbine=10
    )
    results, metrics = optimizer.solve()
    solution_time = time.time() - start_time
    
    if results is None:
        print(f"Optimization failed: Status {metrics['Status']}")
    else:
        # Run simulation
        p_tensor = torch.tensor(results['p'], dtype=torch.float32, device=device)
        q_tensor = torch.tensor(results['q'], dtype=torch.float32, device=device)
        h_tensor = torch.tensor(results['h'], dtype=torch.float32, device=device)
        
        p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_tensor, q_tensor, h_tensor)
        
        da_prices_tensor = torch.tensor(prices_24h, dtype=torch.float32, device=device)
        profit, si_penalty, vol_penalty, op_cost = simulator.calc_profit(
            p_sim, p_tensor, v_low_sim, da_prices_tensor
        )
        
        # Prepare new data
        detailed_new = []
        for hour in range(24):
            detailed_new.append({
                'date': target_date, 'hour': hour, 'power': results['p'][hour],
                'head': results['h'][hour], 'volume': results['v_low'][hour],
                'flow': results['q'][hour], 'price': prices_24h[hour]
            })
        
        benchmark_new = [{
            'Date': target_date, 'Solving Time (s)': solution_time,
            'MIP Gap': metrics.get('MIPGap', 0), 'Binary Variables': metrics.get('NumBinVars', 0),
            'Continuous Variables': metrics.get('NumVars', 0) - metrics.get('NumBinVars', 0),
            'Total Constraints': metrics.get('NumConstrs', 0),
            'Expected Profit (€)': metrics.get('ExpectedProfit', 0),
            'SI Penalty (€)': si_penalty.item(), 'Vol Penalty (€)': vol_penalty.item(),
            'Op Cost (€)': op_cost.item(), 'Ex-post Profit (€)': profit.item()
        }]
        
        # Append to existing CSV files
        try:
            # Load and append detailed results
            detailed_df = pd.read_csv("MIQP_piecewise_results.csv")
            detailed_new_df = pd.DataFrame(detailed_new)
            detailed_updated = pd.concat([detailed_df, detailed_new_df], ignore_index=True)
            detailed_updated.to_csv("MIQP_piecewise_results.csv", index=False)
            
            # Load and append benchmark results
            benchmark_df = pd.read_csv("MIQP_piecewise_benchmark.csv")
            benchmark_new_df = pd.DataFrame(benchmark_new)
            benchmark_updated = pd.concat([benchmark_df, benchmark_new_df], ignore_index=True)
            benchmark_updated.to_csv("MIQP_piecewise_benchmark.csv", index=False)
            
            print(f"SUCCESS! Added {target_date} to CSV files:")
            print(f"Expected profit: {metrics.get('ExpectedProfit', 0):.2f}€")
            print(f"Ex-post profit: {profit.item():.2f}€")
            print(f"Detailed results: {len(detailed_updated)} total rows")
            print(f"Benchmark results: {len(benchmark_updated)} total rows")
            
        except FileNotFoundError as e:
            print(f"CSV file not found: {e}")
            print("Creating new CSV files with Feb 14 data...")
            pd.DataFrame(detailed_new).to_csv("MIQP_piecewise_results.csv", index=False)
            pd.DataFrame(benchmark_new).to_csv("MIQP_piecewise_benchmark.csv", index=False)
# %%
