# %% Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm, trange
import gurobipy as gp
from gurobipy import GRB

# Set device to CPU
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

# %% Load day-ahead prices
def load_prices():
    """Load day-ahead prices from database and set as environment variables."""
    df = pd.read_csv("./Data/price_database.csv")
    price_data = {}
    for _, row in df.iterrows():
        date = row['date']
        prices = eval(row['prices_hourly'])  # Convert string to list
        price_data[date] = prices
    return price_data

# %% MILP Optimizer with Piecewise Linearization
class MILPOptimizer:
    def __init__(self, T, DA_prices, C_op=3.8, M_p=1000, h_init=head_init, h_min=head_min, h_max=head_max, 
                 v_low_init=v_low_init, v_low_target=target_vol_low,
                 n_intervals_v=2, n_intervals_h=2, n_intervals_p=2):
        """
        MILP optimizer with piecewise linearization.
        
        Parameters:
            T (int): Number of time periods.
            DA_prices (list): Day-ahead prices.
            C_op (float): Operational cost coefficient.
            M_p (float): Big-M constant (default 10000).
            h_init (float): Initial head.
            h_min (float): Minimum head.
            h_max (float): Maximum head.
            v_low_init (float): Initial lower reservoir volume.
            v_low_target (float): Target lower reservoir volume.
            n_intervals_v (int): Number of segments for volume-head piecewise linearization.
            n_intervals_h (int): Number of subintervals for head in UPC bilinear interpolation.
            n_intervals_p (int): Number of subintervals for power in UPC bilinear interpolation.
        """
        self.T = T
        self.DA_prices = DA_prices
        self.C_op = C_op
        self.M_p = M_p
        self.h_min = h_min
        self.h_max = h_max
        self.v_low_init = v_low_init
        self.v_low_target = v_low_target
        self.h_init = h_init
        
        self.n_intervals_v = n_intervals_v
        self.n_intervals_h = n_intervals_h
        self.n_intervals_p = n_intervals_p

        self.model = gp.Model("PipelineMILP")
        self._build_model()
    
    def _build_model(self):
        T = self.T
        M_p = self.M_p
        
        # ----------------------------
        # Decision Variables
        # ----------------------------
        self.p_T = self.model.addVars(T, lb=0, name="p_T")      # Turbine power (>=0)
        self.p_P = self.model.addVars(T, lb=-GRB.INFINITY, ub=0, name="p_P")  # Pump power (<=0)
        self.q   = self.model.addVars(T, lb=-GRB.INFINITY, name="q")
        self.h   = self.model.addVars(T, lb=self.h_min, ub=self.h_max, name="h")
        self.v_low = self.model.addVars(T, name="v_low")
        
        self.z_I = self.model.addVars(T, vtype=GRB.BINARY, name="z_I")
        self.z_T = self.model.addVars(T, vtype=GRB.BINARY, name="z_T")
        self.z_P = self.model.addVars(T, vtype=GRB.BINARY, name="z_P")
        
        for t in range(T):
            self.model.addConstr(self.z_I[t] + self.z_T[t] + self.z_P[t] == 1, name=f"mode_sel_{t}")
        
        for t in range(T):
            self.model.addConstr(self.p_T[t] <= M_p * (1 - self.z_I[t]), name=f"idle_pT_{t}")
            self.model.addConstr(self.p_P[t] >= -M_p * (1 - self.z_I[t]), name=f"idle_pP_{t}")
            self.model.addConstr(self.q[t] <=  M_p * (1 - self.z_I[t]), name=f"idle_q_{t}")
        
        # ----------------------------
        # Piecewise Linear Volume-Head Relationship
        # ----------------------------
        v_min = 0.0
        v_max = max_vol_low
        v_breaks = np.linspace(v_min, v_max, self.n_intervals_v + 1)
        H_breaks_v = [float(h_to_v_low_fitted(v)) for v in v_breaks]
        
        # Print the computed v_breaks and H_breaks_v values
        print("v_breaks:", v_breaks)
        print("H_breaks_v:", H_breaks_v)
        
        self.delta_v = self.model.addVars(T, self.n_intervals_v, vtype=GRB.BINARY, name="delta_v")
        for t in range(T):
            self.model.addConstr(gp.quicksum(self.delta_v[t, i] for i in range(self.n_intervals_v)) == 1,
                                 name=f"vol_seg_select_{t}")
            for i in range(self.n_intervals_v):
                self.model.addConstr(self.v_low[t] >= v_breaks[i] - M_p * (1 - self.delta_v[t,i]),
                                     name=f"v_low_seg{i}_lb_{t}")
                self.model.addConstr(self.v_low[t] <= v_breaks[i+1] + M_p * (1 - self.delta_v[t,i]),
                                     name=f"v_low_seg{i}_ub_{t}")
                slope_v = (H_breaks_v[i+1] - H_breaks_v[i]) / (v_breaks[i+1] - v_breaks[i])
                intercept_v = H_breaks_v[i] - slope_v * v_breaks[i]
                self.model.addConstr(self.h[t] <= slope_v * self.v_low[t] + intercept_v + M_p * (1 - self.delta_v[t,i]),
                                     name=f"h_seg{i}_ub_{t}")
                self.model.addConstr(self.h[t] >= slope_v * self.v_low[t] + intercept_v - M_p * (1 - self.delta_v[t,i]),
                                     name=f"h_seg{i}_lb_{t}")
        
        # ----------------------------
        # Bilinear Interpolation for UPC (Unit Performance Curve)
        # ----------------------------
        H_breaks = np.linspace(self.h_min, self.h_max, self.n_intervals_h + 1)
        pT_min_vals = [pos_min_fit[0]*H + pos_min_fit[1] for H in [self.h_min, self.h_max]]
        pT_max_vals = [pos_max_fit[0]*H + pos_max_fit[1] for H in [self.h_min, self.h_max]]
        P_T_breaks = np.linspace(min(pT_min_vals), max(pT_max_vals), self.n_intervals_p + 1)
        pP_min_vals = [neg_min_fit[0]*H + neg_min_fit[1] for H in [self.h_min, self.h_max]]
        pP_max_vals = [neg_max_fit[0]*H + neg_max_fit[1] for H in [self.h_min, self.h_max]]
        P_P_breaks = np.linspace(min(pP_min_vals), max(pP_max_vals), self.n_intervals_p + 1)
        
        Q_turbine = {}
        Q_pump = {}
        for i in range(self.n_intervals_h + 1):
            for j in range(self.n_intervals_p + 1):
                H_val = H_breaks[i]
                P_val = P_T_breaks[j]
                Q_val = predict_q_poly(np.array(P_val), np.array(H_val)).cpu().item()
                Q_turbine[(i,j)] = Q_val
        for i in range(self.n_intervals_h + 1):
            for j in range(self.n_intervals_p + 1):
                H_val = H_breaks[i]
                P_val = P_P_breaks[j]
                Q_val = predict_q_poly(np.array(P_val), np.array(H_val)).cpu().item()
                Q_pump[(i,j)] = Q_val
        
        self.delta_T = self.model.addVars(T, self.n_intervals_h, self.n_intervals_p, vtype=GRB.BINARY, name="delta_T")
        self.delta_P = self.model.addVars(T, self.n_intervals_h, self.n_intervals_p, vtype=GRB.BINARY, name="delta_P")
        for t in range(T):
            self.model.addConstr(gp.quicksum(self.delta_T[t,i,j] for i in range(self.n_intervals_h) for j in range(self.n_intervals_p)) == self.z_T[t],
                                 name=f"T_domain_select_{t}")
            self.model.addConstr(gp.quicksum(self.delta_P[t,i,j] for i in range(self.n_intervals_h) for j in range(self.n_intervals_p)) == self.z_P[t],
                                 name=f"P_domain_select_{t}")
        
        self.alpha_T = self.model.addVars(T, self.n_intervals_h, self.n_intervals_p, 4, lb=0, ub=1, name="alpha_T")
        self.alpha_P = self.model.addVars(T, self.n_intervals_h, self.n_intervals_p, 4, lb=0, ub=1, name="alpha_P")
        
        for t in range(T):
            for i in range(self.n_intervals_h):
                for j in range(self.n_intervals_p):
                    # Turbine Mode constraints
                    self.model.addConstr(self.h[t] >= H_breaks[i] - M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"h_T_seg{i}{j}_lb_{t}")
                    self.model.addConstr(self.h[t] <= H_breaks[i+1] + M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"h_T_seg{i}{j}_ub_{t}")
                    self.model.addConstr(self.p_T[t] >= P_T_breaks[j] - M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"pT_seg{i}{j}_lb_{t}")
                    self.model.addConstr(self.p_T[t] <= P_T_breaks[j+1] + M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"pT_seg{i}{j}_ub_{t}")
                    self.model.addConstr(self.alpha_T[t,i,j,0] + self.alpha_T[t,i,j,1] +
                                           self.alpha_T[t,i,j,2] + self.alpha_T[t,i,j,3] == self.delta_T[t,i,j],
                                           name=f"alphaT_sum_{t}_{i}{j}")
                    head_expr = (H_breaks[i] * (self.alpha_T[t,i,j,0] + self.alpha_T[t,i,j,2]) +
                                 H_breaks[i+1] * (self.alpha_T[t,i,j,1] + self.alpha_T[t,i,j,3]))
                    self.model.addConstr(self.h[t] <= head_expr + M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"h_bilin_T_{t}_{i}{j}_ub")
                    self.model.addConstr(self.h[t] >= head_expr - M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"h_bilin_T_{t}_{i}{j}_lb")
                    power_expr = (P_T_breaks[j] * (self.alpha_T[t,i,j,0] + self.alpha_T[t,i,j,1]) +
                                  P_T_breaks[j+1] * (self.alpha_T[t,i,j,2] + self.alpha_T[t,i,j,3]))
                    self.model.addConstr(self.p_T[t] <= power_expr + M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"pT_bilin_{t}_{i}{j}_ub")
                    self.model.addConstr(self.p_T[t] >= power_expr - M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"pT_bilin_{t}_{i}{j}_lb")
                    flow_expr = (Q_turbine[(i,j)]     * self.alpha_T[t,i,j,0] +
                                 Q_turbine[(i+1,j)]   * self.alpha_T[t,i,j,1] +
                                 Q_turbine[(i,j+1)]   * self.alpha_T[t,i,j,2] +
                                 Q_turbine[(i+1,j+1)] * self.alpha_T[t,i,j,3])
                    self.model.addConstr(self.q[t] <= flow_expr + M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"q_bilin_T_{t}_{i}{j}_ub")
                    self.model.addConstr(self.q[t] >= flow_expr - M_p * (1 - self.delta_T[t,i,j]),
                                         name=f"q_bilin_T_{t}_{i}{j}_lb")
                    
                    # Pump Mode constraints
                    self.model.addConstr(self.h[t] >= H_breaks[i] - M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"h_P_seg{i}{j}_lb_{t}")
                    self.model.addConstr(self.h[t] <= H_breaks[i+1] + M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"h_P_seg{i}{j}_ub_{t}")
                    self.model.addConstr(self.p_P[t] >= P_P_breaks[j] - M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"pP_seg{i}{j}_lb_{t}")
                    self.model.addConstr(self.p_P[t] <= P_P_breaks[j+1] + M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"pP_seg{i}{j}_ub_{t}")
                    self.model.addConstr(self.alpha_P[t,i,j,0] + self.alpha_P[t,i,j,1] +
                                           self.alpha_P[t,i,j,2] + self.alpha_P[t,i,j,3] == self.delta_P[t,i,j],
                                           name=f"alphaP_sum_{t}_{i}{j}")
                    head_expr_P = (H_breaks[i] * (self.alpha_P[t,i,j,0] + self.alpha_P[t,i,j,2]) +
                                   H_breaks[i+1] * (self.alpha_P[t,i,j,1] + self.alpha_P[t,i,j,3]))
                    self.model.addConstr(self.h[t] <= head_expr_P + M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"h_bilin_P_{t}_{i}{j}_ub")
                    self.model.addConstr(self.h[t] >= head_expr_P - M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"h_bilin_P_{t}_{i}{j}_lb")
                    power_expr_P = (P_P_breaks[j] * (self.alpha_P[t,i,j,0] + self.alpha_P[t,i,j,1]) +
                                    P_P_breaks[j+1] * (self.alpha_P[t,i,j,2] + self.alpha_P[t,i,j,3]))
                    self.model.addConstr(self.p_P[t] <= power_expr_P + M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"pP_bilin_{t}_{i}{j}_ub")
                    self.model.addConstr(self.p_P[t] >= power_expr_P - M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"pP_bilin_{t}_{i}{j}_lb")
                    flow_expr_P = (Q_pump[(i,j)]     * self.alpha_P[t,i,j,0] +
                                   Q_pump[(i+1,j)]   * self.alpha_P[t,i,j,1] +
                                   Q_pump[(i,j+1)]   * self.alpha_P[t,i,j,2] +
                                   Q_pump[(i+1,j+1)] * self.alpha_P[t,i,j,3])
                    self.model.addConstr(self.q[t] <= flow_expr_P + M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"q_bilin_P_{t}_{i}{j}_ub")
                    self.model.addConstr(self.q[t] >= flow_expr_P - M_p * (1 - self.delta_P[t,i,j]),
                                         name=f"q_bilin_P_{t}_{i}{j}_lb")
        
        # ----------------------------
        # Volume Dynamics
        # ----------------------------
        for t in range(T):
            if t == 0:
                self.model.addConstr(self.v_low[t] == self.v_low_init + 3600 * self.q[t],
                                     name=f"vol_dyn_{t}")
            else:
                self.model.addConstr(self.v_low[t] == self.v_low[t-1] + 3600 * self.q[t],
                                     name=f"vol_dyn_{t}")
        
        self.model.addConstr(self.v_low[T-1] <= self.v_low_target, name="vol_target")
        
        # ----------------------------
        # Objective
        # ----------------------------
        objective = gp.quicksum(
            (self.p_T[t] + self.p_P[t]) * self.DA_prices[t] -
            self.C_op * (self.p_T[t] + self.p_P[t]) * (self.p_T[t] + self.p_P[t])
            for t in range(T)
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)
        self.model.Params.OutputFlag = 1

    def solve(self):
        """Optimize the MILP and return decision variable values."""
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            results = {
                'p_t_T': [self.p_T[t].X for t in range(self.T)],
                'p_t_P': [self.p_P[t].X for t in range(self.T)],
                'q_t':   [self.q[t].X for t in range(self.T)],
                'h_t':   [self.h[t].X for t in range(self.T)],
                'v_low': [self.v_low[t].X for t in range(self.T)],
                'z_I':   [self.z_I[t].X for t in range(self.T)],
                'z_T':   [self.z_T[t].X for t in range(self.T)],
                'z_P':   [self.z_P[t].X for t in range(self.T)]
            }
            return results
        else:
            print("No optimal solution found!")
            return None

# %% Main execution
if __name__ == "__main__":
    prices = load_prices()
    print(f"Loaded prices for {len(prices)} days")
    
    sample_date = list(prices.keys())[0]
    print(f"\nPrices for {sample_date}: {prices[sample_date]}")
    
    T = 24
    DA_prices = prices[sample_date]
    
    optimizer = MILPOptimizer(T, DA_prices, n_intervals_v=2, n_intervals_h=2, n_intervals_p=2)
    results = optimizer.solve()
    
    if results is not None:
        print("Optimization results:")
        for key, value in results.items():
            print(f"{key}: {value}")
    else:
        # If no solution is found, compute and print the IIS constraints.
        print("The model is infeasible. Computing IIS to identify problematic constraints...")
        optimizer.model.computeIIS()
        optimizer.model.write("model.ilp")
        print("\nConstraints causing infeasibility:")
        for c in optimizer.model.getConstrs():
            if c.IISConstr:
                print(f"{c.ConstrName}: {c}")

# %%
