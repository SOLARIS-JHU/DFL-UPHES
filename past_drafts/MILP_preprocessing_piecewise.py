# %% Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import dill as pickle
import pandas as pd
import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

device = torch.device('cpu')

head_init = 77.0 # Initial head value
v_low_init = h_to_v_low_fitted(head_init) # Initial lower reservoir volume

# %% Load day-ahead prices
# Load day-ahead prices
def load_prices():
    """Load day-ahead prices from database"""
    df = pd.read_csv("./Data/price_database.csv")
    
    price_data = {}
    for _, row in df.iterrows():
        date = row['date']
        prices = eval(row['prices_hourly'])  # Convert string to list
        price_data[date] = prices

    return price_data

def discretize_range(min_val, max_val, n_segments):
    """Create discretization points for a range"""
    return np.linspace(min_val, max_val, n_segments + 1)

def compute_piecewise_coefficients_power_head(predict_q_func, p_range, h_range, n_segments_p, n_segments_h):
    """
    Compute piecewise linear coefficients for the UPC function predicting flow
    from power and head, divided into segments.
    
    Returns:
        Dictionary with coefficients for each segment
    """
    # Create breakpoints for power and head
    p_breaks = discretize_range(p_range[0], p_range[1], n_segments_p)
    h_breaks = discretize_range(h_range[0], h_range[1], n_segments_h)
    
    # Prepare storage for coefficients
    coefs = {}
    
    # For each segment
    for i in range(n_segments_p):
        for j in range(n_segments_h):
            # Get corners of the segment
            p_min, p_max = p_breaks[i], p_breaks[i+1]
            h_min, h_max = h_breaks[j], h_breaks[j+1]
            
            # Create tensors for the corners
            p_corners = torch.tensor([p_min, p_min, p_max, p_max], dtype=torch.float32, device=device)
            h_corners = torch.tensor([h_min, h_max, h_min, h_max], dtype=torch.float32, device=device)
            
            # Calculate flow at the corners
            q_corners = predict_q_func(p_corners, h_corners)
            
            # Build matrix for least squares fit: q = a*p + b*h + c
            A = torch.stack([p_corners, h_corners, torch.ones_like(p_corners)], dim=1)
            b = q_corners.unsqueeze(1)
            
            # Solve for coefficients using least squares
            result = torch.linalg.lstsq(A, b, rcond=None)
            solution = result.solution  # Extract solution from the named tuple
            a, b, c = solution.squeeze().tolist()
            
            # Store coefficients
            coefs[(i, j)] = {
                'a': a,             # Coefficient for power
                'b': b,             # Coefficient for head
                'c': c,             # Constant term
                'p_range': (p_min, p_max),
                'h_range': (h_min, h_max)
            }
    
    return {
        'p_breaks': p_breaks.tolist(),
        'h_breaks': h_breaks.tolist(),
        'coefficients': coefs
    }

def compute_piecewise_coefficients_head_volume(h_to_v_func, h_range, n_segments):
    """
    Compute piecewise linear coefficients for head-volume relationship
    
    Returns:
        Dictionary with slopes and intercepts for each segment
    """
    # Create breakpoints for head
    h_breaks = discretize_range(h_range[0], h_range[1], n_segments)
    
    # Prepare storage for coefficients
    slopes = []
    intercepts = []
    
    # For each segment
    for i in range(n_segments):
        h_min, h_max = h_breaks[i], h_breaks[i+1]
        
        # Calculate volume at segment endpoints
        v_min = h_to_v_func(torch.tensor(h_min, device=device)).item()
        v_max = h_to_v_func(torch.tensor(h_max, device=device)).item()
        
        # Linear approximation: v = m*h + b
        slope = (v_max - v_min) / (h_max - h_min)
        intercept = v_min - slope * h_min
        
        slopes.append(slope)
        intercepts.append(intercept)
    
    return {
        'breaks': h_breaks.tolist(),
        'slopes': slopes,
        'intercepts': intercepts
    }

class PiecewiseMILPOptimizer:
    def __init__(self, T, DA_prices, n_segments_p=5, n_segments_h=3, n_segments_vh=3,
                 C_op=3.8, M_p=10000, h_init=head_init, h_min=head_min, h_max=head_max, 
                 v_low_init=v_low_init, v_low_target=target_vol_low):
        """
        MILP optimizer with piecewise linearization for pumped hydro storage.
        
        Parameters:
            T (int): Number of time periods.
            DA_prices (list): Day-ahead prices for each period.
            n_segments_p (int): Number of power segments for UPC linearization.
            n_segments_h (int): Number of head segments for UPC linearization.
            n_segments_vh (int): Number of segments for volume-head linearization.
            C_op (float): Operational cost coefficient.
            M_p (float): Big-M constant.
            h_init (float): Initial head value.
            h_min (float): Minimum head.
            h_max (float): Maximum head.
            v_low_init (float): Initial lower reservoir volume.
            v_low_target (float): Target lower reservoir volume.
        """
        self.T = T
        self.DA_prices = DA_prices
        self.n_segments_p = n_segments_p
        self.n_segments_h = n_segments_h
        self.n_segments_vh = n_segments_vh
        self.C_op = C_op
        self.M_p = M_p
        self.h_min = h_min
        self.h_max = h_max
        self.v_low_init = v_low_init
        self.v_low_target = v_low_target
        self.h_init = h_init
        
        # Compute piecewise linearization coefficients
        self.compute_linearization_coefficients()
        
        # Create Gurobi model
        self.model = gp.Model("PiecewiseMILP")
        self._build_model()
    
    def compute_linearization_coefficients(self):
        """Compute all piecewise linearization coefficients"""
        # Find power ranges for turbine and pump modes
        h_samples = torch.linspace(self.h_min, self.h_max, 20, device=device)
        
        # Turbine mode power bounds
        p_min_tur = torch.tensor([pos_min(h).item() for h in h_samples])
        p_max_tur = torch.tensor([pos_max(h).item() for h in h_samples])
        min_p_tur = max(0.1, torch.min(p_min_tur).item())  # Ensure positive
        max_p_tur = torch.max(p_max_tur).item()
        
        # Pump mode power bounds
        p_min_pump = torch.tensor([neg_min(h).item() for h in h_samples])
        p_max_pump = torch.tensor([neg_max(h).item() for h in h_samples])
        min_p_pump = torch.min(p_min_pump).item()
        max_p_pump = min(-0.1, torch.max(p_max_pump).item())  # Ensure negative
        
        # Define helper functions for linearization
        def flow_tur_func(p, h):
            """Turbine flow function"""
            return torch.where(p > 0, predict_q_poly(p, h), torch.zeros_like(p))
        
        def flow_pump_func(p, h):
            """Pump flow function"""
            return torch.where(p < 0, predict_q_poly(p, h), torch.zeros_like(p))
        
        # Compute piecewise linearization for turbine UPC
        self.tur_lin = compute_piecewise_coefficients_power_head(
            flow_tur_func,
            (min_p_tur, max_p_tur),
            (self.h_min, self.h_max),
            self.n_segments_p,
            self.n_segments_h
        )
        
        # Compute piecewise linearization for pump UPC
        self.pump_lin = compute_piecewise_coefficients_power_head(
            flow_pump_func,
            (min_p_pump, max_p_pump),
            (self.h_min, self.h_max),
            self.n_segments_p,
            self.n_segments_h
        )
        
        # Compute piecewise linearization for volume-head relationship
        self.vhead_lin = compute_piecewise_coefficients_head_volume(
            h_to_v_low_fitted,
            (self.h_min, self.h_max),
            self.n_segments_vh
        )
        
        # Store power ranges for later use
        self.p_ranges = {
            'turbine': (min_p_tur, max_p_tur),
            'pump': (min_p_pump, max_p_pump)
        }
    
    def _build_model(self):
        """Build the MILP model with piecewise linearization"""
        T = self.T
        M_p = self.M_p
        
        # Decision Variables
        # Power variables
        self.p_T = self.model.addVars(T, lb=0, name="p_T")  # Turbine power (>=0)
        self.p_P = self.model.addVars(T, lb=-GRB.INFINITY, ub=0, name="p_P")  # Pump power (<=0)
        
        # Flow, head, and volume variables
        self.q = self.model.addVars(T, lb=-GRB.INFINITY, name="q")
        self.h = self.model.addVars(T, lb=self.h_min, ub=self.h_max, name="h")
        self.v_low = self.model.addVars(T, name="v_low")
        
        # Mode selection binary variables
        self.z_I = self.model.addVars(T, vtype=GRB.BINARY, name="z_I")  # Idle
        self.z_T = self.model.addVars(T, vtype=GRB.BINARY, name="z_T")  # Turbine
        self.z_P = self.model.addVars(T, vtype=GRB.BINARY, name="z_P")  # Pump
        
        # Binary variables for selecting piecewise segments
        # For volume-head relationship
        self.z_vh = self.model.addVars(T, self.n_segments_vh, vtype=GRB.BINARY, name="z_vh")
        
        # For turbine UPC segments
        self.z_tur = {}
        for t in range(T):
            for i in range(self.n_segments_p):
                for j in range(self.n_segments_h):
                    self.z_tur[t, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"z_tur_{t}_{i}_{j}")
        
        # For pump UPC segments
        self.z_pump = {}
        for t in range(T):
            for i in range(self.n_segments_p):
                for j in range(self.n_segments_h):
                    self.z_pump[t, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"z_pump_{t}_{i}_{j}")
        
        # Mode selection: exactly one mode is active at each time t
        for t in range(T):
            self.model.addConstr(self.z_I[t] + self.z_T[t] + self.z_P[t] == 1, name=f"mode_sel_{t}")
        
        # Ensure one volume-head segment is active per time step
        for t in range(T):
            self.model.addConstr(gp.quicksum(self.z_vh[t, s] for s in range(self.n_segments_vh)) == 1, 
                               name=f"vh_seg_sel_{t}")
        
        # Ensure one turbine UPC segment is active when in turbine mode
        for t in range(T):
            self.model.addConstr(
                gp.quicksum(self.z_tur[t, i, j] for i in range(self.n_segments_p) 
                           for j in range(self.n_segments_h)) == self.z_T[t],
                name=f"tur_seg_sel_{t}"
            )
        
        # Ensure one pump UPC segment is active when in pump mode
        for t in range(T):
            self.model.addConstr(
                gp.quicksum(self.z_pump[t, i, j] for i in range(self.n_segments_p) 
                           for j in range(self.n_segments_h)) == self.z_P[t],
                name=f"pump_seg_sel_{t}"
            )
        
        # Idle Mode Constraints
        for t in range(T):
            self.model.addConstr(self.p_T[t] <= M_p * (1 - self.z_I[t]), name=f"idle_pT_{t}")
            self.model.addConstr(self.p_P[t] >= -M_p * (1 - self.z_I[t]), name=f"idle_pP_{t}")
            self.model.addConstr(self.q[t] <= M_p * (1 - self.z_I[t]), name=f"idle_q_pos_{t}")
            self.model.addConstr(self.q[t] >= -M_p * (1 - self.z_I[t]), name=f"idle_q_neg_{t}")
        
        # Turbine Mode Constraints: Global bounds based on head
        for t in range(T):
            self.model.addConstr(self.p_T[t] >= pos_min_fit[0] * self.h[t] + pos_min_fit[1] * self.z_T[t],
                               name=f"turbine_min_{t}")
            self.model.addConstr(self.p_T[t] <= pos_max_fit[0] * self.h[t] + pos_max_fit[1] * self.z_T[t],
                               name=f"turbine_max_{t}")
        
        # Pump Mode Constraints: Global bounds based on head
        for t in range(T):
            self.model.addConstr(self.p_P[t] >= neg_min_fit[0] * self.h[t] + neg_min_fit[1] * self.z_P[t],
                               name=f"pump_min_{t}")
            self.model.addConstr(self.p_P[t] <= neg_max_fit[0] * self.h[t] + neg_max_fit[1] * self.z_P[t],
                               name=f"pump_max_{t}")
        
        # Piecewise linearization for turbine UPC
        p_breaks_tur = self.tur_lin['p_breaks']
        h_breaks_tur = self.tur_lin['h_breaks']
        
        for t in range(T):
            for i in range(self.n_segments_p):
                for j in range(self.n_segments_h):
                    # Get segment bounds
                    coef = self.tur_lin['coefficients'][(i, j)]
                    p_min, p_max = coef['p_range']
                    h_min, h_max = coef['h_range']
                    a, b, c = coef['a'], coef['b'], coef['c']
                    
                    # Power and head must be within segment bounds if segment is active
                    self.model.addConstr(
                        self.p_T[t] >= p_min - M_p * (1 - self.z_tur[t, i, j]),
                        name=f"tur_p_min_{t}_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.p_T[t] <= p_max + M_p * (1 - self.z_tur[t, i, j]),
                        name=f"tur_p_max_{t}_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.h[t] >= h_min - M_p * (1 - self.z_tur[t, i, j]),
                        name=f"tur_h_min_{t}_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.h[t] <= h_max + M_p * (1 - self.z_tur[t, i, j]),
                        name=f"tur_h_max_{t}_{i}_{j}"
                    )
                    
                    # Flow calculation based on the active segment
                    self.model.addConstr(
                        self.q[t] >= a * self.p_T[t] + b * self.h[t] + c - M_p * (1 - self.z_tur[t, i, j]),
                        name=f"tur_flow_lb_{t}_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.q[t] <= a * self.p_T[t] + b * self.h[t] + c + M_p * (1 - self.z_tur[t, i, j]),
                        name=f"tur_flow_ub_{t}_{i}_{j}"
                    )
        
        # Piecewise linearization for pump UPC
        p_breaks_pump = self.pump_lin['p_breaks']
        h_breaks_pump = self.pump_lin['h_breaks']
        
        for t in range(T):
            for i in range(self.n_segments_p):
                for j in range(self.n_segments_h):
                    # Get segment bounds
                    coef = self.pump_lin['coefficients'][(i, j)]
                    p_min, p_max = coef['p_range']
                    h_min, h_max = coef['h_range']
                    a, b, c = coef['a'], coef['b'], coef['c']
                    
                    # Power and head must be within segment bounds if segment is active
                    self.model.addConstr(
                        self.p_P[t] >= p_min - M_p * (1 - self.z_pump[t, i, j]),
                        name=f"pump_p_min_{t}_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.p_P[t] <= p_max + M_p * (1 - self.z_pump[t, i, j]),
                        name=f"pump_p_max_{t}_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.h[t] >= h_min - M_p * (1 - self.z_pump[t, i, j]),
                        name=f"pump_h_min_{t}_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.h[t] <= h_max + M_p * (1 - self.z_pump[t, i, j]),
                        name=f"pump_h_max_{t}_{i}_{j}"
                    )
                    
                    # Flow calculation based on the active segment
                    self.model.addConstr(
                        self.q[t] >= a * self.p_P[t] + b * self.h[t] + c - M_p * (1 - self.z_pump[t, i, j]),
                        name=f"pump_flow_lb_{t}_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.q[t] <= a * self.p_P[t] + b * self.h[t] + c + M_p * (1 - self.z_pump[t, i, j]),
                        name=f"pump_flow_ub_{t}_{i}_{j}"
                    )
        
        # Piecewise linearization for volume-head relationship
        h_breaks_vh = self.vhead_lin['breaks']
        
        for t in range(T):
            for s in range(self.n_segments_vh):
                # Get segment bounds and coefficients
                h_min = h_breaks_vh[s]
                h_max = h_breaks_vh[s+1]
                slope = self.vhead_lin['slopes'][s]
                intercept = self.vhead_lin['intercepts'][s]
                
                # Head must be within segment bounds if segment is active
                self.model.addConstr(
                    self.h[t] >= h_min - M_p * (1 - self.z_vh[t, s]),
                    name=f"vh_h_min_{t}_{s}"
                )
                self.model.addConstr(
                    self.h[t] <= h_max + M_p * (1 - self.z_vh[t, s]),
                    name=f"vh_h_max_{t}_{s}"
                )
                
                # Volume calculation based on the active segment
                self.model.addConstr(
                    self.v_low[t] >= slope * self.h[t] + intercept - M_p * (1 - self.z_vh[t, s]),
                    name=f"vh_vol_lb_{t}_{s}"
                )
                self.model.addConstr(
                    self.v_low[t] <= slope * self.h[t] + intercept + M_p * (1 - self.z_vh[t, s]),
                    name=f"vh_vol_ub_{t}_{s}"
                )
        
        # Volume dynamics equations
        for t in range(T):
            if t == 0:
                self.model.addConstr(
                    self.v_low[t] == self.v_low_init + 3600 * self.q[t],
                    name=f"vol_dyn_{t}"
                )
            else:
                self.model.addConstr(
                    self.v_low[t] == self.v_low[t-1] + 3600 * self.q[t],
                    name=f"vol_dyn_{t}"
                )
        
        # Final volume constraint
        self.model.addConstr(self.v_low[T-1] <= self.v_low_target, name="vol_target")
        
        # Objective: Maximize profit
        objective = gp.quicksum(
            (self.p_T[t] + self.p_P[t]) * self.DA_prices[t] -
            self.C_op * (self.p_T[t] + self.p_P[t]) * (self.p_T[t] + self.p_P[t])
            for t in range(T)
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)
        
        # Optional output settings
        self.model.Params.OutputFlag = 1
    
    def solve(self):
        """Optimize the MILP model and return results"""
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
            print(f"Optimization error: {self.model.status}")
            return None

def plot_linearization_comparison(nonlinear_func, linear_approx, piecewise_approx, title):
    """
    Plot comparison between nonlinear function, single linear approximation,
    and piecewise linear approximation
    """
    plt.figure(figsize=(10, 6))
    
    # Generate data points for plotting
    x = np.linspace(min(piecewise_approx['breaks']), max(piecewise_approx['breaks']), 100)
    
    # Calculate nonlinear function values
    y_nonlinear = [nonlinear_func(torch.tensor(xi, device=device)).item() for xi in x]
    
    # Calculate single linear approximation
    slope_linear = linear_approx[0]
    intercept_linear = linear_approx[1]
    y_linear = [slope_linear * xi + intercept_linear for xi in x]
    
    # Plot nonlinear function
    plt.plot(x, y_nonlinear, 'b-', linewidth=2, label='Nonlinear function')
    
    # Plot single linear approximation
    plt.plot(x, y_linear, 'g--', linewidth=2, label='Single linear approximation')
    
    # Plot piecewise linear approximation
    for i in range(len(piecewise_approx['breaks']) - 1):
        x_segment = np.linspace(piecewise_approx['breaks'][i], piecewise_approx['breaks'][i+1], 10)
        y_segment = [piecewise_approx['slopes'][i] * xi + piecewise_approx['intercepts'][i] for xi in x_segment]
        if i == 0:
            plt.plot(x_segment, y_segment, 'r-', linewidth=2, label='Piecewise linear approximation')
        else:
            plt.plot(x_segment, y_segment, 'r-', linewidth=2)
    
    # Mark breakpoints
    plt.scatter(piecewise_approx['breaks'], 
               [piecewise_approx['slopes'][min(i, len(piecewise_approx['slopes'])-1)] * 
                piecewise_approx['breaks'][i] + 
                piecewise_approx['intercepts'][min(i, len(piecewise_approx['intercepts'])-1)] 
                for i in range(len(piecewise_approx['breaks']))],
               color='red', s=50)
    
    plt.title(title)
    plt.xlabel('Head (m)')
    plt.ylabel('Volume (m³)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_2d_piecewise_linearization(predict_q_func, piecewise_approx, title):
    """
    Create an interactive 3D visualization comparing the nonlinear function
    and its piecewise linear approximation.
    
    Parameters:
        predict_q_func: Original nonlinear function
        piecewise_approx: Dictionary with piecewise approximation data
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Extract breakpoints
    p_breaks = np.array(piecewise_approx['p_breaks'])
    h_breaks = np.array(piecewise_approx['h_breaks'])
    
    # Range for power and head
    p_min, p_max = p_breaks.min(), p_breaks.max()
    h_min, h_max = h_breaks.min(), h_breaks.max()
    
    # Create grid for visualization
    p_grid = np.linspace(p_min, p_max, 30)
    h_grid = np.linspace(h_min, h_max, 30)
    P, H = np.meshgrid(p_grid, h_grid)
    
    # Calculate nonlinear function values
    P_flat = P.flatten()
    H_flat = H.flatten()
    P_tensor = torch.tensor(P_flat, dtype=torch.float32, device=device)
    H_tensor = torch.tensor(H_flat, dtype=torch.float32, device=device)
    Q_flat = predict_q_func(P_tensor, H_tensor).cpu().numpy()
    Q_nonlinear = Q_flat.reshape(P.shape)
    
    # Calculate piecewise linear approximation
    Q_piecewise = np.zeros_like(Q_nonlinear)
    
    # For each grid point, find the appropriate segment and apply the formula
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            p_val = P[i, j]
            h_val = H[i, j]
            
            # Find segment indices
            p_idx = np.searchsorted(p_breaks, p_val) - 1
            h_idx = np.searchsorted(h_breaks, h_val) - 1
            
            # Handle edge cases
            p_idx = max(0, min(p_idx, len(p_breaks) - 2))
            h_idx = max(0, min(h_idx, len(h_breaks) - 2))
            
            # Apply formula for this segment
            if (p_idx, h_idx) in piecewise_approx['coefficients']:
                coef = piecewise_approx['coefficients'][(p_idx, h_idx)]
                a, b, c = coef['a'], coef['b'], coef['c']
                Q_piecewise[i, j] = a * p_val + b * h_val + c
            else:
                Q_piecewise[i, j] = np.nan
    
    # Create figure
    fig = go.Figure()
    
    # Add the original nonlinear function surface
    fig.add_trace(
        go.Surface(
            x=P, y=H, z=Q_nonlinear,
            colorscale='Viridis',
            opacity=0.7,
            name='Original Nonlinear Function'
        )
    )
    
    # Add the piecewise linear approximation surface
    fig.add_trace(
        go.Surface(
            x=P, y=H, z=Q_piecewise,
            colorscale='Reds',
            opacity=0.5,
            name='Piecewise Linear Approximation'
        )
    )
    
    # Add lines to show segment boundaries
    for p_val in p_breaks:
        # Create vertical planes at each power breakpoint
        h_vals = np.linspace(h_min, h_max, 20)
        p_vals = np.full_like(h_vals, p_val)
        
        # Calculate q values along this line using piecewise approximation
        q_vals = []
        for h_val in h_vals:
            p_idx = np.searchsorted(p_breaks, p_val) - 1
            h_idx = np.searchsorted(h_breaks, h_val) - 1
            p_idx = max(0, min(p_idx, len(p_breaks) - 2))
            h_idx = max(0, min(h_idx, len(h_breaks) - 2))
            
            if (p_idx, h_idx) in piecewise_approx['coefficients']:
                coef = piecewise_approx['coefficients'][(p_idx, h_idx)]
                a, b, c = coef['a'], coef['b'], coef['c']
                q_vals.append(a * p_val + b * h_val + c)
            else:
                q_vals.append(np.nan)
        
        fig.add_trace(
            go.Scatter3d(
                x=p_vals, y=h_vals, z=q_vals,
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False
            )
        )
    
    for h_val in h_breaks:
        # Create horizontal planes at each head breakpoint
        p_vals = np.linspace(p_min, p_max, 20)
        h_vals = np.full_like(p_vals, h_val)
        
        # Calculate q values along this line using piecewise approximation
        q_vals = []
        for p_val in p_vals:
            p_idx = np.searchsorted(p_breaks, p_val) - 1
            h_idx = np.searchsorted(h_breaks, h_val) - 1
            p_idx = max(0, min(p_idx, len(p_breaks) - 2))
            h_idx = max(0, min(h_idx, len(h_breaks) - 2))
            
            if (p_idx, h_idx) in piecewise_approx['coefficients']:
                coef = piecewise_approx['coefficients'][(p_idx, h_idx)]
                a, b, c = coef['a'], coef['b'], coef['c']
                q_vals.append(a * p_val + b * h_val + c)
            else:
                q_vals.append(np.nan)
        
        fig.add_trace(
            go.Scatter3d(
                x=p_vals, y=h_vals, z=q_vals,
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Power (p)',
            yaxis_title='Head (h)',
            zaxis_title='Flow (q)'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def evaluate_piecewise_linearization(predict_q_func, piecewise_approx, n_samples=10000):
    """
    Evaluate the accuracy of the piecewise linearization by comparing
    with the original nonlinear function at random sample points.
    
    Parameters:
        predict_q_func: Original nonlinear function
        piecewise_approx: Dictionary with piecewise approximation data
        n_samples: Number of random samples to evaluate
    
    Returns:
        Dictionary with error metrics
    """
    import numpy as np
    
    # Extract breakpoints
    p_breaks = np.array(piecewise_approx['p_breaks'])
    h_breaks = np.array(piecewise_approx['h_breaks'])
    
    # Range for power and head
    p_min, p_max = p_breaks.min(), p_breaks.max()
    h_min, h_max = h_breaks.min(), h_breaks.max()
    
    # Generate random sample points
    np.random.seed(0)  # For reproducibility
    p_samples = np.random.uniform(p_min, p_max, n_samples)
    h_samples = np.random.uniform(h_min, h_max, n_samples)
    
    # Calculate nonlinear function values
    p_tensor = torch.tensor(p_samples, dtype=torch.float32, device=device)
    h_tensor = torch.tensor(h_samples, dtype=torch.float32, device=device)
    q_true = predict_q_func(p_tensor, h_tensor).cpu().numpy()
    
    # Calculate piecewise linear approximation values
    q_approx = np.zeros_like(q_true)
    
    for i in range(n_samples):
        p_val = p_samples[i]
        h_val = h_samples[i]
        
        # Find segment indices
        p_idx = np.searchsorted(p_breaks, p_val) - 1
        h_idx = np.searchsorted(h_breaks, h_val) - 1
        
        # Handle edge cases
        p_idx = max(0, min(p_idx, len(p_breaks) - 2))
        h_idx = max(0, min(h_idx, len(h_breaks) - 2))
        
        # Apply formula for this segment
        if (p_idx, h_idx) in piecewise_approx['coefficients']:
            coef = piecewise_approx['coefficients'][(p_idx, h_idx)]
            a, b, c = coef['a'], coef['b'], coef['c']
            q_approx[i] = a * p_val + b * h_val + c
        else:
            q_approx[i] = np.nan
    
    # Calculate error metrics
    valid_indices = ~np.isnan(q_approx)
    if not np.any(valid_indices):
        return {"Error": "No valid approximations found"}
    
    q_true_valid = q_true[valid_indices]
    q_approx_valid = q_approx[valid_indices]
    
    mae = np.mean(np.abs(q_true_valid - q_approx_valid))
    rmse = np.sqrt(np.mean((q_true_valid - q_approx_valid) ** 2))
    mape = np.mean(np.abs((q_true_valid - q_approx_valid) / (q_true_valid + 1e-10))) * 100
    
    return {
        "Mean Absolute Error": mae,
        "Root Mean Squared Error": rmse,
        "Mean Absolute Percentage Error": mape,
        "Number of Samples": len(q_true_valid)
    }


# Main execution
if __name__ == "__main__":
    # Load prices
    prices = load_prices()
    sample_date = list(prices.keys())[0]
    T = 24
    DA_prices = prices[sample_date]
    

    
    # Create optimizer with specified number of segments
    optimizer = PiecewiseMILPOptimizer(
        T=T, 
        DA_prices=DA_prices,
        n_segments_p=3,  # Number of power segments
        n_segments_h=3,  # Number of head segments
        n_segments_vh=5  # Number of volume-head segments
    )
    
    # Optional: Plot comparison of linearizations
    plot_linearization_comparison(
        h_to_v_low_fitted,
        h_vlow_coeff_lin,
        optimizer.vhead_lin,
        'Comparison of Volume-Head Linearization Methods'
    )
    
    # Solve the optimization problem
    results = optimizer.solve()
    
    if results is not None:
        print("Optimization completed successfully!")
        
        # Plot power schedule
        plt.figure(figsize=(12, 6))
        plt.plot(range(T), [results['p_t_T'][t] + results['p_t_P'][t] for t in range(T)], 'b-', linewidth=2)
        plt.xlabel('Time period')
        plt.ylabel('Power (MW)')
        plt.title('Optimal Power Schedule')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Print summary of results
        print(f"Total energy generation: {sum(results['p_t_T'])} MWh")
        print(f"Total energy consumption: {abs(sum(results['p_t_P']))} MWh")
        print(f"Final head: {results['h_t'][-1]} m")
        print(f"Final lower reservoir volume: {results['v_low'][-1]} m³")

# %% 