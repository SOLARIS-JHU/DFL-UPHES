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

# %% Piecewise MILP Optimizer with SOS2 constraints
class PiecewiseMILPOptimizerSOS2:
    def __init__(self, T, DA_prices, num_segments_h=5, num_segments_p_pump=5, num_segments_p_turbine=5, 
                 C_op=3.8, M_p=10000, h_init=head_init, h_min=head_min, h_max=head_max, 
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

# %% Piecewise MILP Optimizer with Big-M quadratic constraints
class PiecewiseMILPOptimizer:
    def __init__(self, T, DA_prices, num_segments_h=5, num_segments_p_pump=5, num_segments_p_turbine=5, 
                 C_op=3.8, M_p=10000, h_init=head_init, h_min=head_min, h_max=head_max, 
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

# %% Batch Solving Code
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

def batch_solve_all_dates():
    """Batch solve optimization problems for all dates in the price database."""
    # Create directories if they don't exist
    os.makedirs('./Data', exist_ok=True)
    os.makedirs('./Benchmark', exist_ok=True)
    
    # Load all prices
    price_data = load_prices()
    all_dates = list(price_data.keys())
    
    # Batch solving with PiecewiseMILPOptimizerSOS2
    print("Starting batch solving with SOS2 optimizer...")
    solve_with_optimizer(
        optimizer_class=PiecewiseMILPOptimizerSOS2,
        optimizer_name="SOS2",
        price_data=price_data,
        all_dates=all_dates
    )
    
    # # Batch solving with PiecewiseMILPOptimizer
    # print("Starting batch solving with Big-M quadratic optimizer...")
    # solve_with_optimizer(
    #     optimizer_class=PiecewiseMILPOptimizer,
    #     optimizer_name="big_m_quad",
    #     price_data=price_data,
    #     all_dates=all_dates
    # )

def solve_with_optimizer(optimizer_class, optimizer_name, price_data, all_dates):
    """Solve the optimization problem for all dates with the given optimizer class."""
    # Initialize result lists
    operation_data = []
    benchmark_data = []
    
    for date in all_dates:
        print(f"Solving {optimizer_name} optimizer for date: {date}")
        
        # Get prices for this date
        prices = price_data[date]
        T = len(prices)
        
        # Create the optimizer
        optimizer = optimizer_class(T=T, DA_prices=prices)
        
        # Set Threads parameter (other parameters are set in solve method)
        optimizer.model.Params.Threads = 16
        
        # Solve and time the optimization
        start_time = time.time()
        results = optimizer.solve()
        solve_time = time.time() - start_time
        
        if results is not None:
            # Extract operation data
            for t in range(T):
                operation_data.append([
                    t,                  # Time
                    results['p'][t],    # Power
                    results['h'][t],    # Head
                    results['q'][t],    # Flow
                    results['v_low'][t], # Volume
                    prices[t],          # Price
                    date                # Date
                ])
            
            # Calculate expected profit
            expected_profit = sum(results['p'][t] * prices[t] - optimizer.C_op * results['p'][t] * results['p'][t] for t in range(T))
            
            # Extract benchmark data
            benchmark_data.append({
                'Date': date,
                'SolveTime': solve_time,
                'ExpectedProfit': expected_profit,
                'MIPGap': optimizer.model.MIPGap,
                'Status': optimizer.model.status,
                'NumVars': optimizer.model.NumVars,
                'NumConstrs': optimizer.model.NumConstrs,
                'NumBinVars': optimizer.model.NumBinVars,
                'ObjectiveBound': optimizer.model.ObjBound,
                'ObjectiveValue': optimizer.model.ObjVal
            })
        else:
            print(f"Failed to solve for date: {date}")
            # Add benchmark data for failed solve
            benchmark_data.append({
                'Date': date,
                'SolveTime': solve_time,
                'ExpectedProfit': np.nan,
                'MIPGap': np.nan,
                'Status': optimizer.model.status,
                'NumVars': optimizer.model.NumVars,
                'NumConstrs': optimizer.model.NumConstrs,
                'NumBinVars': optimizer.model.NumBinVars,
                'ObjectiveBound': np.nan,
                'ObjectiveValue': np.nan
            })
    
    # Save operation data to CSV
    operation_df = pd.DataFrame(operation_data, 
                               columns=['Time', 'Power', 'Head', 'Flow', 'Volume', 'Price', 'Date'])
    operation_df.to_csv('./Data/SOS2_database.csv', index=False)
    # operation_df.to_csv(f'./Data/piecewise_opreation_data_{optimizer_name}.csv', index=False)
    
    # Save benchmark data to CSV
    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df.to_csv('./Benchmark/SOS2_database_bm.csv', index=False)
    # benchmark_df.to_csv(f'./Benchmark/piecewise_opreation_data_{optimizer_name}_bm.csv', index=False)
    
    print(f"Completed batch solving with {optimizer_name} optimizer")
    return operation_df, benchmark_df

# Run the batch solving
if __name__ == "__main__":
    batch_solve_all_dates()
# %% Plotting Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import os
from matplotlib.colors import LinearSegmentedColormap

# Load the optimization results
df_sos2 = pd.read_csv('./Data/piecewise_opreation_data_SOS2.csv')

# Calculate the true flow values using predict_q_poly
true_q = []
for i, row in df_sos2.iterrows():
    p = row['Power']
    h = row['Head']
    q_true = predict_q_poly(torch.tensor(float(p)), torch.tensor(float(h))).item()
    true_q.append(q_true)

# Add the true flow values to the dataframe
df_sos2['True_Flow'] = true_q
df_sos2['Flow_Error'] = df_sos2['Flow'] - df_sos2['True_Flow']
df_sos2['Flow_Error_Pct'] = (df_sos2['Flow_Error'] / df_sos2['True_Flow'].abs()) * 100
df_sos2['Flow_Error_Abs'] = df_sos2['Flow_Error'].abs()

# Create output directory for plots
os.makedirs('./Plots', exist_ok=True)

# 1. Scatter plot: Optimized vs. True Flow values
plt.figure(figsize=(10, 8))
plt.scatter(df_sos2['True_Flow'], df_sos2['Flow'], alpha=0.6)
min_val = min(df_sos2['True_Flow'].min(), df_sos2['Flow'].min())
max_val = max(df_sos2['True_Flow'].max(), df_sos2['Flow'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel('True Flow (predict_q_poly)', fontsize=12)
plt.ylabel('Optimized Flow (SOS2)', fontsize=12)
plt.title('Comparison of Optimized vs. True Flow Values', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Plots/flow_comparison_scatter.png', dpi=300)

# 2. Histogram of flow errors
plt.figure(figsize=(10, 6))
plt.hist(df_sos2['Flow_Error'], bins=50, alpha=0.7, color='blue')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Flow Error (Optimized - True)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Flow Errors', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Plots/flow_error_histogram.png', dpi=300)

# 3. Box plot of error by operation mode
plt.figure(figsize=(10, 6))
# Determine operation mode based on Power value
df_sos2['Mode'] = 'Idle'
df_sos2.loc[df_sos2['Power'] > 0, 'Mode'] = 'Turbine'
df_sos2.loc[df_sos2['Power'] < 0, 'Mode'] = 'Pump'

# Create box plot
df_sos2_with_modes = df_sos2[df_sos2['Mode'] != 'Idle']  # Exclude idle mode for better visualization
plt.boxplot([df_sos2_with_modes[df_sos2_with_modes['Mode'] == mode]['Flow_Error_Abs'] 
             for mode in ['Turbine', 'Pump']], 
            labels=['Turbine', 'Pump'])
plt.ylabel('Absolute Flow Error', fontsize=12)
plt.title('Flow Error by Operation Mode', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Plots/flow_error_by_mode.png', dpi=300)

# 4. Heat map of error by Head and Power
plt.figure(figsize=(12, 10))
# Create a custom colormap from blue to white to red
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)

# Set up the plot
plt.scatter(df_sos2['Power'], df_sos2['Head'], 
           c=df_sos2['Flow_Error'], cmap=cm, 
           alpha=0.7, s=50, edgecolors='k', linewidths=0.5)
plt.colorbar(label='Flow Error')
plt.xlabel('Power', fontsize=12)
plt.ylabel('Head', fontsize=12)
plt.title('Flow Error by Power and Head', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Plots/flow_error_heatmap.png', dpi=300)

# 5. Time series plot of optimized vs true flow for a sample date
sample_date = df_sos2['Date'].unique()[0]  # Get the first date as a sample
sample_df = df_sos2[df_sos2['Date'] == sample_date]

plt.figure(figsize=(12, 6))
plt.plot(sample_df['Time'], sample_df['Flow'], 'b-', linewidth=2, label='Optimized Flow')
plt.plot(sample_df['Time'], sample_df['True_Flow'], 'r--', linewidth=2, label='True Flow')
plt.xlabel('Time Period', fontsize=12)
plt.ylabel('Flow', fontsize=12)
plt.title(f'Optimized vs. True Flow Over Time (Date: {sample_date})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('./Plots/flow_timeseries_comparison.png', dpi=300)

# 6. Summary statistics 
error_stats = {
    'Mean Absolute Error': df_sos2['Flow_Error_Abs'].mean(),
    'Max Absolute Error': df_sos2['Flow_Error_Abs'].max(),
    'Mean Percent Error': df_sos2['Flow_Error_Pct'].abs().mean(),
    'Median Absolute Error': df_sos2['Flow_Error_Abs'].median(),
    'Standard Deviation': df_sos2['Flow_Error'].std()
}

# Print and save error statistics
print("\nError Statistics:")
for stat, value in error_stats.items():
    print(f"{stat}: {value}")

# Save error statistics to CSV
pd.DataFrame([error_stats]).to_csv('./Plots/error_statistics.csv', index=False)

# 7. Create visualization of approximation quality across head values
# For selected head values, plot the true UPC curves and the approximated points
selected_heads = [df_sos2['Head'].min(), df_sos2['Head'].mean(), df_sos2['Head'].max()]
selected_heads = [round(h, 1) for h in selected_heads]  # Round for better display

plt.figure(figsize=(15, 10))
colors = ['blue', 'green', 'red']

for i, head in enumerate(selected_heads):
    # Select data points close to this head value (within ±1% tolerance)
    tolerance = 0.01 * head
    head_data = df_sos2[(df_sos2['Head'] >= head - tolerance) & 
                        (df_sos2['Head'] <= head + tolerance)]
    
    # Generate true UPC curve for this head value
    p_values = np.linspace(
        df_sos2['Power'].min() * 1.1,  # Slightly extend range for visualization
        df_sos2['Power'].max() * 1.1,
        100
    )
    true_q_values = [predict_q_poly(torch.tensor(float(p)), torch.tensor(float(head))).item() 
                     for p in p_values]
    
    # Plot true curve
    plt.plot(p_values, true_q_values, color=colors[i], linestyle='-', 
             label=f'True UPC (h={head})')
    
    # Plot optimized points
    if not head_data.empty:
        plt.scatter(head_data['Power'], head_data['Flow'], 
                   color=colors[i], marker='o', s=50, alpha=0.7,
                   label=f'SOS2 Approx (h={head})')

plt.xlabel('Power', fontsize=14)
plt.ylabel('Flow', fontsize=14)
plt.title('Comparison of True UPC Curves vs. SOS2 Approximations', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('./Plots/upc_curve_comparison.png', dpi=300)

# Create a comprehensive summary figure
plt.figure(figsize=(15, 12))

# Plot 1: True vs. Optimized Flow
plt.subplot(2, 2, 1)
plt.scatter(df_sos2['True_Flow'], df_sos2['Flow'], alpha=0.6)
min_val = min(df_sos2['True_Flow'].min(), df_sos2['Flow'].min())
max_val = max(df_sos2['True_Flow'].max(), df_sos2['Flow'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel('True Flow')
plt.ylabel('Optimized Flow')
plt.title('True vs. Optimized Flow')
plt.grid(True, alpha=0.3)

# Plot 2: Error Distribution
plt.subplot(2, 2, 2)
plt.hist(df_sos2['Flow_Error'], bins=30, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Flow Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True, alpha=0.3)

# Plot 3: Error Heatmap
plt.subplot(2, 2, 3)
plt.scatter(df_sos2['Power'], df_sos2['Head'], 
           c=df_sos2['Flow_Error'], cmap=cm, 
           alpha=0.7, s=40, edgecolors='k', linewidths=0.5)
plt.colorbar(label='Error')
plt.xlabel('Power')
plt.ylabel('Head')
plt.title('Error by Power & Head')
plt.grid(True, alpha=0.3)

# Plot 4: Mean Absolute Error by Date
plt.subplot(2, 2, 4)
mae_by_date = df_sos2.groupby('Date')['Flow_Error_Abs'].mean().reset_index()
plt.bar(range(len(mae_by_date)), mae_by_date['Flow_Error_Abs'], alpha=0.7)
plt.xticks(range(len(mae_by_date)), mae_by_date['Date'], rotation=45)
plt.xlabel('Date')
plt.ylabel('Mean Absolute Error')
plt.title('Error by Date')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./Plots/summary_analysis.png', dpi=300)

print("Analysis complete. All plots saved to the './Plots' directory.")

#%% 
# Simulation class for the hydro system
class HydroParameters:
    def __init__(
        self,
        time_horizon=24, # number of time periods
        sampling_rate=50, # number of samples for regression
        δ_p=0.5,
        δ_h=1,
        δ_q=0.5,
        operational_cost=3.8,
        rho=1000,
        g=9.81,
        mu=0.9,
        head_min=head_min,
        head_max=head_max,
        max_vol_up=max_vol_up,
        min_vol_low=min_vol_low,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        target_head=target_head,
        target_vol_low=target_vol_low,
        head_init=head_init,
        v_low_init=v_low_init,
        neg_min_fit=neg_min_fit, 
        neg_max_fit=neg_max_fit,   
        pos_min_fit=pos_min_fit,     
        pos_max_fit=pos_max_fit,
        neg_min=neg_min,
        neg_max=neg_max,
        pos_min=pos_min,
        pos_max=pos_max,
        predict_q_poly=predict_q_poly,
        h_to_v_low_fitted=h_to_v_low_fitted,
        gross_head=gross_head, 
        v_low_to_h_fitted=v_low_to_h_fitted,
    ):
        self.time_horizon = time_horizon
        self.sampling_rate = sampling_rate
        self.operational_cost = operational_cost
        
        self.δ_p = torch.tensor(δ_p, dtype=torch.float32, device=device)
        self.δ_h = torch.tensor(δ_h, dtype=torch.float32, device=device)
        self.δ_q = torch.tensor(δ_q, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rho, dtype=torch.float32, device=device)
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)

        self.head_min = torch.tensor(head_min, dtype=torch.float32, device=device)
        self.head_max = torch.tensor(head_max, dtype=torch.float32, device=device)
        self.max_vol_up = torch.tensor(max_vol_up, dtype=torch.float32, device=device)
        self.min_vol_low = torch.tensor(min_vol_low, dtype=torch.float32, device=device)
        self.ramp_up = torch.tensor(ramp_up, dtype=torch.float32, device=device)
        self.ramp_down = torch.tensor(ramp_down, dtype=torch.float32, device=device)

        self.target_head = torch.tensor(target_head, dtype=torch.float32, device=device)
        self.target_vol_low = torch.tensor(target_vol_low, dtype=torch.float32, device=device)
        
        # Check if inputs are already tensors, if not convert them
        if isinstance(head_init, torch.Tensor):
            self.head_init = head_init.to(device=device, dtype=torch.float32)
        else:
            self.head_init = torch.tensor(head_init, dtype=torch.float32, device=device)
            
        if isinstance(v_low_init, torch.Tensor):
            self.v_low_init = v_low_init.to(device=device, dtype=torch.float32)
        else:
            self.v_low_init = torch.tensor(v_low_init, dtype=torch.float32, device=device)

        self.neg_min_fit = torch.tensor(neg_min_fit, dtype=torch.float32, device=device)
        self.neg_max_fit = torch.tensor(neg_max_fit, dtype=torch.float32, device=device)
        self.pos_min_fit = torch.tensor(pos_min_fit, dtype=torch.float32, device=device)
        self.pos_max_fit = torch.tensor(pos_max_fit, dtype=torch.float32, device=device)

        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.predict_q_poly = predict_q_poly
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.gross_head = gross_head
        self.v_low_to_h_fitted = v_low_to_h_fitted

    def to_cpu(self):
        """Move all PyTorch tensors to CPU"""
        for attr_name in dir(self):
            # Skip private attributes, methods, and callable attributes
            if attr_name.startswith('_') or callable(getattr(self, attr_name)):
                continue
            
            try:
                attr = getattr(self, attr_name)
                
                # Handle tensors
                if isinstance(attr, torch.Tensor):
                    setattr(self, attr_name, attr.cpu())
                
                # Handle lists of tensors
                elif isinstance(attr, list):
                    new_list = []
                    for item in attr:
                        if isinstance(item, torch.Tensor):
                            new_list.append(item.cpu())
                        else:
                            new_list.append(item)
                    setattr(self, attr_name, new_list)
                
                # Handle dictionaries containing tensors
                elif isinstance(attr, dict):
                    new_dict = {}
                    for key, value in attr.items():
                        if isinstance(value, torch.Tensor):
                            new_dict[key] = value.cpu()
                        else:
                            new_dict[key] = value
                    setattr(self, attr_name, new_dict)
            except: # Skip attributes that cannot be accessed or operated on
                pass 
        
        return self  # Return self to support method chaining
    
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
            q (torch.Tensor): Hourly flow schedule [time_horizon]
            h (torch.Tensor): Hourly head schedule [time_horizon]
        
        Returns:
            tuple: Calibrated hourly (p, q, h, v_low) schedules.
        """
        TH = self.params.time_horizon
        
        # Initialize lists for each state
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        # Start states
        v_current = self.params.v_low_init  # user-chosen initial reservoir volume
        v_list.append(v_current)

        for i in range(TH):
            # Current state values
            h_current = h[i]
            p_current = p[i]
            
            # a) Base: idle => q=0
            q_candidate = torch.zeros_like(p_current)
            # Initialize p_clamped as zero for idle mode
            p_clamped = torch.zeros_like(p_current)

            # b) For turbine mode (p_current>0), clamp p between pos_min(h) and pos_max(h)
            #    then get q via polynomial
            if p_current > 0.5:  # Turbine mode
                p_min_turb = self.params.pos_min(h_current)
                p_max_turb = self.params.pos_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            
            # c) For pump mode (p_current<0), clamp p between neg_min(h) and neg_max(h)
            elif p_current < -0.5:  # Pump mode
                p_min_pump = self.params.neg_min(h_current)
                p_max_pump = self.params.neg_max(h_current)
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
                # Use p_clamped only if we're not in idle mode
                p_final = p_clamped
                q_final = q_candidate
                # Update head based on new volume
                h_next = self.params.v_low_to_h_fitted(v_next)
            
            # Append states for this hour
            p_list.append(p_final)
            q_list.append(q_final)
            h_list.append(h_next)
            v_list.append(v_next.item())
            
            # Update current volume for next iteration
            v_current = v_next
        
        # Convert lists to tensors
        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list[:-1])  # Remove the extra head value
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32)
        
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
        surplus_penalty_multiplier = -0.75
        shortage_penalty_multiplier = -1.5

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

# Evaluate optimization performance with penalties
import pandas as pd
import numpy as np
import torch
import os

def evaluate_optimization_performance():
    """
    Evaluate the performance of the SOS2 optimization by calculating:
    - Expected Profit
    - SI Penalty
    - Volume Penalty
    - Operational Cost
    - Ex-post Profit (Expected Profit - SI Penalty - Volume Penalty)
    
    Appends results to the benchmark CSV file.
    """
    print("Evaluating SOS2 optimization performance...")
    
    # Load operation data
    operation_df = pd.read_csv('./Data/piecewise_opreation_data_SOS2.csv')
    
    # Load benchmark data
    benchmark_path = './Benchmark/piecewise_opreation_data_SOS2_bm.csv'
    if os.path.exists(benchmark_path):
        benchmark_df = pd.read_csv(benchmark_path)
    else:
        print(f"Benchmark file {benchmark_path} not found.")
        return
    
    # Initialize results list for all dates
    results_list = []
    
    # Process each date
    for date in operation_df['Date'].unique():
        print(f"Processing date: {date}")
        
        # Filter data for current date
        date_df = operation_df[operation_df['Date'] == date]
        
        # Extract optimization results
        p_opt = torch.tensor(date_df['Power'].values, dtype=torch.float32)
        h_opt = torch.tensor(date_df['Head'].values, dtype=torch.float32)
        q_opt = torch.tensor(date_df['Flow'].values, dtype=torch.float32)
        v_low_opt = torch.tensor(date_df['Volume'].values, dtype=torch.float32)
        price = torch.tensor(date_df['Price'].values, dtype=torch.float32)
        
        # Get initial values for simulation
        head_init = date_df.iloc[0]['Head']
        v_low_init = date_df.iloc[0]['Volume']
        
        # Create parameters with same values as optimizer
        params = HydroParameters(
            time_horizon=len(date_df),
            head_init=head_init,
            v_low_init=v_low_init
        )
        
        # Create simulation layer
        simulator = SimulationLayer(params)
        
        # Simulate operation to get actual values
        p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_opt, q_opt, h_opt)
        
        # Calculate profit components
        total_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
            p_sim, p_opt, v_low_sim, price
        )
        
        # Convert tensor to scalar
        if isinstance(total_profit, torch.Tensor):
            total_profit = total_profit.item()
        if isinstance(SI_penalty, torch.Tensor):
            SI_penalty = SI_penalty.item()
        if isinstance(volume_penalty, torch.Tensor):
            volume_penalty = volume_penalty.item()
        if isinstance(operating_cost, torch.Tensor):
            operating_cost = operating_cost.item()
        
        # Get expected profit from benchmark data
        expected_profit = benchmark_df.loc[benchmark_df['Date'] == date, 'ExpectedProfit'].values[0]
        
        # Append results
        results_list.append({
            'Date': date,
            'ExpectedProfit': expected_profit,
            'SIPenalty': SI_penalty,
            'VolumePenalty': volume_penalty,
            'OperationalCost': operating_cost,
            'ExPostProfit': total_profit
        })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results_list)
    
    # Merge with benchmark data
    merged_df = pd.merge(benchmark_df, results_df, on='Date', how='left')
    
    # Save updated benchmark file
    merged_df.to_csv(benchmark_path, index=False)
    
    # Print brief summary
    print("\nPerformance Summary:")
    print(f"Average Expected Profit: {results_df['ExpectedProfit'].mean():.2f}")
    print(f"Average Ex-post Profit: {results_df['ExPostProfit'].mean():.2f}")
    
    return results_df

# Run the evaluation
if __name__ == "__main__":
    evaluate_optimization_performance()

# %% Run optimization for 2024 price data
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

def load_prices_2024():
    """Load day-ahead prices from 2024 database and set as environment variables."""
    # Read price database
    df = pd.read_csv("./Data/price_data_2024.csv")
    
    # Process each day's prices
    price_data = {}
    for _, row in df.iterrows():
        date = row['date']
        # Split comma-separated string to list
        prices = [float(price) for price in row['prices_hourly'].split(",")]
        price_data[date] = prices

    return price_data

def batch_solve_2024_dates():
    """Batch solve optimization problems for all dates in the 2024 price database."""
    # Create directories if they don't exist
    os.makedirs('./Data', exist_ok=True)
    os.makedirs('./Benchmark', exist_ok=True)
    
    # Load all 2024 prices
    price_data = load_prices_2024()
    all_dates = list(price_data.keys())
    print(f"Loaded 2024 prices for {len(all_dates)} days")
    
    # Batch solving with PiecewiseMILPOptimizerSOS2
    print("Starting batch solving with SOS2 optimizer for 2024 data...")
    solve_with_optimizer_2024(
        optimizer_class=PiecewiseMILPOptimizerSOS2,
        optimizer_name="SOS2_2024",
        price_data=price_data,
        all_dates=all_dates
    )

def solve_with_optimizer_2024(optimizer_class, optimizer_name, price_data, all_dates):
    """Solve the optimization problem for all 2024 dates with the given optimizer class."""
    # Initialize result lists
    operation_data = []
    benchmark_data = []
    
    # Use 10 segments instead of default 5
    num_segments = 10
    optimizer_name = f"{optimizer_name}_{num_segments}seg"
    
    for date in all_dates:
        print(f"Solving {optimizer_name} optimizer for date: {date}")
        
        # Get prices for this date
        prices = price_data[date]
        T = len(prices)
        
        # Create the optimizer with 10 segments
        optimizer = optimizer_class(
            T=T, 
            DA_prices=prices,
            num_segments_h=num_segments,            # 10 segments for head
            num_segments_p_pump=num_segments,       # 10 segments for pump power
            num_segments_p_turbine=num_segments     # 10 segments for turbine power
        )
        
        # Set Threads parameter (other parameters are set in solve method)
        optimizer.model.Params.Threads = 16
        
        # Solve and time the optimization
        start_time = time.time()
        results = optimizer.solve()
        solve_time = time.time() - start_time
        
        if results is not None:
            # Extract operation data
            for t in range(T):
                operation_data.append([
                    t,                  # Time
                    results['p'][t],    # Power
                    results['h'][t],    # Head
                    results['q'][t],    # Flow
                    results['v_low'][t], # Volume
                    prices[t],          # Price
                    date                # Date
                ])
            
            # Calculate expected profit
            expected_profit = sum(results['p'][t] * prices[t] - optimizer.C_op * results['p'][t] * results['p'][t] for t in range(T))
            
            # Extract benchmark data
            benchmark_data.append({
                'Date': date,
                'SolveTime': solve_time,
                'ExpectedProfit': expected_profit,
                'MIPGap': optimizer.model.MIPGap,
                'Status': optimizer.model.status,
                'NumVars': optimizer.model.NumVars,
                'NumConstrs': optimizer.model.NumConstrs,
                'NumBinVars': optimizer.model.NumBinVars,
                'ObjectiveBound': optimizer.model.ObjBound,
                'ObjectiveValue': optimizer.model.ObjVal
            })
        else:
            print(f"Failed to solve for date: {date}")
            # Add benchmark data for failed solve
            benchmark_data.append({
                'Date': date,
                'SolveTime': solve_time,
                'ExpectedProfit': np.nan,
                'MIPGap': np.nan,
                'Status': optimizer.model.status,
                'NumVars': optimizer.model.NumVars,
                'NumConstrs': optimizer.model.NumConstrs,
                'NumBinVars': optimizer.model.NumBinVars,
                'ObjectiveBound': np.nan,
                'ObjectiveValue': np.nan
            })
    
    # Save operation data to CSV
    operation_df = pd.DataFrame(operation_data, 
                               columns=['Time', 'Power', 'Head', 'Flow', 'Volume', 'Price', 'Date'])
    operation_df.to_csv(f'./Data/piecewise_operation_data_{optimizer_name}.csv', index=False)
    
    # Save benchmark data to CSV
    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df.to_csv(f'./Benchmark/piecewise_operation_data_{optimizer_name}_bm.csv', index=False)
    
    print(f"Completed batch solving with {optimizer_name} optimizer for 2024 data")
    return operation_df, benchmark_df

def evaluate_optimization_performance_2024():
    """
    Evaluate the performance of the SOS2 optimization for 2024 data by calculating:
    - Expected Profit
    - SI Penalty
    - Volume Penalty
    - Operational Cost
    - Ex-post Profit (Expected Profit - SI Penalty - Volume Penalty)
    
    Appends results to the benchmark CSV file.
    """
    print("Evaluating SOS2 optimization performance for 2024 data...")
    
    # Load operation data
    operation_df = pd.read_csv('./Data/piecewise_operation_data_SOS2_2024.csv')
    
    # Load benchmark data
    benchmark_path = './Benchmark/piecewise_operation_data_SOS2_2024_bm.csv'
    if os.path.exists(benchmark_path):
        benchmark_df = pd.read_csv(benchmark_path)
    else:
        print(f"Benchmark file {benchmark_path} not found.")
        return
    
    # Initialize results list for all dates
    results_list = []
    
    # Process each date
    for date in operation_df['Date'].unique():
        print(f"Processing date: {date}")
        
        # Filter data for current date
        date_df = operation_df[operation_df['Date'] == date]
        
        # Extract optimization results
        p_opt = torch.tensor(date_df['Power'].values, dtype=torch.float32)
        h_opt = torch.tensor(date_df['Head'].values, dtype=torch.float32)
        q_opt = torch.tensor(date_df['Flow'].values, dtype=torch.float32)
        v_low_opt = torch.tensor(date_df['Volume'].values, dtype=torch.float32)
        price = torch.tensor(date_df['Price'].values, dtype=torch.float32)
        
        # Get initial values for simulation
        head_init = date_df.iloc[0]['Head']
        v_low_init = date_df.iloc[0]['Volume']
        
        # Create parameters with same values as optimizer
        params = HydroParameters(
            time_horizon=len(date_df),
            head_init=head_init,
            v_low_init=v_low_init
        )
        
        # Create simulation layer
        simulator = SimulationLayer(params)
        
        # Simulate operation to get actual values
        p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_opt, q_opt, h_opt)
        
        # Calculate profit components
        total_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
            p_sim, p_opt, v_low_sim, price
        )
        
        # Convert tensor to scalar
        if isinstance(total_profit, torch.Tensor):
            total_profit = total_profit.item()
        if isinstance(SI_penalty, torch.Tensor):
            SI_penalty = SI_penalty.item()
        if isinstance(volume_penalty, torch.Tensor):
            volume_penalty = volume_penalty.item()
        if isinstance(operating_cost, torch.Tensor):
            operating_cost = operating_cost.item()
        
        # Get expected profit from benchmark data
        expected_profit = benchmark_df.loc[benchmark_df['Date'] == date, 'ExpectedProfit'].values[0]
        
        # Append results
        results_list.append({
            'Date': date,
            'ExpectedProfit': expected_profit,
            'SIPenalty': SI_penalty,
            'VolumePenalty': volume_penalty,
            'OperationalCost': operating_cost,
            'ExPostProfit': total_profit
        })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results_list)
    
    # Merge with benchmark data
    merged_df = pd.merge(benchmark_df, results_df, on='Date', how='left')
    
    # Save updated benchmark file
    merged_df.to_csv(benchmark_path, index=False)
    
    # Print brief summary
    print("\nPerformance Summary for 2024 data:")
    print(f"Average Expected Profit: {results_df['ExpectedProfit'].mean():.2f}")
    print(f"Average Ex-post Profit: {results_df['ExPostProfit'].mean():.2f}")
    
    # Create plots directory
    os.makedirs('./Plots_2024', exist_ok=True)
    
    # Create summary plots for 2024 data
    create_summary_plots_2024(operation_df)
    
    return results_df

def create_summary_plots_2024(operation_df):
    """Create summary plots for 2024 optimization results"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Load the predict_q_poly function from the preprocess.pkl file
    with open('preprocess.pkl', 'rb') as f:
        preprocess_data = pickle.load(f)
        predict_q_poly = preprocess_data[21]  # This index should contain predict_q_poly
    
    # Calculate the true flow values using predict_q_poly
    true_q = []
    for i, row in operation_df.iterrows():
        p = float(row['Power'])
        h = float(row['Head'])
        
        # Use proper tensor shape and handle errors
        try:
            p_tensor = torch.tensor(p, dtype=torch.float32)
            h_tensor = torch.tensor(h, dtype=torch.float32)
            q_true = predict_q_poly(p_tensor, h_tensor).item()
        except Exception as e:
            print(f"Error calculating flow for row {i}: {e}")
            # If error occurs, use optimized flow as fallback
            q_true = row['Flow']
        
        true_q.append(q_true)
    
    # Add the true flow values to the dataframe
    operation_df['True_Flow'] = true_q
    operation_df['Flow_Error'] = operation_df['Flow'] - operation_df['True_Flow']
    operation_df['Flow_Error_Pct'] = (operation_df['Flow_Error'] / operation_df['True_Flow'].abs().clip(0.01)) * 100
    operation_df['Flow_Error_Abs'] = operation_df['Flow_Error'].abs()
    
    # 1. Plot profit comparison across dates
    plt.figure(figsize=(12, 6))
    
    # Determine operation mode based on Power value
    operation_df['Mode'] = 'Idle'
    operation_df.loc[operation_df['Power'] > 0, 'Mode'] = 'Turbine'
    operation_df.loc[operation_df['Power'] < 0, 'Mode'] = 'Pump'
    
    # Scatter plot: Optimized vs. True Flow
    plt.figure(figsize=(10, 8))
    plt.scatter(operation_df['True_Flow'], operation_df['Flow'], alpha=0.6)
    min_val = min(operation_df['True_Flow'].min(), operation_df['Flow'].min())
    max_val = max(operation_df['True_Flow'].max(), operation_df['Flow'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True Flow (predict_q_poly)', fontsize=12)
    plt.ylabel('Optimized Flow (SOS2)', fontsize=12)
    plt.title('Comparison of Optimized vs. True Flow Values (2024)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./Plots_2024/flow_comparison_scatter_2024.png', dpi=300)
    plt.close()
    
    # Box plot of error by operation mode
    plt.figure(figsize=(10, 6))
    operation_df_with_modes = operation_df[operation_df['Mode'] != 'Idle']  # Exclude idle mode
    operation_df_with_modes_grouped = operation_df_with_modes.groupby('Mode')
    
    # Create boxplot data
    turbine_errors = operation_df_with_modes[operation_df_with_modes['Mode'] == 'Turbine']['Flow_Error_Abs']
    pump_errors = operation_df_with_modes[operation_df_with_modes['Mode'] == 'Pump']['Flow_Error_Abs']
    
    if not turbine_errors.empty and not pump_errors.empty:
        plt.boxplot([turbine_errors, pump_errors], labels=['Turbine', 'Pump'])
        plt.ylabel('Absolute Flow Error', fontsize=12)
        plt.title('Flow Error by Operation Mode (2024)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./Plots_2024/flow_error_by_mode_2024.png', dpi=300)
    plt.close()
    
    # Heat map of error by Head and Power
    plt.figure(figsize=(12, 10))
    # Create a custom colormap
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cm = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
    
    plt.scatter(operation_df['Power'], operation_df['Head'], 
               c=operation_df['Flow_Error'], cmap=cm, 
               alpha=0.7, s=50, edgecolors='k', linewidths=0.5)
    plt.colorbar(label='Flow Error')
    plt.xlabel('Power', fontsize=12)
    plt.ylabel('Head', fontsize=12)
    plt.title('Flow Error by Power and Head (2024)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./Plots_2024/flow_error_heatmap_2024.png', dpi=300)
    plt.close()
    
    # Error statistics
    error_stats = {
        'Mean Absolute Error': operation_df['Flow_Error_Abs'].mean(),
        'Max Absolute Error': operation_df['Flow_Error_Abs'].max(),
        'Mean Percent Error': operation_df['Flow_Error_Pct'].abs().mean(),
        'Median Absolute Error': operation_df['Flow_Error_Abs'].median(),
        'Standard Deviation': operation_df['Flow_Error'].std()
    }
    
    # Print and save error statistics
    print("\nFlow Error Statistics for 2024 data:")
    for stat, value in error_stats.items():
        print(f"{stat}: {value}")
    
    # Save error statistics to CSV
    pd.DataFrame([error_stats]).to_csv('./Plots_2024/error_statistics_2024.csv', index=False)
    
    # Create a comprehensive summary figure
    plt.figure(figsize=(15, 12))
    
    # Plot 1: True vs. Optimized Flow
    plt.subplot(2, 2, 1)
    plt.scatter(operation_df['True_Flow'], operation_df['Flow'], alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True Flow')
    plt.ylabel('Optimized Flow')
    plt.title('True vs. Optimized Flow (2024)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error Distribution
    plt.subplot(2, 2, 2)
    plt.hist(operation_df['Flow_Error'], bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Flow Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution (2024)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error Heatmap
    plt.subplot(2, 2, 3)
    plt.scatter(operation_df['Power'], operation_df['Head'], 
               c=operation_df['Flow_Error'], cmap=cm, 
               alpha=0.7, s=40, edgecolors='k', linewidths=0.5)
    plt.colorbar(label='Error')
    plt.xlabel('Power')
    plt.ylabel('Head')
    plt.title('Error by Power & Head (2024)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Mean Absolute Error by Date
    plt.subplot(2, 2, 4)
    mae_by_date = operation_df.groupby('Date')['Flow_Error_Abs'].mean().reset_index()
    plt.bar(range(len(mae_by_date)), mae_by_date['Flow_Error_Abs'], alpha=0.7)
    plt.xticks(range(len(mae_by_date)), mae_by_date['Date'], rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mean Absolute Error')
    plt.title('Error by Date (2024)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./Plots_2024/summary_analysis_2024.png', dpi=300)
    plt.close()
    
    print("Analysis complete. All plots saved to the './Plots_2024' directory.")

# Run the 2024 optimization and evaluation
if __name__ == "__main__":
    print("Running optimization with 2024 price data...")
    batch_solve_2024_dates()
    evaluate_optimization_performance_2024()
    print("Completed optimization and evaluation for 2024 data!")

#%% Run and save simulation for 2024 price data with SOS2 optimizer
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
            q (torch.Tensor): Hourly flow schedule [time_horizon]
            h (torch.Tensor): Hourly head schedule [time_horizon]
        
        Returns:
            tuple: Calibrated hourly (p, q, h, v_low) schedules.
        """
        TH = self.params.time_horizon
        
        # Initialize lists for each state
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        # Start states
        v_current = self.params.v_low_init  # user-chosen initial reservoir volume
        v_list.append(v_current)

        for i in range(TH):
            # Current state values
            h_current = h[i]
            p_current = p[i]
            
            # a) Base: idle => q=0
            q_candidate = torch.zeros_like(p_current)
            p_clamped = torch.zeros_like(p_current)
            
            # b) For turbine mode (p_current>0), clamp p between pos_min(h) and pos_max(h)
            #    then get q via polynomial
            if p_current > 0.5:  # Turbine mode
                p_min_turb = self.params.pos_min(h_current)
                p_max_turb = self.params.pos_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            
            # c) For pump mode (p_current<0), clamp p between neg_min(h) and neg_max(h)
            elif p_current < -0.5:  # Pump mode
                p_min_pump = self.params.neg_min(h_current)
                p_max_pump = self.params.neg_max(h_current)
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
            h_list.append(h_next)
            v_list.append(v_next.item())
            
            # Update current volume for next iteration
            v_current = v_next
        
        # Convert lists to tensors
        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list[:-1])  # Remove the extra head value
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32)
        
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

class HydroParameters:
    def __init__(
        self,
        time_horizon=24, # number of time periods
        sampling_rate=50, # number of samples for regression
        δ_p=0.5,
        δ_h=1,
        δ_q=0.5,
        operational_cost=3.8,
        rho=1000,
        g=9.81,
        mu=0.9,
        head_min=head_min,
        head_max=head_max,
        max_vol_up=max_vol_up,
        min_vol_low=min_vol_low,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        target_head=target_head,
        target_vol_low=target_vol_low,
        head_init=head_init,
        v_low_init=v_low_init,
        neg_min_fit=neg_min_fit, 
        neg_max_fit=neg_max_fit,   
        pos_min_fit=pos_min_fit,     
        pos_max_fit=pos_max_fit,
        neg_min=neg_min,
        neg_max=neg_max,
        pos_min=pos_min,
        pos_max=pos_max,
        predict_q_poly=predict_q_poly,
        h_to_v_low_fitted=h_to_v_low_fitted,
        gross_head=gross_head, 
        v_low_to_h_fitted=v_low_to_h_fitted,
    ):
        self.time_horizon = time_horizon
        self.sampling_rate = sampling_rate
        self.operational_cost = operational_cost
        
        self.δ_p = torch.tensor(δ_p, dtype=torch.float32, device=device)
        self.δ_h = torch.tensor(δ_h, dtype=torch.float32, device=device)
        self.δ_q = torch.tensor(δ_q, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rho, dtype=torch.float32, device=device)
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)

        self.head_min = torch.tensor(head_min, dtype=torch.float32, device=device)
        self.head_max = torch.tensor(head_max, dtype=torch.float32, device=device)
        self.max_vol_up = torch.tensor(max_vol_up, dtype=torch.float32, device=device)
        self.min_vol_low = torch.tensor(min_vol_low, dtype=torch.float32, device=device)
        self.ramp_up = torch.tensor(ramp_up, dtype=torch.float32, device=device)
        self.ramp_down = torch.tensor(ramp_down, dtype=torch.float32, device=device)

        self.target_head = torch.tensor(target_head, dtype=torch.float32, device=device)
        self.target_vol_low = torch.tensor(target_vol_low, dtype=torch.float32, device=device)
        self.head_init = head_init.clone().detach().to(device=device, dtype=torch.float32)
        self.v_low_init = v_low_init.clone().detach().to(device=device, dtype=torch.float32)

        self.neg_min_fit = torch.tensor(neg_min_fit, dtype=torch.float32, device=device)
        self.neg_max_fit = torch.tensor(neg_max_fit, dtype=torch.float32, device=device)
        self.pos_min_fit = torch.tensor(pos_min_fit, dtype=torch.float32, device=device)
        self.pos_max_fit = torch.tensor(pos_max_fit, dtype=torch.float32, device=device)

        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.predict_q_poly = predict_q_poly
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.gross_head = gross_head
        self.v_low_to_h_fitted = v_low_to_h_fitted

    def to_cpu(self):
        """Move all PyTorch tensors to CPU"""
        for attr_name in dir(self):
            # Skip private attributes, methods, and callable attributes
            if attr_name.startswith('_') or callable(getattr(self, attr_name)):
                continue
            
            try:
                attr = getattr(self, attr_name)
                
                # Handle tensors
                if isinstance(attr, torch.Tensor):
                    setattr(self, attr_name, attr.cpu())
                
                # Handle lists of tensors
                elif isinstance(attr, list):
                    new_list = []
                    for item in attr:
                        if isinstance(item, torch.Tensor):
                            new_list.append(item.cpu())
                        else:
                            new_list.append(item)
                    setattr(self, attr_name, new_list)
                
                # Handle dictionaries containing tensors
                elif isinstance(attr, dict):
                    new_dict = {}
                    for key, value in attr.items():
                        if isinstance(value, torch.Tensor):
                            new_dict[key] = value.cpu()
                        else:
                            new_dict[key] = value
                    setattr(self, attr_name, new_dict)
            except: # Skip attributes that cannot be accessed or operated on
                pass 
        
        return self  # Return self to support method chaining

if __name__ == "__main__":
    # Check which optimizer results file to use - standard or 10-segment version
    optimizer_name = "SOS2_2024_10seg"  # Use the 10-segment version if available
    operation_file = f'./Data/piecewise_operation_data_{optimizer_name}.csv'
    benchmark_file = f'./Benchmark/piecewise_operation_data_{optimizer_name}_bm.csv'
    
    if not os.path.exists(operation_file):
        optimizer_name = "SOS2_2024"  # Fall back to standard version
        operation_file = f'./Data/piecewise_operation_data_{optimizer_name}.csv'
        benchmark_file = f'./Benchmark/piecewise_operation_data_{optimizer_name}_bm.csv'
    
    print(f"Using operation data from: {operation_file}")
    
    # Load operation data
    df_op = pd.read_csv(operation_file)
    
    # Load benchmark data
    df_benchmark = pd.read_csv(benchmark_file)
    
    # Create new columns for results
    results = {
        'Date': [],
        'SimProfit': [],
        'SIPenalty': [],
        'VolumePenalty': [],
        'OperatingCost': [],
        'Revenue': []
    }
    
    # Process each date
    for date in df_benchmark['Date'].unique():
        print(f"Simulating operation for {date}...")
        
        # Filter data for this day
        day_data = df_op[df_op['Date'] == date]
        
        if day_data.empty:
            print(f"No data found for {date}, skipping.")
            continue
        
        # Extract arrays
        p_opt = torch.tensor(day_data['Power'].values, dtype=torch.float32)
        h_opt = torch.tensor(day_data['Head'].values, dtype=torch.float32)
        q_opt = torch.tensor(day_data['Flow'].values, dtype=torch.float32)
        da_price = torch.tensor(day_data['Price'].values, dtype=torch.float32)
        
        # Get initial values
        head_init = torch.tensor(day_data.iloc[0]['Head'], dtype=torch.float32)
        v_low_init = torch.tensor(day_data.iloc[0]['Volume'], dtype=torch.float32)
        
        # Initialize parameters
        params = HydroParameters(
            time_horizon=len(day_data),
            head_init=head_init,
            v_low_init=v_low_init
        )
        
        # Initialize simulation layer
        sim = SimulationLayer(params)
        
        # Simulate operation
        p_sim, q_sim, h_sim, v_low_sim = sim.simulate_operation(p_opt, q_opt, h_opt)
        
        # Calculate profit and penalties
        total_profit, si_penalty, volume_penalty, operating_cost = sim.calc_profit(
            p_sim, p_opt, v_low_sim, da_price
        )
        
        # Calculate revenue
        revenue = torch.sum(da_price * p_sim).item()
        
        # Store results
        results['Date'].append(date)
        results['SimProfit'].append(total_profit.item())
        results['SIPenalty'].append(si_penalty.item())
        results['VolumePenalty'].append(volume_penalty.item())
        results['OperatingCost'].append(operating_cost.item())
        results['Revenue'].append(revenue)
        
        # Print first day details for validation
        if date == df_benchmark['Date'].iloc[0]:
            print("\nValidation for first day:")
            print(f"Optimized Power: Mean={p_opt.mean():.4f}, Min={p_opt.min():.4f}, Max={p_opt.max():.4f}")
            print(f"Simulated Power: Mean={p_sim.mean():.4f}, Min={p_sim.min():.4f}, Max={p_sim.max():.4f}")
            print(f"Diff: Mean={(p_sim - p_opt).mean():.4f}, Max Abs={(p_sim - p_opt).abs().max():.4f}")
            print(f"Initial Head: {h_opt[0]:.4f}, Final Head: {h_sim[-1]:.4f}")
            print(f"Profit: ${total_profit.item():.2f}, SI Penalty: ${si_penalty.item():.2f}")
            print(f"Volume Penalty: ${volume_penalty.item():.2f}, Operating Cost: ${operating_cost.item():.2f}")
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Merge with benchmark data
    df_merged = pd.merge(df_benchmark, df_results, on='Date', how='left')
    
    # Save updated benchmark data
    df_merged.to_csv(benchmark_file, index=False)
    
    print(f"\nUpdated benchmark data saved with simulation results to {benchmark_file}")
    
    # Display summary statistics
    print("\nSimulation Results Summary:")
    print(f"Average Simulated Profit: ${df_results['SimProfit'].mean():.2f}")
    print(f"Average SI Penalty: ${df_results['SIPenalty'].mean():.2f}")
    print(f"Average Volume Penalty: ${df_results['VolumePenalty'].mean():.2f}")
    print(f"Average Operating Cost: ${df_results['OperatingCost'].mean():.2f}")
    print(f"Average Revenue: ${df_results['Revenue'].mean():.2f}")
    
    # Plot comparison of simulated vs optimized profit
    plt.figure(figsize=(10, 6))
    plt.scatter(df_merged['ObjectiveValue'], df_merged['SimProfit'], alpha=0.6)
    plt.plot([df_merged['ObjectiveValue'].min(), df_merged['ObjectiveValue'].max()], 
             [df_merged['ObjectiveValue'].min(), df_merged['ObjectiveValue'].max()], 
             'r--')
    plt.xlabel('Optimized Profit ($)')
    plt.ylabel('Simulated Profit ($)')
    plt.title(f'Comparison of Optimized vs. Simulated Profit ({optimizer_name})')
    plt.grid(True)
    plt.savefig(f'./Plots_2024/profit_comparison_{optimizer_name}.png', dpi=300)
    plt.close()
    
    # Plot penalty distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_results['SIPenalty'], bins=20, alpha=0.5, label='SI Penalty')
    plt.hist(df_results['VolumePenalty'], bins=20, alpha=0.5, label='Volume Penalty')
    plt.xlabel('Penalty ($)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Penalties in 2024 Data ({optimizer_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./Plots_2024/penalty_distribution_{optimizer_name}.png', dpi=300)
    plt.close()
    
    # Plot daily simulation results for a sample date
    sample_date = df_results['Date'].iloc[0]
    sample_data = df_op[df_op['Date'] == sample_date]
    
    if not sample_data.empty:
        plt.figure(figsize=(12, 8))
        
        # Plot power and prices
        ax1 = plt.subplot(211)
        ax1.plot(sample_data['Time'], sample_data['Power'], 'b-', linewidth=2, label='Optimized Power')
        ax1.set_ylabel('Power (MW)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(sample_data['Time'], sample_data['Price'], 'r--', linewidth=1.5, label='Price')
        ax2.set_ylabel('Price ($/MWh)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot head and volume
        ax3 = plt.subplot(212)
        ax3.plot(sample_data['Time'], sample_data['Head'], 'g-', linewidth=2, label='Head')
        ax3.set_ylabel('Head (m)', color='g')
        ax3.set_xlabel('Time (hour)')
        ax3.tick_params(axis='y', labelcolor='g')
        
        ax4 = ax3.twinx()
        ax4.plot(sample_data['Time'], sample_data['Volume'], 'm--', linewidth=1.5, label='Volume')
        ax4.set_ylabel('Volume (m³)', color='m')
        ax4.tick_params(axis='y', labelcolor='m')
        
        # Add legend
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'./Plots_2024/daily_results_{optimizer_name}_{sample_date}.png', dpi=300)
        plt.close()
    
    # Compare performance with the linearized model results
    linear_benchmark_file = './Benchmark/global_linearized_operational_data_2024.csv'
    if os.path.exists(linear_benchmark_file):
        df_linear = pd.read_csv(linear_benchmark_file)
        
        # Create comparison dataframe with common dates
        common_dates = set(df_merged['Date']).intersection(set(df_linear['Date']))
        
        if common_dates:
            comparison_data = {
                'Date': list(common_dates),
                'SOS2_ExpectedProfit': [],
                'SOS2_SimProfit': [],
                'Linear_ExpectedProfit': [],
                'Linear_SimProfit': []
            }
            
            for date in common_dates:
                sos2_row = df_merged[df_merged['Date'] == date].iloc[0]
                linear_row = df_linear[df_linear['Date'] == date].iloc[0]
                
                comparison_data['SOS2_ExpectedProfit'].append(sos2_row['ExpectedProfit'])
                comparison_data['SOS2_SimProfit'].append(sos2_row['SimProfit'])
                comparison_data['Linear_ExpectedProfit'].append(linear_row['ExpectedProfit'])
                comparison_data['Linear_SimProfit'].append(linear_row['SimProfit'])
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Save comparison to CSV
            df_comparison.to_csv('./Plots_2024/model_comparison_2024.csv', index=False)
            
            # Plot comparison
            plt.figure(figsize=(12, 6))
            
            x = range(len(df_comparison))
            width = 0.2
            
            plt.bar([i-width*1.5 for i in x], df_comparison['Linear_ExpectedProfit'], width, 
                   label='Linear Expected Profit', color='blue', alpha=0.7)
            plt.bar([i-width*0.5 for i in x], df_comparison['Linear_SimProfit'], width, 
                   label='Linear Simulated Profit', color='lightblue', alpha=0.7)
            plt.bar([i+width*0.5 for i in x], df_comparison['SOS2_ExpectedProfit'], width, 
                   label='SOS2 Expected Profit', color='red', alpha=0.7)
            plt.bar([i+width*1.5 for i in x], df_comparison['SOS2_SimProfit'], width, 
                   label='SOS2 Simulated Profit', color='lightcoral', alpha=0.7)
            
            plt.xlabel('Date')
            plt.ylabel('Profit ($)')
            plt.title('Profit Comparison: Linear vs. SOS2 Models (2024)')
            plt.xticks(x, df_comparison['Date'], rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig('./Plots_2024/profit_model_comparison_2024.png', dpi=300)
            plt.close()
            
            # Print average comparison
            print("\nModel Comparison (Averages):")
            print(f"Linear Expected Profit: ${df_comparison['Linear_ExpectedProfit'].mean():.2f}")
            print(f"Linear Simulated Profit: ${df_comparison['Linear_SimProfit'].mean():.2f}")
            print(f"SOS2 Expected Profit: ${df_comparison['SOS2_ExpectedProfit'].mean():.2f}")
            print(f"SOS2 Simulated Profit: ${df_comparison['SOS2_SimProfit'].mean():.2f}")
            
            # Calculate improvement percentages
            expected_improvement = ((df_comparison['SOS2_ExpectedProfit'].mean() / 
                                    df_comparison['Linear_ExpectedProfit'].mean()) - 1) * 100
            simulated_improvement = ((df_comparison['SOS2_SimProfit'].mean() / 
                                     df_comparison['Linear_SimProfit'].mean()) - 1) * 100
            
            print(f"SOS2 Expected Profit Improvement: {expected_improvement:.2f}%")
            print(f"SOS2 Simulated Profit Improvement: {simulated_improvement:.2f}%")
    
    print("\nAll analyses complete and saved to the Plots_2024 directory!")

# %% Analysis of Piecewise Linearization Accuracy
import pandas as pd
import numpy as np
import torch

# Load the datasets
piecewise_data = pd.read_csv('./Data/piecewise_operation_data_SOS2_2024_10seg.csv')
global_data = pd.read_csv('./Data/database_no_piecewise_2024.csv')

print("Piecewise data columns:", piecewise_data.columns.tolist())
print("Global data columns:", global_data.columns.tolist())
print(f"Piecewise data shape: {piecewise_data.shape}")
print(f"Global data shape: {global_data.shape}")

# Clean the data: set Power and Flow values close to zero to exactly zero
def clean_near_zero_values(data, threshold=0.2):
    """Set Power and Flow values between -threshold and +threshold to 0"""
    data_clean = data.copy()
    
    # Count values that will be changed
    power_near_zero = ((data_clean['Power'] >= -threshold) & (data_clean['Power'] <= threshold)).sum()
    flow_near_zero = ((data_clean['Flow'] >= -threshold) & (data_clean['Flow'] <= threshold)).sum()
    
    print(f"Setting {power_near_zero} Power values and {flow_near_zero} Flow values near zero to exactly 0")
    
    # Set near-zero values to exactly zero
    data_clean.loc[(data_clean['Power'] >= -threshold) & (data_clean['Power'] <= threshold), 'Power'] = 0
    data_clean.loc[(data_clean['Flow'] >= -threshold) & (data_clean['Flow'] <= threshold), 'Flow'] = 0
    
    return data_clean

print("\nCleaning piecewise data...")
piecewise_data = clean_near_zero_values(piecewise_data)

print("\nCleaning global data...")
global_data = clean_near_zero_values(global_data)

# Calculate MSE for UPC relationship (Flow prediction)
def calculate_upc_mse(data, method_name):
    """Calculate MSE for UPC relationship q = predict_q_poly(p, h)"""
    mse_values = []
    for _, row in data.iterrows():
        p = row['Power']
        h = row['Head']
        q_actual = row['Flow']
        
        # Predict flow using the nonlinear UPC function
        q_predicted = predict_q_poly(p, h).item()
        
        mse_values.append((q_predicted - q_actual) ** 2)
    
    mse = np.mean(mse_values)
    print(f"{method_name} UPC MSE: {mse:.8f}")
    return mse

# Calculate MSE for Volume-Head relationship
def calculate_vh_mse(data, method_name):
    """Calculate MSE for volume-head relationship v_low = h_to_v_low_fitted(h)"""
    if 'Volume' not in data.columns:
        print(f"{method_name}: Volume column not found, calculating from head and flow dynamics...")
        return calculate_vh_mse_from_dynamics(data, method_name)
    
    mse_values = []
    for _, row in data.iterrows():
        h = row['Head']
        v_actual = row['Volume']
        
        # Predict volume using the nonlinear volume-head function
        v_predicted = h_to_v_low_fitted(torch.tensor(h)).item()
        
        mse_values.append((v_predicted - v_actual) ** 2)
    
    mse = np.mean(mse_values)
    print(f"{method_name} Volume-Head MSE: {mse:.8f}")
    return mse

def calculate_vh_mse_from_dynamics(data, method_name):
    """Calculate volume from flow dynamics and then compute MSE"""
    # Group by date to calculate volume dynamics for each day
    mse_values = []
    
    for date in data['Date'].unique():
        date_data = data[data['Date'] == date].sort_values('Time').reset_index(drop=True)
        
        # Calculate actual volumes from flow dynamics
        v_actual = [v_low_init]  # Start with initial volume
        for i, row in date_data.iterrows():
            if i > 0:
                v_prev = v_actual[-1]
                q = row['Flow']
                v_curr = v_prev + 3600 * q  # Volume dynamics
                v_actual.append(v_curr)
        
        # Calculate MSE for this date
        for i, row in date_data.iterrows():
            h = row['Head']
            v_pred = h_to_v_low_fitted(torch.tensor(h)).item()
            
            if i < len(v_actual):
                mse_values.append((v_pred - v_actual[i]) ** 2)
    
    mse = np.mean(mse_values)
    print(f"{method_name} Volume-Head MSE (calculated): {mse:.8f}")
    return mse

# Calculate MSE for both methods
print("\nCalculating MSE values...")
piecewise_upc_mse = calculate_upc_mse(piecewise_data, "Piecewise Linearization")
piecewise_vh_mse = calculate_vh_mse(piecewise_data, "Piecewise Linearization")

global_upc_mse = calculate_upc_mse(global_data, "Global Linearization")
global_vh_mse = calculate_vh_mse(global_data, "Global Linearization")

# Create LaTeX table
latex_table = f"""\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|c|}}
\\hline
Method & UPC MSE & Volume-Head MSE \\\\
\\hline
Piecewise Linearization & {piecewise_upc_mse:.2e} & {piecewise_vh_mse:.2e} \\\\
Global Linearization & {global_upc_mse:.2e} & {global_vh_mse:.2e} \\\\
\\hline
\\end{{tabular}}
\\caption{{Mean Squared Error Comparison for Nonlinear Function Approximations}}
\\label{{tab:mse_comparison}}
\\end{{table}}"""

print("\n" + "="*60)
print("LaTeX Table:")
print("="*60)
print(latex_table)

# Print detailed summary statistics
print("\n" + "="*60)
print("DETAILED SUMMARY:")
print("="*60)
print(f"Piecewise Linearization:")
print(f"  - UPC MSE: {piecewise_upc_mse:.8f}")
print(f"  - Volume-Head MSE: {piecewise_vh_mse:.8f}")
print(f"\nGlobal Linearization:")
print(f"  - UPC MSE: {global_upc_mse:.8f}")
print(f"  - Volume-Head MSE: {global_vh_mse:.8f}")

# Calculate improvement ratios
upc_improvement = ((global_upc_mse - piecewise_upc_mse) / global_upc_mse) * 100
vh_improvement = ((global_vh_mse - piecewise_vh_mse) / global_vh_mse) * 100

print(f"\nImprovement of Piecewise over Global:")
print(f"  - UPC MSE improvement: {upc_improvement:.2f}%")
print(f"  - Volume-Head MSE improvement: {vh_improvement:.2f}%")

if upc_improvement > 0:
    print(f"  → Piecewise linearization is {upc_improvement:.2f}% more accurate for UPC")
else:
    print(f"  → Global linearization is {-upc_improvement:.2f}% more accurate for UPC")

if vh_improvement > 0:
    print(f"  → Piecewise linearization is {vh_improvement:.2f}% more accurate for Volume-Head")
else:
    print(f"  → Global linearization is {-vh_improvement:.2f}% more accurate for Volume-Head")