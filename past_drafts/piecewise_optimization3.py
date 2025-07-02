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

# %% Load day-ahead prices
def load_prices():
    """Load day-ahead prices from database and set as environment variables."""
    # Read price database
    df = pd.read_csv("./Data/price_database_2.csv")
    
    # Process each day's prices
    price_data = {}
    for _, row in df.iterrows():
        date = row['date']
        prices = eval(row['prices_hourly'])  # Convert string to list
        price_data[date] = prices

    return price_data

# %% Piecewise MILP Optimizer
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
        self.model.Params.TimeLimit = 2400  # 40 minute time limit

        
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

# Main execution and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Load day-ahead prices
    prices = load_prices()
    # Get earliest date by sorting the keys
    sample_date = sorted(prices.keys())[0]
    DA_prices = prices[sample_date] 
    print(f"Day-ahead prices for {sample_date}: {DA_prices}")
    
    # Set optimization parameters
    T = 24  # 24-hour horizon
    
    # Create optimizer with configurable number of segments
    optimizer = PiecewiseMILPOptimizer(
        T=T, 
        DA_prices=DA_prices,
        num_segments_h=5,                # Number of head segments
        num_segments_p_pump=5,           # Number of power segments for pump mode
        num_segments_p_turbine=5,        # Number of power segments for turbine mode
        C_op=3.8,                        # Operational cost coefficient
        M_p=10000,                       # Big-M constant
        h_init=head_init,                # Initial head
        v_low_init=v_low_init,           # Initial lower reservoir volume
        v_low_target=target_vol_low      # Target lower reservoir volume
    )
    
    # Solve the optimization problem
    results = optimizer.solve()
    
    if results is not None:
        # Calculate profit
        profit = sum(results['p'][t] * DA_prices[t] - optimizer.C_op * results['p'][t]**2 for t in range(T))
        print(f"Total profit: {profit:.2f}")
        
        # Visualize results
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Plot 1: Power and prices
        ax1 = axs[0]
        ax1.bar(range(T), results['p'], color=['red' if p < 0 else 'green' for p in results['p']], alpha=0.6)
        ax1.set_ylabel('Power (MW)')
        ax1.set_title('Optimal Power Schedule')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(range(T), DA_prices, 'b-', marker='o')
        ax1_twin.set_ylabel('Price (€/MWh)', color='b')
        ax1_twin.tick_params(axis='y', labelcolor='b')
        
        # Plot 2: Flow rate
        ax2 = axs[1]
        ax2.bar(range(T), results['q'], color=['red' if q < 0 else 'green' for q in results['q']], alpha=0.6)
        ax2.set_ylabel('Flow (m³/s)')
        ax2.set_title('Flow Rate')
        
        # Plot 3: Head
        ax3 = axs[2]
        ax3.plot(range(T), results['h'], 'k-', marker='o')
        ax3.set_ylabel('Head (m)')
        ax3.set_title('Head')
        ax3.set_ylim([optimizer.h_min - 1, optimizer.h_max + 1])
        
        # Plot 4: Lower reservoir volume
        ax4 = axs[3]
        ax4.plot(range(T), results['v_low'], 'k-', marker='o')
        ax4.axhline(y=optimizer.v_low_target, color='r', linestyle='--', label='Target Volume')
        ax4.set_ylabel('Volume (m³)')
        ax4.set_xlabel('Time (h)')
        ax4.set_title('Lower Reservoir Volume')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print mode selection
        print("\nMode Selection:")
        for t in range(T):
            if results['z_I'][t] > 0.5:
                mode = "Idle"
            elif results['z_T'][t] > 0.5:
                mode = "Turbine"
            elif results['z_P'][t] > 0.5:
                mode = "Pump"
            else:
                mode = "Unknown"
            print(f"t={t}: {mode}")
    
    # # %% print power, flow, head, volume, and price
    # print("Power: ", results['p'])
    # print("Flow: ", results['q'])
    # print("Head: ", results['h'])
    # print("Volume: ", results['v_low'])
    # print("Day-ahead Prices: ", DA_prices)

# %% Batch Solver for All Dates
def solve_and_save_all_dates():
    """
    Solve the optimization problem for all dates in the price data
    and save the results to a CSV file.
    """
    import pandas as pd
    import time
    
    # Load day-ahead prices for all dates
    all_prices = load_prices()
    dates = list(all_prices.keys())
    print(f"Found {len(dates)} dates to process")
    
    # Initialize results dataframe
    columns = ['Time', 'Power', 'Head', 'Flow', 'Price', 'Date']
    results_df = pd.DataFrame(columns=columns)
    
    # Process each date
    for date_idx, date in enumerate(dates):
        start_time = time.time()
        print(f"Processing date {date} ({date_idx+1}/{len(dates)})")
        
        # Get prices for this date
        DA_prices = all_prices[date]
        T = len(DA_prices)  # Number of time periods
        
        # Create and solve optimizer
        optimizer = PiecewiseMILPOptimizer(
            T=T, 
            DA_prices=DA_prices,
            num_segments_h=5,
            num_segments_p_pump=5,
            num_segments_p_turbine=5,
            C_op=3.8,
            M_p=10000,
            h_init=head_init,
            v_low_init=v_low_init,
            v_low_target=target_vol_low
        )
        
        results = optimizer.solve()
        
        if results is not None:
            # Calculate profit
            profit = sum(results['p'][t] * DA_prices[t] - optimizer.C_op * results['p'][t]**2 for t in range(T))
            print(f"  - Total profit: {profit:.2f}")
            
            # Collect results for this date
            for t in range(T):
                row = {
                    'Time': t,
                    'Power': results['p'][t],
                    'Head': results['h'][t],
                    'Flow': results['q'][t],
                    'Price': DA_prices[t],
                    'Date': date
                }
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        else:
            print(f"  - Optimization failed for {date}")
        
        elapsed = time.time() - start_time
        print(f"  - Completed in {elapsed:.2f} seconds")
    
    # Save results to CSV
    output_file = "./Data/database_piecewise.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results_df

# Add this to the main block
if __name__ == "__main__":
    # Comment out the visualization code
    # Replace with batch processing
    print("Starting batch processing for all dates...")
    all_results = solve_and_save_all_dates()
    print("Batch processing complete!")
    
    # Optional: Some summary statistics from the results
    print("\nSummary Statistics:")
    dates = all_results['Date'].unique()
    print(f"Processed {len(dates)} dates")
    
    # Group by date and calculate average power and profit per date
    summary = all_results.groupby('Date').agg({
        'Power': ['mean', 'min', 'max'],
        'Head': ['mean', 'min', 'max'],
        'Flow': ['mean']
    })
    
    print(summary)
    
# %% Visualization
# Plot the result of Power with the day-ahead prices
import matplotlib.pyplot as plt
import seaborn as sns

# Extract the results
p = results['p']
DA_prices = prices[sample_date]

# Plot the power and day-ahead prices on different Y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Time Period')
ax1.set_ylabel('Power (MW)', color=color)
ax1.step(range(len(p)), p, label="Power (MW)", color=color, where='mid')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Day-ahead Prices ($/MWh)', color=color)
ax2.step(range(len(DA_prices)), DA_prices, label="Day-ahead Prices ($/MWh)", color=color, where='mid')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("Power and Day-ahead Prices of date: " + sample_date)
plt.show()
# %% Visualization of Linearized Subdomains
import numpy as np
import plotly.graph_objects as go
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# -----------------------------------------------------------------------------
# 1. Volume-Head Relationship Linearization (unchanged)
# -----------------------------------------------------------------------------
h_dense = np.linspace(optimizer.h_min, optimizer.h_max, 200)
v_dense = [h_to_v_low_fitted(torch.tensor(h)).item() for h in h_dense]
plt.figure(figsize=(8, 6))
plt.plot(h_dense, v_dense, label="Original Volume-Head Curve", color='blue')
plt.plot(optimizer.h_samples, optimizer.v_low_samples, 'o-', label="Linearization Points", color='red')
plt.xlabel("Head (m)")
plt.ylabel("Volume")
plt.title("Volume-Head Relationship Linearization")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# 2. Pump Mode: Interactive 3D Plot with Plotly
# -----------------------------------------------------------------------------
# Gather pump grid values to set axis limits.
pump_p_vals = []
pump_q_vals = []
for i in range(optimizer.num_segments_h + 1):
    pump_p_vals.extend(optimizer.pump_grid[i]['p'])
    pump_q_vals.extend(optimizer.pump_grid[i]['q'])
p_min = min(pump_p_vals)
p_max = max(pump_p_vals)
q_min = min(pump_q_vals)
q_max = max(pump_q_vals)
h_min = optimizer.h_min
h_max = optimizer.h_max

# Build the original nonlinear surface for pump mode.
h_vals = np.linspace(h_min, h_max, 50)
p_min_vals = [neg_min(h).item() for h in h_vals]
p_max_vals = [neg_max(h).item() for h in h_vals]
p_common_min = min(p_min_vals)
p_common_max = max(p_max_vals)
p_grid = np.linspace(p_common_min, p_common_max, 50)
H_mesh, P_mesh = np.meshgrid(h_vals, p_grid)
q_func = np.vectorize(lambda p, h: predict_q_poly(p, h).item())
Q_mesh = q_func(P_mesh, H_mesh)

# Create the Plotly figure for pump mode.
fig_pump = go.Figure()

# Add the original nonlinear surface.
fig_pump.add_trace(go.Surface(x=P_mesh, y=H_mesh, z=Q_mesh,
                              colorscale='Viridis', opacity=0.7,
                              name='Original Nonlinear Surface'))

# Loop over each quadrilateral in the pump grid.
for i in range(optimizer.num_segments_h):
    h_lower = optimizer.h_samples[i]
    h_upper = optimizer.h_samples[i+1]
    for j in range(optimizer.num_segments_p_pump):
        # Extract corner values from the pump grid.
        p_bl = optimizer.pump_grid[i]['p'][j]
        q_bl = optimizer.pump_grid[i]['q'][j]
        p_br = optimizer.pump_grid[i]['p'][j+1]
        q_br = optimizer.pump_grid[i]['q'][j+1]
        p_tl = optimizer.pump_grid[i+1]['p'][j]
        q_tl = optimizer.pump_grid[i+1]['q'][j]
        p_tr = optimizer.pump_grid[i+1]['p'][j+1]
        q_tr = optimizer.pump_grid[i+1]['q'][j+1]
        
        # Create 2x2 arrays for the surface patch.
        X = np.array([[p_bl, p_br],
                      [p_tl, p_tr]])
        Y = np.array([[h_lower, h_lower],
                      [h_upper, h_upper]])
        Z = np.array([[q_bl, q_br],
                      [q_tl, q_tr]])
        
        # Add the piecewise plane as a surface.
        fig_pump.add_trace(go.Surface(x=X, y=Y, z=Z,
                                      colorscale='Reds', opacity=0.7,
                                      showscale=False, name='Piecewise Plane'))
        
        # Add an outline with corner markers.
        outline_x = [p_bl, p_br, p_tr, p_tl, p_bl]
        outline_y = [h_lower, h_lower, h_upper, h_upper, h_lower]
        outline_z = [q_bl, q_br, q_tr, q_tl, q_bl]
        fig_pump.add_trace(go.Scatter3d(x=outline_x, y=outline_y, z=outline_z,
                                        mode='lines+markers',
                                        marker=dict(size=4, color='black'),
                                        line=dict(color='black', width=2),
                                        name='Piecewise Outline'))

fig_pump.update_layout(title='Pump Mode: Piecewise Linearization 3D',
                      scene=dict(
                          xaxis_title='Power (MW)',
                          yaxis_title='Head (m)',
                          zaxis_title='Flow (q)',
                          xaxis=dict(range=[p_min, p_max]),
                          yaxis=dict(range=[h_min, h_max]),
                          zaxis=dict(range=[q_min, q_max])
                      ),
                      autosize=True)
fig_pump.show()

# -----------------------------------------------------------------------------
# 3. Turbine Mode: Interactive 3D Plot with Plotly
# -----------------------------------------------------------------------------
# Gather turbine grid values to set axis limits.
turb_p_vals = []
turb_q_vals = []
for i in range(optimizer.num_segments_h + 1):
    turb_p_vals.extend(optimizer.turbine_grid[i]['p'])
    turb_q_vals.extend(optimizer.turbine_grid[i]['q'])
p_min_turb = min(turb_p_vals)
p_max_turb = max(turb_p_vals)
q_min_turb = min(turb_q_vals)
q_max_turb = max(turb_q_vals)

# Build the original nonlinear surface for turbine mode.
h_vals = np.linspace(h_min, h_max, 50)
p_min_vals_turb = [pos_min(h).item() for h in h_vals]
p_max_vals_turb = [pos_max(h).item() for h in h_vals]
p_common_min_turb = min(p_min_vals_turb)
p_common_max_turb = max(p_max_vals_turb)
p_grid = np.linspace(p_common_min_turb, p_common_max_turb, 50)
H_mesh, P_mesh = np.meshgrid(h_vals, p_grid)
q_func = np.vectorize(lambda p, h: predict_q_poly(p, h).item())
Q_mesh = q_func(P_mesh, H_mesh)

# Create the Plotly figure for turbine mode.
fig_turbine = go.Figure()
fig_turbine.add_trace(go.Surface(x=P_mesh, y=H_mesh, z=Q_mesh,
                                 colorscale='Viridis', opacity=0.7,
                                 name='Original Nonlinear Surface'))

# Loop over each quadrilateral in the turbine grid.
for i in range(optimizer.num_segments_h):
    h_lower = optimizer.h_samples[i]
    h_upper = optimizer.h_samples[i+1]
    for j in range(optimizer.num_segments_p_turbine):
        p_bl = optimizer.turbine_grid[i]['p'][j]
        q_bl = optimizer.turbine_grid[i]['q'][j]
        p_br = optimizer.turbine_grid[i]['p'][j+1]
        q_br = optimizer.turbine_grid[i]['q'][j+1]
        p_tl = optimizer.turbine_grid[i+1]['p'][j]
        q_tl = optimizer.turbine_grid[i+1]['q'][j]
        p_tr = optimizer.turbine_grid[i+1]['p'][j+1]
        q_tr = optimizer.turbine_grid[i+1]['q'][j+1]
        
        X = np.array([[p_bl, p_br],
                      [p_tl, p_tr]])
        Y = np.array([[h_lower, h_lower],
                      [h_upper, h_upper]])
        Z = np.array([[q_bl, q_br],
                      [q_tl, q_tr]])
        
        fig_turbine.add_trace(go.Surface(x=X, y=Y, z=Z,
                                        colorscale='Greens', opacity=0.7,
                                        showscale=False, name='Piecewise Plane'))
        
        outline_x = [p_bl, p_br, p_tr, p_tl, p_bl]
        outline_y = [h_lower, h_lower, h_upper, h_upper, h_lower]
        outline_z = [q_bl, q_br, q_tr, q_tl, q_bl]
        fig_turbine.add_trace(go.Scatter3d(x=outline_x, y=outline_y, z=outline_z,
                                           mode='lines+markers',
                                           marker=dict(size=4, color='black'),
                                           line=dict(color='black', width=2),
                                           name='Piecewise Outline'))
        
fig_turbine.update_layout(title='Turbine Mode: Piecewise Linearization 3D',
                          scene=dict(
                              xaxis_title='Power (MW)',
                              yaxis_title='Head (m)',
                              zaxis_title='Flow (q)',
                              xaxis=dict(range=[p_min_turb, p_max_turb]),
                              yaxis=dict(range=[h_min, h_max]),
                              zaxis=dict(range=[q_min_turb, q_max_turb])
                          ),
                          autosize=True)
fig_turbine.show()

# -----------------------------------------------------------------------------
# 4. 2D Plots of the Piecewise Grids (p–h)
# -----------------------------------------------------------------------------

# 4.1 Pump Mode 2D Plot (p vs. head)
plt.figure(figsize=(8, 6))
plt.title("Pump Mode: 2D Piecewise Grid (p-h)")
ax = plt.gca()

# Loop over each quadrilateral in pump mode.
for i in range(optimizer.num_segments_h):
    h_lower = optimizer.h_samples[i]
    h_upper = optimizer.h_samples[i+1]
    for j in range(optimizer.num_segments_p_pump):
        p_bl = optimizer.pump_grid[i]['p'][j]
        p_br = optimizer.pump_grid[i]['p'][j+1]
        p_tl = optimizer.pump_grid[i+1]['p'][j]
        p_tr = optimizer.pump_grid[i+1]['p'][j+1]
        
        # Define the corner coordinates.
        corners = [(p_bl, h_lower), (p_br, h_lower), (p_tr, h_upper), (p_tl, h_upper)]
        poly = Polygon(corners, closed=True, fill=None, edgecolor='red', linewidth=2)
        ax.add_patch(poly)

plt.xlabel("Power (MW)")
plt.ylabel("Head (m)")
plt.grid(True)
plt.xlim(min(pump_p_vals) - 0.05 * (max(pump_p_vals)-min(pump_p_vals)), 
         max(pump_p_vals) + 0.05 * (max(pump_p_vals)-min(pump_p_vals)))
plt.ylim(h_min - 1, h_max + 1)
plt.show()

# 4.2 Turbine Mode 2D Plot (p vs. head)
plt.figure(figsize=(8, 6))
plt.title("Turbine Mode: 2D Piecewise Grid (p-h)")
ax = plt.gca()

# Loop over each quadrilateral in turbine mode.
for i in range(optimizer.num_segments_h):
    h_lower = optimizer.h_samples[i]
    h_upper = optimizer.h_samples[i+1]
    for j in range(optimizer.num_segments_p_turbine):
        p_bl = optimizer.turbine_grid[i]['p'][j]
        p_br = optimizer.turbine_grid[i]['p'][j+1]
        p_tl = optimizer.turbine_grid[i+1]['p'][j]
        p_tr = optimizer.turbine_grid[i+1]['p'][j+1]
        
        corners = [(p_bl, h_lower), (p_br, h_lower), (p_tr, h_upper), (p_tl, h_upper)]
        poly = Polygon(corners, closed=True, fill=None, edgecolor='green', linewidth=2)
        ax.add_patch(poly)

plt.xlabel("Power (MW)")
plt.ylabel("Head (m)")
plt.grid(True)
plt.xlim(min(turb_p_vals) - 0.05 * (max(turb_p_vals)-min(turb_p_vals)), 
         max(turb_p_vals) + 0.05 * (max(turb_p_vals)-min(turb_p_vals)))
plt.ylim(h_min - 1, h_max + 1)
plt.show()

# %%
