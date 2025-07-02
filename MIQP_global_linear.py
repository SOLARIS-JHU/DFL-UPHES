# %% Import libraries
import torch
import dill as pickle
import pandas as pd
import sys
# torch.autograd.set_detect_anomaly(True)
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import dill as pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cpu")

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = 77.0 # Initial head value
v_low_init = h_to_v_low_fitted(head_init) # Initial lower reservoir volume
# convert v_low_init to float
v_low_init = float(v_low_init)
# convert h_vlow_coeff_lin to python array
h_vlow_coeff_lin = h_vlow_coeff_lin.detach().numpy()

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

if __name__ == "__main__":
    prices = load_prices()
    print(f"Loaded prices for {len(prices)} days")
    
    # Example: retrieve prices
    sample_date = list(prices.keys())[0]
    print(f"\nPrices for {sample_date}: {prices[sample_date]}")


# %% MILP Optimizer


class MILPOptimizer:
    def __init__(self, T, DA_prices, C_op=3.8, M_p=10000, h_init=head_init, h_min=head_min, h_max=head_max, v_low_init=v_low_init, v_low_target=target_vol_low, h_target=target_head):
        """
        MILP optimizer for initial pipeline points.
        
        Parameters:
            T (int): Number of time periods.
            DA_prices (list): Day-ahead prices for each period.
            C_op (float): Operational cost coefficient.
            M_p (float): Big-M constant.
            h_min (float): Minimum head.
            h_max (float): Maximum head.
            v_low_init (float): Initial lower reservoir volume.
            v_low_target (float): Target lower reservoir volume.
            h_target (float): Target head value.
        """
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

        # Decision Variables
        # Continuous power variables (split into turbine and pump components)
        self.p_T = self.model.addVars(T, lb=0, name="p_T")  # Turbine power (>=0)
        self.p_P = self.model.addVars(T, lb=-GRB.INFINITY, ub=0, name="p_P")    # Pump power (<=0)
        # Flow, head and lower reservoir volume variables
        self.q   = self.model.addVars(T, lb=-GRB.INFINITY, name="q")
        self.h   = self.model.addVars(T, lb=self.h_min, ub=self.h_max, name="h")
        self.v_low = self.model.addVars(T, name="v_low")
        
        # Binary variables for mode selection:
        # z_t^I: Idle, z_t^T: Turbine, z_t^P: Pump.
        self.z_I = self.model.addVars(T, vtype=GRB.BINARY, name="z_I")
        self.z_T = self.model.addVars(T, vtype=GRB.BINARY, name="z_T")
        self.z_P = self.model.addVars(T, vtype=GRB.BINARY, name="z_P")
        
        # Mode selection: exactly one mode is active at each time t.
        for t in range(T):
            self.model.addConstr(self.z_I[t] + self.z_T[t] + self.z_P[t] == 1, name=f"mode_sel_{t}")
        
        # Idle Mode Constraints: if idle (z_I=1) then p_t^T, p_t^P, and q_t are forced to zero.
        for t in range(T):
            # self.model.addConstr(self.p_T[t] <= M_p * (1 - self.z_I[t]), name=f"idle_pT_{t}")
            # self.model.addConstr(self.p_P[t] >= M_p * (1 - self.z_I[t]), name=f"idle_pP_{t}")
            self.model.addConstr(self.q[t] <=  M_p * (1 - self.z_I[t]), name=f"idle_q_{t}")
        
        # Turbine Mode Constraints:
        for t in range(T):
            self.model.addConstr(self.p_T[t] >= pos_min_fit @ [self.h[t], 1.0] * self.z_T[t],
                     name=f"turbine_min_{t}")
            self.model.addConstr(self.p_T[t] <= pos_max_fit @ [self.h[t], 1.0] * self.z_T[t],
                     name=f"turbine_max_{t}")
        
        # Pump Mode Constraints:
        for t in range(T):
            self.model.addConstr(self.p_P[t] >= neg_min_fit @ [self.h[t], 1.0] * self.z_P[t],
                                 name=f"pump_min_{t}")
            self.model.addConstr(self.p_P[t] <= neg_max_fit @ [self.h[t], 1.0] * self.z_P[t],
                                 name=f"pump_max_{t}")
        
        # Flow Relation Constraints 
        for t in range(T):
            # Turbine flow prediction
            q_tur = coefs_tur_lin @ [self.p_T[t], self.h[t]] + intercept_tur_lin
            # Pump flow prediction
            q_pump = coefs_pump_lin @ [self.p_P[t], self.h[t]] + intercept_pump_lin
            # Since only one of z_T or z_P is active (unless idle, in which case q_t=0),
            # we model the flow as the sum of the contributions:
            self.model.addConstr(
            self.q[t] == q_tur * self.z_T[t] + q_pump * self.z_P[t],
            name=f"flow_{t}"
            )
        
        # Volume-Head Relationship and Dynamics
        for t in range(T):
            # Link lower reservoir volume to head using linear fit
            self.model.addConstr(self.v_low[t] == h_vlow_coeff_lin @ [self.h[t],1], name=f"vol_head_{t}")
            # Dynamics: v_low[t] = v_low[t-1] + 3600 * q_t[t]
            if t == 0:
                self.model.addConstr(self.v_low[t] == self.v_low_init + 3600 * self.q[t],
                            name=f"vol_dyn_{t}")
            else:
                self.model.addConstr(self.v_low[t] == self.v_low[t-1] + 3600 * self.q[t],
                            name=f"vol_dyn_{t}")
        
        # # Initial Head Constraint:
        # self.model.addConstr(self.h[0] == self.h_init, name="init_head")
        
        # Final Volume Constraint:
        self.model.addConstr(self.v_low[T-1] <= self.v_low_target, name="vol_target")
        
        # Final Head Constraint:
        self.model.addConstr(self.h[T-1] >= self.h_target, name="head_target")
        
        # Set the Objective:
        # Maximize: sum_t [(p_t^T + p_t^P)*lambda_DA_t - C_op*(p_t^T + p_t^P)^2]
        objective = gp.quicksum(
            (self.p_T[t] + self.p_P[t]) * self.DA_prices[t] -
            self.C_op * (self.p_T[t] + self.p_P[t]) * (self.p_T[t] + self.p_P[t])
            for t in range(T)
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)
        
        # Optional: set output parameters
        self.model.Params.OutputFlag = 1

    def solve(self):
        """Optimize the MILP and return the decision variable values and metrics."""
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
            metrics['ExpectedProfit'] = self.model.objVal  # Assuming profit equals objective value
            
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
            
            return results, metrics
        else:
            print("No optimal solution found!")
            return None, metrics

# %% Main execution - Run optimizer for all dates
# Main execution - Run optimizer for all dates
if __name__ == "__main__":
    T = 24  # Number of time periods in a day
    all_results = {}
    all_metrics = {}
    
    for date, DA_prices in prices.items():
        print(f"Running optimization for {date}...")
        optimizer = MILPOptimizer(T, DA_prices)
        results, metrics = optimizer.solve()
        
        if results is not None:
            all_results[date] = results
            all_metrics[date] = metrics
        else:
            print(f"No optimal solution found for {date}!")
            all_metrics[date] = metrics  # Still store metrics even if no solution
    
    # Create benchmark directory if it doesn't exist
    import os
    os.makedirs("Benchmark", exist_ok=True)
    
    # Create the benchmark CSV file
    benchmark_df = pd.DataFrame({
        'Date': list(all_metrics.keys()),
        'SolveTime': [metrics['SolveTime'] for metrics in all_metrics.values()],
        'ExpectedProfit': [metrics['ExpectedProfit'] for metrics in all_metrics.values()],
        'MIPGap': [metrics['MIPGap'] for metrics in all_metrics.values()],
        'Status': [metrics['Status'] for metrics in all_metrics.values()],
        'NumVars': [metrics['NumVars'] for metrics in all_metrics.values()],
        'NumConstrs': [metrics['NumConstrs'] for metrics in all_metrics.values()],
        'NumBinVars': [metrics['NumBinVars'] for metrics in all_metrics.values()],
        'ObjectiveBound': [metrics['ObjectiveBound'] for metrics in all_metrics.values()],
        'ObjectiveValue': [metrics['ObjectiveValue'] for metrics in all_metrics.values()]
    })
    
    # Save benchmark metrics to CSV
    benchmark_df.to_csv("Benchmark/global_linearized_operational_data.csv", index=False)
    
    print("Benchmark metrics saved to Benchmark/global_linearized_operational_data.csv")

# # %% Test MILP optimizer for a single date
# if __name__ == "__main__":
#     T = 24
#     DA_prices = prices[sample_date]  

#     optimizer = MILPOptimizer(T, DA_prices)
#     results = optimizer.solve()
    
#     # # check for infeasibility
#     # optimizer.model.computeIIS()
#     # optimizer.model.write("model.ilp")
#     # print("IIS Constraints:")
#     # for c in optimizer.model.getConstrs():
#     #     if c.IISConstr:
#     #         print(f"{c.ConstrName}: {c}")

#     # print("\nIIS Bounds:")
#     # for v in optimizer.model.getVars():
#     #     if v.IISLB:
#     #         print(f"Lower Bound Infeasible: {v.VarName} >= {v.LB}")
#     #     if v.IISUB:
#     #         print(f"Upper Bound Infeasible: {v.VarName} <= {v.UB}")


#     if results is not None:
#         print("Optimization results:")
#         for key, value in results.items():
#             print(f"{key}: {value}")

# %% Post processing with idle mode correction
if __name__ == "__main__":
    # Process all results to ensure idle mode values are exactly 0
    for date, results in all_results.items():
        # Get the idle mode indicators
        z_I_values = results['z_I']
        
        # Create corrected results
        corrected_p_t_T = results['p_t_T'].copy()
        corrected_p_t_P = results['p_t_P'].copy()
        corrected_q_t = results['q_t'].copy()
        
        # Force values to exactly 0 when in idle mode
        for t in range(len(z_I_values)):
            if z_I_values[t] > 0.5:  # If idle mode is active (binary variable close to 1)
                corrected_p_t_T[t] = 0.0
                corrected_p_t_P[t] = 0.0
                corrected_q_t[t] = 0.0
        
        # Update the results dictionary
        results['p_t_T'] = corrected_p_t_T
        results['p_t_P'] = corrected_p_t_P
        results['q_t'] = corrected_q_t
    
    # Print the corrected results for verification
    date = list(all_results.keys())[0]
    results = all_results[date]
    
    print(f"\nCorrected Results for {date}:")
    print("-" * 80)
    print("Time | Power (MW) | Flow (m³/s) | Head (m) | Mode")
    print("-" * 80)
    
    for t in range(len(results['p_t_T'])):
        power = results['p_t_T'][t] + results['p_t_P'][t]
        flow = results['q_t'][t]
        head = results['h_t'][t]
        
        # Determine mode
        if results['z_I'][t] > 0.5:
            mode = "Idle"
        elif results['z_T'][t] > 0.5:
            mode = "Turbine"
        else:
            mode = "Pump"
            
        print(f"{t:4d} | {power:10.4f} | {flow:10.4f} | {head:8.4f} | {mode}")
    
    # Update the DataFrame creation code as well
    df_list = []
    for date, results in all_results.items():
        # Create power values with proper zeros for idle mode
        power_values = [p_t_T + p_t_P for p_t_T, p_t_P in zip(results['p_t_T'], results['p_t_P'])]
        
        df = pd.DataFrame({
            'Time': range(T),
            'Power': power_values,
            'Head': results['h_t'],
            'Flow': results['q_t'],
            'Price': prices[date],
            'Date': date,
            'Mode': ["Idle" if z_I > 0.5 else ("Turbine" if z_T > 0.5 else "Pump") 
                    for z_I, z_T in zip(results['z_I'], results['z_T'])]
        })
        df_list.append(df)
    
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv("./Data/database_no_piecewise.csv", index=False)
    print("\nResults saved to ./Data/database_no_piecewise.csv")

# %% Modified plotting code to show modes clearer
if __name__ == "__main__":
    date = list(all_results.keys())[0]
    results = all_results[date]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create corrected power array
    power = [p_t_T + p_t_P for p_t_T, p_t_P in zip(results['p_t_T'], results['p_t_P'])]
    
    # Create an array to highlight different modes
    turbine_periods = [t for t in range(T) if results['z_T'][t] > 0.5]
    pump_periods = [t for t in range(T) if results['z_P'][t] > 0.5]
    idle_periods = [t for t in range(T) if results['z_I'][t] > 0.5]
    
    # Plot power
    ax1.plot(range(T), power, label="Power", color='b')
    
    # Highlight different modes with background colors
    for t in turbine_periods:
        ax1.axvspan(t-0.5, t+0.5, alpha=0.2, color='green')
    for t in pump_periods:
        ax1.axvspan(t-0.5, t+0.5, alpha=0.2, color='red')
    for t in idle_periods:
        ax1.axvspan(t-0.5, t+0.5, alpha=0.2, color='gray')
    
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Power (MW)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(range(T), prices[date], label="DA Price", color='r', linestyle='--')
    ax2.set_ylabel("DA Price ($/MWh)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add a legend for modes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Turbine Mode'),
        Patch(facecolor='red', alpha=0.2, label='Pump Mode'),
        Patch(facecolor='gray', alpha=0.2, label='Idle Mode')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Add a legend for lines
    lines_legend = [
        plt.Line2D([0], [0], color='b', label='Power'),
        plt.Line2D([0], [0], color='r', linestyle='--', label='DA Price')
    ]
    ax2.legend(handles=lines_legend, loc='upper right')

    fig.suptitle(f"Power and DA Price-Time Plot with Operation Modes ({date})")
    plt.tight_layout()
    plt.show()

# %% 
# Plot results
if __name__ == "__main__":
    plt.figure(figsize=(10, 6))
    for date, results in all_results.items():
        power = [p_t_T + p_t_P for p_t_T, p_t_P in zip(results['p_t_T'], results['p_t_P'])]
        plt.plot(range(T), power, label=f"Power - {date}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Power (MW)")
    plt.title("Power-Time Plot")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for date, results in all_results.items():
        plt.plot(range(T), results['h_t'], label=f"Head - {date}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Head (m)")
    plt.title("Head-Time Plot")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for date, results in all_results.items():
        plt.plot(range(T), results['q_t'], label=f"Flow - {date}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Flow (m³/s)")
    plt.title("Flow-Time Plot")
    plt.legend()
    plt.show()

# %%
# plot the first day's results

if __name__ == "__main__":
    date = list(all_results.keys())[0]
    results = all_results[date]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    power = [p_t_T + p_t_P for p_t_T, p_t_P in zip(results['p_t_T'], results['p_t_P'])]
    ax1.plot(range(T), power, label="Power", color='b')
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Power (MW)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(range(T), prices[date], label="DA Price", color='r')
    ax2.set_ylabel("DA Price ($/MWh)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.suptitle(f"Power and DA Price-Time Plot ({date})")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(T), results['h_t'], label="Head")
    plt.xlabel("Time (hours)")
    plt.ylabel("Head (m)")
    plt.title(f"Head-Time Plot ({date})")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(T), results['q_t'], label="Flow")
    plt.xlabel("Time (hours)")
    plt.ylabel("Flow (m³/s)")
    plt.title(f"Flow-Time Plot ({date})")
    plt.legend()
    plt.show()

# %% Convert results to DataFrame and save as CSV
if __name__ == "__main__":
    df_list = []
    for date, results in all_results.items():
        df = pd.DataFrame({
            'Time': range(T),
            'Power': [p_t_T + p_t_P for p_t_T, p_t_P in zip(results['p_t_T'], results['p_t_P'])],
            'Head': results['h_t'],
            'Flow': results['q_t'],
            'Price': prices[date],
            'Date': date
        })
        df_list.append(df)
    
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv("./Data/database_no_piecewise.csv", index=False)
    print("Results saved to ./Data/database_no_piecewise.csv")

# %% Load and run 2024 data
def load_prices_2024():
    """Load day-ahead prices from 2024 database and set as environment variables."""
    # Read price database
    df = pd.read_csv("./Data/price_data_2024.csv")
    
    # Process each day's prices
    price_data = {}
    for _, row in df.iterrows():
        date = row['date']
        prices = [float(price) for price in row['prices_hourly'].split(",")]  # Convert comma-separated string to list
        price_data[date] = prices

    return price_data

if __name__ == "__main__":
    # Load 2024 prices
    prices_2024 = load_prices_2024()
    print(f"Loaded 2024 prices for {len(prices_2024)} days")
    
    # Example: retrieve prices
    sample_date_2024 = list(prices_2024.keys())[0]
    print(f"\nPrices for {sample_date_2024}: {prices_2024[sample_date_2024]}")

    # Run optimizer for all 2024 dates
    T = 24  # Number of time periods in a day
    all_results_2024 = {}
    all_metrics_2024 = {}
    
    for date, DA_prices in prices_2024.items():
        print(f"Running optimization for {date}...")
        optimizer = MILPOptimizer(T, DA_prices)
        results, metrics = optimizer.solve()
        
        if results is not None:
            all_results_2024[date] = results
            all_metrics_2024[date] = metrics
        else:
            print(f"No optimal solution found for {date}!")
            all_metrics_2024[date] = metrics  # Still store metrics even if no solution
    
    # Create benchmark directory if it doesn't exist
    import os
    os.makedirs("Benchmark", exist_ok=True)
    
    # Create the benchmark CSV file
    benchmark_df = pd.DataFrame({
        'Date': list(all_metrics_2024.keys()),
        'SolveTime': [metrics['SolveTime'] for metrics in all_metrics_2024.values()],
        'ExpectedProfit': [metrics['ExpectedProfit'] for metrics in all_metrics_2024.values()],
        'MIPGap': [metrics['MIPGap'] for metrics in all_metrics_2024.values()],
        'Status': [metrics['Status'] for metrics in all_metrics_2024.values()],
        'NumVars': [metrics['NumVars'] for metrics in all_metrics_2024.values()],
        'NumConstrs': [metrics['NumConstrs'] for metrics in all_metrics_2024.values()],
        'NumBinVars': [metrics['NumBinVars'] for metrics in all_metrics_2024.values()],
        'ObjectiveBound': [metrics['ObjectiveBound'] for metrics in all_metrics_2024.values()],
        'ObjectiveValue': [metrics['ObjectiveValue'] for metrics in all_metrics_2024.values()]
    })
    
    # Save benchmark metrics to CSV
    benchmark_df.to_csv("Benchmark/global_linearized_operational_data_2024.csv", index=False)
    
    print("Benchmark metrics saved to Benchmark/global_linearized_operational_data_2024.csv")

    # Process all results to ensure idle mode values are exactly 0
    for date, results in all_results_2024.items():
        # Get the idle mode indicators
        z_I_values = results['z_I']
        
        # Create corrected results
        corrected_p_t_T = results['p_t_T'].copy()
        corrected_p_t_P = results['p_t_P'].copy()
        corrected_q_t = results['q_t'].copy()
        
        # Force values to exactly 0 when in idle mode
        for t in range(len(z_I_values)):
            if z_I_values[t] > 0.5:  # If idle mode is active (binary variable close to 1)
                corrected_p_t_T[t] = 0.0
                corrected_p_t_P[t] = 0.0
                corrected_q_t[t] = 0.0
        
        # Update the results dictionary
        results['p_t_T'] = corrected_p_t_T
        results['p_t_P'] = corrected_p_t_P
        results['q_t'] = corrected_q_t
    
    # Print the corrected results for verification
    date = list(all_results_2024.keys())[0]
    results = all_results_2024[date]
    
    print(f"\nCorrected Results for {date}:")
    print("-" * 80)
    print("Time | Power (MW) | Flow (m³/s) | Head (m) | Mode")
    print("-" * 80)
    
    for t in range(len(results['p_t_T'])):
        power = results['p_t_T'][t] + results['p_t_P'][t]
        flow = results['q_t'][t]
        head = results['h_t'][t]
        
        # Determine mode
        if results['z_I'][t] > 0.5:
            mode = "Idle"
        elif results['z_T'][t] > 0.5:
            mode = "Turbine"
        else:
            mode = "Pump"
            
        print(f"{t:4d} | {power:10.4f} | {flow:10.4f} | {head:8.4f} | {mode}")
    
    # Update the DataFrame creation code as well
    df_list = []
    for date, results in all_results_2024.items():
        # Create power values with proper zeros for idle mode
        power_values = [p_t_T + p_t_P for p_t_T, p_t_P in zip(results['p_t_T'], results['p_t_P'])]
        
        df = pd.DataFrame({
            'Time': range(T),
            'Power': power_values,
            'Head': results['h_t'],
            'Flow': results['q_t'],
            'Price': prices_2024[date],
            'Date': date,
            'Mode': ["Idle" if z_I > 0.5 else ("Turbine" if z_T > 0.5 else "Pump") 
                    for z_I, z_T in zip(results['z_I'], results['z_T'])]
        })
        df_list.append(df)
    
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv("./Data/database_no_piecewise_2024.csv", index=False)
    print("\nResults saved to ./Data/database_no_piecewise_2024.csv")

    # Plot results for 2024 data
    date = list(all_results_2024.keys())[0]
    results = all_results_2024[date]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create corrected power array
    power = [p_t_T + p_t_P for p_t_T, p_t_P in zip(results['p_t_T'], results['p_t_P'])]
    
    # Create an array to highlight different modes
    turbine_periods = [t for t in range(T) if results['z_T'][t] > 0.5]
    pump_periods = [t for t in range(T) if results['z_P'][t] > 0.5]
    idle_periods = [t for t in range(T) if results['z_I'][t] > 0.5]
    
    # Plot power
    ax1.plot(range(T), power, label="Power", color='b')
    
    # Highlight different modes with background colors
    for t in turbine_periods:
        ax1.axvspan(t-0.5, t+0.5, alpha=0.2, color='green')
    for t in pump_periods:
        ax1.axvspan(t-0.5, t+0.5, alpha=0.2, color='red')
    for t in idle_periods:
        ax1.axvspan(t-0.5, t+0.5, alpha=0.2, color='gray')
    
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Power (MW)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(range(T), prices_2024[date], label="DA Price", color='r', linestyle='--')
    ax2.set_ylabel("DA Price ($/MWh)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add a legend for modes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Turbine Mode'),
        Patch(facecolor='red', alpha=0.2, label='Pump Mode'),
        Patch(facecolor='gray', alpha=0.2, label='Idle Mode')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Add a legend for lines
    lines_legend = [
        plt.Line2D([0], [0], color='b', label='Power'),
        plt.Line2D([0], [0], color='r', linestyle='--', label='DA Price')
    ]
    ax2.legend(handles=lines_legend, loc='upper right')

    fig.suptitle(f"Power and DA Price-Time Plot with Operation Modes ({date}) - 2024 Data")
    plt.tight_layout()
    plt.savefig("./Data/power_price_plot_2024.png", dpi=300)
    plt.close()

# %% Simulate and append results of 2024 data
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
    # Initialize parameters with modified initialization for float values
    params = HydroParameters(
        head_init=torch.tensor(head_init, dtype=torch.float32, device=device),
        v_low_init=torch.tensor(v_low_init, dtype=torch.float32, device=device)
    )
    
    # Initialize simulation layer
    sim = SimulationLayer(params)
    
    # Load optimized operation data
    df_op = pd.read_csv("./Data/database_no_piecewise_2024.csv")
    
    # Load benchmark data
    df_benchmark = pd.read_csv("Benchmark/global_linearized_operational_data_2024.csv")
    
    # Create new columns for results
    results = {
        'Date': [],
        'SimProfit': [],
        'SIPenalty': [],
        'VolumePenalty': [],
        'OperatingCost': []
    }
    
    # Process each day
    for date in df_benchmark['Date'].unique():
        print(f"Simulating operation for {date}...")
        
        # Filter data for this day
        day_data = df_op[df_op['Date'] == date]
        
        # Extract arrays
        p_opt = torch.tensor(day_data['Power'].values, dtype=torch.float32)
        h_opt = torch.tensor(day_data['Head'].values, dtype=torch.float32)
        q_opt = torch.tensor(day_data['Flow'].values, dtype=torch.float32)
        da_price = torch.tensor(day_data['Price'].values, dtype=torch.float32)
        
        # Simulate operation
        p_sim, q_sim, h_sim, v_low_sim = sim.simulate_operation(p_opt, q_opt, h_opt)
        
        # Calculate profit and penalties
        total_profit, si_penalty, volume_penalty, operating_cost = sim.calc_profit(
            p_sim, p_opt, v_low_sim, da_price
        )
        
        # Store results
        results['Date'].append(date)
        results['SimProfit'].append(total_profit.item())
        results['SIPenalty'].append(si_penalty.item())
        results['VolumePenalty'].append(volume_penalty.item())
        results['OperatingCost'].append(operating_cost.item())
        
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
    df_merged.to_csv("Benchmark/global_linearized_operational_data_2024.csv", index=False)
    
    print("\nUpdated benchmark data saved with simulation results.")
    
    # Display summary statistics
    print("\nSimulation Results Summary:")
    print(f"Average Simulated Profit: ${df_results['SimProfit'].mean():.2f}")
    print(f"Average SI Penalty: ${df_results['SIPenalty'].mean():.2f}")
    print(f"Average Volume Penalty: ${df_results['VolumePenalty'].mean():.2f}")
    print(f"Average Operating Cost: ${df_results['OperatingCost'].mean():.2f}")
    
    # Plot comparison of simulated vs optimized profit
    plt.figure(figsize=(10, 6))
    plt.scatter(df_merged['ObjectiveValue'], df_merged['SimProfit'], alpha=0.6)
    plt.plot([df_merged['ObjectiveValue'].min(), df_merged['ObjectiveValue'].max()], 
             [df_merged['ObjectiveValue'].min(), df_merged['ObjectiveValue'].max()], 
             'r--')
    plt.xlabel('Optimized Profit ($)')
    plt.ylabel('Simulated Profit ($)')
    plt.title('Comparison of Optimized vs. Simulated Profit')
    plt.grid(True)
    plt.savefig('./Data/profit_comparison_2024.png', dpi=300)
    plt.close()
    
    # Plot penalty distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_results['SIPenalty'], bins=20, alpha=0.5, label='SI Penalty')
    plt.hist(df_results['VolumePenalty'], bins=20, alpha=0.5, label='Volume Penalty')
    plt.xlabel('Penalty ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Penalties in 2024 Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('./Data/penalty_distribution_2024.png', dpi=300)
    plt.close()
    
    print("\nPlots saved to Data directory.")