# %% Import libraries
import torch
import dill as pickle
import pandas as pd
import sys
# torch.autograd.set_detect_anomaly(True)
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = 77.0 # Initial head value
v_low_init = h_to_v_low_fitted(head_init) # Initial lower reservoir volume

# %% Load day-ahead prices
def load_prices():
    """Load day-ahead prices from database and set as environment variables."""
    # Read price database
    df = pd.read_csv("./Data/price_database.csv")
    
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

# target_vol_low = max_vol_low

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
        """Optimize the MILP and return the decision variable values."""
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

# %% Main execution - Run optimizer for all dates
if __name__ == "__main__":
    T = 24  # Number of time periods in a day
    all_results = {}
    
    for date, DA_prices in prices.items():
        print(f"Running optimization for {date}...")
        optimizer = MILPOptimizer(T, DA_prices)
        results = optimizer.solve()
        
        if results is not None:
            all_results[date] = results
        else:
            print(f"No optimal solution found for {date}!")
    
    # Save results to a pickle file for future analysis
    with open("milp_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    print("Optimization completed for all dates.")

# %% Test MILP optimizer for a single date
if __name__ == "__main__":
    T = 24
    DA_prices = prices[sample_date]  

    optimizer = MILPOptimizer(T, DA_prices)
    results = optimizer.solve()
    
    # # check for infeasibility
    # optimizer.model.computeIIS()
    # optimizer.model.write("model.ilp")
    # print("IIS Constraints:")
    # for c in optimizer.model.getConstrs():
    #     if c.IISConstr:
    #         print(f"{c.ConstrName}: {c}")

    # print("\nIIS Bounds:")
    # for v in optimizer.model.getVars():
    #     if v.IISLB:
    #         print(f"Lower Bound Infeasible: {v.VarName} >= {v.LB}")
    #     if v.IISUB:
    #         print(f"Upper Bound Infeasible: {v.VarName} <= {v.UB}")


    if results is not None:
        print("Optimization results:")
        for key, value in results.items():
            print(f"{key}: {value}")

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

# %%
