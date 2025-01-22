# %%
# Initialization
import torch
import dill as pickle
import cvxpy as cp
import numpy as np
import pandas as pd
import sympy as sp
from pathlib import Path
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer
import sys
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go
import torch.nn as nn
from cvxpy import vstack

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly,neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

# %% Regression Layer setup
# Regression Layer setup

class RegressionLayer(nn.Module):
    def __init__(self, Time, δh, δp, δq, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, head_min, head_max, UPC_sampling_rate=400):
        super(RegressionLayer, self).__init__()
        self.Time = Time
        self.δh = δh
        self.δp = δp
        self.δq = δq
        self.neg_min_fit = torch.tensor(neg_min_fit, dtype=torch.float32)
        self.neg_max_fit = torch.tensor(neg_max_fit, dtype=torch.float32)
        self.pos_min_fit = torch.tensor(pos_min_fit, dtype=torch.float32)
        self.pos_max_fit = torch.tensor(pos_max_fit, dtype=torch.float32)
        self.head_min = head_min
        self.head_max = head_max
        self.UPC_sampling_rate = UPC_sampling_rate

    def least_squares_UPC_torch(self, p_samples, h_samples, q_values):
        X = torch.stack([p_samples, h_samples, torch.ones_like(p_samples)], dim=1)
        y = q_values.unsqueeze(1)
        XTX = torch.matmul(X.t(), X)
        XTX_inv = torch.inverse(XTX)
        XTy = torch.matmul(X.t(), y)
        beta = torch.matmul(XTX_inv, XTy)
        return beta.squeeze()

    def least_squares_v_low_torch(self, h_samples, v_low_samples):
        X = torch.stack([h_samples, torch.ones_like(h_samples)], dim=1)
        y = v_low_samples.unsqueeze(1)
        XTX = torch.matmul(X.t(), X)
        XTX_inv = torch.inverse(XTX)
        XTy = torch.matmul(X.t(), y)
        beta = torch.matmul(XTX_inv, XTy)
        return beta.squeeze()

    def forward(self, power, head):
        c, d, e = {}, {}, {}
        a, b = {}, {}
        
        for t in range(self.Time):
            # UPC regression
            h_samples = torch.linspace(max(self.head_min, head[t] - self.δh), min(self.head_max, head[t] + self.δh), self.UPC_sampling_rate)
            p_samples = torch.linspace(power[t] - self.δp, power[t] + self.δp, self.UPC_sampling_rate)

            p_mesh, h_mesh = torch.meshgrid(p_samples, h_samples, indexing='ij')
            p_flat = p_mesh.flatten()
            h_flat = h_mesh.flatten()

            mask = ((self.neg_min_fit[0] * h_flat + self.neg_min_fit[1] <= p_flat) & 
                    (p_flat <= self.neg_max_fit[0] * h_flat + self.neg_max_fit[1])) | \
                   ((self.pos_min_fit[0] * h_flat + self.pos_min_fit[1] <= p_flat) & 
                    (p_flat <= self.pos_max_fit[0] * h_flat + self.pos_max_fit[1]))

            p_valid = p_flat[mask]
            h_valid = h_flat[mask]

            if p_valid.numel() > 0:
                # Use the imported predict_q_poly function
                q_values = torch.tensor([predict_q_poly(p.item(), h.item()) for p, h in zip(p_valid, h_valid)], dtype=torch.float32)
                beta = self.least_squares_UPC_torch(p_valid, h_valid, q_values)
                c[t], d[t], e[t] = beta.tolist()
            else:
                c[t], d[t], e[t] = 0, 0, 0  # Default values if no valid points

            # v_low regression
            h_samples = torch.linspace(max(self.head_min, head[t] - self.δh), min(self.head_max, head[t] + self.δh), self.UPC_sampling_rate)
            # Use the imported h_to_v_low_fitted function
            v_low_samples = torch.tensor([h_to_v_low_fitted(h.item()) for h in h_samples], dtype=torch.float32)
            beta = self.least_squares_v_low_torch(h_samples, v_low_samples)
            a[t], b[t] = beta.tolist()

        # Create tensors for c, d, e, a, b coefficients
        c_tensor = torch.tensor([c[t] for t in range(self.Time)], dtype=torch.float32)
        d_tensor = torch.tensor([d[t] for t in range(self.Time)], dtype=torch.float32)
        e_tensor = torch.tensor([e[t] for t in range(self.Time)], dtype=torch.float32)
        a_tensor = torch.tensor([a[t] for t in range(self.Time)], dtype=torch.float32)
        b_tensor = torch.tensor([b[t] for t in range(self.Time)], dtype=torch.float32)

        return c_tensor, d_tensor, e_tensor, a_tensor, b_tensor
    
# %% Test the RegressionLayer
# Test the RegressionLayer

def test_RegressionLayer():
    # Define test parameters
    Time = 24
    δh = 20
    δp = 5
    δq = 7
    head_min = 50
    head_max = 99

    # Create an instance of the layer
    layer = RegressionLayer(Time, δh, δp, δq, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, head_min, head_max)

    # Create sample input data
    power = torch.tensor([-6.77, -7.01, -7.32, -7.63, -7.95, -8.26, -8.19, 4.27, 4.11, 4.43, 4.23, 4.01, 3.78, 3.55, 
                          3.37, 3.3, 3.23, 4.17, 4.8, 4.55, 3.91, 3.66, 2.64, 2.57], dtype=torch.float32)
    head = torch.tensor([76.96, 79.39, 81.82, 84.25, 86.67, 89.12, 91.47, 90.13, 88.82, 87.34, 85.89, 84.48, 83.13, 81.85, 
                         80.6, 79.35, 78.09, 76.37, 74.23, 72.18, 70.49, 68.9, 67.77, 66.67], dtype=torch.float32)

    # Run the layer
    c, d, e, a, b = layer(power, head)

    # Print results
    print("UPC regression coefficients:")
    print("c:", c)
    print("d:", d)
    print("e:", e)
    print("\nv_low regression coefficients:")
    print("a:", a)
    print("b:", b)

    # Check shapes
    assert c.shape == (Time,), f"Expected shape ({Time},), got {c.shape}"
    assert d.shape == (Time,), f"Expected shape ({Time},), got {d.shape}"
    assert e.shape == (Time,), f"Expected shape ({Time},), got {e.shape}"
    assert a.shape == (Time,), f"Expected shape ({Time},), got {a.shape}"
    assert b.shape == (Time,), f"Expected shape ({Time},), got {b.shape}"

    print("-------------Test completed-------------")

# Run the test
if __name__ == "__main__":
    test_RegressionLayer()

# %% Optimization Layer

class OptimizationLayer(nn.Module):
    def __init__(self, Time=24):
        super(OptimizationLayer, self).__init__()
        
        # Basic constants
        self.Time = Time
        self.rho = 1000    # Density of water in kg/m³
        self.g = 9.81      # Gravity in m/s²
        self.mu = 0.9      # Efficiency/penalty factor
        self.time_step = 1 # Time step in hours
        
        # Operating limits
        self.head_min = 50
        self.head_max = 99
        self.target_head = 66.67
        self.h_init = 76.96 # Initial head value
        
        # Trust region sizes
        self.δp = 5  # MW
        self.δh = 20 # m
        self.δq = 7  # m³/s
        
        # UPC boundary coefficients (constants)
        self.neg_min_fit = neg_min_fit
        self.neg_max_fit = neg_max_fit
        self.pos_min_fit = pos_min_fit
        self.pos_max_fit = pos_max_fit
        
        # CVXPY Parameters
        self.DA_price_hour_cp = cp.Parameter(24)  # Day-ahead prices
        self.C_op = cp.Parameter(nonneg=True)  # Cost coefficient
        self.power_prev = cp.Parameter(24)  # Previous power schedule for warm start
        self.head_prev = cp.Parameter(24)  # Previous head values for warm start
        
        # Variables
        self.p_cp = cp.Variable(24)  # power for each hour
        self.q_cp = cp.Variable(24)  # flow rate for each hour
        self.h_cp = cp.Variable(24)  # head at the end of each hour
        self.v_low_cp = cp.Variable(24)  # lower basin volume at the end of each hour
        self.volume_deficit = cp.Variable()  # volume deficit variable
        
        # Set up the optimization layer
        self.layer = self._setup_layer()
        
    def _setup_layer(self):
        """Set up the CvxpyLayer with the optimization problem"""
        # Variables (using the class variables)
        p = self.p_cp
        q = self.q_cp
        h = self.h_cp
        v_low = self.v_low_cp
        vol_deficit = self.volume_deficit
        
        # Initial conditions
        v_low_init = h_to_v_low_fitted(self.h_init)
        target_vol_low = h_to_v_low_fitted(self.target_head)
        
        # Initialize constraint lists
        constraints_UPC = []
        constraints_TR = []
        constraints_LRA = []
        constraints_vol = []
        constraints_VD = []
        
        # Volume deficit constraints
        constraints_VD += [
            vol_deficit >= 0,
            vol_deficit >= v_low[23] - target_vol_low
        ]
        
        # Initial volume balance constraint
        constraints_vol += [v_low[0] == v_low_init + q[0] * 3600]
        
        # Time-dependent constraints
        for t in range(self.Time):
            # UPC boundary constraints based on previous power values
            # Use multiplication instead of conditional statements
            # For turbine mode (power > 0)
            pos_indicator = cp.pos(self.power_prev[t])
            constraints_UPC += [
                pos_indicator * (p[t] - (self.pos_min_fit[0] * h[t] + self.pos_min_fit[1])) >= 0,
                pos_indicator * (self.pos_max_fit[0] * h[t] + self.pos_max_fit[1] - p[t]) >= 0
            ]
            
            # For pump mode (power < 0)
            neg_indicator = cp.neg(self.power_prev[t])
            constraints_UPC += [
                neg_indicator * (p[t] - (self.neg_min_fit[0] * h[t] + self.neg_min_fit[1])) >= 0,
                neg_indicator * (self.neg_max_fit[0] * h[t] + self.neg_max_fit[1] - p[t]) >= 0
            ]
            
            # For idle mode (power = 0)
            idle_indicator = (1 - pos_indicator - neg_indicator)
            constraints_UPC += [
                idle_indicator * p[t] == 0,
                idle_indicator * q[t] == 0
            ]
            
            # Head limits
            constraints_UPC += [
                h[t] >= self.head_min,
                h[t] <= self.head_max
            ]
            
            # Trust region constraints
            constraints_TR += [
                p[t] <= self.power_prev[t] + self.δp,
                p[t] >= self.power_prev[t] - self.δp,
                h[t] <= self.head_prev[t] + self.δh,
                h[t] >= self.head_prev[t] - self.δh
            ]
            
            # Linear regression approximation constraints
            # constraints_LRA += [
            #     q[t] == op[t]['q_TA'](p[t], h[t]),
            #     v_low[t] == op[t]['v_low_TA'](h[t])
            # ]
            
            # Volume balance constraints for t > 0
            if t > 0:
                constraints_vol += [
                    v_low[t] == v_low[t-1] + q[t] * 3600
                ]
        
        # Combine all constraints
        constraints = constraints_UPC + constraints_TR + constraints_LRA + constraints_vol + constraints_VD
        
        # Objective function components
        revenue = self.DA_price_hour_cp @ p * self.time_step
        energy_loss = vol_deficit * self.rho * self.g * self.target_head * self.mu / 3600000000
        volume_penalty = energy_loss * cp.max(self.DA_price_hour_cp)
        operational_costs = self.C_op * cp.sum_squares(p)
        
        # Complete objective
        objective = cp.Maximize(revenue - volume_penalty - operational_costs)
        
        # Create and return CvxpyLayer
        problem = cp.Problem(objective, constraints)
        return CvxpyLayer(
            problem,
            parameters=[self.DA_price_hour_cp, self.h_init, self.C_op, self.power_prev, self.head_prev],
            variables=[p, q, h, v_low, vol_deficit]
        )
    
    def forward(self, DA_price, h_init, power, head, C_op=3.8):
        """
        Forward pass of the optimization layer with warm start
        
        Args:
            DA_price (torch.Tensor): Day-ahead prices [Time]
            h_init (torch.Tensor): Initial head value [1]
            power (torch.Tensor): Previous power schedule for warm start [Time]
            head (torch.Tensor): Previous head values for warm start [Time]
            C_op (float): Operational cost coefficient (default: 3.8 €/MWh)
        
        Returns:
            tuple: Optimized (power, flow, head, volume, deficit)
        """
        try:
            # Initialize variables with warm start values
            self.p_cp.value = power
            self.h_cp.value = head
            
            # Solve optimization
            p_sol, q_sol, h_sol, v_low_sol, vol_def_sol = self.layer(
                DA_price, h_init, C_op, power, head
            )
            return p_sol, q_sol, h_sol, v_low_sol, vol_def_sol
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return None
        

# %% Test the OptimizationLayer
def test_optimization_layer():
    """Test the OptimizationLayer with warm starting capability"""
    print("\n-------------Testing OptimizationLayer-------------")
    
    # Initialize parameters
    Time = 24
    
    # Create sample input data (warm start values)
    power = torch.tensor([-6.77, -7.01, -7.32, -7.63, -7.95, -8.26, -8.19, 4.27, 4.11, 4.43, 
                         4.23, 4.01, 3.78, 3.55, 3.37, 3.3, 3.23, 4.17, 4.8, 4.55, 
                         3.91, 3.66, 2.64, 2.57], dtype=torch.float32, requires_grad=True)
    
    head = torch.tensor([76.96, 79.39, 81.82, 84.25, 86.67, 89.12, 91.47, 90.13, 88.82, 87.34, 
                        85.89, 84.48, 83.13, 81.85, 80.6, 79.35, 78.09, 76.37, 74.23, 72.18,
                        70.49, 68.9, 67.77, 66.67], dtype=torch.float32)
    
    # Convert DA price to tensor and ensure it's float32
    DA_price_tensor = torch.tensor(DA_price_hour, dtype=torch.float32)
    
    # Initial head value (first value from head tensor)
    h_init = torch.tensor(76.96, dtype=torch.float32)
    
    # Initialize OptimizationLayer
    print("Initializing OptimizationLayer...")
    optimization_layer = OptimizationLayer(Time=Time)
    
    # Print input shapes and values
    print("\nInput shapes and values:")
    print(f"DA_price: {DA_price_tensor.shape}")
    print(f"h_init: {h_init.item():.2f} m")
    print(f"power: {power.shape}")
    print(f"head: {head.shape}")
    
    # Run optimization
    print("\nRunning optimization...")
    try:
        solutions = optimization_layer(DA_price_tensor, h_init, power, head)
        
        if solutions is not None:
            p_sol, q_sol, h_sol, v_low_sol, vol_def_sol = solutions
            
            # Convert solutions to numpy for easier handling
            p_sol_np = p_sol.detach().numpy()
            q_sol_np = q_sol.detach().numpy()
            h_sol_np = h_sol.detach().numpy()
            v_low_sol_np = v_low_sol.detach().numpy()
            
            # Print optimization results
            print("\nOptimization successfully completed!")
            print("\nSolution shapes:")
            print(f"Power schedule: {p_sol.shape}")
            print(f"Flow rate: {q_sol.shape}")
            print(f"Head values: {h_sol.shape}")
            print(f"Lower basin volume: {v_low_sol.shape}")
            print(f"Volume deficit: {vol_def_sol.item():.4f} m³")
            
            # Compare original and optimized schedules
            print("\nSchedule comparison (first 5 hours):")
            print("Hour | Original Power | Optimized Power | Original Head | Optimized Head")
            print("-" * 65)
            for t in range(5):
                print(f"{t:3d} | {power[t]:13.2f} | {p_sol_np[t]:14.2f} | {head[t]:12.2f} | {h_sol_np[t]:13.2f}")
            
            # Calculate objective components
            revenue = np.sum(DA_price_tensor.numpy() * p_sol_np * optimization_layer.time_step)
            energy_loss = vol_def_sol.item() * optimization_layer.rho * optimization_layer.g * optimization_layer.target_head * optimization_layer.mu / 3600000000
            volume_penalty = energy_loss * np.max(DA_price_tensor.numpy())
            operational_costs = 3.8 * np.sum(p_sol_np ** 2)
            
            print("\nObjective components:")
            print(f"Revenue: {revenue:.2f} €")
            print(f"Volume penalty: {volume_penalty:.2f} €")
            print(f"Operational costs: {operational_costs:.2f} €")
            print(f"Net profit: {(revenue - volume_penalty - operational_costs):.2f} €")
            
            # Verify trust region constraints
            print("\nVerifying trust region constraints...")
            power_violations = np.abs(p_sol_np - power.numpy()) > optimization_layer.δp + 1e-6
            head_violations = np.abs(h_sol_np - head.numpy()) > optimization_layer.δh + 1e-6
            
            if np.any(power_violations):
                print("Warning: Power trust region violations detected!")
                violated_hours = np.where(power_violations)[0]
                for hour in violated_hours:
                    print(f"Hour {hour}: |{p_sol_np[hour]:.2f} - {power[hour]:.2f}| > {optimization_layer.δp}")
            else:
                print("Power trust region constraints satisfied.")
                
            if np.any(head_violations):
                print("Warning: Head trust region violations detected!")
                violated_hours = np.where(head_violations)[0]
                for hour in violated_hours:
                    print(f"Hour {hour}: |{h_sol_np[hour]:.2f} - {head[hour]:.2f}| > {optimization_layer.δh}")
            else:
                print("Head trust region constraints satisfied.")
            
            # Plot results
            plot_optimization_results(power, head, p_sol, h_sol, q_sol, v_low_sol)
            
        else:
            print("Optimization returned None - check for errors in the solve.")
            
    except Exception as e:
        print(f"Error during optimization: {str(e)}")

def plot_optimization_results(power, head, p_sol, h_sol, q_sol, v_low_sol):
    """Plot the optimization results"""
    plt.figure(figsize=(15, 12))
    
    # Convert all tensors to numpy arrays for plotting
    power_np = power.numpy()
    head_np = head.numpy()
    p_sol_np = p_sol.detach().numpy()
    h_sol_np = h_sol.detach().numpy()
    q_sol_np = q_sol.detach().numpy()
    v_low_sol_np = v_low_sol.detach().numpy()
    
    # Plot power schedules
    plt.subplot(3, 1, 1)
    plt.plot(range(24), power_np, 'b-', label='Original Power')
    plt.plot(range(24), p_sol_np, 'r--', label='Optimized Power')
    plt.title('Power Schedule Comparison')
    plt.xlabel('Hour')
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.grid(True)
    
    # Plot head values
    plt.subplot(3, 1, 2)
    plt.plot(range(24), head_np, 'b-', label='Original Head')
    plt.plot(range(24), h_sol_np, 'r--', label='Optimized Head')
    plt.title('Head Value Comparison')
    plt.xlabel('Hour')
    plt.ylabel('Head (m)')
    plt.legend()
    plt.grid(True)
    
    # Plot flow rates and lower basin volume
    ax1 = plt.subplot(3, 1, 3)
    line1 = ax1.plot(range(24), q_sol_np, 'g-', label='Flow Rate')
    ax1.set_ylabel('Flow Rate (m³/s)')
    ax1.set_xlabel('Hour')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(range(24), v_low_sol_np, 'b--', label='Lower Basin Volume')
    ax2.set_ylabel('Volume (m³)')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the test
    test_optimization_layer()