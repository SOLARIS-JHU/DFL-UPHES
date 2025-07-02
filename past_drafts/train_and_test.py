# %%
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import sys
from tqdm import tqdm, trange 
import torch.optim as optim  
from datetime import datetime, timedelta
import random
# torch.autograd.set_detect_anomaly(True)

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly,neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

from pipeline_learnable_weight_ver import HydroParameters, RegressionLayer, OptiLayer, SimulationLayer, Pipeline, read_da_price, hourly_to_quarterly

# %% Define test data
# Create sample input data
power = torch.tensor([-6.77, -7.01, -7.32, -7.63, -7.95, -8.26, -8.19, 4.27, 4.11, 4.43, 4.23, 4.01, 3.78, 3.55, 
                        3.37, 3.3, 3.23, 4.17, 4.8, 4.55, 3.91, 3.66, 2.64, 2.57], dtype=torch.float32, requires_grad=True)
head = torch.tensor([76.96, 79.39, 81.82, 84.25, 86.67, 89.12, 91.47, 90.13, 88.82, 87.34, 85.89, 84.48, 83.13, 81.85, 
                        80.6, 79.35, 78.09, 76.37, 74.23, 72.18, 70.49, 68.9, 67.77, 66.67], dtype=torch.float32, requires_grad=True)
flow = torch.tensor([-10.24, -10.14, -10.12, -10.11, -10.11, -10.18, -9.79, 5.55, 5.48, 6.16, 6.06, 5.87, 5.61, 5.36, 
                        5.19, 5.21, 5.24, 7.15, 8.95, 8.53, 7.04, 6.64, 4.68, 4.61], dtype=torch.float32, requires_grad=True)

head_init = 77.0 # Initial head value
v_low_init = h_to_v_low_fitted(head_init) # Initial lower reservoir volume

# %% Test the pipeline forward pass
def plot_optimization_simulation_results(p_opt, q_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, max_vol_low=max_vol_low, save_path="optimization_simulation_results.svg"):
    """
    Plot optimization and simulation results comparison with upper reservoir volume and save as SVG
   
    Args:
        p_opt (torch.Tensor): Optimized power schedule (hourly, size=24)
        q_opt (torch.Tensor): Optimized flow schedule (hourly, size=24)
        p_sim_clb (torch.Tensor): Simulated power schedule (per minute, size=1440)
        q_sim_clb (torch.Tensor): Simulated flow schedule (per minute, size=1440)
        h_sim_clb (torch.Tensor): Simulated head schedule (per minute, size=1440)
        v_low_clb (torch.Tensor): Simulated lower reservoir volume (per minute, size=1440)
        max_vol_low (float): Maximum volume of reservoirs (default=588000)
        save_path (str): Path where to save the SVG file (default="optimization_simulation_results.svg")
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
   
    # Create figure with 4 subplots (added one for better separation of volumes)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16))
   
    # Create time arrays
    t_hours = np.arange(24)
    t_minutes = np.arange(len(p_sim_clb)) / 60  # Convert to hours

    # Calculate upper reservoir volume
    v_up_clb = max_vol_low - v_low_clb.detach().numpy()
    
    # Plot 1: Power comparison
    ax1_opt = ax1
    ax1_sim = ax1.twinx()
   
    # Plot optimization results
    line1 = ax1_opt.step(t_hours, p_opt.detach().numpy(), 'r-', label='Optimized Power', where='post')
    # Plot simulation results
    line2 = ax1_sim.plot(t_minutes, p_sim_clb.detach().numpy(), 'b-', alpha=0.6, label='Simulated Power')
   
    # Add labels and legend
    ax1_opt.set_xlabel('Time (hours)')
    ax1_opt.set_ylabel('Optimized Power (MW)')
    ax1_sim.set_ylabel('Simulated Power (MW)')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    ax1.set_title('Power Schedule Comparison')
    ax1.grid(True)

    # Plot 2: Flow comparison
    ax2_opt = ax2
    ax2_sim = ax2.twinx()
   
    # Plot optimization results
    line3 = ax2_opt.step(t_hours, q_opt.detach().numpy(), 'r-', label='Optimized Flow', where='post')
    # Plot simulation results
    line4 = ax2_sim.plot(t_minutes, q_sim_clb.detach().numpy(), 'b-', alpha=0.6, label='Simulated Flow')
   
    # Add labels and legend
    ax2_opt.set_xlabel('Time (hours)')
    ax2_opt.set_ylabel('Optimized Flow (m³/s)')
    ax2_sim.set_ylabel('Simulated Flow (m³/s)')
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.set_title('Flow Schedule Comparison')
    ax2.grid(True)

    # Plot 3: Head
    # Plot head with single y-axis
    line5 = ax3.plot(t_minutes, h_sim_clb.detach().numpy(), 'g-', label='Head')
   
    # Add labels and legend
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Head (m)')
    ax3.legend(loc='upper right')
    ax3.set_title('Head Profile')
    ax3.grid(True)

    # Plot 4: Reservoir Volumes
    # Create a shared axis for both volumes
    line6 = ax4.plot(t_minutes, v_low_clb.detach().numpy(), 'b-', label='Lower Reservoir')
    line7 = ax4.plot(t_minutes, v_up_clb, 'r-', label='Upper Reservoir')
    
    # Add horizontal line for maximum volume
    ax4.axhline(y=max_vol_low, color='k', linestyle='--', alpha=0.5, label='Maximum Volume')
    
    # Add labels and legend
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Volume (m³)')
    ax4.legend(loc='upper right')
    ax4.set_title('Reservoir Volumes')
    ax4.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure as SVG
    # Ensure the directory exists
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with high DPI and vector format
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {save_path}")
    
    # Display the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close(fig)

if __name__ == "__main__":
    params = HydroParameters()
    pipeline = Pipeline(params)
    profit, p_opt, q_opt, h_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h = pipeline.forward(power, head, DA_price_hour, DA_price_quarter)

    plot_optimization_simulation_results(p_opt, q_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb)

    # print optimal power, flow, head, and volume
    print("Optimized Power Schedule:")
    print(p_opt.detach().numpy())
    print("\nOptimized Flow Schedule:")
    print(q_opt.detach().numpy())
    print("\nOptimized Head Schedule:")
    print(h_sim_clb.detach().numpy())
    print("\nOptimized Lower Reservoir Volume Schedule:")
    print(v_low_clb.detach().numpy())
    print("\nProfit:")
    print(profit)


# %% Test backpropagation of SimulationLayer

# Wrap p_opt, q_opt, h_opt with requires_grad for gradient computation
p_opt_test = p_opt.detach().clone().requires_grad_(True)
q_opt_test = q_opt.detach().clone().requires_grad_(True)
h_opt_test = h_opt.detach().clone().requires_grad_(True)

# Create SimulationLayer instance for forward simulation
sim_layer = SimulationLayer(params)

# Run forward simulation
p_sim_clb_test, q_sim_clb_test, h_sim_clb_test, v_low_clb_test = sim_layer.simulate_operation(
    p_opt_test, q_opt_test, h_opt_test
)

# Compute profit scalar for differentiation
test_profit = -sim_layer.calc_profit(
    p_sim_clb_test, p_opt_test, v_low_clb_test, DA_price_quarter
)

# Run backpropagation
print(">>> Attempting backprop on test_profit:")
test_profit.backward()

# Display gradients
print("Gradient wrt p_opt_test:\n", p_opt_test.grad)
print("Gradient wrt q_opt_test:\n", q_opt_test.grad)
print("Gradient wrt h_opt_test:\n", h_opt_test.grad)

# %% Test backpropagation of OptiLayer
params = HydroParameters()
opti_layer = OptiLayer(params)

# Create test tensors with gradients
c_test = c.detach().clone().requires_grad_(False)
d_test = d.detach().clone().requires_grad_(False)
e_test = e.detach().clone().requires_grad_(False)
a_test = a.detach().clone().requires_grad_(False)
b_test = b.detach().clone().requires_grad_(False)
w_p_test = w_p.detach().clone().requires_grad_(True)
w_h_test = w_h.detach().clone().requires_grad_(True)
w_q_test = w_q.detach().clone().requires_grad_(True)

# Run optimization
p_opt_test, q_opt_test, h_opt_test, v_opt_test = opti_layer.forward(
    DA_price_hour, c_test, d_test, e_test, a_test, b_test,
    power, head, flow,
    w_p_test, w_h_test, w_q_test
)

# Calculate test loss and backpropagate
test_loss = -torch.mean(p_opt_test)
print(">>> Attempting backprop on test_loss:")
test_loss.backward()

# Display gradients
print("\nGradients for regression coefficients:")
print("c gradient:", c_test.grad)
print("d gradient:", d_test.grad)
print("e gradient:", e_test.grad)
print("a gradient:", a_test.grad)
print("b gradient:", b_test.grad)

print("\nGradients for weights:")
print("w_p gradient:", w_p_test.grad)
print("w_h gradient:", w_h_test.grad)
print("w_q gradient:", w_q_test.grad)

# %% Test backpropagation of RegressionLayer
# Initialize parameters and create regression layer
params = HydroParameters()
regression_layer = RegressionLayer(params)

# Create test inputs with gradients enabled
power_test = power.detach().clone().requires_grad_(True)
head_test = head.detach().clone().requires_grad_(True)

# Run regression
c_test, d_test, e_test, a_test, b_test = regression_layer.run_regression(power_test, head_test)

# Create loss function
loss = c_test.sum() + d_test.sum() + e_test.sum() + a_test.sum() + b_test.sum()
print("Attempting backprop on regression loss")
loss.backward()

# Check gradients
print("Grad wrt power_test:", power_test.grad)
print("Grad wrt head_test:", head_test.grad)

# %% debugging regression layer
def debug_regression_layer_fixed(params, power, head):
    """
    Fixed version of regression layer that maintains gradient flow
    """
    TH = params.time_horizon
    device = power.device
    
    print("Initial tensors require grad:")
    print(f"power.requires_grad: {power.requires_grad}")
    print(f"head.requires_grad: {head.requires_grad}")
    
    for t in range(1):  # Test with just first timestep for debugging
        # Instead of detaching, use the original tensor values
        p_center = power[t]
        h_center = head[t]
        
        # Create sample points that depend on the input tensors
        delta_p = params.δ_p
        num_samples = params.sampling_rate
        p_steps = torch.linspace(-delta_p, delta_p, num_samples, device=device)
        p_samples = p_center + p_steps  # This maintains gradient connection
        
        # Similar for head samples
        h_lo = torch.maximum(params.head_min * torch.ones_like(h_center), 
                           h_center - params.δ_h)
        h_hi = torch.minimum(params.head_max * torch.ones_like(h_center), 
                           h_center + params.δ_h)
        h_steps = torch.linspace(0, 1, num_samples, device=device)
        h_samples = h_lo + (h_hi - h_lo) * h_steps  # This maintains gradient connection
        
        print("\nAfter creating sample points:")
        print(f"p_samples.requires_grad: {p_samples.requires_grad}")
        print(f"h_samples.requires_grad: {h_samples.requires_grad}")
        
        # Create meshgrid
        p_mesh, h_mesh = torch.meshgrid(p_samples, h_samples, indexing="ij")
        p_flat = p_mesh.flatten()
        h_flat = h_mesh.flatten()
        
        print("\nAfter meshgrid:")
        print(f"p_flat.requires_grad: {p_flat.requires_grad}")
        print(f"h_flat.requires_grad: {h_flat.requires_grad}")
        
        # Filter valid regions using a soft mask approach
        def soft_and(a, b):
            return a * b
        
        def soft_or(a, b):
            return a + b - a * b
        
        # Convert boolean conditions to soft constraints
        mask_turbine = soft_and(
            (p_flat >= params.pos_min_fit[0]*h_flat + params.pos_min_fit[1]).float(),
            (p_flat <= params.pos_max_fit[0]*h_flat + params.pos_max_fit[1]).float()
        )
        mask_pump = soft_and(
            (p_flat >= params.neg_min_fit[0]*h_flat + params.neg_min_fit[1]).float(),
            (p_flat <= params.neg_max_fit[0]*h_flat + params.neg_max_fit[1]).float()
        )
        soft_mask = soft_or(mask_turbine, mask_pump)
        
        # Apply soft mask
        p_valid = p_flat * soft_mask
        h_valid = h_flat * soft_mask
        
        print("\nAfter masking:")
        print(f"p_valid.requires_grad: {p_valid.requires_grad}")
        print(f"h_valid.requires_grad: {h_valid.requires_grad}")
        
        # Calculate q values with vectorized operations
        q_values = torch.zeros_like(p_valid)
        q_values = predict_q_poly(p_valid, h_valid)
        q_values = q_values * soft_mask  # Apply mask to q values
        
        print("\nAfter q_values calculation:")
        print(f"q_values.requires_grad: {q_values.requires_grad}")
        
        # Compute regression with gradient tracking
        ones = torch.ones_like(p_valid)
        X = torch.stack([p_valid, h_valid, ones], dim=1)
        y = q_values.view(-1, 1)
        
        print("\nBefore least squares:")
        print(f"X.requires_grad: {X.requires_grad}")
        print(f"y.requires_grad: {y.requires_grad}")
        
        # Add small regularization for numerical stability
        XTX = torch.matmul(X.t(), X)
        XTy = torch.matmul(X.t(), y)
        epsilon = 1e-6
        reg_matrix = epsilon * torch.eye(3, device=XTX.device)
        XTX_reg = XTX + reg_matrix
        
        beta = torch.matmul(torch.inverse(XTX_reg), XTy)
        print("\nAfter least squares:")
        print(f"beta.requires_grad: {beta.requires_grad}")
        
        return beta

# Test the fixed version
def test_regression_debug_fixed():
    params = HydroParameters()
    power_test = torch.tensor([-6.77], requires_grad=True)
    head_test = torch.tensor([76.96], requires_grad=True)
    
    beta = debug_regression_layer_fixed(params, power_test, head_test)
    
    if beta is not None:
        loss = beta.sum()
        print("\nAttempting backpropagation...")
        loss.backward()
        
        print("\nFinal gradients:")
        print(f"power_test.grad: {power_test.grad}")
        print(f"head_test.grad: {head_test.grad}")

test_regression_debug_fixed()

# %% Test backpropagation of the whole pipeline
def test_pipeline_backprop():
    # Initialize parameters and pipeline
    params = HydroParameters()
    pipeline = Pipeline(params)
    
    # Create optimizer for the weight network
    optimizer = torch.optim.Adam(pipeline.weight_network.parameters(), lr=0.001)
    
    # Test data (using existing variables from earlier in the code)
    test_power = power.detach().clone()
    test_head = head.detach().clone()
    
    print("Starting pipeline backpropagation test...")
    
    # Training loop
    for epoch in range(5):  # Test with 5 epochs
        optimizer.zero_grad()
        
        # Forward pass through pipeline
        profit, p_opt, q_opt, h_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h = pipeline.forward(
            test_power, test_head, DA_price_hour, DA_price_quarter
        )
        
        # Loss is negative profit (since we want to maximize profit)
        loss = -profit
        
        # Backward pass
        loss.backward()
        
        # Print gradients of weight network parameters
        print(f"\nEpoch {epoch + 1}")
        print("Weight network gradients:")
        for name, param in pipeline.weight_network.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad shape={param.grad.shape}, grad mean={param.grad.mean():.6f}")
            else:
                print(f"{name}: No gradient")
        
        # Print loss
        print(f"Loss: {loss.item():.2f}")
        
        # Update weights
        optimizer.step()

if __name__ == "__main__":
    test_pipeline_backprop()

#
