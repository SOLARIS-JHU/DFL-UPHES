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

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly,neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

# %% Define the output of the predictor
# Define the output of the predictor

# create dictionary for operational point
op = {}

# Scheduling parameters
Time = 24  # 24h time as operational range
time_step = 1 # 1 hour as the minimum time step


# Just for test
head_op = [76.96, 79.39, 81.82, 84.25, 86.67, 89.12, 91.47, 90.13, 88.82, 87.34, 85.89, 84.48, 83.13, 81.85, 80.6, 79.35, 78.09, 76.37, 74.23, 72.18, 70.49, 68.9, 67.77, 66.67]
power_op = [-6.77, -7.01, -7.32, -7.63, -7.95, -8.26, -8.19, 4.27, 4.11, 4.43, 4.23, 4.01, 3.78, 3.55, 3.37, 3.3, 3.23, 4.17, 4.8, 4.55, 3.91, 3.66, 2.64, 2.57] # test
# flow_op = np.zeros(Time)

for t in range(Time):  # enumerate the operation period
    # Determine the operational mode based on the power output
    mode = 'turbine' if power_op[t] > 0 else 'pump' if power_op[t] < 0 else 'idle'

    # Store data in the dictionary
    op[t] = {
        'mode': mode,  # operational mode
        'power': power_op[t], # power obtained from NN? Is it necessary?
        'head': head_op[t], # head obtained from NN
        # 'flow': flow_op[t] # t * 10  # flow rate obtained from NN
    }

# Deviation for trust region in optimization
δp = 3  # MW
δh = 5    # m
δq = 2  # m^3/s

# %% head-v_low trust region linear regression with least square
# head-v_low trust region linear regression with least square

def v_h_linear_regression(h_to_v_low_fitted, op=op, Time=Time, 
                          head_min=head_min, head_max=head_max, 
                          δh=δh, sample_size=100):
    """
    Perform linear regression for each time period and update the operational dictionary.
    This version is compatible with PyTorch and doesn't use NumPy.
    
    Args:
    h_to_v_low_fitted (function): Function to calculate v_low from h.
    op (dict): Operational dictionary.
    Time (int, optional): The number of time periods. Default is 24.
    head_min (float, optional): Minimum head value. Default is 50.0.
    head_max (float, optional): Maximum head value. Default is 100.0.
    δh (float, optional): Head deviation for trust region. Default is 20.0.
    sample_size (int, optional): Number of samples for linear regression. Default is 100.
    
    Returns:
    dict: Updated operational dictionary with linear regression results.
    """
    for t in range(Time):
        # Generate h_range without numpy
        h_start = max(head_min, op[t]['head'] - δh)
        h_end = min(head_max, op[t]['head'] + δh)
        h_step = (h_end - h_start) / (sample_size - 1)
        h_range = [h_start + i * h_step for i in range(sample_size)]
        
        # Calculate v_low_range
        v_low_range = [h_to_v_low_fitted(h) for h in h_range]
        
        # Calculate sums for linear regression
        n = len(h_range)
        sum_x = sum(h_range)
        sum_y = sum(v_low_range)
        sum_xy = sum(h * v for h, v in zip(h_range, v_low_range))
        sum_x2 = sum(h ** 2 for h in h_range)
        
        # Calculate coefficients
        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        b = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x ** 2)
        
        # Store the lambda function in op
        op[t]['v_low_TA'] = lambda h_val, a=a, b=b: a * h_val + b 
        
        print(f"Time {t}: v_low = {a:.4f}*h + {b:.4f}")
    
    return op

def plot_v_h_results(h_to_v_low_fitted, op=op, Time=Time, δh=δh):
    """
    Plot the results of the linear regression against the global polynomial fit.
    
    Args:
    h_to_v_low_fitted (function): Function to calculate v_low from h.
    op (dict): Operational dictionary.
    Time (int): The number of time periods.
    δh (float): Head deviation for trust region.
    """
    global_h_range = np.linspace(50, 99, 1000)
    global_v_low_pred = h_to_v_low_fitted(global_h_range)
    
    plt.figure(figsize=(12, 8))
    plt.plot(global_h_range, global_v_low_pred, label='Polynomial Fit', color='r', linewidth=2, alpha=0.7)
    
    colors = plt.cm.viridis(np.linspace(0, 1, Time))
    for t in range(Time):
        local_h = np.linspace(op[t]['head'] - δh, op[t]['head'] + δh, 100)
        local_v_low = op[t]['v_low_TA'](local_h)
        plt.plot(local_h, local_v_low, label=f'Time {t}: Linear Fit', color=colors[t])
    
    plt.xlabel('head (h)')
    plt.ylabel('v_low')
    plt.title('Comparison of Polynomial and Hourly Linear Fits')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()

# Perform linear regression
op = v_h_linear_regression(h_to_v_low_fitted, op, Time, head_min, head_max, δh)

# Plot results
plot_v_h_results(h_to_v_low_fitted, op, Time, δh)

# %% UPC trust region linear regression with least square
# UPC trust region linear regression with least square

def least_squares_UPC_torch(p_samples, h_samples, q_values):
    """
    Perform least squares regression for UPC using PyTorch.
    
    Args:
    p_samples (torch.Tensor): Power samples
    h_samples (torch.Tensor): Head samples
    q_values (torch.Tensor): Flow values
    
    Returns:
    torch.Tensor: Coefficients [c, d, e] of the plane q = c*p + d*h + e
    """
    n = p_samples.shape[0]
    X = torch.stack([p_samples, h_samples, torch.ones_like(p_samples)], dim=1)
    y = q_values.unsqueeze(1)

    # Compute (X^T X)^(-1) X^T y
    XTX = torch.matmul(X.t(), X)
    XTX_inv = torch.inverse(XTX)
    XTy = torch.matmul(X.t(), y)
    beta = torch.matmul(XTX_inv, XTy)

    return beta.squeeze()

# Main processing loop
c, d, e = {}, {}, {}
problems = {}
UPC_sampling_rate = 400

for t in range(Time):
    h_samples = torch.linspace(op[t]['head'] - δh, op[t]['head'] + δh, UPC_sampling_rate)
    p_samples = torch.linspace(op[t]['power'] - δp, op[t]['power'] + δp, UPC_sampling_rate)

    # Create a mesh grid of p and h
    p_mesh, h_mesh = torch.meshgrid(p_samples, h_samples)
    p_flat = p_mesh.flatten()
    h_flat = h_mesh.flatten()

    # Apply constraints
    neg_min = torch.tensor(neg_min_fit)
    neg_max = torch.tensor(neg_max_fit)
    pos_min = torch.tensor(pos_min_fit)
    pos_max = torch.tensor(pos_max_fit)

    mask = ((neg_min[0] * h_flat + neg_min[1] <= p_flat) & (p_flat <= neg_max[0] * h_flat + neg_max[1])) | \
           ((pos_min[0] * h_flat + pos_min[1] <= p_flat) & (p_flat <= pos_max[0] * h_flat + pos_max[1]))

    p_valid = p_flat[mask]
    h_valid = h_flat[mask]

    if p_valid.numel() > 0:
        q_values = torch.tensor([predict_q_poly(p.item(), h.item()) for p, h in zip(p_valid, h_valid)])

        # Calculate coefficients using least squares
        beta = least_squares_UPC_torch(p_valid, h_valid, q_values)
        c[t], d[t], e[t] = beta.tolist()

        # Store the lambda function in op
        op[t]['q_TA'] = lambda p_val, h_val, ct=c[t], dt=d[t], et=e[t]: ct * p_val + dt * h_val + et
        print(f"Time {t}: Flow model: q = {c[t]:.4f}*p + {d[t]:.4f}*h + {e[t]:.4f}")
    else:
        print(f"Time {t}: No valid points found within constraints.")

# Calculate op['flow'] based on op['power'] and op['head'] with op['q_TA']
for t in range(Time):
    op[t]['flow'] = op[t]['q_TA'](op[t]['power'], op[t]['head'])

# Plotting function remains unchanged
def plot_3d_surface_interactive(x_valid, y_valid, z_valid, model, title, coefficients):
    # ... [The rest of the plotting function remains unchanged]
    pass

# Example of plotting with additional planes
if __name__ == '__main__':
    import numpy as np
    import plotly.graph_objects as go
    
    results_pump = prepare_and_fit_model('./Data/UPCs/temp/Mod_Francis_pump_temp.xlsx')
    results_turbine = prepare_and_fit_model('./Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx')

    # Collect the coefficients from the smaller area fits
    coefficients_pump = [(c[t], d[t], e[t]) for t in range(Time)]
    coefficients_turbine = [(c[t], d[t], e[t]) for t in range(Time)]

    plot_3d_surface_interactive(*results_pump[1:4], results_pump[0], 'Pump Model', coefficients_pump)
    plot_3d_surface_interactive(*results_turbine[1:4], results_turbine[0], 'Turbine Model', coefficients_turbine)