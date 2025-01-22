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
flow_op = np.zeros(Time)

for t in range(Time):  # enumerate the operation period
    # Determine the operational mode based on the power output
    mode = 'turbine' if power_op[t] > 0 else 'pump' if power_op[t] < 0 else 'idle'

    # Store data in the dictionary
    op[t] = {
        'mode': mode,
        # 'UPC_TA': UPC_TA[t],
        # 'head_TA': head_TA[t],
        'power': power_op[t], # power obtained from NN? Is it necessary?
        'head': head_op[t], # t * 2 + 50-2,     # head obtained from NN
        # v_low should be calculated by head
        'flow': flow_op[t] # t * 10  # flow rate obtained from NN
    }

# Deviation for trust region in optimization
δp = 5  # MW
δh = 5    # m
δq = 7  # m^3/s

# %% head-v_low trust region linear regression with cvxpy
# head-v_low trust region linear regression with cvxpy

# Generate initial data for h and calculate v_low
h_range = [np.linspace(max(head_min, op[t]['head'] - δh), min(head_max, op[t]['head'] + δh), 100) for t in range(Time)]
v_low_range = [h_to_v_low_fitted(h) for h in h_range]  # Assuming polynomial is the fitted function

# Perform linear regression using cvxpy for each time period
h={}; v_low={}; a={}; b={}; objective={}; constraints={} #initialize 24 cvxpy prob.
for t in range(Time):
    h[t] = cp.Parameter(shape=(100,), value=h_range[t])
    v_low[t] = cp.Parameter(shape=(100,), value=v_low_range[t])
    a[t] = cp.Variable()  # Slope
    b[t] = cp.Variable()  # Intercept

    # Objective: minimize the sum of squared residuals
    objective[t] = cp.Minimize(cp.sum_squares(a[t] * h[t] + b[t] - v_low[t]))
    constraints[t] = []  # No constraints for simple linear regression

    # Setup and solve the problem
    problem = cp.Problem(objective[t], constraints[t])
    problem.solve()

    # Store the lambda function in op
    op[t]['v_low_TA'] = lambda h_val, a=a[t].value, b=b[t].value: a * h_val + b 
    print(f"Time {t}: v_low = {a[t].value:.4f}*h + {b[t].value:.4f}")

# Plot & test

# Assume polynomial is already defined from earlier fit
global_h_range = np.linspace(min([min(h) for h in h_range]), max([max(h) for h in h_range]), 1000)
global_v_low_pred = h_to_v_low_fitted(global_h_range)
plt.figure(figsize=(12, 8))
plt.plot(global_h_range, global_v_low_pred, label='Polynomial Fit', color='r', linewidth=2, alpha=0.7)

# Plot fitted lines
colors = plt.cm.viridis(np.linspace(0, 1, Time))  # color mapping for time periods
for t in range(Time):
    # Local h and v_low values for plotting
    local_h = np.linspace(op[t]['head'] - δh, op[t]['head'] + δh, 100)
    local_v_low = op[t]['v_low_TA'](local_h)  # Calculate v_low using the fitted lambda function

    plt.plot(local_h, local_v_low, label=f'Time {t}: Linear Fit', color=colors[t])

plt.xlabel('head (h)')
plt.ylabel('v_low')
plt.title('Comparison of Polynomial and Hourly Linear Fits')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# %% head-v_low trust region linear regression with least square
# head-v_low trust region linear regression with least square

# Generate initial data for h and calculate v_low
h_range = [np.linspace(max(head_min, op[t]['head'] - δh), min(head_max, op[t]['head'] + δh), 100) for t in range(Time)]
v_low_range = [h_to_v_low_fitted(h) for h in h_range]  # Assuming polynomial is the fitted function

# Perform linear regression using cvxpy for each time period
h={}; v_low={}; a={}; b={}; objective={}; constraints={} #initialize 24 cvxpy prob.

# Perform linear regression for each time period
for t in range(Time):
    x = h_range[t]
    y = v_low_range[t]
    n = len(x)

    # Compute the sums needed for the formulas
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    # Compute coefficients a and b
    a[t] = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b[t] = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x ** 2)

    # Store the lambda function in op
    op[t]['v_low_TA'] = lambda h_val, a=a[t], b=b[t]: a * h_val + b 
    print(f"Time {t}: v_low = {a[t]:.4f}*h + {b[t]:.4f}")

# Plot & test

# Assume polynomial is already defined from earlier fit
global_h_range = np.linspace(50, 99, 1000)
global_v_low_pred = h_to_v_low_fitted(global_h_range)
plt.figure(figsize=(12, 8))
plt.plot(global_h_range, global_v_low_pred, label='Polynomial Fit', color='r', linewidth=2, alpha=0.7)

# Plot fitted lines
colors = plt.cm.viridis(np.linspace(0, 1, Time))  # color mapping for time periods
for t in range(Time):
    # Local h and v_low values for plotting
    local_h = np.linspace(op[t]['head'] - δh, op[t]['head'] + δh, 100)
    local_v_low = op[t]['v_low_TA'](local_h)  # Calculate v_low using the fitted lambda function

    plt.plot(local_h, local_v_low, label=f'Time {t}: Linear Fit', color=colors[t])

plt.xlabel('head (h)')
plt.ylabel('v_low')
plt.title('Comparison of Polynomial and Hourly Linear Fits')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# %% UPC trust region linear regression with cvxpy
# UPC trust region linear regression with cvxpy

c = {}; d = {}; e = {}
problems = {}
UPC_sampling_rate = 400 # 400^2 samples per trust region (before truncated by BC)

for t in range(Time):
    h_samples = np.linspace(op[t]['head'] - δh, op[t]['head'] + δh, UPC_sampling_rate)  
    p_samples = np.linspace(op[t]['power'] - δp, op[t]['power'] + δp, UPC_sampling_rate) 

    points = [(p, h) for h in h_samples for p in p_samples 
              if (np.dot([h, 1], neg_min_fit) <= p <= np.dot([h, 1], neg_max_fit)) or
                 (np.dot([h, 1], pos_min_fit) <= p <= np.dot([h, 1], pos_max_fit))]

    if points:
        q_values = []
        for p, h in points:
            q = predict_q_poly(p, h) # predict q based on p & h
            q_values.append(q)

        p, h = zip(*points)  # Unzip the points
        p = np.array(p)
        h = np.array(h)
        q = np.array(q_values)  # q = predict_q_poly(p, h)

        # Define the optimization variables
        c[t] = cp.Variable()
        d[t] = cp.Variable()
        e[t] = cp.Variable()

        objective = cp.Minimize(cp.sum_squares(c[t] * p + d[t] * h + e[t] - q))
        problems[t] = cp.Problem(objective)
        problems[t].solve()

        # Store the lambda function in op
        op[t]['q_TA'] = lambda p_val, h_val, ct=c[t].value, dt=d[t].value, et=e[t].value: ct * p_val + dt * h_val + et
        print(f"Time {t}: Flow model: q = {c[t].value:.4f}*p + {d[t].value:.4f}*h + {e[t].value:.4f}")

        # check if the problem is DPP
        print(f"prob is DCP: {problems[t].is_dcp()}")
        print(f"prob is DGP: {problems[t].is_dgp()}")
        print(f"prob is DPP: {problems[t].is_dpp()}")
    else:
        print(f"Time {t}: No valid points found within constraints.")

# calcualte op['flow'] based on the op['power'] and op['head'] with op['q_TA']
for t in range(Time):
    op[t]['flow'] = op[t]['q_TA'](op[t]['power'], op[t]['head'])
    
# %% UPC trust region linear regression with least square
# UPC trust region linear regression with least square

def least_squares_UPC(p_samples, h_samples, q_values):
    # Form the matrix X and vector y
    X = np.column_stack((p_samples, h_samples, np.ones_like(p_samples)))
    y = np.array(q_values)

    # Compute the coefficients using the normal equation
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta  # Returns coefficients [c, d, e]

c = {}; d = {}; e = {}
problems = {}
UPC_sampling_rate = 400  # 400^2 samples per trust region (before truncated by BC)

for t in range(Time):
    h_samples = np.linspace(op[t]['head'] - δh, op[t]['head'] + δh, UPC_sampling_rate)
    p_samples = np.linspace(op[t]['power'] - δp, op[t]['power'] + δp, UPC_sampling_rate)

    points = [(p, h) for h in h_samples for p in p_samples
              if (np.dot([h, 1], neg_min_fit) <= p <= np.dot([h, 1], neg_max_fit)) or
                 (np.dot([h, 1], pos_min_fit) <= p <= np.dot([h, 1], pos_max_fit))]

    if points:
        p, h = zip(*points)  # Unzip the points
        q_values = [predict_q_poly(p_val, h_val) for p_val, h_val in points]

        # Calculate coefficients using least squares
        beta = least_squares_UPC(np.array(p), np.array(h), np.array(q_values))
        c[t], d[t], e[t] = beta  # Unpack the coefficients

        # Store the lambda function in op
        op[t]['q_TA'] = lambda p_val, h_val, ct=c[t], dt=d[t], et=e[t]: ct * p_val + dt * h_val + et
        print(f"Time {t}: Flow model: q = {c[t]:.4f}*p + {d[t]:.4f}*h + {e[t]:.4f}")
    else:
        print(f"Time {t}: No valid points found within constraints.")

# calculate op['flow'] based on the op['power'] and op['head'] with op['q_TA']
for t in range(Time):
    op[t]['flow'] = op[t]['q_TA'](op[t]['power'], op[t]['head'])

#%% Plot the fitted planes
# Plot the fitted planes
def plot_3d_surface_interactive(x_valid, y_valid, z_valid, model, title, coefficients):
    # Create the main model surface
    x_surf, y_surf = np.meshgrid(np.linspace(x_valid.min(), x_valid.max(), 50),
                                 np.linspace(y_valid.min(), y_valid.max(), 50))
    xy_surf = np.vstack([x_surf.ravel(), y_surf.ravel()]).T
    z_surf = model.predict(xy_surf).reshape(x_surf.shape)
    z_min = z_valid.min()
    z_max = z_valid.max()

    fig = go.Figure(data=[
        go.Scatter3d(x=x_valid, y=y_valid, z=z_valid, mode='markers', name='Original Data', 
                     marker=dict(size=1, color=z_valid, colorscale='Plasma', cmin=z_min, cmax=z_max)),
        go.Surface(x=x_surf, y=y_surf, z=z_surf, name='Fitted Surface', 
                   colorscale='Viridis', cmin=z_min, cmax=z_max, opacity=0.7)
    ])

    # Add smaller planes for each time period
    for t, (ct, dt, et) in enumerate(coefficients):
        # Define the plane within a restricted domain around the operational point
        x_small_surf, y_small_surf = np.meshgrid(
            np.linspace(op[t]['power'] - 0.5, op[t]['power'] + 0.5, 10),
            np.linspace(op[t]['head'] - 5, op[t]['head'] + 5, 10)
        )
        z_small_surf = ct * x_small_surf + dt * y_small_surf + et
        fig.add_trace(go.Surface(
            x=x_small_surf, y=y_small_surf, z=z_small_surf, name=f'Time {t} Plane',
            showscale=False, opacity=0.5
        ))
        
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Power (p)',
            yaxis_title='Head (h)',
            zaxis_title='Flow (q)',
            zaxis=dict(range=[z_min-1, z_max+1])
        )
    )
    fig.show()

# Example of plotting with additional planes
if __name__ == '__main__':
    results_pump = prepare_and_fit_model('./Data/UPCs/temp/Mod_Francis_pump_temp.xlsx')
    results_turbine = prepare_and_fit_model('./Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx')

    # Collect the coefficients from the smaller area fits
    coefficients_pump = [(c[t], d[t], e[t]) for t in range(Time)]
    coefficients_turbine = [(c[t], d[t], e[t]) for t in range(Time)]

    plot_3d_surface_interactive(*results_pump[1:4], results_pump[0], 'Pump Model', coefficients_pump)
    plot_3d_surface_interactive(*results_turbine[1:4], results_turbine[0], 'Turbine Model', coefficients_turbine)


# %% Optimization
# Optimization

# ---------------Basic Parameters---------------

# time step
t_step = cp.Parameter(nonneg=True)
t_step.value = time_step

# assign an operational cost(constant parameter)
operational_cost = 3.8 # €/MWh
C_op = cp.Parameter(nonneg=True)
C_op.value = operational_cost
  
# ---------------Variables---------------
# Impossible to warm start with a initial guess op[t]['power'] from NN (2021.04)
p_cp = cp.Variable(Time) # power for turbine is positive, vise versa.
q_cp = cp.Variable(Time) # flow rate for turbine is positive, vise versa.
h_cp = cp.Variable(Time+1) # power for turbine is positive, vise versa.
v_low_cp = cp.Variable(Time+1) # Lower basin volume

# Initialize numpy arrays to hold the initial values
initial_p = np.zeros(Time)
initial_q = np.zeros(Time)
initial_h = np.zeros(Time+1)

# Populate the arrays with initial values from 'op'
for t in range(Time):
    initial_p[t] = op[t]['power']
    initial_q[t] = op[t]['flow']
    initial_h[t] = op[t]['head']

# Assign these initial values to the CVXPY variables
p_cp.value = initial_p
q_cp.value = initial_q
h_cp.value = initial_h

# ---------------Objective Function---------------
# objective = cp.Minimize(C_op * cp.sum(cp.abs(p_cp)) * t_step) # Linear form
objective = cp.Minimize( C_op * cp.sum_squares(p_cp) ) # Quadratic form

# ---------------Constraints---------------
constraints_UPC = []
constraints_TR = []
constraints_LRA = []
constraints_vol = []
constraints = []

# constraints_UPC based on operational mode and linear regression boundaries
for t in range(Time):
    mode = op[t]['mode']
    match mode:
        case 'turbine':
            # In turbine mode, power must be within the positive power boundaries derived from linear regression
            constraints_UPC += [
                p_cp[t] >= pos_min_fit[0] * h_cp[t] + pos_min_fit[1],  # Power should not be less than the minimum boundary
                p_cp[t] <= pos_max_fit[0] * h_cp[t] + pos_max_fit[1],  # Power should not exceed the maximum boundary
                # p_cp[t] > 0  # Ensure power is positive in turbine mode
                # h_cp[t+1] >= head_min, # Minimum head value
                # h_cp[t+1] <= head_max # Maximum head value
            ]
        
        case 'pump':
            # In pump mode, power must be within the negative power boundaries derived from linear regression
            constraints_UPC += [
                p_cp[t] >= neg_min_fit[0] * h_cp[t] + neg_min_fit[1],  # Power should not exceed the minimum boundary (more negative)
                p_cp[t] <= neg_max_fit[0] * h_cp[t] + neg_max_fit[1],  # Power should not be less than the maximum boundary (less negative)
                # p_cp[t] < 0  # Ensure power is negative in pump mode
                # h_cp[t+1] >= head_min, # Minimum head value
                # h_cp[t+1] <= head_max # Maximum head value
            ]
        
        case 'idle':
            # In idle mode, power and flow should be zero, head within a specific range
            constraints_UPC += [
                p_cp[t] == 0, # Power output should be zero
                q_cp[t] == 0, # Flow rate should be zero
                # h_cp[t] >= head_min, # Minimum head value
                # h_cp[t] <= head_max # Maximum head value
                # h_cp[t+1] == h_cp[t] # Head should remain constant
            ]

    constraints_UPC += [h_cp[t+1] >= head_min, h_cp[t+1] <= head_max]
    
    # constraints_TR for Trust Region
    constraints_TR += [
        p_cp[t] <= op[t]['power'] + δp,
        p_cp[t] >= op[t]['power'] - δp,
        q_cp[t] <= op[t]['flow'] + δq,
        q_cp[t] >= op[t]['flow'] - δq
        # h_cp[t] <= op[t]['head'] + δh,
        # h_cp[t] >= op[t]['head'] - δh,
    ]
    
    # constraints_LRA for Linear Regression Approximation
    constraints_LRA += [
        q_cp[t] == op[t]['q_TA'](p_cp[t], h_cp[t]),
        v_low_cp[t] == op[t]['v_low_TA'](h_cp[t])
    ]

    # Constraints_vol for water volume balance
    # if t > 0:
    constraints_vol += [
        v_low_cp[t+1] == v_low_cp[t] + q_cp[t]*3600, # 3600s in 1 time_step(1h)
    ]

# last_ts_cstr = [v_low_cp[Time] == op[Time]['v_low_TA'](h_cp[Time])]


# merge all constraints
constraints = constraints_UPC + constraints_TR + constraints_LRA + constraints_vol

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.ECOS, verbose=True)
result = prob.value

# Post-processing
print("Optimal value:", result)
for t in range(Time):
    print(f"Time {t}: Power = {p_cp[t].value}, Head = {h_cp[t].value}, Flow = {q_cp[t].value}")

# %%
# ---------------Parameters for First Hour---------------
p_first = cp.Parameter()
q_first = cp.Parameter()
h_first = cp.Parameter()
v_low_first = cp.Parameter()

# Set the values for the first hour

h_first.value = op[0]['head']
v_low_first.value = op[0]['v_low_TA'](h_first.value) 

# ---------------Variables (from second hour onwards)---------------
p_cp = cp.Variable(Time)  # power for turbine is positive, vice versa.
q_cp = cp.Variable(Time)  # flow rate for turbine is positive, vice versa.
h_cp = cp.Variable(Time+1)  # head (includes the last hour)
v_low_cp = cp.Variable(Time+1)  # Lower basin volume (includes the last hour)

# ---------------Objective Function---------------
# Note! start summing from the second hour
objective = cp.Maximize(-t_step * C_op * cp.sum_squares(p_cp))  # Quadratic form

# ---------------Constraints---------------
constraints_UPC = []
constraints_TR = []
constraints_LRA = []
constraints_vol = []

# Set the first hour's head and v_low
constraints = [
    h_cp[0] == h_first,
    v_low_cp[0] == v_low_first
]

# Constraints for hours 2 to Time
for t in range(Time - 1):  # t here represents the index in p_cp, q_cp (so it's actually t+1 in the original time scale)
    mode = op[t+1]['mode']  # Use t+1 to get the correct mode
    
    # constraints_UPC based on operational mode and linear regression boundaries
    match mode:
        case 'turbine':
            constraints_UPC += [
                p_cp[t] >= pos_min_fit[0] * h_cp[t+1] + pos_min_fit[1], # left boundary
                p_cp[t] <= pos_max_fit[0] * h_cp[t+1] + pos_max_fit[1], # right boundary
                h_cp[t+1] >= head_min, # upper boundary
                h_cp[t+1] <= head_max # lower boundary
            ]
        
        case 'pump':
            constraints_UPC += [
                p_cp[t] >= neg_min_fit[0] * h_cp[t+1] + neg_min_fit[1], # left boundary
                p_cp[t] <= neg_max_fit[0] * h_cp[t+1] + neg_max_fit[1], # right boundary
                h_cp[t+1] >= head_min, # upper boundary
                h_cp[t+1] <= head_max # lower boundary
            ]
        
        case 'idle':
            constraints_UPC += [
                p_cp[t] == 0, 
                q_cp[t] == 0,
                h_cp[t+1] == h_cp[t] # Head should remain constant
            ]
    
    # constraints_TR for Trust Region
    constraints_TR += [
        p_cp[t] <= op[t+1]['power'] + δp,
        p_cp[t] >= op[t+1]['power'] - δp,
        h_cp[t+1] <= op[t+1]['head'] + δh,
        h_cp[t+1] >= op[t+1]['head'] - δh,
        q_cp[t] <= op[t+1]['flow'] + δq,
        q_cp[t] >= op[t+1]['flow'] - δq
    ]
    
    # constraints_LRA for Linear Regression Approximation
    constraints_LRA += [
        q_cp[t] == op[t+1]['q_TA'](p_cp[t], h_cp[t+1]),
        v_low_cp[t+1] == op[t+1]['v_low_TA'](h_cp[t+1])
    ]

    # Constraints_vol for water volume balance
    constraints_vol += [
        v_low_cp[t+1] == v_low_cp[t] + q_cp[t]*3600,  # 3600s in 1 time_step(1h)
    ]

# Add constraint for the first hour's volume balance
constraints_vol = [v_low_cp[0] == v_low_first + q_first*3600] + constraints_vol

# Merge all constraints
constraints += constraints_UPC + constraints_TR + constraints_LRA + constraints_vol

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.ECOS)
result = prob.value

# Post-processing
print("Optimal value:", result)
print(f"Time 0: Power = {p_first.value}, Head = {h_first.value}, Flow = {q_first.value}")
for t in range(Time - 1):
    print(f"Time {t+1}: Power = {p_cp[t].value}, Head = {h_cp[t+1].value}, Flow = {q_cp[t].value}")


# %% Modified Optimization
# Modified Optimization 

# Convert PyTorch tensor to numpy array first, then create parameter
DA_price_hour_np = DA_price_hour.numpy()  # Convert torch tensor to numpy array
DA_price_hour_cp = cp.Parameter(24)
DA_price_hour_cp.value = DA_price_hour_np  # Assign numpy array to parameter

# Maximum price as a constant
max_price = np.max(DA_price_hour_np)  # Maximum price from DA prices

target_head = 66.67  # Set target head value here

# Target volume for penalty calculation
target_vol_low = h_to_v_low_fitted(target_head)

# ---------------Basic Parameters---------------

# Physical constants
rho = 1000    # Density of water in kg/m³
g = 9.81      # Gravity in m/s²
mu = 0.9      # Efficiency/penalty factor
time_step = 1  # time step as a constant (1 hour)

# assign an operational cost (constant parameter)
operational_cost = 3.8  # €/MWh
C_op = cp.Parameter(nonneg=True)
C_op.value = operational_cost

# Initial head as a parameter
h_init = cp.Parameter()
h_init.value = 76.96  # Set your desired initial head value here

# Calculate initial v_low using h_to_v_low_fitted
v_low_init = h_to_v_low_fitted(h_init.value)


# ---------------Variables---------------
# Decision variables (24 elements each)
p_cp = cp.Variable(24)  # power for each hour
q_cp = cp.Variable(24)  # flow rate for each hour

# State variables (24 elements each, representing end-of-hour states)
h_cp = cp.Variable(24)    # head at the end of each hour
v_low_cp = cp.Variable(24)  # lower basin volume at the end of each hour

# Initialize arrays with initial values
initial_p = np.zeros(24)
initial_q = np.zeros(24)
initial_h = np.zeros(24)

# Variable for volume deficit
volume_deficit = cp.Variable()  # This will be constrained to be positive

# Populate arrays with initial values from 'op'
for t in range(24):
    initial_p[t] = op[t]['power']
    initial_q[t] = op[t]['flow']
    initial_h[t] = op[t]['head']

# Assign initial values to variables
p_cp.value = initial_p
q_cp.value = initial_q
h_cp.value = initial_h

# ---------------Objective Function---------------
# Revenue term: DA_price_hour_cp @ p_cp * time_step
revenue = DA_price_hour_cp @ p_cp * time_step

# Energy loss from volume deficit (MWh)
energy_loss = volume_deficit * rho * g * target_head * mu / 3600000000

# Volume penalty using maximum DA price
volume_penalty = energy_loss * max_price

# Operational cost term: C_op * cp.sum_squares(p_cp)
operational_costs = C_op * cp.sum_squares(p_cp)
# operational_costs = C_op * cp.sum(cp.abs(p_cp)) # should we take linear form????????????????????

# Complete objective function: maximize revenue - volume_penalty - operational_costs
objective = cp.Maximize(revenue - volume_penalty - operational_costs)


# ---------------Constraints---------------
constraints_UPC = [] # Constraints for the operational mode
constraints_TR = [] # Constraints for the trust region
constraints_LRA = [] # Constraints for the linear regression approximation
constraints_vol = [] #  Constraints for the volume balance
constraints_VD = [] # Constraints for the volume deficit
constraints = []

# Volume deficit constraints
final_volume = v_low_cp[23]  # Volume at the end of the last hour
constraints_VD += [
    volume_deficit >= 0,  # Must be non-negative
    volume_deficit >= final_volume - target_vol_low  # Must be at least as large as the deficit
]

# Constraints for all 24 hours
for t in range(24):
    mode = op[t]['mode']
    
    # Operational mode constraints
    match mode:
        case 'turbine':
            constraints_UPC += [
                p_cp[t] >= pos_min_fit[0] * h_cp[t] + pos_min_fit[1],
                p_cp[t] <= pos_max_fit[0] * h_cp[t] + pos_max_fit[1],
            ]
        
        case 'pump':
            constraints_UPC += [
                p_cp[t] >= neg_min_fit[0] * h_cp[t] + neg_min_fit[1],
                p_cp[t] <= neg_max_fit[0] * h_cp[t] + neg_max_fit[1],
            ]
        
        case 'idle':
            constraints_UPC += [
                p_cp[t] == 0,
                q_cp[t] == 0
            ]
    
    # Head limits for all modes
    constraints_UPC += [
        h_cp[t] >= head_min,
        h_cp[t] <= head_max
    ]
    
    # Trust region constraints
    constraints_TR += [
        p_cp[t] <= op[t]['power'] + δp,
        p_cp[t] >= op[t]['power'] - δp,
        q_cp[t] <= op[t]['flow'] + δq,
        q_cp[t] >= op[t]['flow'] - δq
    ]
    
    # Linear regression approximation constraints
    constraints_LRA += [
        q_cp[t] == op[t]['q_TA'](p_cp[t], h_cp[t]),
        v_low_cp[t] == op[t]['v_low_TA'](h_cp[t])
    ]
    
    # Volume balance constraints
    if t > 0:
        constraints_vol += [
            v_low_cp[t] == v_low_cp[t-1] + q_cp[t] * 3600  # Convert flow rate (m³/s) to volume (m³)
        ]
    else:
        # For the first hour, use the initial volume calculated from h_init
        constraints_vol += [
            v_low_cp[0] == v_low_init + q_cp[0] * 3600  # Convert flow rate (m³/s) to volume (m³)
        ]

# Merge all constraints
constraints = constraints_UPC + constraints_TR + constraints_LRA + constraints_vol + constraints_VD

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.ECOS)
result = prob.value

# Post-processing
print(f"Initial head: {h_init.value}")
print(f"Initial lower basin volume: {v_low_init}")
print(f"Target head: {target_head}")
print(f"Target volume: {target_vol_low}")
print(f"Maximum DA price: {max_price}")
print("Optimal value:", result)
for t in range(24):
    print(f"Hour {t+1}: Power = {p_cp[t].value}, Head = {h_cp[t].value}, Flow = {q_cp[t].value}")

# %%
# draw lines with neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit
x = np.linspace(head_min, head_max, 100)
y1 = pos_min_fit[0] * x + pos_min_fit[1]
y2 = pos_max_fit[0] * x + pos_max_fit[1]
y3 = neg_min_fit[0] * x + neg_min_fit[1]
y4 = neg_max_fit[0] * x + neg_max_fit[1]

plt.figure(figsize=(12, 8))
plt.plot(x, y1, label='Positive Min', color='b', linestyle='--')
plt.plot(x, y2, label='Positive Max', color='g', linestyle='--')
plt.plot(x, y3, label='Negative Min', color='r', linestyle='--')
plt.plot(x, y4, label='Negative Max', color='black', linestyle='--')
plt.xlabel('Head (h)')
plt.ylabel('Power (p)')
plt.title('Linear Regression Boundaries for UPC')
plt.legend()
plt.grid(True)
plt.show()

#%% DPP check

# check if the problem is DPP
print(f"prob is DCP: {prob.is_dcp()}")
print(f"prob is DGP: {prob.is_dgp()}")
print(f"prob is DPP: {prob.is_dpp()}")
# %%

# Function to evaluate and print the properties of constraints
def check_constraint_properties(constraint_list, description):
    print(f"Checking properties for {description}:")
    for constraint in constraint_list:
        print(f"  DCP: {constraint.is_dcp()}, DGP: {constraint.is_dgp()}, DPP: {constraint.is_dpp()}")

# Apply the function to each list of constraints
check_constraint_properties(constraints_UPC, "UPC Constraints")
check_constraint_properties(constraints_TR, "Trust Region Constraints")
check_constraint_properties(constraints_LRA, "Linear Regression Approximation Constraints")
check_constraint_properties(constraints_vol, "Volume Balance Constraints")
check_constraint_properties(constraints, "All Constraints")

# check for the objective function
print(f"Objective Function DCP: {objective.is_dcp()}, DGP: {objective.is_dgp()}, DPP: {objective.is_dpp()}")

# %%
# Test op['q_TA'] and op['v_low_TA'] with the op['power'] and op['head']
for t in range(Time):
    print(f"Time {t}: Flow = {op[t]['q_TA'](op[t]['power'], op[t]['head'])}, V_low = {op[t]['v_low_TA'](op[t]['head'])}")
    
#%% debug
# Initialize a flag to check if all components are DGP compliant
all_dgp = True

# Check if the objective function is DGP
if objective.is_dgp():
    print("The objective function is DGP compliant.")
else:
    print("The objective function is NOT DGP compliant.")
    all_dgp = False

# Check each constraint for DGP compliance
for index, constraint in enumerate(constraints):
    if constraint.is_dgp():
        print(f"Constraint {index} is DGP compliant.")
    else:
        print(f"Constraint {index} is NOT DGP compliant.")
        all_dgp = False

# Report overall compliance
if all_dgp:
    print("All components are DGP compliant.")
else:
    print("Some components are NOT DGP compliant.")


# %% 
# Taylor approximation for q_TA(p,h) with CVXPY layers
def create_q_taylor_layers(num_samples, p_samples, h_samples, q_values):
    q_layers = []
    for t in range(num_samples):
        p = cp.Parameter(shape=(len(p_samples[t]),))
        h = cp.Parameter(shape=(len(h_samples[t]),))
        q = cp.Parameter(shape=(len(q_values[t]),))
        
        # Coefficients of the linear function
        a = cp.Variable()
        b = cp.Variable()
        c = cp.Variable()
        
        # Linear approximation: q = a*p + b*h + c
        objective = cp.Minimize(cp.sum_squares(a * p + b * h + c - q))
        constraints = []
        
        # Define and solve the CVXPY problem
        problem = cp.Problem(objective, constraints)
        layer = CvxpyLayer(problem, parameters=[p, h, q], variables=[a, b, c])
        q_layers.append(layer)
    
    return q_layers

# Taylor approximation for v_low_TA(h) with CVXPY layers
def create_v_low_taylor_layers(num_samples, h_samples, v_low_values):
    v_low_layers = []
    for t in range(num_samples):
        h = cp.Parameter(shape=(len(h_samples[t]),))
        v_low = cp.Parameter(shape=(len(v_low_values[t]),))
        
        # Coefficients of the linear function
        d = cp.Variable()
        e = cp.Variable()
        
        # Linear approximation: v_low = d*h + e
        objective = cp.Minimize(cp.sum_squares(d * h + e - v_low))
        constraints = []
        
        # Define and solve the CVXPY problem
        problem = cp.Problem(objective, constraints)
        layer = CvxpyLayer(problem, parameters=[h, v_low], variables=[d, e])
        v_low_layers.append(layer)
    
    return v_low_layers

#%%
class TaylorApproximationNetwork(torch.nn.Module):
    def __init__(self, head_op, power_op, delta_h, delta_p, num_samples=100):
        super().__init__()
        self.head_op = head_op
        self.power_op = power_op
        self.delta_h = delta_h
        self.delta_p = delta_p
        self.num_samples = num_samples

        self.q_layers = self.create_q_taylor_layers()
        self.v_low_layers = self.create_v_low_taylor_layers()

    def create_q_taylor_layers(self):
        q_layers = []
        for t in range(len(self.head_op)):
            h_samples = np.linspace(max(head_min, self.head_op[t] - self.delta_h), 
                                    min(head_max, self.head_op[t] + self.delta_h), self.num_samples)
            p_samples = np.linspace(self.power_op[t] - self.delta_p, 
                                    self.power_op[t] + self.delta_p, self.num_samples) # torch.linspace
            q_values = [predict_q_poly(p, h) for p, h in zip(p_samples, h_samples)]
            
            c, d, e = cp.Variable(), cp.Variable(), cp.Variable()
            p = cp.Parameter(shape=(self.num_samples,), value=p_samples)
            h = cp.Parameter(shape=(self.num_samples,), value=h_samples)
            q = cp.Parameter(shape=(self.num_samples,), value=q_values)

            objective = cp.Minimize(cp.sum_squares(c * p + d * h + e - q))
            constraints = []
            problem = cp.Problem(objective, constraints)
            
            # Diagnostic prints
            print("Problem parameters:", problem.parameters())
            print("Passing to layer:", [p, h, q])
            
            q_layers.append(CvxpyLayer(problem, [p, h, q], [c * p + d * h + e]))
        return q_layers

    def create_v_low_taylor_layers(self):
        v_low_layers = []
        for t in range(len(self.head_op)):
            h_samples = np.linspace(max(head_min, self.head_op[t] - self.delta_h), 
                                    min(head_max, self.head_op[t] + self.delta_h), self.num_samples)
            v_low_values = h_to_v_low_fitted(h_samples)
            
            a, b = cp.Variable(), cp.Variable()
            h = cp.Parameter(shape=(self.num_samples,), value=h_samples)
            v_low = cp.Parameter(shape=(self.num_samples,), value=v_low_values)

            objective = cp.Minimize(cp.sum_squares(a * h + b - v_low))
            problem = cp.Problem(objective)
            v_low_layers.append(CvxpyLayer(problem, [h], [a*h + b]))
        return v_low_layers

    def forward(self, head_op, power_op):
        q_approximations = [layer(h, p)[0] for layer, h, p in zip(self.q_layers, head_op, power_op)]
        v_low_approximations = [layer(h)[0] for layer, h in zip(self.v_low_layers, head_op)]
        return q_approximations, v_low_approximations
    
# %%
# Create the Taylor approximation network
taylor_network = TaylorApproximationNetwork(head_op, power_op, δh, δp)
q_approximations, v_low_approximations = taylor_network.forward(head_op, power_op)

# Print outputs
print("Q Approximations:", q_approximations)
print("V_low Approximations:", v_low_approximations)