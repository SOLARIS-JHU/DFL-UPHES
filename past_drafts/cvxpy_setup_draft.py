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
        'power': power_op[t], 
        'head': head_op[t], # t * 2 + 50-2,     # head obtained from NN
        'flow': flow_op[t] # t * 10  # flow rate obtained from NN
    }

# Deviation for trust region in optimization
δp = 5  # MW
δh = 20    # m
δq = 7  # m^3/s

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
UPC_sampling_rate = 100  # 100^2 samples per trust region (before truncated by BC)

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

# # Variable for volume deficit
# volume_deficit = cp.Variable()  # This will be constrained to be positive

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

# # Energy loss from volume deficit (MWh)
# energy_loss = volume_deficit * rho * g * target_head * mu / 3600000000

# Operational cost term(quadratic): C_op * cp.sum_squares(p_cp)
operational_costs = C_op * cp.sum_squares(p_cp)


# Complete objective function: maximize revenue - operational_costs
objective = cp.Maximize(revenue - operational_costs)


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
    final_volume <= target_vol_low  # Must be at least as large as the deficit
]

# Constraints for all 24 hours
for t in range(24):
    
    # Operational mode constraints
    if op[t]['power'] > 0:
        constraints_UPC += [
            p_cp[t] >= pos_min_fit[0] * h_cp[t] + pos_min_fit[1],
            p_cp[t] <= pos_max_fit[0] * h_cp[t] + pos_max_fit[1],
            q_cp[t] == op[t]['q_TA'](p_cp[t], h_cp[t])
        ]
    
    elif op[t]['power'] < 0:
        constraints_UPC += [
            p_cp[t] >= neg_min_fit[0] * h_cp[t] + neg_min_fit[1],
            p_cp[t] <= neg_max_fit[0] * h_cp[t] + neg_max_fit[1],
            q_cp[t] == op[t]['q_TA'](p_cp[t], h_cp[t])
        ]
    
    else:
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
        # q_cp[t] <= op[t]['flow'] + δq,
        # q_cp[t] >= op[t]['flow'] - δq,
        h_cp[t] <= op[t]['head'] + δh,
        h_cp[t] >= op[t]['head'] - δh
    ]
    
    # Linear regression approximation constraints
    constraints_LRA += [
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
    print(f"Hour {t+1}: Power = {p_cp[t].value}, Head = {h_cp[t].value}, Flow = {q_cp[t].value}, v_low = {v_low_cp[t].value}")


# %%
# Print the inital head, initial lower basin volume, target head, target volume, and maximum DA price
print(f"Initial head: {h_init.value}")
print(f"Initial lower basin volume: {v_low_init}")
print(f"Target head: {target_head}")
print(f"Target volume: {target_vol_low}")
print(f"Maximum DA price: {max_price}")

# %%
