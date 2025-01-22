# %% import data
# Initialization
import torch
import torch.nn as nn
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
import dill as pickle
from scipy.optimize import fsolve

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

print(target_head)

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

#%%
# test
head_example = 80  # Example head value 
print(f"Maximum Positive Power: {pos_max(head_example)}")



#%%
# Testing data
p = torch.tensor([
    -6.77, -7.01, -7.32, -7.63, -7.95, -8.26, -8.19, 
     4.27,  4.11,  4.43,  4.23,  4.01,  3.78,  3.55, 
     3.37,  3.3,   3.23,  4.17,  4.8,   4.55,  3.91, 
     3.66,  2.64,  2.57
])

q = torch.tensor([
    -10.24, -10.14, -10.12, -10.11, -10.11, -10.18, -9.79, 
     5.55,   5.48,   6.16,   6.06,   5.87,   5.61,   5.36, 
     5.19,   5.21,   5.24,   7.15,   8.95,   8.53,   7.04, 
     6.64,   4.68,   4.61
])

h = torch.tensor([
    76.96, 79.39, 81.82, 84.25, 86.67, 89.12, 91.47, 
    90.13, 88.82, 87.34, 85.89, 84.48, 83.13, 81.85,     
    80.6,  79.35, 78.09, 76.37, 74.23, 72.18, 70.49, 
    68.9,  67.77, 66.67
])


# %%
# Initial reservoir state
v_low_clb = torch.zeros(1441)

# solve for the initial v_low from h (gross head), outside pipline
ini_gross_head = h[0]  # h[0] is initial gross head

def hv_eq(v_low, target_gross_head = ini_gross_head):
    return gross_head(h_up=None, h_low=None, v_up=None, v_low=v_low) - target_gross_head

initial_guess = 80.794  # half-full as initial guess
v_low_clb[0] = torch.tensor(fsolve(hv_eq, initial_guess), dtype=torch.float32)

#%%
# sim test
p_sim = p.repeat_interleave(60)
q_sim = q.repeat_interleave(60)
h_sim = h.repeat_interleave(60)

q_sim_clb = torch.zeros(1441) 
h_sim_clb = torch.zeros(1441)

# Calibrate power ouputs with ramping capacity
p_sim_clb = p_sim.clone()

# Add the new element to the tensor(time 0 next day)
p_sim_clb = torch.cat([p_sim_clb, p_sim_clb[-1].unsqueeze(0)])

# Idle for 1 min before change modes
for i in range(1440, 0, -1):
    if p_sim_clb[i] * p_sim_clb[i-1] < 0: 
        p_sim_clb[i-1] = 0

#%%
# Iterate backwards through the day, adjusting power values
for hour in range(23, -1, -1):  # from 23 (last hour) to 0 (first hour)
    hour_start = hour * 60
    hour_end = hour_start + 60 
    
    # Ensure the first minute of the hour matches p[hour]
    p_sim_clb[hour_start] = p[hour]
    
    # Backward adjustment for the rest of the hour
    for i in range(hour_end, hour_start, -1):
        if p_sim_clb[i] - p_sim_clb[i-1] > ramp_down:
            p_sim_clb[i-1] = p_sim_clb[i] - ramp_down
        elif p_sim_clb[i] - p_sim_clb[i-1] < -ramp_up:
            p_sim_clb[i-1] = p_sim_clb[i] + ramp_up

# p_sim_clb now contains the calibrated power values
print(p_sim_clb)

#%%
# Calibrate with real reservoir properties
q_sim_clb[0] = q_sim[0]
h_sim_clb[0] = h_sim[0]
p_sim_clb[0] = p_sim[0]

for i in range(1440):
    # Turbine mode
    if p_sim_clb[i] > 0:
        
        # Predict the flow and head based on the current state
        if p_sim_clb[i] > pos_min(h_sim_clb[i]) and p_sim_clb[i] < pos_max(h_sim_clb[i]):
            q_sim_clb[i] = predict_q_poly(p_sim_clb[i], h_sim_clb[i])
        elif p_sim_clb[i] < pos_min(h_sim_clb[i]):
            p_sim_clb[i] = pos_min(h_sim_clb[i])
            q_sim_clb[i] = predict_q_poly(p_sim_clb[i], h_sim_clb[i])
        elif p_sim_clb[i] > pos_max(h_sim_clb[i]):
            p_sim_clb[i] = pos_max(h_sim_clb[i])
            q_sim_clb[i] = predict_q_poly(p_sim_clb[i], h_sim_clb[i])
    
    # Pump mode
    elif p_sim_clb[i] < 0:
        
        if p_sim_clb[i] > neg_min(h_sim_clb[i]) and p_sim_clb[i] < neg_max(h_sim_clb[i]):
            q_sim_clb[i] = predict_q_poly(p_sim_clb[i], h_sim_clb[i])
        elif p_sim_clb[i] < neg_min(h_sim_clb[i]):
            p_sim_clb[i] = neg_min(h_sim_clb[i])
            q_sim_clb[i] = predict_q_poly(p_sim_clb[i], h_sim_clb[i])
        elif p_sim_clb[i] > neg_max(h_sim_clb[i]):
            p_sim_clb[i] = neg_max(h_sim_clb[i])
            q_sim_clb[i] = predict_q_poly(p_sim_clb[i], h_sim_clb[i])
    
    
    # Update the volume of the lower reservoir
    v_low_clb[i+1] = v_low_clb[i] + q_sim_clb[i] * 60  # assuming q_sim_clb is in m^3/s, convert to m^3/min
    
    # Check if v_low_clb is within limits
    if v_low_clb[i+1] > max_vol_up or v_low_clb[i+1] < min_vol_low:
        # Set to idle mode if out of bounds
        p_sim_clb[i] = 0
        q_sim_clb[i] = 0
        h_sim_clb[i+1] = h_sim_clb[i]
        v_low_clb[i+1] = v_low_clb[i]  # No change in volume
    else:
        # Update head for valid volume
        h_sim_clb[i+1] = gross_head(v_low=v_low_clb[i+1])

#%%
# debug test
print(predict_q_poly(-6.7513,77),
    neg_min(77),
    neg_max(77))

# %%
# Simulation Profit
def calculate_simulation_profit(
        DA_price_quarter=DA_price_quarter, 
        p_sim_clb=p_sim_clb, 
        p_opti_hour=p, 
        surplus_penalty_multiplier=-1/2, 
        shortage_penalty_multiplier=-2,
        v_low_clb=v_low_clb,
        target_vol_low=target_vol_low,
        target_head=target_head,
        rho=1000,  # Density of water in kg/m3
        g=9.81,    # Gravity in m/s^2
        mu=0.9,    # Efficiency or penalty factor
        C_op = 3.8 # Operating cost in EUR/MWh
        ):

    # Truncate the last element of p_sim_clb
    p_sim_clb = p_sim_clb[:-1]  # Remove the last element
    
    # Expand p_opti_hour from hourly to minute-wise by repeating each value 60 times
    p_opti_minute = p_opti_hour.repeat_interleave(60)
    
    # Sum every 15 minutes to aggregate the minute-wise data to quarter-hourly totals
    e_sim_quarter = p_sim_clb.view(-1, 15).sum(dim=1) * 0.25  # Convert MW to MWh for each quarter
    e_opti_quarter = p_opti_minute.view(-1, 15).sum(dim=1) * 0.25  # Convert MW to MWh for each quarter
    
    # Calculate the revenue for each quarter-hour
    revenue_per_quarter = DA_price_quarter * e_sim_quarter  # Revenue calculation in EUR, adjusted for MWh
    
    # Determine the System Imbalance (SI) price 
    SI_price = torch.where(e_sim_quarter < e_opti_quarter,  # Shortage in simulation
                           shortage_penalty_multiplier * DA_price_quarter,  # Lower output penalty
                           surplus_penalty_multiplier * DA_price_quarter)  # Higher output penalty

    # Calculate the penalty for each quarter-hour
    penalty_per_quarter = (e_sim_quarter - e_opti_quarter) * SI_price  # Penalty calculation adjusted for MWh
    
    # Sum the penalties over all quarter-hours to get the total penalty
    SI_penalty = penalty_per_quarter.sum()

    # Calculate volume penalty
    volume_deficit = max(0, v_low_clb[-1] - target_vol_low)  # Ensure no penalty if above target?
    energy_loss = rho * volume_deficit * g * target_head * mu / 3600000000  # Convert from J to MWh
    volume_penalty = energy_loss * max(DA_price_quarter)

    # Calcolate the operating cost
    operating_cost = C_op * torch.sum(p_sim_clb ** 2) / 60
    
    # Deduct the total penalty from the revenue to get the total daily profit
    total_daily_profit = revenue_per_quarter.sum() - SI_penalty - volume_penalty - operating_cost

    return total_daily_profit, SI_penalty, e_sim_quarter , e_opti_quarter

# %%

# Create arrays for plotting
p_sim_clb = p_sim_clb[:1440].detach().numpy()
v_low_clb = v_low_clb[:1440].detach().numpy()
q_sim_clb = q_sim_clb[:1440].detach().numpy()
h_sim_clb = h_sim_clb[:1440].detach().numpy()
upper_vol_clb = max_vol_up - v_low_clb

# Setup the figure and subplots
plt.figure(figsize=(12, 10))

# Plot p_sim_clb - Power Simulation over Time
plt.subplot(4, 1, 1)
plt.plot(p_sim_clb, label='Power Simulation', color='red')
plt.xlabel('Time (minutes)')
plt.ylabel('Power (MW)')
plt.title('Power Simulation over Time')
plt.legend()

# Plot q_sim_clb - Flow Simulation over Time
plt.subplot(4, 1, 2)
plt.plot(q_sim_clb, label='Flow Simulation', color='blue')
plt.xlabel('Time (minutes)')
plt.ylabel('Flow (m³/s)')
plt.title('Flow Simulation over Time')
plt.legend()

# Plot h_sim_clb - Net Head Simulation over Time
plt.subplot(4, 1, 3)
plt.plot(h_sim_clb, label='Net Head Simulation', color='green')
plt.xlabel('Time (minutes)')
plt.ylabel('Net Head (m)')
plt.title('Net Head Simulation over Time')
plt.legend()

# Plot v_low_clb and upper_vol_clb - Basin Volume Simulation over Time
plt.subplot(4, 1, 4)
plt.plot(v_low_clb, label='Lower Basin Volume', color='magenta')
plt.plot(upper_vol_clb, label='Upper Basin Volume', color='cyan')
plt.xlabel('Time (minutes)')
plt.ylabel('Volume (m³)')
plt.title('Basin Volume Simulation over Time')
plt.legend()

# Adjust the layout and save as SVG
plt.tight_layout()
plt.savefig("simulation_plots.svg", format='svg')  # Save the figure as SVG
plt.show()  # Show the plot
