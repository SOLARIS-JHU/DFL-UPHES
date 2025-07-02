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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol

# %% 
# Read DA Prices

def read_da_price(date, file_path="./Data/Day-ahead Prices_202301010000-202401010000-2.csv"):
    """
    Input: "MM.DD.YYYY", "./Data/Day-ahead Prices_202301010000-202401010000-2.csv"
    """
    # Load the data
    data = pd.read_csv(file_path)
    
    # Convert the date string to datetime format for easier filtering
    data['Date'] = pd.to_datetime(data['MTU (CET/CEST)'].str[:10], format='%d.%m.%Y')
    
    # Filter the data for the given date
    filtered_data = data[data['Date'] == date]['Day-ahead Price [EUR/MWh]']
    
    # Convert the series to a torch tensor
    tensor_data = torch.tensor(filtered_data.values, dtype=torch.float).to(device)
    
    return tensor_data

def hourly_to_quarterly(tensor_data):
    # Repeat each element in the tensor 4 times
    quarterly_data = tensor_data.repeat_interleave(4)
    return quarterly_data

sample_date = "04.25.2023"
DA_price_hour = read_da_price(sample_date)
DA_price_quarter = hourly_to_quarterly(DA_price_hour)

# %% Linear regression on UPC boundaries (outside thepipeline)
# Linear regression on UPC boundaries
# Note: Only works for UPCs with boundaries as 👇
# --------------------------------------------------------------------->p
# -10         /       /           50|         \            \        10
#            /       /              |          \            \
#           /       /               |           \            \
#          /       /                |            \            \
#         /       /                 |             \            \
#        /       /                  |              \            \
#       /       /                   |               \            \
#      /_______/                  99|                \____________\
#                                   |
#                                  h↓

def get_UPC_bound():
    """
    Load boundary data for pump and turbine operations from an Excel file.

    Returns:
        tuple: Contains two numpy arrays:
            - boundaries_neg: Array of [head, min value, max value] for pump operation,
            - boundaries_pos: Array of [head, min value, max value] for turbine operation.
    """
    # Designated file path & load data
    current_dir = Path(__file__).parent
    UPC_file_path = current_dir / 'Data/UPCs/Mod_Francis_joint.xlsx'
    UPC_df = pd.read_excel(UPC_file_path, sheet_name='Flow rate')

    # Extracting boundaries from DataFrame
    h_values = UPC_df.iloc[:, 0]
    p_values = np.linspace(-10, 10, UPC_df.shape[1] - 1)
    boundaries_neg = []
    boundaries_pos = []

    # Determine non-NaN boundaries for each h
    for i, row in UPC_df.iterrows():
        non_nan_indices = np.where(~pd.isna(row[1:]))[0]
        if (non_nan_indices.size > 0) and (non_nan_indices[0] < len(p_values) // 2):
            min_p_neg = p_values[non_nan_indices[0]]
            max_p_neg = p_values[non_nan_indices[non_nan_indices < (len(p_values) // 2)][-1]]
            min_p_pos = p_values[non_nan_indices[non_nan_indices > (len(p_values) // 2)][0]]
            max_p_pos = p_values[non_nan_indices[-1]]
            boundaries_neg.append((h_values[i], min_p_neg, max_p_neg))
            boundaries_pos.append((h_values[i], min_p_pos, max_p_pos))

    boundaries_neg = np.array(boundaries_neg)
    boundaries_pos = np.array(boundaries_pos)
    return boundaries_neg, boundaries_pos

def LR_UPC_bound(boundaries_neg, boundaries_pos):
    """
    Perform linear regression to find coefficients for the minimum and maximum
    power boundaries for both negative (pump mode) and positive (turbine mode) power values.

    Args:
        boundaries_neg (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for positive power mode.

    Returns:
        tuple: 
        - h_fit (numpy.ndarray): Array of the range of head values used for fitting.
        - p_neg_min_fit, p_neg_max_fit (tuple): Coefficients of the fitted line (slope, intercept)
            for the minimum and maximum negative power boundaries.
        - p_pos_min_fit, p_pos_max_fit (tuple): Coefficients of the fitted line (slope, intercept)
            for the minimum and maximum positive power boundaries.
    """

    # Perform linear fitting
    h_fit = np.array([min(boundaries_neg[:, 0]), max(boundaries_neg[:, 0])])
    p_neg_min_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 1], 1)
    p_neg_max_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 2], 1)
    p_pos_min_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 1], 1)
    p_pos_max_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 2], 1)

    return h_fit, p_neg_min_fit, p_neg_max_fit, p_pos_min_fit, p_pos_max_fit

def plot_UPC_boundaries(boundaries_neg, boundaries_pos, h_fit, p_neg_min_fit, p_neg_max_fit, p_pos_min_fit, p_pos_max_fit):
    """
    Plot both actual and fitted data regions based on boundary data for pump and turbine operational modes.

    Args:
        boundaries_neg (numpy.ndarray): Array containing actual boundary data for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing actual boundary data for positive power mode.
        h_fit (numpy.ndarray): Array of the range of head values used for plotting the fitted lines.
        p_neg_min_fit (tuple): Coefficients (slope, intercept) of the fitted line for the minimum negative power boundary.
        p_neg_max_fit (tuple): Coefficients (slope, intercept) of the fitted line for the maximum negative power boundary.
        p_pos_min_fit (tuple): Coefficients (slope, intercept) of the fitted line for the minimum positive power boundary.
        p_pos_max_fit (tuple): Coefficients (slope, intercept) of the fitted line for the maximum positive power boundary.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate fitted regions using the linear coefficients
    neg_min_line = np.polyval(p_neg_min_fit, h_fit)
    neg_max_line = np.polyval(p_neg_max_fit, h_fit)
    pos_min_line = np.polyval(p_pos_min_fit, h_fit)
    pos_max_line = np.polyval(p_pos_max_fit, h_fit)
    
    # Plotting actual data regions
    for i in range(len(boundaries_neg) - 1):
        # Ensure only one legend entry for actual regions
        ax.fill_betweenx([boundaries_neg[i][0], boundaries_neg[i+1][0]],
                         [boundaries_neg[i][1], boundaries_neg[i+1][1]],
                         [boundaries_neg[i][2], boundaries_neg[i+1][2]], 
                         color='blue', alpha=0.3, edgecolor='none', 
                         label='Actual Region (p<0)' if i == 0 else "")
        ax.fill_betweenx([boundaries_pos[i][0], boundaries_pos[i+1][0]],
                         [boundaries_pos[i][1], boundaries_pos[i+1][1]],
                         [boundaries_pos[i][2], boundaries_pos[i+1][2]], 
                         color='red', alpha=0.3, edgecolor='none', 
                         label='Actual Region (p>0)' if i == 0 else "")

    # Draw fitted regions with borders
    ax.fill(np.append(neg_min_line, neg_max_line[::-1]), np.append(h_fit, h_fit[::-1]),
            color='cyan', alpha=0.5, edgecolor='blue', linewidth=2, label='Fitted Region (p<0)')
    ax.fill(np.append(pos_min_line, pos_max_line[::-1]), np.append(h_fit, h_fit[::-1]),
            color='orange', alpha=0.5, edgecolor='red', linewidth=2, label='Fitted Region (p>0)')

    # Adding legends with specific settings
    legend = ax.legend(frameon=True, framealpha=1, edgecolor='black')
    for legobj in legend.legend_handles:
        legobj.set_linewidth(1.3)  # Set uniform line width for fitted region markers

    # Adding labels and titles
    ax.set_xlabel('Power p')
    ax.set_ylabel('Head h')
    ax.set_title('Actual and Fitted Data Regions in p-h Space')

    # Show plot
    plt.show()

# If this script is the main entry point, execute the plot function
if __name__ == '__main__':
    boundaries_neg, boundaries_pos = get_UPC_bound()
    h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit = LR_UPC_bound(boundaries_neg, boundaries_pos)
    plot_UPC_boundaries(boundaries_neg, boundaries_pos, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit)

# neg_min_line = np.polyval(neg_min_fit, h_fit)

# %% Linear regression(high poly for simulation) on UPC boundaries (outside thepipeline)
# Linear regression on UPC boundaries
# Note: Only works for UPCs with boundaries as 👇
# --------------------------------------------------------------------->p
# -10         /       /           50|         \            \        10
#            /       /              |          \            \
#           /       /               |           \            \
#          /       /                |            \            \
#         /       /                 |             \            \
#        /       /                  |              \            \
#       /       /                   |               \            \
#      /_______/                  99|                \____________\
#                                   |
#                                  h↓

def poly_LR_UPC_bound(boundaries_neg, boundaries_pos, degree=4):
    """
    Perform polynomial regression to find coefficients for the minimum and maximum
    power boundaries for both negative (pump mode) and positive (turbine mode) power values.

    Args:
        boundaries_neg (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for positive power mode.
        degree (int): Degree of the polynomial model to be fitted.

    Returns:
        tuple: 
        - h_poly_fit (numpy.ndarray): Array of the range of head values used for fitting.
        - poly_neg_min_fit, poly_neg_max_fit (tuple): Coefficients of the fitted polynomial
            for the minimum and maximum negative power boundaries.
        - poly_pos_min_fit, poly_pos_max_fit (tuple): Coefficients of the fitted polynomial
            for the minimum and maximum positive power boundaries.
    """

    h_poly_fit = np.linspace(min(boundaries_neg[:, 0]), max(boundaries_neg[:, 0]), 100)
    poly_neg_min_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 1], degree)
    poly_neg_max_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 2], degree)
    poly_pos_min_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 1], degree)
    poly_pos_max_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 2], degree)

    poly_neg_min_fit = torch.tensor(poly_neg_min_fit, dtype=torch.float32, device=device)
    poly_neg_max_fit = torch.tensor(poly_neg_max_fit, dtype=torch.float32, device=device)
    poly_pos_min_fit = torch.tensor(poly_pos_min_fit, dtype=torch.float32, device=device)
    poly_pos_max_fit = torch.tensor(poly_pos_max_fit, dtype=torch.float32, device=device)

    return h_poly_fit, poly_neg_min_fit, poly_neg_max_fit, poly_pos_min_fit, poly_pos_max_fit

def LR_UPC_bound(boundaries_neg, boundaries_pos, degree=5):
    """
    Perform polynomial regression to find coefficients for the minimum and maximum
    power boundaries for both negative (pump mode) and positive (turbine mode) power values.

    Args:
        boundaries_neg (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for positive power mode.
        degree (int): Degree of the polynomial model to be fitted.

    Returns:
        tuple: 
        - h_fit (numpy.ndarray): Array of the range of head values used for fitting.
        - p_neg_min_fit, p_neg_max_fit (tuple): Coefficients of the fitted polynomial
            for the minimum and maximum negative power boundaries.
        - p_pos_min_fit, p_pos_max_fit (tuple): Coefficients of the fitted polynomial
            for the minimum and maximum positive power boundaries.
    """

    # Perform polynomial fitting
    h_fit = np.linspace(min(boundaries_neg[:, 0]), max(boundaries_neg[:, 0]), 100)
    p_neg_min_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 1], degree)
    p_neg_max_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 2], degree)
    p_pos_min_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 1], degree)
    p_pos_max_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 2], degree)

    return h_fit, p_neg_min_fit, p_neg_max_fit, p_pos_min_fit, p_pos_max_fit

def plot_poly_UPC_boundaries(boundaries_neg, boundaries_pos, h_poly_fit, poly_neg_min_fit, poly_neg_max_fit, poly_pos_min_fit, poly_pos_max_fit):
    """
    Plot both actual and fitted data regions based on boundary data for pump and turbine operational modes using polynomial fits.

    Args:
        boundaries_neg (numpy.ndarray): Array containing actual boundary data for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing actual boundary data for positive power mode.
        h_poly_fit (numpy.ndarray): Array of the range of head values used for plotting the fitted polynomials.
        poly_neg_min_fit (tuple): Coefficients (slope, intercept) of the fitted polynomial for the minimum negative power boundary.
        poly_neg_max_fit (tuple): Coefficients (slope, intercept) of the fitted polynomial for the maximum negative power boundary.
        poly_pos_min_fit (tuple): Coefficients (slope, intercept) of the fitted polynomial for the minimum positive power boundary.
        poly_pos_max_fit (tuple): Coefficients (slope, intercept) of the fitted polynomial for the maximum positive power boundary.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate fitted regions using the polynomial coefficients
    neg_min_line = np.polyval(poly_neg_min_fit.cpu().numpy(), h_poly_fit)
    neg_max_line = np.polyval(poly_neg_max_fit.cpu().numpy(), h_poly_fit)
    pos_min_line = np.polyval(poly_pos_min_fit.cpu().numpy(), h_poly_fit)
    pos_max_line = np.polyval(poly_pos_max_fit.cpu().numpy(), h_poly_fit)
    
    # Plotting actual data regions
    for i in range(len(boundaries_neg) - 1):
        ax.fill_betweenx([boundaries_neg[i][0], boundaries_neg[i+1][0]],
                         [boundaries_neg[i][1], boundaries_neg[i+1][1]],
                         [boundaries_neg[i][2], boundaries_neg[i+1][2]], 
                         color='blue', alpha=0.3, edgecolor='none', 
                         label='Actual Region (p<0)' if i == 0 else "")
        ax.fill_betweenx([boundaries_pos[i][0], boundaries_pos[i+1][0]],
                         [boundaries_pos[i][1], boundaries_pos[i+1][1]],
                         [boundaries_pos[i][2], boundaries_pos[i+1][2]], 
                         color='red', alpha=0.3, edgecolor='none', 
                         label='Actual Region (p>0)' if i == 0 else "")

    # Draw fitted regions with borders
    ax.fill(np.append(neg_min_line, neg_max_line[::-1]), np.append(h_poly_fit, h_poly_fit[::-1]),
            color='cyan', alpha=0.5, edgecolor='blue', linewidth=2, label='Fitted Region (p<0)')
    ax.fill(np.append(pos_min_line, pos_max_line[::-1]), np.append(h_poly_fit, h_poly_fit[::-1]),
            color='orange', alpha=0.5, edgecolor='red', linewidth=2, label='Fitted Region (p>0)')

    # Set grid with major and minor ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1.0))
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.grid(which='major', linestyle='-', linewidth=0.75)

    # Mark p=0 axis line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    legend = ax.legend(frameon=True, framealpha=1, edgecolor='black')
    ax.set_xlabel('Power p')
    ax.set_ylabel('Head h')
    ax.set_title('Polynomial Fit of UPC Boundaries in p-h Space')

    plt.show()

# Load the UPC boundary data
boundaries_neg, boundaries_pos = get_UPC_bound()

# Perform polynomial regression to find the boundary fits
h_poly_fit, poly_neg_min_fit, poly_neg_max_fit, poly_pos_min_fit, poly_pos_max_fit = poly_LR_UPC_bound(boundaries_neg, boundaries_pos)

# Convert numpy arrays to torch tensors
poly_neg_min_fit = torch.tensor(poly_neg_min_fit, dtype=torch.float32, device=device)
poly_neg_max_fit = torch.tensor(poly_neg_max_fit, dtype=torch.float32, device=device)
poly_pos_min_fit = torch.tensor(poly_pos_min_fit, dtype=torch.float32, device=device)
poly_pos_max_fit = torch.tensor(poly_pos_max_fit, dtype=torch.float32, device=device)

def neg_min(h, coefficients=poly_neg_min_fit):
    """p >= neg_min(h), in pump mode"""
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def neg_max(h, coefficients=poly_neg_max_fit):
    """p <= neg_max(h), in pump mode"""
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def pos_min(h, coefficients=poly_pos_min_fit):
    """p >= pos_min(h), in turbine mode"""
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def pos_max(h, coefficients=poly_pos_max_fit):
    """p <= pos_max(h), in turbine mode"""
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

# If this script is the main entry point, execute the plot function
if __name__ == '__main__':

    # Example: Calculate boundary powers for a specific head value
    head_example = 80  # Example head value
    min_neg_power = neg_min(head_example)
    max_neg_power = neg_max(head_example)
    min_pos_power = pos_min(head_example)
    max_pos_power = pos_max(head_example)

    print(f"Head: {head_example}")
    print(f"Minimum Negative Power: {min_neg_power}")
    print(f"Maximum Negative Power: {max_neg_power}")
    print(f"Minimum Positive Power: {min_pos_power}")
    print(f"Maximum Positive Power: {max_pos_power}")

    # Plot the results using the polynomial fitted boundaries
    plot_poly_UPC_boundaries(boundaries_neg, boundaries_pos, h_poly_fit, poly_neg_min_fit, poly_neg_max_fit, poly_pos_min_fit, poly_pos_max_fit)

# neg_min_line = np.polyval(neg_min_fit, h_fit)

# %% Fit h-v_low data (outside the pipeline)
# Fit h-v_low data (UPCs data is fit by Origin)

v_low_range = np.linspace(0, max_vol_low, 100000)
h_values = np.array([gross_head(v_low=v) for v in v_low_range])

# Fit data with 15th order polynomial
degree = 5
coefficients = np.polyfit(h_values, v_low_range, degree)
h_v_poly = np.poly1d(coefficients)
h_v_coeffs = torch.tensor(coefficients, dtype=torch.float32, device=device)

def h_to_v_low_fitted(head=None):
    if head is None:
        return None
        
    # Convert input to tensor if it's not already
    if not isinstance(head, torch.Tensor):
        head = torch.tensor(head, dtype=torch.float32, device=device)
    
    # Evaluate the polynomial using Horner's method
    # For a 15th degree polynomial: c[0]*x^15 + c[1]*x^14 + ... + c[14]*x + c[15]
    result = h_v_coeffs[0]  # Start with the highest power coefficient (x^15)
    for i in range(1, len(h_v_coeffs)):
        result = result * head + h_v_coeffs[i]
    
    return result

# Convert h_values to torch tensor
h_values_tensor = torch.tensor(h_values, dtype=torch.float32, device=device)

# Predicted v_low volume
v_low_pred_tensor = h_to_v_low_fitted(h_values_tensor)
v_low_pred = v_low_pred_tensor.cpu().numpy()  # Convert back to numpy for plotting

# Print fitted function
print("Fitted function form:")
print("v_low =", h_v_poly)

# Calculate residuals, R-squared and Adjusted R-squared
residuals = v_low_range - v_low_pred
chi_squared = np.sum((residuals ** 2) / v_low_pred)  # simplified
reduced_chi_squared = chi_squared / (len(v_low_range) - (degree + 1))
r_squared = np.corrcoef(v_low_range, v_low_pred)[0, 1]**2
adjusted_r_squared = 1 - (1 - r_squared) * (len(v_low_range) - 1) / (len(v_low_range) - (degree + 1) - 1)

# Print index
print("Reduced Chi-Square:", reduced_chi_squared)
print("R-Square (COD):", r_squared)
print("Adjusted R-Square:", adjusted_r_squared)

# plot
plt.figure(figsize=(6, 6))
plt.plot(h_values, v_low_range, '-', alpha=0.7, label='Original data')
plt.plot(h_values, v_low_pred, '-', alpha=0.7, label='Fitted curve')
plt.xlabel('head')
plt.ylabel('v_low')
plt.title('V_Low vs. Head Polynomial Fitting')
plt.legend()
plt.show()



# test
print(h_to_v_low_fitted(torch.tensor(99, device=device)))
print(h_to_v_low_fitted(torch.tensor(50, device=device)))

# %% Fit v_low-to-h data (inverse of h-to-v_low data)
# Fit v_low-to-h data (UPCs data is fit by Origin)

# Fit data with 5th order polynomial for the inverse mapping (matching the degree used for h-to-v_low)
degree_inverse = 5
coefficients_inverse = np.polyfit(v_low_range, h_values, degree_inverse)
v_low_h_poly = np.poly1d(coefficients_inverse)
v_low_h_coeffs = torch.tensor(coefficients_inverse, dtype=torch.float32, device=device)

def v_low_to_h_fitted(v_low=None):
    if v_low is None:
        return None
        
    # Convert input to tensor if it's not already
    if not isinstance(v_low, torch.Tensor):
        v_low = torch.tensor(v_low, dtype=torch.float32, device=device)
    elif v_low.device != v_low_h_coeffs.device:
        v_low = v_low.to(v_low_h_coeffs.device)
    
    # Evaluate the polynomial using Horner's method
    result = v_low_h_coeffs[0]  # Start with the highest power coefficient
    for i in range(1, len(v_low_h_coeffs)):
        result = result * v_low + v_low_h_coeffs[i]
    
    return result

# Convert v_low_range to torch tensor
v_low_range_tensor = torch.tensor(v_low_range, dtype=torch.float32, device=device)

# Predicted head values using PyTorch function
h_pred_tensor = v_low_to_h_fitted(v_low_range_tensor)
h_pred = h_pred_tensor.cpu().numpy()  # Convert back to numpy for plotting

# Print fitted function
print("Fitted function form:")
print("h =", v_low_h_poly)

# Calculate residuals, R-squared and Adjusted R-squared
residuals_inverse = h_values - h_pred
chi_squared_inverse = np.sum((residuals_inverse ** 2) / h_pred)  # simplified
reduced_chi_squared_inverse = chi_squared_inverse / (len(h_values) - (degree_inverse + 1))
r_squared_inverse = np.corrcoef(h_values, h_pred)[0, 1]**2
adjusted_r_squared_inverse = 1 - (1 - r_squared_inverse) * (len(h_values) - 1) / (len(h_values) - (degree_inverse + 1) - 1)

# Print index
print("Reduced Chi-Square:", reduced_chi_squared_inverse)
print("R-Square (COD):", r_squared_inverse)
print("Adjusted R-Square:", adjusted_r_squared_inverse)

# plot
plt.figure(figsize=(6, 6))
plt.plot(v_low_range, h_values, '-', alpha=0.7, label='Original data')
plt.plot(v_low_range, h_pred, '-', alpha=0.7, label='Fitted curve')
plt.xlabel('v_low')
plt.ylabel('head')
plt.title('Head vs. V_Low Polynomial Fitting (Inverse)')
plt.legend()
plt.show()

# Test the function
print("Testing v_low_to_h_fitted with a sample value:")
print(v_low_to_h_fitted(588000))  # Example value within the range

# Test inverse relationship
test_v1 = h_to_v_low_fitted(99)
test_v2 = h_to_v_low_fitted(50)
print(f"v_low corresponding to h=99: {test_v1}")
print(f"v_low corresponding to h=50: {test_v2}")
print(f"h corresponding to v_low={test_v1}: {v_low_to_h_fitted(test_v1)}")
print(f"h corresponding to v_low={test_v2}: {v_low_to_h_fitted(test_v2)}")

# %% Fit UPC data (outside pipeline)
# Fit UPC data (outside pipeline)

def prepare_and_fit_model(file_path, poly_degree=5):
    # Load data from the specified file
    data = pd.read_excel(file_path)
    x_values = np.array(data.columns[1:], dtype=float)  # x is Power
    y_values = np.array(data.iloc[:, 0], dtype=float)   # y is Head
    X, Y = np.meshgrid(x_values, y_values)
    z_flat = data.iloc[:, 1:].values.flatten()
    
    # Filter valid data points
    valid_indices = ~np.isnan(z_flat)
    x_valid = X.flatten()[valid_indices]
    y_valid = Y.flatten()[valid_indices]
    z_valid = z_flat[valid_indices]
    
    # Fit model
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    XY_valid = np.vstack([x_valid, y_valid]).T
    model = make_pipeline(poly, LinearRegression())
    model.fit(XY_valid, z_valid)

    # Additional data for formulas
    coefs = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    feature_names = poly.get_feature_names_out(['p', 'h'])

    # Predictions for calculating R^2 and other statistics
    z_pred = model.predict(XY_valid)
    sst = np.sum((z_valid - np.mean(z_valid)) ** 2)
    ssr = np.sum((z_valid - z_pred) ** 2)
    r2 = 1 - (ssr / sst)
    n = len(z_valid)
    p = poly.n_output_features_
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    reduced_chi_squared = ssr / (n - p)
    
    return model, x_valid, y_valid, z_valid, r2, adjusted_r2, reduced_chi_squared, coefs, intercept, feature_names

def plot_3d_surface_interactive(x_valid, y_valid, z_valid, model, title):
    # Create mesh grid for the surface
    x_surf, y_surf = np.meshgrid(np.linspace(x_valid.min(), x_valid.max(), 50),
                                 np.linspace(y_valid.min(), y_valid.max(), 50))
    xy_surf = np.vstack([x_surf.ravel(), y_surf.ravel()]).T
    z_surf = model.predict(xy_surf).reshape(x_surf.shape)

    # Determine the range of z values for plotting
    z_min = z_valid.min()
    z_max = z_valid.max()

    # Create the interactive figure
    fig = go.Figure(data=[
        go.Scatter3d(x=x_valid, y=y_valid, z=z_valid, mode='markers', name='Original Data', 
                     marker=dict(size=1, color=z_valid,colorscale='Plasma',cmin=z_min,cmax=z_max)),
        go.Surface(x=x_surf, y=y_surf, z=z_surf, name='Fitted Surface', 
                   colorscale='Viridis', cmin=z_min, cmax=z_max,opacity=0.7)
    ])
    
    # Update layout with dynamic z-axis range and title
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Power (p)',
            yaxis_title='Head (h)',
            zaxis_title='Flow (q)',
            zaxis=dict(range=[z_min-1, z_max+1])  # Set the z-axis range based on actual data
        )
    )

    fig.show()

def print_model_formula(coefs, intercept, feature_names):
    terms = [f"{coef:.2f}*{name}" for coef, name in zip(coefs, feature_names)]
    formula = " + ".join(terms)
    formula = f"q = {intercept:.2f} + {formula}"
    return formula

# Perform polynomial linear regression on UPC(Outside pipeline)
if __name__ == '__main__':
    # Fit and plot models for both pump and turbine
    results_pump = prepare_and_fit_model('./Data/UPCs/temp/Mod_Francis_pump_temp.xlsx')
    results_turbine = prepare_and_fit_model('./Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx')
    
    # Plot the fitted models
    plot_3d_surface_interactive(*results_pump[1:4], results_pump[0], 'Pump Model')
    plot_3d_surface_interactive(*results_turbine[1:4], results_turbine[0], 'Turbine Model')

    # Print statistical measures and formulas
    print("Pump Model - R^2: {:f}, Adjusted R^2: {:f}, Reduced Chi-Squared: {:f}".format(results_pump[4], results_pump[5], results_pump[6]))
    print("Pump Model Formula:")
    print(print_model_formula(results_pump[7], results_pump[8], results_pump[9]))
    
    print("Turbine Model - R^2: {:f}, Adjusted R^2: {:f}, Reduced Chi-Squared: {:f}".format(results_turbine[4], results_turbine[5], results_turbine[6]))
    print("Turbine Model Formula:")
    print(print_model_formula(results_turbine[7], results_turbine[8], results_turbine[9]))

    # Convert coefficients and intercepts to PyTorch tensors
    coefs_tur = torch.tensor(results_turbine[7], dtype=torch.float32, device=device)
    intercept_tur = torch.tensor(results_turbine[8], dtype=torch.float32, device=device)
    coefs_pump = torch.tensor(results_pump[7], dtype=torch.float32, device=device)
    intercept_pump = torch.tensor(results_pump[8], dtype=torch.float32, device=device)


# ✨Predict by manually generated features (Used in the pipeline)

def predict_q_poly(p, h, coefs_tur=coefs_tur,
                  intercept_tur=intercept_tur,
                  coefs_pump=coefs_pump,
                  intercept_pump=intercept_pump, 
                  poly_degree=5):
    """
    Vectorized prediction of q values from p and h
    
    Args:
        p: torch tensor of power values of any shape
        h: torch tensor of head values (same shape as p)
        coefs_tur, intercept_tur: turbine model parameters
        coefs_pump, intercept_pump: pump model parameters
        poly_degree: degree of polynomial features
        
    Returns:
        torch tensor of predicted flow values (same shape as inputs)
    """
    # Ensure inputs are torch tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32, device=torch.device('cpu'))
    if not isinstance(h, torch.Tensor):
        h = torch.tensor(h, dtype=torch.float32, device=torch.device('cpu'))
        
    # Generate power matrix
    batch_size = p.numel()
    max_power = poly_degree + 1
    
    # Precompute all possible power combinations (p^a * h^b, a + b <= poly_degree)
    powers = torch.arange(max_power, device=p.device)
    p_pows = torch.pow(p.unsqueeze(-1), powers)  # [..., max_power]
    h_pows = torch.pow(h.unsqueeze(-1), powers)  # [..., max_power]
    
    # Generate polynomial feature matrix (avoid loops)
    terms = []
    for total_degree in range(1, poly_degree + 1):
        for a in range(total_degree, -1, -1):
            b = total_degree - a
            if b <= total_degree and a + b <= poly_degree:
                terms.append(p_pows[..., a] * h_pows[..., b])
    
    features = torch.stack(terms, dim=-1)  # [..., num_features]
    
    # Combine turbine and pump coefficient calculations
    q_tur = torch.einsum('...f,f->...', features, coefs_tur) + intercept_tur
    q_pump = torch.einsum('...f,f->...', features, coefs_pump) + intercept_pump
    
    # Vectorized conditional selection
    mask_tur = (p > 0)
    mask_pump = (p < 0)
    q = torch.where(mask_tur, q_tur, torch.where(mask_pump, q_pump, torch.zeros_like(p).to(p.device)))
        
    return q

# Test the function
if __name__ == '__main__':
    # Define example inputs
    p_example = torch.tensor([-5.89, 0.0, 5.89], dtype=torch.float32, device=device)
    h_example = torch.tensor([78, 67, 91], dtype=torch.float32, device=device)
    # Expected outputs [-8.984, 0.0, 7.906] 

    # Predict the flow q
    q_predicted = predict_q_poly(p_example, h_example)
    print(f"Predicted flow (q) for p={p_example.tolist()}, h={h_example.tolist()}:\n{q_predicted.tolist()}")

# %% Fit h-v_low data (linear)
# Fit h-v_low data (linear)
v_low_range_lin = np.linspace(0, max_vol_low, 2)
h_values_lin = np.array([gross_head(v_low=v) for v in v_low_range_lin])

# Fit data with 1th order polynomial
degree_lin = 1
h_vlow_coeff_lin = np.polyfit(h_values_lin, v_low_range_lin, degree_lin)
h_v_poly_lin = np.poly1d(h_vlow_coeff_lin)
h_v_coeffs_lin = torch.tensor(h_vlow_coeff_lin, dtype=torch.float32, device=device)

# Predicted v_low volume
v_low_pred_lin = h_v_poly_lin(h_values_lin)

# Print fitted function
print("Fitted function form:")
print("v_low =", h_v_poly_lin)

# Calculate residuals, R-squared and Adjusted R-squared
residuals_lin = v_low_range_lin - v_low_pred_lin
chi_squared_lin = np.sum((residuals_lin ** 2) / v_low_pred_lin)  # simplified
reduced_chi_squared_lin = chi_squared_lin / (len(v_low_range_lin) - (degree_lin + 1))
r_squared_lin = np.corrcoef(v_low_range_lin, v_low_pred_lin)[0, 1]**2
adjusted_r_squared_lin = 1 - (1 - r_squared_lin) * (len(v_low_range_lin) - 1) / (len(v_low_range_lin) - (degree_lin + 1) - 1)

# Print index
print("Reduced Chi-Square:", reduced_chi_squared_lin)
print("R-Square (COD):", r_squared_lin)
print("Adjusted R-Square:", adjusted_r_squared_lin)

# plot
plt.figure(figsize=(6, 6))
plt.plot(h_values, v_low_range, '-', alpha=0.7, label='Original data')
plt.plot(h_values_lin, v_low_pred_lin, '-', alpha=0.7, label='Fitted curve')
plt.xlabel('head')
plt.ylabel('v_low')
plt.title('V_Low vs. Head Polynomial Fitting (Linear)')
plt.legend()
plt.show()

def h_to_v_low_lin(head=None):
    return h_v_poly_lin(head)

# test
print(h_to_v_low_lin(99))
print(h_to_v_low_lin(50))



# %% Fit UPC data (linear)
# Fit UPC data (linear)

def prepare_and_fit_model_linear(file_path):
    return prepare_and_fit_model(file_path, poly_degree=1)



# Perform linear regression on UPC (Outside pipeline)
if __name__ == '__main__':
    # Fit and plot models for both pump and turbine
    results_pump_linear = prepare_and_fit_model_linear('./Data/UPCs/temp/Mod_Francis_pump_temp.xlsx')
    results_turbine_linear = prepare_and_fit_model_linear('./Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx')
    
    # Plot the fitted models
    plot_3d_surface_interactive(*results_pump_linear[1:4], results_pump_linear[0], 'Pump Model (Linear)')
    plot_3d_surface_interactive(*results_turbine_linear[1:4], results_turbine_linear[0], 'Turbine Model (Linear)')

    # Print statistical measures and formulas
    print("Pump Model (Linear) - R^2: {:f}, Adjusted R^2: {:f}, Reduced Chi-Squared: {:f}".format(results_pump_linear[4], results_pump_linear[5], results_pump_linear[6]))
    print("Pump Model Formula (Linear):")
    print(print_model_formula(results_pump_linear[7], results_pump_linear[8], results_pump_linear[9]))
    
    print("Turbine Model (Linear) - R^2: {:f}, Adjusted R^2: {:f}, Reduced Chi-Squared: {:f}".format(results_turbine_linear[4], results_turbine_linear[5], results_turbine_linear[6]))
    print("Turbine Model Formula (Linear):")
    print(print_model_formula(results_turbine_linear[7], results_turbine_linear[8], results_turbine_linear[9]))

    # Convert coefficients and intercepts to PyTorch tensors
    coefs_tur_linear = torch.tensor(results_turbine_linear[7], dtype=torch.float32, device=device)
    intercept_tur_linear = torch.tensor(results_turbine_linear[8], dtype=torch.float32, device=device)
    coefs_pump_linear = torch.tensor(results_pump_linear[7], dtype=torch.float32, device=device)
    intercept_pump_linear = torch.tensor(results_pump_linear[8], dtype=torch.float32, device=device)

    coefs_tur_lin = results_turbine_linear[7]
    intercept_tur_lin = results_turbine_linear[8]
    coefs_pump_lin = results_pump_linear[7]
    intercept_pump_lin = results_pump_linear[8]

# Create 2 functions predict_q_linear_tur and predict_q_linear_pump
def predict_q_linear_tur(p, h, coefs=coefs_tur_linear, intercept=intercept_tur_linear):
    """
    Linear prediction of flow rate for turbine mode.
    """
    # Create feature matrix [p, h]
    features = torch.stack([p, h], dim=-1)
    
    # Compute linear prediction q = c_p*p + c_h*h + intercept
    q = torch.einsum('...d,d->...', features, coefs) + intercept
    
    # Zero out predictions for non-turbine mode (p ≤ 0)
    mask_tur = (p > 0)
    return torch.where(mask_tur, q, torch.zeros_like(q).to(device))

def predict_q_linear_pump(p, h, coefs=coefs_pump_linear, intercept=intercept_pump_linear):
    """
    Linear prediction of flow rate for pump mode.
    """
    # Create feature matrix [p, h]
    features = torch.stack([p, h], dim=-1)
    
    # Compute linear prediction q = c_p*p + c_h*h + intercept
    q = torch.einsum('...d,d->...', features, coefs) + intercept
    
    # Zero out predictions for non-pump mode (p ≥ 0)
    mask_pump = (p < 0)
    return torch.where(mask_pump, q, torch.zeros_like(q).to(device))

if __name__ == '__main__':
    # test the functions
    p_example = torch.tensor([-5.89, 0.0, 5.89], dtype=torch.float32, device=device)
    h_example = torch.tensor([78, 67, 91], dtype=torch.float32, device=device)
    q_predicted_tur = predict_q_linear_tur(p_example, h_example)
    q_predicted_pump = predict_q_linear_pump(p_example, h_example)
    print(f"Predicted flow (q) for p={p_example.tolist()}, h={h_example.tolist()} (Turbine):\n{q_predicted_tur.tolist()}")
    print(f"Predicted flow (q) for p={p_example.tolist()}, h={h_example.tolist()} (Pump):\n{q_predicted_pump.tolist()}")


# %% 
# Save preprocessing functions and variables
with open('preprocess.pkl', 'wb') as f:
    pickle.dump((v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound), f)

print('---------------Preprocessing Completed---------------')
# %%
