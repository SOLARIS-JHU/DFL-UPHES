#%%
""" In this file, solve all the cubic equations with Cardano's formula 
instead of numerical methods to decrease the time complexity. """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from cardano_method import CubicEquation
# from scipy.optimize import fsolve

""" Due to the geometry of the upper and lower reservoirs, their head levels
are both monotonously increasing in relations to their corresponding volumes.
Therefore, ∃! real number solution in the operating range. """
def solve_cubic_in_range(coeffs, lower_bound, upper_bound):
    """
    Solve a cubic equation with coefficients and returns the real root within a specified range.
    Parameters:
        coeffs (list): List of coefficients [a, b, c, d] for the cubic equation ax^3 + bx^2 + cx + d = 0.
        lower_bound (float): Lower bound of the range to find the real root.
        upper_bound (float): Upper bound of the range to find the real root.
    Returns:
        float or None: The real root within the given range, if it exists, otherwise None.
    """
    # solve cubic eq. with Cardano's method
    equation = CubicEquation(coeffs)
    roots = equation.answers
    
    # find the root of real numbers within operating range
    real_roots_in_range = [root.real for root in roots if root.imag == 0 and lower_bound <= root.real <= upper_bound]

    return real_roots_in_range[0] if real_roots_in_range else None

# # testing
# coefficients = [1, -6, 11, -6]  # cubic eq.: x^3 - 6x^2 + 11x - 6 = 0
# lower = 1.5  # lower bound
# upper = 2.5  # upper bound
# real_root = solve_cubic_in_range(coefficients, lower, upper)
# print("The real root in the given range is:", real_root)

# # read UPHES portfolio
# portfolio_path = '../Data/portfolio_UPHES.xlsx'
# df = pd.read_excel(portfolio_path,sheet_name='Sheet1')

# pi = np.pi
# r = df.iloc[17, 5] # radius of the upper reservoir bottom
# m = df.iloc[21, 2] # slope coefficient of the upper reservoir
# h_dead_up = df.iloc[22, 2]
# h_normal_up = df.iloc[23, 2]
# height_up = h_normal_up - h_dead_up # height of the utilizable storage capacity 
# R = df.iloc[27,2] # radius of underground mine pits
# height_low = 2 * R # height of the utilizable storage capacity 
# n = df.iloc[26, 2] # number of underground mine pits
# h_dead_low = df.iloc[28, 2]
# h_normal_low = df.iloc[29, 2]
# max_vol_up = df.iloc[14, 2]
# max_vol_low = df.iloc[16, 2]
# max_vol = min(max_vol_up,max_vol_low)

def load_portfolio_data():

    # Designated file path
    current_dir = Path(__file__).parent
    portfolio_path = current_dir / '../Data/portfolio_UPHES.xlsx'

    # Declare global variables
    global pi, r, m, h_dead_up, h_normal_up, height_up, head_min, head_max, ramp_down, ramp_up, target_vol_up
    global R, height_low, n, h_dead_low, h_normal_low, max_vol_up, min_vol_up, min_vol_low, a, max_vol_low, max_vol

    # Read the Excel file
    df = pd.read_excel(portfolio_path, sheet_name='Sheet1')

    # Extract variables from the dataframe
    pi = np.pi
    r = df.iloc[17, 5]  # Radius of the upper reservoir bottom
    m = df.iloc[21, 2]  # Slope coefficient of the upper reservoir
    h_dead_up = df.iloc[22, 2]
    h_normal_up = df.iloc[23, 2]
    height_up = h_normal_up - h_dead_up  # Height of the utilizable storage capacity
    R = df.iloc[27, 2]  # Radius of underground mine pits
    height_low = 2 * R  # Height of the utilizable storage capacity
    n = df.iloc[26, 2]  # Number of underground mine pits
    h_dead_low = df.iloc[28, 2]
    h_normal_low = df.iloc[29, 2]
    max_vol_up = df.iloc[14, 2]
    max_vol_low = df.iloc[16, 2]
    min_vol_up = df.iloc[13,2]
    min_vol_low = df.iloc[15,2]
    a = df.iloc[13,2]
    max_vol = min(max_vol_up, max_vol_low)
    head_min = df.iloc[11,2]
    head_max = df.iloc[12,2]
    ramp_up = df.iloc[6,2]
    ramp_down = df.iloc[7,2]
    target_vol_up = df.iloc[3,2]

# testing
portfolio_path = '../Data/portfolio_UPHES.xlsx'
load_portfolio_data()
# Variables can now be directly used afterwards
print(target_vol_up)  # Print the radius of the upper reservoir bottom


#%%
# Upper reservoir

def head_to_vol_up(head):
    """Input upper basin head level, output upper basin volume."""
    if head < h_dead_up or head > h_normal_up:
        raise ValueError("Head level out of operational range.")
    height = head - h_dead_up
    current_radius = r + m * height
    return (1/3) * pi * height * (current_radius**2 + current_radius * r + r**2)

# using Cardano's method to solve cubic eq.
def vol_to_head_up(volume): 
    """Input upper basin volume, output upper basin head level."""
    if volume < 0 or volume > max_vol_up:
        raise ValueError("Volume out of operational range.")
    coeffs = [m**2, 3*m*r, 3*r**2, -3*volume/pi]
    height = solve_cubic_in_range(coeffs, 0, h_normal_up - h_dead_up)
    head = height + h_dead_up
    return head

# # using fsolve for real numerical solution of a cubic eq.
# def vol_to_head_up(volume): 
#     """Input upper basin volume, output upper basin head level."""
#     if volume < 0 or volume > max_vol_up:
#         raise ValueError("Volume out of operational range.")
#     def equation(h):
#         return head_to_vol_up(h) - volume
#     head_initial_guess = volume / (pi * r**2)
    
#     # iterates from a cylinder with radius of r
#     head = fsolve(equation, head_initial_guess)
#     return head

# # testing upper basin
# test_volume = 588000 / 2  # half full volmune
# test_height = vol_to_head_up(test_volume)
# computed_volume = head_to_vol_up(test_height)
# print(test_height, computed_volume)

#%%
# Lower reservoir

def head_to_vol_low(head):
    """Input lower basin head level, output lower basin volume."""
    if head < h_dead_low or head > h_normal_low:
        raise ValueError("Head level out of operational range.")
    height = head - h_dead_low
    return n * pi * height**2 * (3 * R - height) / 3

# using Cardano's method to solve cubic eq.
def vol_to_head_low(volume):
    """Input lower basin volume, output lower basin head level."""
    if volume < 0 or volume > max_vol_low:
        raise ValueError("Volume out of operational range.")
    coeffs = [pi/3, -pi*R, 0, volume/n]
    height = solve_cubic_in_range(coeffs, 0, h_normal_low - h_dead_low)
    head = height + h_dead_low
    return head

# # using fsolve for real numerical solution of a cubic eq.
# def volume_to_height(volume):
#     """Input lower basin volume, output lower basin head level."""
#     if volume < 0 or volume > max_vol_low:
#         raise ValueError("Volume out of operational range.")
#     # eq. for spherical cap
#     def equation(h):
#         return (3*R - h) * h**2 - 3 * volume / pi / n
    
#     # iterates from the half full state
#     initial_guess = R / 2
#     from scipy.optimize import fsolve
#     height = fsolve(equation, initial_guess)
#     head = height + h_dead_low
#     return head

# # tesing lower basin
# test_volume = 294000  # half full volume
# computed_height = vol_to_head_low(test_volume)
# computed_volume = head_to_vol_low(computed_height)
# print(computed_height, computed_volume)

# %%
# general functions

def gross_head(h_up=None, h_low=None, v_up=None, v_low=None):
    """
    Calculate the gross head between the upper and lower reservoirs.
    Given the sum of the volumes of the upper and lower reservoirs is constant and equal to max_vol,
    only one of the parameters h_up, h_low, v_up, or v_low is necessary to compute the gross head.

    Args:
    h_up (float): Head level of the upper reservoir (optional).
    h_low (float): Head level of the lower reservoir (optional).
    v_up (float): Volume of the upper reservoir (optional).
    v_low (float): Volume of the lower reservoir (optional).

    Returns:
    float: The gross head (h_up - h_low).
    """
    if h_up is None and h_low is None:
        if v_up is None and v_low is None:
            raise ValueError("At least one parameter (h_up, h_low, v_up, or v_low) must be provided.")
        elif v_up is not None and v_low is not None: 
            h_up = vol_to_head_up(v_up)
            h_low = vol_to_head_low(v_low)
        elif v_up is not None:
            h_up = vol_to_head_up(v_up)
            v_low = max_vol - v_up
            h_low = vol_to_head_low(v_low)
        elif v_low is not None: #
            h_low = vol_to_head_low(v_low)
            v_up = max_vol - v_low
            h_up = vol_to_head_up(v_up)
    elif h_up is not None and h_low is None:
        if v_low is not None:
            h_low = vol_to_head_low(v_low) 
        elif v_up is not None:
            v_low = max_vol - v_up
            h_low = vol_to_head_low(v_low)
        else:
            v_up = head_to_vol_up(h_up)
            v_low = max_vol - v_up
            h_low = vol_to_head_low(v_low)
    elif h_low is not None and h_up is None:
        if v_up is not None:
            h_up = vol_to_head_up(v_up)
        elif v_low is not None:
            v_up = max_vol - v_low
            h_up = vol_to_head_up(v_up)
        else:
            v_low = head_to_vol_low(h_low)
            v_up = max_vol - v_low
            h_up = vol_to_head_up(v_up)

    return h_up - h_low

# # testing gross_head
# print(gross_head(v_up=300000))  # Calculate using upper volume
# print(gross_head(h_low=33.45177))     # Calculate using lower head level

def get_v_up(h_up=None, h_low=None, v_low=None):
    """
    Calculate the volume of the upper reservoir based on the provided parameters.

    Args:
    h_up (float): Head level of the upper reservoir (optional).
    h_low (float): Head level of the lower reservoir (optional).
    v_low (float): Volume of the lower reservoir (optional).

    Returns:
    float: The volume of the upper reservoir.
    """
    if v_low is not None:
        return max_vol - v_low
    elif h_up is not None:
        return head_to_vol_up(h_up)
    elif h_low is not None:
        v_low = head_to_vol_low(h_low)
        return max_vol - v_low
    else:
        raise ValueError("Insufficient parameters to calculate upper reservoir volume.")

def get_v_low(gross_head=None, h_up=None, h_low=None, v_up=None):
    """
    Calculate the volume of the lower reservoir based on the provided parameters.

    Args:
    h_up (float): Head level of the upper reservoir (optional).
    h_low (float): Head level of the lower reservoir (optional).
    v_up (float): Volume of the upper reservoir (optional).

    Returns:
    float: The volume of the lower reservoir.
    """
    
    if v_up is not None:
        return max_vol - v_up
    elif h_low is not None:
        return head_to_vol_low(h_low)
    elif h_up is not None:
        v_up = head_to_vol_up(h_up)
        return max_vol - v_up
    else:
        raise ValueError("Insufficient parameters to calculate lower reservoir volume.")

# # testing volume functions
# print(get_v_up(h_low=33.45177))  # empty upper basin
# print(get_v_low(h_up=91.9908328350036)) # half volume
#%%
# Plot v-h relations for both reservoirs

if __name__ == "__main__":

    load_portfolio_data()

    # assign head range
    h_up_range = np.linspace(h_dead_up, h_normal_up, 100)
    h_low_range = np.linspace(h_dead_low, h_normal_low, 100)

    # calculate corresponding volume
    v_up_values = [head_to_vol_up(h) for h in h_up_range]
    v_low_values = [head_to_vol_low(h) for h in h_low_range]

    plt.figure(figsize=(12, 6))

    # subplot for upper reservoir v-h curve
    plt.subplot(1, 2, 1)
    plt.plot(v_up_values, h_up_range, label='Upper Reservoir', color='blue')
    plt.xlabel('Volume (m³)')
    plt.ylabel('Head Level (m)')
    plt.title('Upper Reservoir Volume vs. Head Level')
    plt.grid(True)

    # subplt for lower reservoir v-h curve
    plt.subplot(1, 2, 2)
    plt.plot(v_low_values, h_low_range, label='Lower Reservoir', color='green')
    plt.xlabel('Volume (m³)')
    plt.ylabel('Head Level (m)')
    plt.title('Lower Reservoir Volume vs. Head Level')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

#%%
# plot gross head-volume relations for both reservoirs

if __name__ == "__main__":
    # assign volume range
    v_up_range = np.linspace(0, max_vol_up, 100)
    v_low_range = np.linspace(0, max_vol_low, 100)

    # calculate gross head
    gh_from_v_up = [gross_head(v_up=v) for v in v_up_range]
    gh_from_v_low = [gross_head(v_low=v) for v in v_low_range]

    plt.figure(figsize=(12, 6))

    # subplot for upper basin gross head
    plt.subplot(1, 2, 1)
    plt.plot(v_up_range, gh_from_v_up, label='Gross Head from Upper Volume', color='blue')
    plt.xlabel('Upper Reservoir Volume (m³)')
    plt.ylabel('Gross Head (m)')
    plt.title('Gross Head vs. Upper Reservoir Volume')
    plt.grid(True)
    plt.legend()

    # subplot for lower basin gross head
    plt.subplot(1, 2, 2)
    plt.plot(v_low_range, gh_from_v_low, label='Gross Head from Lower Volume', color='red')
    plt.xlabel('Lower Reservoir Volume (m³)')
    plt.ylabel('Gross Head (m)')
    plt.title('Gross Head vs. Lower Reservoir Volume')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# %%
# Calculate target volume and target head
global target_vol_low, target_head
target_vol_low = get_v_low(v_up=target_vol_up)
target_head = gross_head(v_up=target_vol_up)
