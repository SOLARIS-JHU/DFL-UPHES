# %% Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import sys
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# torch.autograd.set_detect_anomaly(True)

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

# %% Read database
# Function to read data from CSV file (add to pipeline_learnable_weight_ver.py)
def load_historical_data(file_path="./Data/database_no_piecewise.csv", with_coefficients=False):
    """
    Load historical optimization data from CSV.
    
    Parameters:
        file_path (str): Path to the CSV database file
        with_coefficients (bool): Whether the file contains linearization coefficients
        
    Returns:
        dict: Dictionary with data grouped by date
             Keys are date strings, values are dictionaries with arrays of:
             power, head, flow, price, and coefficients (if available)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['Time', 'Power', 'Head', 'Flow', 'Price', 'Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for coefficients if requested
        if with_coefficients:
            coeff_columns = ['c', 'd', 'e', 'a', 'b']
            missing_coeffs = [col for col in coeff_columns if col not in df.columns]
            if missing_coeffs:
                raise ValueError(f"Missing coefficient columns: {missing_coeffs}")
        
        # Convert Date to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Add 'Mode' column if not present
        if 'Mode' not in df.columns:
            # Determine mode based on power and flow values
            conditions = [
                (abs(df['Power']) < 0.01),  # Idle mode (power close to zero)
                (df['Power'] > 0),          # Turbine mode (positive power)
                (df['Power'] < 0)           # Pump mode (negative power)
            ]
            choices = ['Idle', 'Turbine', 'Pump']
            df['Mode'] = np.select(conditions, choices, default='Unknown')
        
        # Group data by date
        data_by_date = {}
        for date, group in df.groupby('Date'):
            # Sort by Time to ensure correct order
            group = group.sort_values('Time')
            
            # Convert date to string format
            date_str = date.strftime('%Y-%m-%d')
            
            # Create dictionary for this date
            date_data = {
                'power': torch.tensor(group['Power'].values, dtype=torch.float32),
                'head': torch.tensor(group['Head'].values, dtype=torch.float32),
                'flow': torch.tensor(group['Flow'].values, dtype=torch.float32),
                'price': torch.tensor(group['Price'].values, dtype=torch.float32),
                'mode': group['Mode'].values
            }
            
            # Add coefficients if available
            if with_coefficients:
                date_data.update({
                    'c': torch.tensor(group['c'].values, dtype=torch.float32),
                    'd': torch.tensor(group['d'].values, dtype=torch.float32),
                    'e': torch.tensor(group['e'].values, dtype=torch.float32),
                    'a': torch.tensor(group['a'].values, dtype=torch.float32),
                    'b': torch.tensor(group['b'].values, dtype=torch.float32)
                })
            
            data_by_date[date_str] = date_data
        
        print(f"Successfully loaded data for {len(data_by_date)} days.")
        return data_by_date
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# %% Read day-ahead prices
def read_da_price(date, file_path="./Data/Belgium.csv"):
    """
    Input: "YYYY-MM-DD"
    """
    data = pd.read_csv(file_path)
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    filtered_data = data[data['Datetime (UTC)'].dt.date == pd.to_datetime(date).date()]
    return torch.tensor(filtered_data['Price (EUR/MWhe)'].values[:24], dtype=torch.float32)

def hourly_to_quarterly(tensor_data):
    return tensor_data.repeat_interleave(4)

# # Example usage:
# sample_date = "2022-01-01"
# DA_price_hour = read_da_price(sample_date)
# DA_price_quarter = hourly_to_quarterly(DA_price_hour)
# print(DA_price_hour)
# print(DA_price_quarter)

# %% Define test data
# # Create sample input data
# power = torch.tensor([-6.77, -7.01, -7.32, -7.63, -7.95, -8.26, -8.19, 4.27, 4.11, 4.43, 4.23, 4.01, 3.78, 3.55, 
#                         3.37, 3.3, 3.23, 4.17, 4.8, 4.55, 3.91, 3.66, 2.64, 2.57], dtype=torch.float32, requires_grad=True)
# head = torch.tensor([76.96, 79.39, 81.82, 84.25, 86.67, 89.12, 91.47, 90.13, 88.82, 87.34, 85.89, 84.48, 83.13, 81.85, 
#                         80.6, 79.35, 78.09, 76.37, 74.23, 72.18, 70.49, 68.9, 67.77, 66.67], dtype=torch.float32, requires_grad=True)
# flow = torch.tensor([-10.24, -10.14, -10.12, -10.11, -10.11, -10.18, -9.79, 5.55, 5.48, 6.16, 6.06, 5.87, 5.61, 5.36, 
#                         5.19, 5.21, 5.24, 7.15, 8.95, 8.53, 7.04, 6.64, 4.68, 4.61], dtype=torch.float32, requires_grad=True)

head_init = 77.0 # Initial head value
v_low_init = h_to_v_low_fitted(head_init) # Initial lower reservoir volume

# %% Pipeline
# Pipeline
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
        gross_head=gross_head
    ):
        self.time_horizon = time_horizon
        self.sampling_rate = sampling_rate
        self.operational_cost = operational_cost
        
        self.δ_p = torch.tensor(δ_p, dtype=torch.float32)
        self.δ_h = torch.tensor(δ_h, dtype=torch.float32)
        self.δ_q = torch.tensor(δ_q, dtype=torch.float32)
        self.rho = torch.tensor(rho, dtype=torch.float32)
        self.g = torch.tensor(g, dtype=torch.float32)
        self.mu = torch.tensor(mu, dtype=torch.float32)

        self.head_min = torch.tensor(head_min, dtype=torch.float32)
        self.head_max = torch.tensor(head_max, dtype=torch.float32)
        self.max_vol_up = torch.tensor(max_vol_up, dtype=torch.float32)
        self.min_vol_low = torch.tensor(min_vol_low, dtype=torch.float32)
        self.ramp_up = torch.tensor(ramp_up, dtype=torch.float32)
        self.ramp_down = torch.tensor(ramp_down, dtype=torch.float32)

        self.target_head = torch.tensor(target_head, dtype=torch.float32)
        self.target_vol_low = torch.tensor(target_vol_low, dtype=torch.float32)
        self.head_init = torch.tensor(head_init, dtype=torch.float32)
        self.v_low_init = torch.tensor(v_low_init, dtype=torch.float32)

        self.neg_min_fit = torch.tensor(neg_min_fit, dtype=torch.float32)
        self.neg_max_fit = torch.tensor(neg_max_fit, dtype=torch.float32)
        self.pos_min_fit = torch.tensor(pos_min_fit, dtype=torch.float32)
        self.pos_max_fit = torch.tensor(pos_max_fit, dtype=torch.float32)

        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.predict_q_poly = predict_q_poly
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.gross_head = gross_head

class RegressionLayer: # with hard constraints
    def __init__(self, params: HydroParameters):
        self.params = params

    def least_squares_UPC_torch(self, p_samples, h_samples, q_values):
        """
        Perform least squares for q = c*p + d*h + e with gradient tracking.
        Ensures proper shape handling for matrix operations.
        """
        # Ensure inputs are tensors with gradients and proper shapes
        p_samples = p_samples.detach().clone().requires_grad_(True)
        h_samples = h_samples.detach().clone().requires_grad_(True)
        q_values = q_values.detach().clone().requires_grad_(True)
        
        # Reshape inputs to ensure proper dimensions
        p_samples = p_samples.view(-1)  # Flatten to 1D
        h_samples = h_samples.view(-1)  # Flatten to 1D
        q_values = q_values.view(-1)    # Flatten to 1D
        
        # Create design matrix with gradient tracking
        ones = torch.ones_like(p_samples, requires_grad=True)
        X = torch.stack([p_samples, h_samples, ones], dim=1)  # Shape: [n_samples, 3]
        y = q_values.view(-1, 1)  # Shape: [n_samples, 1]
        
        # Compute least squares solution with gradient tracking
        XTX = torch.matmul(X.t(), X)  # Shape: [3, 3]
        XTy = torch.matmul(X.t(), y)  # Shape: [3, 1]
        
        # Add small regularization for numerical stability
        epsilon = 1e-6
        reg_matrix = epsilon * torch.eye(3, device=XTX.device)
        XTX_reg = XTX + reg_matrix
        
        beta = torch.matmul(torch.inverse(XTX_reg), XTy)
        return beta.squeeze()

    def least_squares_v_low_torch(self, h_samples, v_low_samples):
        """
        Perform least squares for v_low = a*h + b with gradient tracking.
        Ensures proper shape handling for matrix operations.
        """
        # Ensure inputs are tensors with gradients and proper shapes
        h_samples = h_samples.detach().clone().requires_grad_(True)
        v_low_samples = v_low_samples.detach().clone().requires_grad_(True)
        
        # Reshape inputs to ensure proper dimensions
        h_samples = h_samples.view(-1)      # Flatten to 1D
        v_low_samples = v_low_samples.view(-1)  # Flatten to 1D
        
        # Create design matrix with gradient tracking
        ones = torch.ones_like(h_samples, requires_grad=True)
        X = torch.stack([h_samples, ones], dim=1)  # Shape: [n_samples, 2]
        y = v_low_samples.view(-1, 1)  # Shape: [n_samples, 1]
        
        # Compute least squares solution with gradient tracking
        XTX = torch.matmul(X.t(), X)  # Shape: [2, 2]
        XTy = torch.matmul(X.t(), y)  # Shape: [2, 1]
        
        # Add small regularization for numerical stability
        epsilon = 1e-6
        reg_matrix = epsilon * torch.eye(2, device=XTX.device)
        XTX_reg = XTX + reg_matrix
        
        beta = torch.matmul(torch.inverse(XTX_reg), XTy)
        return beta.squeeze()

    def run_regression(self, power, head):
        """
        Run regression with gradient tracking enabled.
        Handles batch operations properly.
        """
        TH = self.params.time_horizon
        device = power.device
        c_list, d_list, e_list = [], [], []
        a_list, b_list = [], []

        for t in range(TH):
            # Sample points around current operating point
            p_center = power[t].detach()
            h_center = head[t].detach()
            
            # Create sample points with gradient tracking
            p_samples = torch.linspace(
                float(p_center - self.params.δ_p),
                float(p_center + self.params.δ_p),
                self.params.sampling_rate,
                device=device,
                requires_grad=True
            )
            
            h_lo = torch.max(self.params.head_min, h_center - self.params.δ_h)
            h_hi = torch.min(self.params.head_max, h_center + self.params.δ_h)
            h_samples = torch.linspace(
                float(h_lo), 
                float(h_hi), 
                self.params.sampling_rate,
                device=device,
                requires_grad=True
            )
            
            # Create meshgrid
            p_mesh, h_mesh = torch.meshgrid(p_samples, h_samples, indexing="ij")
            p_flat = p_mesh.flatten()
            h_flat = h_mesh.flatten()

            # Filter valid regions
            mask_turbine = (
                (p_flat >= self.params.pos_min_fit[0]*h_flat + self.params.pos_min_fit[1]) &
                (p_flat <= self.params.pos_max_fit[0]*h_flat + self.params.pos_max_fit[1])
            )
            mask_pump = (
                (p_flat >= self.params.neg_min_fit[0]*h_flat + self.params.neg_min_fit[1]) &
                (p_flat <= self.params.neg_max_fit[0]*h_flat + self.params.neg_max_fit[1])
            )
            mask = mask_turbine | mask_pump
            
            if not mask.any():
                # Handle case with no valid points
                c_list.append(torch.zeros(1, device=device, requires_grad=True))
                d_list.append(torch.zeros(1, device=device, requires_grad=True))
                e_list.append(torch.zeros(1, device=device, requires_grad=True))
            else:
                p_valid = p_flat[mask]
                h_valid = h_flat[mask]
                
                # Vectorized q_values calculation
                q_values = predict_q_poly(p_valid, h_valid)
                
                beta = self.least_squares_UPC_torch(p_valid, h_valid, q_values)
                c_list.append(beta[0])
                d_list.append(beta[1])
                e_list.append(beta[2])

            # Regression for v_low
            h_samples_2 = torch.linspace(
                float(h_lo), 
                float(h_hi), 
                self.params.sampling_rate,
                device=device,
                requires_grad=True
            )
            
            # Calculate v_low values
            v_low_values = torch.tensor([
                self.params.h_to_v_low_fitted(h.item()) 
                for h in h_samples_2
            ], dtype=torch.float32, device=device, requires_grad=True)
            
            beta_v = self.least_squares_v_low_torch(h_samples_2, v_low_values)
            a_list.append(beta_v[0])
            b_list.append(beta_v[1])

        # Stack results with gradient tracking
        c_tensor = torch.stack(c_list)
        d_tensor = torch.stack(d_list)
        e_tensor = torch.stack(e_list)
        a_tensor = torch.stack(a_list)
        b_tensor = torch.stack(b_list)

        return c_tensor, d_tensor, e_tensor, a_tensor, b_tensor
    
class TaylorRegressionLayer:
    def __init__(self, params: HydroParameters):
        self.params = params
        
    def compute_UPC_derivatives(self, p, h):
        """
        Compute partial derivatives of q = predict_q_poly(p, h) at point (p, h)
        using numerical differentiation since predict_q_poly may not be directly differentiable.
        
        Args:
            p (float): Power value to evaluate derivative at
            h (float): Head value to evaluate derivative at
            
        Returns:
            tuple: (dq/dp, dq/dh) evaluated at (p, h)
        """
        eps = 1e-6  # Small value for numerical differentiation
        
        # Compute dq/dp using central difference
        q_p_plus = self.params.predict_q_poly(torch.tensor(p + eps), torch.tensor(h))
        q_p_minus = self.params.predict_q_poly(torch.tensor(p - eps), torch.tensor(h))
        dq_dp = (q_p_plus - q_p_minus) / (2 * eps)
        
        # Compute dq/dh using central difference
        q_h_plus = self.params.predict_q_poly(torch.tensor(p), torch.tensor(h + eps))
        q_h_minus = self.params.predict_q_poly(torch.tensor(p), torch.tensor(h - eps))
        dq_dh = (q_h_plus - q_h_minus) / (2 * eps)
        
        # Convert to tensors with gradient tracking
        dq_dp = torch.tensor(float(dq_dp), requires_grad=True)
        dq_dh = torch.tensor(float(dq_dh), requires_grad=True)
        
        return dq_dp, dq_dh

    def compute_volume_derivatives(self, h):
        """
        Compute derivative of v_low = h_to_v_low_fitted(h) at point h
        using numerical differentiation.
        
        Args:
            h (float): Head value to evaluate derivative at
            
        Returns:
            torch.Tensor: dv/dh evaluated at h
        """
        eps = 1e-6  # Small value for numerical differentiation
        
        # Compute dv/dh using central difference
        v_plus = self.params.h_to_v_low_fitted(h + eps)
        v_minus = self.params.h_to_v_low_fitted(h - eps)
        dv_dh = (v_plus - v_minus) / (2 * eps)
        
        # Convert to tensor with gradient tracking
        return torch.tensor(float(dv_dh), requires_grad=True)

    def run_regression(self, power, head):
        """
        Compute local linear approximations using numerical derivatives
        around each operating point.
        
        Args:
            power (torch.Tensor): Power schedule [time_horizon]
            head (torch.Tensor): Head schedule [time_horizon]
            
        Returns:
            tuple: Coefficients (c, d, e, a, b) for linear approximations
                  q ≈ c*p + d*h + e
                  v_low ≈ a*h + b
        """
        TH = self.params.time_horizon
        device = power.device
        
        c_list = []
        d_list = []
        e_list = []
        a_list = []
        b_list = []
        
        for t in range(TH):
            p_t = float(power[t])  # Convert to Python float
            h_t = float(head[t])   # Convert to Python float
            
            if abs(p_t) < 1e-6:  # Idle mode
                c_list.append(torch.zeros(1, device=device, requires_grad=True))
                d_list.append(torch.zeros(1, device=device, requires_grad=True))
                e_list.append(torch.zeros(1, device=device, requires_grad=True))
            else:
                try:
                    # Compute UPC derivatives
                    dq_dp, dq_dh = self.compute_UPC_derivatives(p_t, h_t)
                    
                    # Get q value at operating point
                    q_t = float(self.params.predict_q_poly(
                        torch.tensor(p_t), 
                        torch.tensor(h_t)
                    ))
                    
                    # Compute coefficients ensuring gradient tracking
                    c_list.append(dq_dp.to(device))
                    d_list.append(dq_dh.to(device))
                    e_list.append(torch.tensor(
                        q_t - float(dq_dp)*p_t - float(dq_dh)*h_t,
                        device=device,
                        requires_grad=True
                    ))
                except Exception as e:
                    print(f"Error in UPC derivatives at t={t}: {e}")
                    raise
            
            try:
                # Compute volume derivatives
                dv_dh = self.compute_volume_derivatives(h_t)
                v_t = float(self.params.h_to_v_low_fitted(h_t))
                
                # Compute coefficients ensuring gradient tracking
                a_list.append(dv_dh.to(device))
                b_list.append(torch.tensor(
                    v_t - float(dv_dh)*h_t,
                    device=device,
                    requires_grad=True
                ))
            except Exception as e:
                print(f"Error in volume derivatives at t={t}: {e}")
                raise
        
        # Stack results maintaining gradient tracking
        c = torch.stack(c_list)
        d = torch.stack(d_list)
        e = torch.stack(e_list)
        a = torch.stack(a_list)
        b = torch.stack(b_list)
        
        return c, d, e, a, b

'''
self.cvxpylayer_solution = self.cvxpylayer(*parameters.values(), solver_args={"solve_method": "ECOS","verbose": True,"max_iters": 2000000})
'''
class OptiLayer:
    def __init__(self, params: HydroParameters):
        """
        A class that constructs (and caches) a CVXPY problem for optimization.
        """
        self.params = params
        self.layer = None
        self.power_init = None
        self.head_init = None
        self.flow_init = None

    def initialize_layer(self, power, head, flow):
        """
        Only build the CVXPY problem if needed.
        """
        if (self.layer is None 
            or not torch.allclose(self.power_init, power) 
            or not torch.allclose(self.head_init, head)):
            
            self.power_init = power.detach()
            self.head_init = head.detach()
            self.flow_init = flow.detach()
            self.layer = self._build_cvxpy()

    def _build_cvxpy(self):
        TH = self.params.time_horizon
        # Define CVXPY variables
        p_var = cp.Variable(TH)
        q_var = cp.Variable(TH)
        h_var = cp.Variable(TH)
        v_low_var = cp.Variable(TH)

        # Define CVXPY parameters
        DA_price_param = cp.Parameter(TH)
        c_param = cp.Parameter(TH)
        d_param = cp.Parameter(TH)
        e_param = cp.Parameter(TH)
        a_param = cp.Parameter(TH)
        b_param = cp.Parameter(TH)
        w_p_param = cp.Parameter(TH, nonneg=True)
        w_h_param = cp.Parameter(TH, nonneg=True)
        w_q_param = cp.Parameter(TH, nonneg=True)

        # Warm starts
        p_var.value = self.power_init.tolist()
        h_var.value = self.head_init.tolist()

        # Objective
        revenue = DA_price_param @ p_var
        cost = self.params.operational_cost * cp.sum_squares(p_var)

        power_dev_pen = cp.sum(w_p_param @ cp.square(p_var - self.power_init))
        head_dev_pen = cp.sum(w_h_param @ cp.square(h_var - self.head_init))
        flow_dev_pen = cp.sum(w_q_param @ cp.square(q_var - self.flow_init))

        objective = cp.Maximize(
            revenue 
            - cost
            - power_dev_pen
            - head_dev_pen
            - flow_dev_pen
        )

        # Constraints
        constraints = []
        for t in range(TH):
            # Mode constraints based on sign of power_init
            if self.power_init[t] == 0:
                constraints += [p_var[t] == 0, q_var[t] == 0]
            elif self.power_init[t] > 0:  # Turbine
                constraints += [
                    p_var[t] >= self.params.pos_min_fit[0] * h_var[t] + self.params.pos_min_fit[1],
                    p_var[t] <= self.params.pos_max_fit[0] * h_var[t] + self.params.pos_max_fit[1],
                    q_var[t] == c_param[t] * p_var[t] + d_param[t]*h_var[t] + e_param[t],
                ]
            else:  # Pump
                constraints += [
                    p_var[t] >= self.params.neg_min_fit[0] * h_var[t] + self.params.neg_min_fit[1],
                    p_var[t] <= self.params.neg_max_fit[0] * h_var[t] + self.params.neg_max_fit[1],
                    q_var[t] == c_param[t] * p_var[t] + d_param[t]*h_var[t] + e_param[t],
                ]

            # Head and volume constraints
            constraints += [
                h_var[t] >= self.params.head_min,
                h_var[t] <= self.params.head_max,
                v_low_var[t] == a_param[t] * h_var[t] + b_param[t],
            ]

            # Volume balance
            if t == 0:
                constraints += [v_low_var[0] == self.params.v_low_init + q_var[0] * 3600]
            else:
                constraints += [v_low_var[t] == v_low_var[t-1] + q_var[t] * 3600]

        # Final volume constraint
        constraints += [v_low_var[TH-1] <= self.params.target_vol_low]

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        layer = CvxpyLayer(
            problem,
            parameters=[DA_price_param, c_param, d_param, e_param, a_param, b_param, w_p_param, w_h_param, w_q_param],
            variables=[p_var, q_var, h_var, v_low_var]
        )
        return layer

    def forward(self, DA_prices, c, d, e, a, b, power, head, flow, w_p, w_h, w_q):
        self.initialize_layer(power, head, flow)
        
        try:
            # Solve with more robust settings
            (p_opt, q_opt, h_opt, v_opt) = self.layer(
                DA_prices, c, d, e, a, b, 
                w_p, w_h, w_q,
                solver_args={
                    "solve_method": "ECOS",
                    "max_iters": 200000,  # Increased iterations
                    "reltol": 1e-5,       # Tighter tolerances
                    "abstol": 1e-5,
                    "feastol": 1e-5,
                    "verbose": True
                }
            )
        except Exception as e:
            print(f"\n⚠️ Solver error: {e}")
            print("Problematic parameters:")
            print(f"DA_prices: {DA_prices.detach().numpy().round(2)}")
            print(f"c: {c.detach().numpy().round(2)}")
            print(f"d: {d.detach().numpy().round(2)}")
            print(f"e: {e.detach().numpy().round(2)}")
            print(f"a: {a.detach().numpy().round(2)}")
            print(f"b: {b.detach().numpy().round(2)}")
            print(f"w_p: {w_p.detach().numpy().round(2)}")
            print(f"w_h: {w_h.detach().numpy().round(2)}")
            print(f"w_q: {w_q.detach().numpy().round(2)}\n")
            raise

        # Check for numerical issues
        if any(torch.isnan(tensor).any() for tensor in [p_opt, q_opt, h_opt, v_opt]):
            print("\n❌ NaN detected in solution. Parameters:")
            print(f"c[0]: {c[0].item():.2f}, d[0]: {d[0].item():.2f}, e[0]: {e[0].item():.2f}")
            print(f"w_p[0]: {w_p[0].item():.2f}, w_h[0]: {w_h[0].item():.2f}, w_q[0]: {w_q[0].item():.2f}")

        return p_opt, q_opt, h_opt, v_opt

class SimulationLayer:
    def __init__(self, params):
        """
        A class for minute-by-minute simulation of the operation, 
        using the same parameters object as the other modules.
        """
        self.params = params

    def simulate_operation(self, p, q, h):
        """
        Simulate minute-by-minute operation with physical constraints and calibration.
        
        Args:
            p (torch.Tensor): Hourly power schedule [time_horizon]
            q (torch.Tensor): Hourly flow schedule [time_horizon]
            h (torch.Tensor): Hourly head schedule [time_horizon]
        
        Returns:
            tuple: Calibrated minute-wise (p, q, h, v_low) schedules.
                Each returned tensor has length time_horizon * 60.
        """

        # 1) Expand to minute-level
        p_60 = p.repeat_interleave(60)  # shape: [time_horizon*60]
        q_60 = q.repeat_interleave(60)
        h_60 = h.repeat_interleave(60)

        # 2) Insert an extra element for “end-of-day”
        p_sim = torch.cat([p_60, p_60[-1].unsqueeze(0)])  # shape: [T*60 + 1]
        
        # 3) Add idle minutes between mode changes
        prod_next = p_sim[:-1] * p_sim[1:]  # p_sim[0:-1]*p_sim[1:] to see if it's < 0
        idle_mask = (prod_next < 0)
        # If sign changes, set that minute's power to 0
        p_no_mode_flip = torch.where(
            idle_mask,
            torch.zeros_like(p_sim[:-1]),
            p_sim[:-1]
        )
        # Re-append the last element
        p_no_mode_flip = torch.cat([p_no_mode_flip, p_no_mode_flip[-1].unsqueeze(0)])

        # 4) Backward ramping adjustment
        p_ramped = p_no_mode_flip.clone()  # new reference so we don't do in-place

        def backward_ramp_1hr(segment, p_hour_val, ramp_up, ramp_down):
            """
            segment: shape [60], representing the current hour’s minute-resolution power.
            p_hour_val: the original hourly power we want at minute 0 of this hour-block
            ramp_up, ramp_down: ramping constraints
            Returns a ramped segment of shape [60].
            """
            seg_out = segment.clone()
            # Force the first minute to match the original hourly power
            seg_out[0] = p_hour_val

            # Walk backward over the range [59..1], adjusting each prior minute
            for i in reversed(range(1, 60)):
                diff = seg_out[i] - seg_out[i-1]
                # If diff > ramp_down, seg_out[i-1] = seg_out[i] - ramp_down
                # else if diff < -ramp_up, seg_out[i-1] = seg_out[i] + ramp_up
                # else leave seg_out[i-1] alone
                seg_out[i-1] = torch.where(
                    diff > ramp_down,
                    seg_out[i] - ramp_down,
                    torch.where(
                        diff < -ramp_up,
                        seg_out[i] + ramp_up,
                        seg_out[i-1]
                    )
                )
            return seg_out

        # total minutes = self.params.time_horizon * 60
        for hour in reversed(range(self.params.time_horizon)):
            hour_start = hour * 60
            hour_end = hour_start + 60
            # isolate that hour’s 60-min slice
            hr_segment = p_ramped[hour_start:hour_end]
            # ramp-correct it
            new_segment = backward_ramp_1hr(
                hr_segment,
                p[hour],  # the original hourly power for that block
                self.params.ramp_up,
                self.params.ramp_down
            )
            # reassemble
            p_ramped = torch.cat([
                p_ramped[:hour_start],
                new_segment,
                p_ramped[hour_end:]
            ])
        
        # 5) Forward simulation:

        # Initialize lists for each state
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        # Start states
        T_minutes = len(p_ramped) - 1  # total steps minus the appended “last state”
        v_init = self.params.v_low_init  # user-chosen initial reservoir volume
        p_list.append(p_ramped[0])      # power at minute 0
        q_list.append(q_60[0])          # flow at minute 0 (initial guess)
        h_list.append(h_60[0])          # head at minute 0
        v_list.append(v_init)

        for i in range(T_minutes):
            # Current states from the last appended item
            # p_prev = p_list[-1]
            # q_prev = q_list[-1]
            h_prev = h_list[-1]
            v_prev = v_list[-1]

            # Proposed new power from the ramped schedule:
            p_new = p_ramped[i]
            # We'll figure out q_new based on p_new, mode constraints, etc.
            
            # a) Base: idle => q=0
            # q_candidate = torch.zeros_like(p_new)

            # b) For turbine mode (p_new>0), clamp p between pos_min(h) and pos_max(h)
            #    then get q via polynomial
            p_min_turb = self.params.pos_min(h_prev)
            p_max_turb = self.params.pos_max(h_prev)
            p_new_turb = torch.clamp(p_new, min=p_min_turb, max=p_max_turb)
            q_turb = predict_q_poly(p_new_turb.unsqueeze(0), h_prev.unsqueeze(0)).squeeze(0)

            # c) For pump mode (p_new<0), clamp p between neg_min(h) and neg_max(h)
            p_min_pump = self.params.neg_min(h_prev)
            p_max_pump = self.params.neg_max(h_prev)
            p_new_pump = torch.clamp(p_new, min=p_min_pump, max=p_max_pump)
            q_pump = predict_q_poly(p_new_pump.unsqueeze(0), h_prev.unsqueeze(0)).squeeze(0)

            # Combine these with torch.where logic:
            # If p_new>0 => turbine scenario, if p_new<0 => pump scenario, else idle
            is_turbine = (p_new > 0)
            is_pump    = (p_new < 0)
            # (If exactly zero, stay idle.)

            # Final p_new after clamping:
            p_final = torch.where(
            is_turbine,
            p_new_turb,  # clamp to pos_min/max
            torch.where(
                is_pump,
                p_new_pump,  # clamp to neg_min/max
                torch.zeros_like(p_new)  # idle
            )
            )
            # Final q_new:
            q_final = torch.where(
            is_turbine,
            q_turb,
            torch.where(
                is_pump,
                q_pump,
                torch.zeros_like(q_turb)  # idle
            )
            )

            # d) Update volume
            #    v_next = v_prev + q_final * 60
            v_candidate = v_prev + q_final * 60

            # e) If v_candidate out of bounds => revert to idle (p=0, q=0, no volume change, no head change)
            out_of_bounds = (v_candidate > self.params.max_vol_up) | (v_candidate < self.params.min_vol_low)

            p_next = torch.where(out_of_bounds, torch.zeros_like(p_final), p_final)
            q_next = torch.where(out_of_bounds, torch.zeros_like(q_final), q_final)
            v_next = torch.where(out_of_bounds, v_prev, v_candidate)

            # f) Update head from final volume
            #    If out_of_bounds => h_next = h_prev else h_next = self.params.gross_head(...)
            h_candidate = self.params.gross_head(v_low=v_next)
            h_next = torch.where(out_of_bounds, h_prev, h_candidate)

            # Append these new states
            p_list.append(p_next)
            q_list.append(q_next)
            v_list.append(v_next.item())  # Convert to Python float for appending
            h_list.append(h_next)

            # # Debug print 
            # print(f"Minute {i}: v_low={v_next.item():.3f}")

        p_sim_clb = torch.stack(p_list[:-1])  # length T_minutes
        q_sim_clb = torch.stack(q_list[:-1])
        h_sim_clb = torch.stack(h_list[:-1])
        v_low_clb = torch.tensor(v_list[:-1], dtype=torch.float32)  # Convert to Tensor

        return p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb

    def calc_profit(self, 
                    p_sim_clb, p_opt, v_low_clb, 
                    DA_price_quarter):
        """
        Calculate the daily profit from the final simulation.
        """

        # Expand p_opt from hourly to minute
        p_opt_minute = p_opt.repeat_interleave(60)

        # E.g. quarter-hour intervals => 15 minutes
        e_sim_quarter = p_sim_clb.view(-1, 15).sum(dim=1) * 0.25
        e_opt_quarter = p_opt_minute.view(-1, 15).sum(dim=1) * 0.25

        # Calculate revenue
        revenue_per_quarter = DA_price_quarter * e_sim_quarter

        # Determine the System Imbalance (SI) price
        surplus_penalty_multiplier = -0.5
        shortage_penalty_multiplier = -2

        SI_price = torch.where(
            e_sim_quarter < e_opt_quarter, # Shortage in simulation
            shortage_penalty_multiplier * DA_price_quarter, # Lower output penalty
            surplus_penalty_multiplier * DA_price_quarter # Higher output penalty
        )
        penalty_per_quarter = (e_sim_quarter - e_opt_quarter) * SI_price # Penalty calculation adjusted for MWh
        SI_penalty = penalty_per_quarter.sum()

        # Volume penalty
        volume_deficit = max(0, v_low_clb[-1] - self.params.target_vol_low) # Ensure no penalty if above target
        energy_loss = self.params.rho * volume_deficit * self.params.g * self.params.target_head * self.params.mu / 3.6e9 # Convert from J to MWh
        volume_penalty = energy_loss * torch.max(DA_price_quarter)

        # Operating cost
        operating_cost = self.params.operational_cost * torch.sum(p_sim_clb**2) / 60

        total_profit = revenue_per_quarter.sum() - SI_penalty - volume_penalty - operating_cost
        return total_profit

class LSTMWeightPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, dropout=0.2, time_horizon=24):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        
        # Simplified architecture for easier gradient flow
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Simpler feed-forward layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3 * time_horizon),  # 3 weights per timestep
            nn.Softplus()  # Ensure positive weights
        )
        
        # Initialize weights with slightly larger values
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=1.5)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
                
    def forward(self, x):
        # Add batch dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # LSTM forward pass
        output, _ = self.lstm(x, (h0, c0))
        
        # Use only the last timestep output
        last_output = output[:, -1, :]
        
        # Get weights through feed-forward layers
        weights = self.fc(last_output)
        
        # Reshape weights
        weights = weights.view(-1, 3, self.time_horizon)
        w_p, w_q, w_h = weights[:, 0, :], weights[:, 1, :], weights[:, 2, :]
        
        # Remove batch dimension if it was added
        if x.size(0) == 1:
            w_p, w_q, w_h = w_p.squeeze(0), w_q.squeeze(0), w_h.squeeze(0)
        
        return w_p, w_q, w_h
    
    def predict_weights(self, DA_prices, power, flow, head):
        """
        Predict weights for a sequence of inputs.
        
        Args:
            DA_prices (torch.Tensor): Day-ahead prices [time_horizon]
            power (torch.Tensor): Power values [time_horizon]
            flow (torch.Tensor): Flow values [time_horizon]
            head (torch.Tensor): Head values [time_horizon]
            
        Returns:
            tuple: (w_p, w_q, w_h) weights for each timestep
        """
        # Stack features into sequence
        x = torch.stack([DA_prices, power, flow, head], dim=1)  # [time_horizon, 4]
        
        with torch.no_grad():
            w_p, w_q, w_h = self.forward(x)
        
        return w_p, w_q, w_h

class Pipeline:
    def __init__(self, params: HydroParameters):
        self.params = params
        
        # Initialize LSTM weight predictor
        self.weight_network = LSTMWeightPredictor(
            input_size=4,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            time_horizon=params.time_horizon
        )
        
        # Rest of initialization
        self.regression = RegressionLayer(params)
        self.optimizer = OptiLayer(params)
        self.simulator = SimulationLayer(params)
    
    def forward(self, power_init, head_init, DA_prices, DA_price_quarter):
        """
        Forward pass through the pipeline.
        """
        # 1) Predict initial flow from (p,h)
        flow_init = predict_q_poly(power_init, head_init)
        
        # 2) Predict penalty weights using LSTM
        w_p, w_q, w_h = self.weight_network.predict_weights(
            DA_prices, power_init, flow_init, head_init
        )
        
        # Rest of the forward pass remains the same...
        c, d, e, a, b = self.regression.run_regression(power_init, head_init)
        
        p_opt, q_opt, h_opt, v_opt = self.optimizer.forward(
            DA_prices, c, d, e, a, b,
            power_init, head_init, flow_init,
            w_p, w_h, w_q
        )
        
        p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb = self.simulator.simulate_operation(
            p_opt, q_opt, h_opt
        )
        
        profit = self.simulator.calc_profit(
            p_sim_clb, p_opt, v_low_clb, DA_price_quarter
        )
        
        return profit, p_opt, q_opt, h_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h

# %%


# %% Training 
import torch.optim as optim  
from datetime import datetime, timedelta
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau 

def generate_random_dates_2022():
    """
    Generate 10 random dates from 2022 in "YYYY-MM-DD" format
    """
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    date_range = (end_date - start_date).days + 1
    random_days = random.sample(range(date_range), 10)
    
    random_dates = [
        (start_date + timedelta(days=day)).strftime("%Y-%m-%d") 
        for day in random_days
    ]
    return random_dates

''' 
Early stopping:

validation loss is increasing for a certain number of epochs,
the model is likely overfitting the training data.
5 epochs of increasing validation loss is a common threshold for early stopping.

1. In 2024, 10 typical date (clustering), 3 extreme date, (Highest variance, lowest nagative DA Price, highest DA Price)
2. Within 2023 find the closest date(Mean squared err) of the sampling 2024. (MILP to create 'historycal data')
3. 'historical data' as the hyperparameter of the pipeline

1. back propagation
2. epocs on 10 days of decisions (double for loop: database of 10 days; epocs)
'''
def train_weights(pipeline, num_epochs=100, learning_rate=0.001, patience=5):
    """
    Complete training function with gradient monitoring, dynamic learning rate adjustment,
    and comprehensive tracking of training metrics.
    
    Args:
        pipeline: Pipeline instance with LSTM weight predictor
        num_epochs: Maximum number of epochs to train
        learning_rate: Initial learning rate
        patience: Number of epochs to wait before early stopping
    
    Returns:
        dict: Training results including losses, profits, and monitoring metrics
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Initialize optimizer with gradient clipping
    optimizer = optim.Adam(pipeline.weight_network.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True,
        min_lr=1e-5
    )
    
    # Generate dates and split data
    all_dates = generate_random_dates_2022()
    train_size = int(0.8 * len(all_dates))
    training_dates = all_dates[:train_size]
    validation_dates = all_dates[train_size:]
    
    # Initialize tracking variables
    epoch_losses = []
    validation_losses = []
    daily_profits = {date: [] for date in all_dates}
    learning_rates = []
    weight_histories = {'w_p': [], 'w_q': [], 'w_h': []}
    gradient_norms = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    consecutive_increases = 0
    best_model_state = None
    
    # Progress bar for epochs
    epoch_pbar = trange(num_epochs, desc='Training Progress')
    
    for epoch in epoch_pbar:
        # Training phase
        pipeline.weight_network.train()
        epoch_loss = 0
        epoch_grad_norm = 0
        
        # Iterate over training days
        day_pbar = tqdm(training_dates, desc=f'Epoch {epoch+1}', leave=False)
        for date in day_pbar:
            try:
                # Get price data
                DA_price_hour = read_da_price(date)
                DA_price_quarter = hourly_to_quarterly(DA_price_hour)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                profit, *_, w_p, w_q, w_h = pipeline.forward(power, head, DA_price_hour, DA_price_quarter)
                
                # Store weights for monitoring (first timestep)
                if date == training_dates[0]:
                    weight_histories['w_p'].append(w_p[0].item())
                    weight_histories['w_q'].append(w_q[0].item())
                    weight_histories['w_h'].append(w_h[0].item())
                
                # Compute loss with regularization
                l2_lambda = 0.01
                l2_reg = torch.tensor(0., device=profit.device)
                for param in pipeline.weight_network.parameters():
                    l2_reg += torch.norm(param)
                loss = -profit + l2_lambda * l2_reg
                
                # Backward pass
                loss.backward()
                
                # Compute and store gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    pipeline.weight_network.parameters(), 
                    max_norm=1.0
                )
                epoch_grad_norm += grad_norm.item()
                
                # Optimizer step
                optimizer.step()
                
                # Track metrics
                current_loss = loss.item()
                epoch_loss += current_loss
                daily_profits[date].append(profit.item())
                
                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                day_pbar.set_postfix({
                    'Loss': f'{current_loss:.2f}',
                    'Grad': f'{grad_norm:.2e}',
                    'LR': f'{current_lr:.2e}'
                })
                
                # Print detailed information periodically
                if len(weight_histories['w_p']) % 5 == 0:
                    print(f"\nGradient norm: {grad_norm:.4f}")
                    print(f"Current weights (t=0): w_p={w_p[0]:.4f}, w_q={w_q[0]:.4f}, w_h={w_h[0]:.4f}")
                
            except Exception as e:
                print(f"\nError in training for date {date}: {str(e)}")
                continue
        
        # Calculate average training metrics
        avg_train_loss = epoch_loss / len(training_dates)
        avg_grad_norm = epoch_grad_norm / len(training_dates)
        epoch_losses.append(avg_train_loss)
        gradient_norms.append(avg_grad_norm)
        
        # Validation phase
        pipeline.weight_network.eval()
        val_loss = 0
        
        with torch.no_grad():
            for date in validation_dates:
                try:
                    DA_price_hour = read_da_price(date)
                    DA_price_quarter = hourly_to_quarterly(DA_price_hour)
                    
                    profit, *_ = pipeline.forward(power, head, DA_price_hour, DA_price_quarter)
                    current_val_loss = -profit.item()
                    val_loss += current_val_loss
                    daily_profits[date].append(profit.item())
                    
                except Exception as e:
                    print(f"\nError in validation for date {date}: {str(e)}")
                    continue
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(validation_dates)
        validation_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            consecutive_increases = 0
            best_model_state = {
                key: value.cpu().clone() 
                for key, value in pipeline.weight_network.state_dict().items()
            }
        else:
            consecutive_increases += 1
            if consecutive_increases >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                pipeline.weight_network.load_state_dict(best_model_state)
                break
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.2f}',
            'Val Loss': f'{avg_val_loss:.2f}',
            'LR': f'{current_lr:.2e}',
            'Grad': f'{avg_grad_norm:.2e}'
        })
        
        # Print detailed monitoring information
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1} Statistics:")
            print(f"Average Gradient Norm: {avg_grad_norm:.4f}")
            print(f"Weight Changes (first timestep):")
            print(f"  w_p: {weight_histories['w_p'][-5:]}")
            print(f"  w_q: {weight_histories['w_q'][-5:]}")
            print(f"  w_h: {weight_histories['w_h'][-5:]}")
    
    return {
        'epoch_losses': epoch_losses,
        'validation_losses': validation_losses,
        'daily_profits': daily_profits,
        'training_dates': training_dates,
        'validation_dates': validation_dates,
        'learning_rates': learning_rates,
        'weight_histories': weight_histories,
        'gradient_norms': gradient_norms
    }

def plot_training_results(training_results):
    """
    Plot training metrics including validation loss and learning rates
    """
    import matplotlib.pyplot as plt
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Training and Validation Loss curves
    ax1.plot(training_results['epoch_losses'], label='Training Loss')
    ax1.plot(training_results['validation_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Learning Rate over epochs
    ax2.plot(training_results['learning_rates'], color='green')
    ax2.set_title('Learning Rate over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')  # Use log scale for learning rate
    ax2.grid(True)
    
    # Plot 3: Daily profits
    for date in training_results['training_dates']:
        ax3.plot(training_results['daily_profits'][date], 
                label=f'{date} (Train)', 
                alpha=0.7)
    for date in training_results['validation_dates']:
        ax3.plot(training_results['daily_profits'][date], 
                label=f'{date} (Val)', 
                linestyle='--',
                alpha=0.7)
    ax3.set_title('Daily Profits over Epochs')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Profit')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Plot final weights
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    w_p, w_q, w_h = training_results['final_weights']
    ax.plot(w_p, label='w_p')
    ax.plot(w_q, label='w_q')
    ax.plot(w_h, label='w_h')
    ax.set_title('Final Weights after Training')
    ax.set_xlabel('Time Horizon')
    ax.set_ylabel('Weight Value')
    ax.legend()
    ax.grid(True)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Initialize parameters and pipeline
    params = HydroParameters()
    pipeline = Pipeline(params)
    
    # Train the weights
    training_results = train_weights(pipeline, num_epochs=100)
    
    # Plot results
    plot_training_results(training_results)
    
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
#%%
    # Compare TaylorRegressionLayer and RegressionLayer
    print("\nComparing TaylorRegressionLayer and RegressionLayer:")
    
    # Initialize both regression layers
    taylor_reg = TaylorRegressionLayer(params)
    standard_reg = RegressionLayer(params)
    
    # Get coefficients from both methods
    c_taylor, d_taylor, e_taylor, a_taylor, b_taylor = taylor_reg.run_regression(power, head)
    c_std, d_std, e_std, a_std, b_std = standard_reg.run_regression(power, head)
    
    # Compare UPC coefficients
    print("\nUPC Coefficients Comparison:")
    print("Time | Taylor (c,d,e) | Standard (c,d,e)")
    print("-" * 50)
    for t in range(len(c_taylor)):
        print(f"{t:2d} | {c_taylor[t].item():6.3f}, {d_taylor[t].item():6.3f}, {e_taylor[t].item():6.3f} | "
              f"{c_std[t].item():6.3f}, {d_std[t].item():6.3f}, {e_std[t].item():6.3f}")
    
    # Compare volume coefficients
    print("\nVolume Coefficients Comparison:")
    print("Time | Taylor (a,b) | Standard (a,b)")
    print("-" * 40)
    for t in range(len(a_taylor)):
        print(f"{t:2d} | {a_taylor[t].item():6.3f}, {b_taylor[t].item():6.3f} | "
              f"{a_std[t].item():6.3f}, {b_std[t].item():6.3f}")
    
    # Calculate average absolute differences
    avg_c_diff = torch.mean(torch.abs(c_taylor - c_std)).item()
    avg_d_diff = torch.mean(torch.abs(d_taylor - d_std)).item()
    avg_e_diff = torch.mean(torch.abs(e_taylor - e_std)).item()
    avg_a_diff = torch.mean(torch.abs(a_taylor - a_std)).item()
    avg_b_diff = torch.mean(torch.abs(b_taylor - b_std)).item()
    
    print("\nAverage Absolute Differences:")
    print(f"c coefficient: {avg_c_diff:.4f}")
    print(f"d coefficient: {avg_d_diff:.4f}")
    print(f"e coefficient: {avg_e_diff:.4f}")
    print(f"a coefficient: {avg_a_diff:.4f}")
    print(f"b coefficient: {avg_b_diff:.4f}")
