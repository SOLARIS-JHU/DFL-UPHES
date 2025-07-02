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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = torch.tensor(77.0, device=device)  # Initial head value
v_low_init = torch.tensor(h_to_v_low_fitted(head_init), device=device)  # Initial lower reservoir volume

def hourly_to_quarterly(tensor_data):
    return tensor_data.repeat_interleave(4)

# Read database
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
                'power': torch.tensor(group['Power'].values, dtype=torch.float32, device=device),
                'head': torch.tensor(group['Head'].values, dtype=torch.float32, device=device),
                'flow': torch.tensor(group['Flow'].values, dtype=torch.float32, device=device),
                'price': torch.tensor(group['Price'].values, dtype=torch.float32, device=device),
                'mode': group['Mode'].values
            }
            
            # Add coefficients if available
            if with_coefficients:
                date_data.update({
                    'c': torch.tensor(group['c'].values, dtype=torch.float32, device=device),
                    'd': torch.tensor(group['d'].values, dtype=torch.float32, device=device),
                    'e': torch.tensor(group['e'].values, dtype=torch.float32, device=device),
                    'a': torch.tensor(group['a'].values, dtype=torch.float32, device=device),
                    'b': torch.tensor(group['b'].values, dtype=torch.float32, device=device)
                })
            
            data_by_date[date_str] = date_data
        
        print(f"Successfully loaded data for {len(data_by_date)} days.")
        return data_by_date
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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
        gross_head=gross_head, 
        v_low_to_h_fitted=v_low_to_h_fitted,
    ):
        self.time_horizon = time_horizon
        self.sampling_rate = sampling_rate
        self.operational_cost = operational_cost
        
        self.δ_p = torch.tensor(δ_p, dtype=torch.float32, device=device)
        self.δ_h = torch.tensor(δ_h, dtype=torch.float32, device=device)
        self.δ_q = torch.tensor(δ_q, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rho, dtype=torch.float32, device=device)
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)

        self.head_min = torch.tensor(head_min, dtype=torch.float32, device=device)
        self.head_max = torch.tensor(head_max, dtype=torch.float32, device=device)
        self.max_vol_up = torch.tensor(max_vol_up, dtype=torch.float32, device=device)
        self.min_vol_low = torch.tensor(min_vol_low, dtype=torch.float32, device=device)
        self.ramp_up = torch.tensor(ramp_up, dtype=torch.float32, device=device)
        self.ramp_down = torch.tensor(ramp_down, dtype=torch.float32, device=device)

        self.target_head = torch.tensor(target_head, dtype=torch.float32, device=device)
        self.target_vol_low = torch.tensor(target_vol_low, dtype=torch.float32, device=device)
        self.head_init = torch.tensor(head_init, dtype=torch.float32, device=device)
        self.v_low_init = torch.tensor(v_low_init, dtype=torch.float32, device=device)

        self.neg_min_fit = torch.tensor(neg_min_fit, dtype=torch.float32, device=device)
        self.neg_max_fit = torch.tensor(neg_max_fit, dtype=torch.float32, device=device)
        self.pos_min_fit = torch.tensor(pos_min_fit, dtype=torch.float32, device=device)
        self.pos_max_fit = torch.tensor(pos_max_fit, dtype=torch.float32, device=device)

        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.predict_q_poly = predict_q_poly
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.gross_head = gross_head
        self.v_low_to_h_fitted = v_low_to_h_fitted

    def to_cpu(self):
        """Move all PyTorch tensors to CPU"""
        for attr_name in dir(self):
            # Skip private attributes, methods, and callable attributes
            if attr_name.startswith('_') or callable(getattr(self, attr_name)):
                continue
            
            try:
                attr = getattr(self, attr_name)
                
                # Handle tensors
                if isinstance(attr, torch.Tensor):
                    setattr(self, attr_name, attr.cpu())
                
                # Handle lists of tensors
                elif isinstance(attr, list):
                    new_list = []
                    for item in attr:
                        if isinstance(item, torch.Tensor):
                            new_list.append(item.cpu())
                        else:
                            new_list.append(item)
                    setattr(self, attr_name, new_list)
                
                # Handle dictionaries containing tensors
                elif isinstance(attr, dict):
                    new_dict = {}
                    for key, value in attr.items():
                        if isinstance(value, torch.Tensor):
                            new_dict[key] = value.cpu()
                        else:
                            new_dict[key] = value
                    setattr(self, attr_name, new_dict)
            except: # Skip attributes that cannot be accessed or operated on
                pass 
        
        return self  # Return self to support method chaining

class RegressionLayer:
    def __init__(self, params: HydroParameters):
        self.params = params

    def least_squares_UPC_torch(self, p_samples, h_samples, q_values):
        """
        Perform least squares for q = c*p + d*h + e with gradient tracking.
        Ensures proper shape handling for matrix operations.
        """
        # Implementation remains the same
        # ...

    def least_squares_v_low_torch(self, h_samples, v_low_samples):
        """
        Perform least squares for v_low = a*h + b with gradient tracking.
        Ensures proper shape handling for matrix operations.
        """
        # Implementation remains the same
        # ...

    def run_regression(self, power, head):
        """
        Run regression with gradient tracking enabled.
        Handles batch operations properly with proper error handling.
        """
        TH = self.params.time_horizon
        device = power.device
        c_list, d_list, e_list = [], [], []
        a_list, b_list = [], []

        # Process each time step individually
        for t in range(TH):
            try:
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
                    # Handle case with no valid points - create zero tensors with proper shapes
                    c_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                    d_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                    e_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                else:
                    p_valid = p_flat[mask]
                    h_valid = h_flat[mask]
                    
                    # Vectorized q_values calculation
                    q_values = self.params.predict_q_poly(p_valid, h_valid)
                    
                    beta = self.least_squares_UPC_torch(p_valid, h_valid, q_values)
                    
                    # Convert to scalar tensors
                    c_list.append(beta[0].clone())
                    d_list.append(beta[1].clone())
                    e_list.append(beta[2].clone())

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
                
                # Convert to scalar tensors
                a_list.append(beta_v[0].clone())
                b_list.append(beta_v[1].clone())
                
            except Exception as e:
                # On error, add default values to maintain consistent list lengths
                print(f"Error at time step {t}: {e}")
                c_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                d_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                e_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                a_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                b_list.append(torch.tensor(0.0, device=device, requires_grad=True))

        # Verify all tensors have consistent shapes before stacking
        for i, tensor_list in enumerate([c_list, d_list, e_list, a_list, b_list]):
            list_name = ["c_list", "d_list", "e_list", "a_list", "b_list"][i]
            shapes = [t.shape for t in tensor_list]
            if len(set(shapes)) > 1:
                # Shapes are inconsistent - let's fix them
                print(f"Warning: Inconsistent shapes in {list_name}: {shapes}")
                for j in range(len(tensor_list)):
                    if tensor_list[j].dim() == 0:  # It's a scalar
                        tensor_list[j] = tensor_list[j].reshape(1)
                    elif tensor_list[j].dim() > 1:  # It's a multi-dimensional tensor
                        tensor_list[j] = tensor_list[j].reshape(-1)[0].reshape(1)

        try:
            # Stack results with gradient tracking
            c_tensor = torch.stack(c_list)
            d_tensor = torch.stack(d_list)
            e_tensor = torch.stack(e_list)
            a_tensor = torch.stack(a_list)
            b_tensor = torch.stack(b_list)
        except RuntimeError as e:
            # If stacking still fails, create tensors manually with proper dimensions
            print(f"Stacking error: {e}. Attempting to create tensors manually.")
            
            # Convert to simple Python floats and recreate tensors
            c_values = [float(c.item()) for c in c_list]
            d_values = [float(d.item()) for d in d_list]
            e_values = [float(e.item()) for e in e_list]
            a_values = [float(a.item()) for a in a_list]
            b_values = [float(b.item()) for b in b_list]
            
            c_tensor = torch.tensor(c_values, device=device, requires_grad=True)
            d_tensor = torch.tensor(d_values, device=device, requires_grad=True)
            e_tensor = torch.tensor(e_values, device=device, requires_grad=True)
            a_tensor = torch.tensor(a_values, device=device, requires_grad=True)
            b_tensor = torch.tensor(b_values, device=device, requires_grad=True)

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
        q_p_plus = self.params.predict_q_poly(torch.tensor(p + eps, device=device), torch.tensor(h, device=device))
        q_p_minus = self.params.predict_q_poly(torch.tensor(p - eps, device=device), torch.tensor(h, device=device))
        dq_dp = (q_p_plus - q_p_minus) / (2 * eps)
        
        # Compute dq/dh using central difference
        q_h_plus = self.params.predict_q_poly(torch.tensor(p, device=device), torch.tensor(h + eps, device=device))
        q_h_minus = self.params.predict_q_poly(torch.tensor(p, device=device), torch.tensor(h - eps, device=device))
        dq_dh = (q_h_plus - q_h_minus) / (2 * eps)
        
        # Convert to tensors with gradient tracking
        dq_dp = torch.tensor(float(dq_dp), device=device, requires_grad=True)
        dq_dh = torch.tensor(float(dq_dh), device=device, requires_grad=True)
        
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
        return torch.tensor(float(dv_dh), device=device, requires_grad=True)

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
                        torch.tensor(p_t, device=device), 
                        torch.tensor(h_t, device=device)
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
        self.params = params.to_cpu()
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
            
            self.power_init = power.detach().cpu()
            self.head_init = head.detach().cpu()
            self.flow_init = flow.detach().cpu()
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
        
        DA_prices_cpu = DA_prices.cpu()
        c_cpu = c.cpu()
        d_cpu = d.cpu()
        e_cpu = e.cpu()
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        w_p_cpu = w_p.cpu()
        w_h_cpu = w_h.cpu()
        w_q_cpu = w_q.cpu()
        
        try:
            # Solve with more robust settings
            (p_opt, q_opt, h_opt, v_opt) = self.layer(
                DA_prices_cpu, c_cpu, d_cpu, e_cpu, a_cpu, b_cpu, 
                w_p_cpu, w_h_cpu, w_q_cpu,
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
            print(f"DA_prices: {DA_prices.detach().cpu().numpy().round(2)}")
            print(f"c: {c.detach().cpu().numpy().round(2)}")
            print(f"d: {d.detach().cpu().numpy().round(2)}")
            print(f"e: {e.detach().cpu().numpy().round(2)}")
            print(f"a: {a.detach().cpu().numpy().round(2)}")
            print(f"b: {b.detach().cpu().numpy().round(2)}")
            print(f"w_p: {w_p.detach().cpu().numpy().round(2)}")
            print(f"w_h: {w_h.detach().cpu().numpy().round(2)}")
            print(f"w_q: {w_q.detach().cpu().numpy().round(2)}\n")
            raise

        # Check for numerical issues
        if any(torch.isnan(tensor).any() for tensor in [p_opt, q_opt, h_opt, v_opt]):
            print("\n❌ NaN detected in solution. Parameters:")
            print(f"c[0]: {c[0].item():.2f}, d[0]: {d[0].item():.2f}, e[0]: {e[0].item():.2f}")
            print(f"w_p[0]: {w_p[0].item():.2f}, w_h[0]: {w_h[0].item():.2f}, w_q[0]: {w_q[0].item():.2f}")

        # Threshold processing - adjust values close to zero to exactly 0
        threshold = 0.1
        p_opt_thresholded = torch.where(torch.abs(p_opt) < threshold, torch.zeros_like(p_opt), p_opt)
        q_opt_thresholded = torch.where(torch.abs(q_opt) < threshold, torch.zeros_like(q_opt), q_opt)

        # Calculate profit from the optimization
        revenue = torch.sum(DA_prices_cpu * p_opt_thresholded)
        operating_cost = self.params.operational_cost * torch.sum(p_opt_thresholded**2)
        
        # Calculate the penalty terms from the objective function
        power_dev_pen = torch.sum(w_p_cpu * torch.square(p_opt_thresholded - self.power_init))
        head_dev_pen = torch.sum(w_h_cpu * torch.square(h_opt - self.head_init))
        flow_dev_pen = torch.sum(w_q_cpu * torch.square(q_opt_thresholded - self.flow_init))
        
        # Calculate the complete objective function value
        optimized_objective = revenue - operating_cost - power_dev_pen - head_dev_pen - flow_dev_pen
        
        # Calculate the profit without penalties (for comparison with simulator)
        optimized_profit = revenue - operating_cost

        return p_opt_thresholded, q_opt_thresholded, h_opt, v_opt, optimized_profit, optimized_objective        

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
            q_candidate = torch.zeros_like(p_new)

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
            h_candidate = self.params.v_low_to_h_fitted(v_next)
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
        e_sim_quarter = p_sim_clb.view(-1, 15).mean(dim=1) * 0.25
        e_opt_quarter = p_opt_minute.view(-1, 15).mean(dim=1) * 0.25

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

        total_profit = revenue_per_quarter.sum() - operating_cost - SI_penalty - volume_penalty 
        return total_profit

class WeightPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2, time_horizon=24, archetype='LSTM'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        self.archetype = archetype.upper()
        
        # Select neural network architecture based on archetype
        if self.archetype == 'LSTM':
            # LSTM architecture
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.archetype == 'RNN':
            # RNN architecture
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.archetype == 'FC':
            # Fully connected network
            self.fc_layers = nn.Sequential(
                nn.Linear(input_size * time_horizon, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unsupported archetype: {archetype}. Choose from 'LSTM', 'RNN', or 'FC'.")
        
        # Output layer is the same for all architectures
        self.output = nn.Sequential(
            nn.Linear(hidden_size, 3 * time_horizon),
            nn.Softplus()  # Ensure positive weights
        )
        
        # Initialize weights
        self._init_weights()
                
    def _init_weights(self):
        """Initialize weights based on the selected architecture"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=1.5) # Xavier initialization
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
                
    def forward(self, x):
        # Add batch dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        if self.archetype in ['LSTM', 'RNN']:
            # For recurrent architectures
            if self.archetype == 'LSTM':
                # Initialize hidden and cell states for LSTM
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                # LSTM forward pass
                output, _ = self.rnn(x, (h0, c0))
            else:  # RNN
                # Initialize hidden state for RNN
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                # RNN forward pass
                output, _ = self.rnn(x, h0)
            
            # Use only the last timestep output
            last_output = output[:, -1, :]
            
        else:  # FC architecture
            # Flatten the input for fully connected layers
            batch_size = x.size(0)
            x_flat = x.reshape(batch_size, -1)  # Flatten time and feature dimensions
            last_output = self.fc_layers(x_flat)
        
        # Get weights through output layer
        weights = self.output(last_output)
        
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
    def __init__(self, params: HydroParameters, use_precomputed_coefficients=True, archetype='LSTM'):
        self.params = params
        self.use_precomputed_coefficients = use_precomputed_coefficients
        self.archetype = archetype.upper()
        
        # Load historical data
        if use_precomputed_coefficients:
            self.historical_data = load_historical_data(
                file_path="./Data/database_no_piecewise_with_coeff.csv",
                with_coefficients=True
            )
        else:
            self.historical_data = None
        
        # Initialize neural network weight predictor with the specified architecture
        self.weight_network = WeightPredictor(
            input_size=4,
            hidden_size=32,
            num_layers=1,
            dropout=0.2,
            time_horizon=params.time_horizon,
            archetype=self.archetype
        ).to(device)
        
        # Rest of initialization
        self.regression = RegressionLayer(params)
        self.optimizer = OptiLayer(params)
        self.simulator = SimulationLayer(params)
    
    def get_precomputed_coefficients(self, date_str, hour=None):
        """
        Retrieve precomputed linearization coefficients from historical data.
        
        Parameters:
            date_str (str): Date string in format 'YYYY-MM-DD'
            hour (int, optional): Specific hour to get coefficients for.
                                  If None, returns coefficients for all hours.
        
        Returns:
            tuple: (c, d, e, a, b) tensors. If hour is specified, returns values for that hour.
                  If hour is None, returns tensors of shape [24].
        """
        if not self.use_precomputed_coefficients or self.historical_data is None:
            return None, None, None, None, None
        
        if date_str not in self.historical_data:
            print(f"Warning: No data found for date {date_str}")
            return None, None, None, None, None
        
        date_data = self.historical_data[date_str]
        if 'c' not in date_data:
            print(f"Warning: No coefficient data found for date {date_str}")
            return None, None, None, None, None
            
        c = date_data['c']
        d = date_data['d']
        e = date_data['e']
        a = date_data['a']
        b = date_data['b']
        
        if hour is not None:
            if hour < 0 or hour >= len(c):
                print(f"Warning: Invalid hour {hour}")
                return None, None, None, None, None
            return c[hour], d[hour], e[hour], a[hour], b[hour]
        
        return c, d, e, a, b
    
    def forward(self, power_init, head_init, DA_prices, DA_price_quarter, date_str=None):
        """
        Forward pass through the pipeline.
        
        Parameters:
            power_init (torch.Tensor): Initial power values [time_horizon]
            head_init (torch.Tensor): Initial head values [time_horizon]
            DA_prices (torch.Tensor): Day-ahead prices [time_horizon]
            DA_price_quarter (torch.Tensor): Quarter-hourly prices [time_horizon*4]
            date_str (str, optional): Date string for retrieving precomputed coefficients
        """
        # 1) Predict initial flow from (p,h)
        flow_init = predict_q_poly(power_init, head_init)
    
        # 2) Predict penalty weights using neural network
        w_p, w_q, w_h = self.weight_network.predict_weights(
            DA_prices, power_init, flow_init, head_init
        )
        
        # 3) Get linearization coefficients - either precomputed or computed on-the-fly
        if self.use_precomputed_coefficients and date_str is not None:
            c, d, e, a, b = self.get_precomputed_coefficients(date_str)
            
            # Fall back to computing if precomputed coefficients are not available
            if c is None:
                print(f"No precomputed coefficients found for {date_str}, computing on-the-fly")
                c, d, e, a, b = self.regression.run_regression(power_init, head_init)
        else:
            # Compute linearization coefficients on-the-fly
            c, d, e, a, b = self.regression.run_regression(power_init, head_init)
        
        # 4) Run optimization
        p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective = self.optimizer.forward(
            DA_prices, c, d, e, a, b,
            power_init, head_init, flow_init,
            w_p, w_h, w_q
        )
        
        # Move tensors to GPU if available
        p_opt = p_opt.to(device)
        q_opt = q_opt.to(device)
        h_opt = h_opt.to(device)
        v_opt = v_opt.to(device)

        # Check for NaN values in the optimized tensors
        if any(torch.isnan(tensor).any() for tensor in [p_opt, q_opt, h_opt, v_opt]):
            print("\n❌ NaN detected in optimized solution. Parameters:")
            print(f"c[0]: {c[0].item():.2f}, d[0]: {d[0].item():.2f}, e[0]: {e[0].item():.2f}")
            print(f"w_p[0]: {w_p[0].item():.2f}, w_h[0]: {w_h[0].item():.2f}, w_q[0]: {w_q[0].item():.2f}")
        
        # 5) Simulate operation
        p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb = self.simulator.simulate_operation(
            p_opt, q_opt, h_opt
        )
        
        # 6) Calculate profit
        profit = self.simulator.calc_profit(
            p_sim_clb, p_opt, v_low_clb, DA_price_quarter
        )
        
        return profit, p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h
    
# %% Test the pipeline with precomputed coefficients for the first day
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

if __name__ == "__main__":
    # Initialize parameters
    params = HydroParameters()
    
    # Initialize pipeline with precomputed coefficients
    print("Initializing pipeline with precomputed coefficients...")
    pipeline = Pipeline(params, use_precomputed_coefficients=True)
    
    # Check if historical data was loaded successfully
    if pipeline.historical_data is None or len(pipeline.historical_data) == 0:
        print("Error: Failed to load historical data with precomputed coefficients")
        exit(1)
    
    # Get the first date in the database
    first_date = sorted(list(pipeline.historical_data.keys()))[0]
    print(f"Testing pipeline with precomputed coefficients for date: {first_date}")
    
    # Get the data for the first date
    first_day_data = pipeline.historical_data[first_date]
    
    # Extract power, head, price for the first day
    power_init = first_day_data['power']
    head_init = first_day_data['head']
    flow_init = first_day_data['flow']  # Original flow from the database
    price = first_day_data['price']
    
    # Create quarter-hourly prices (repeating each hourly price 4 times)
    price_quarter = hourly_to_quarterly(price)
    
    # Print the initial data summary
    print(f"\nInitial data summary for {first_date}:")
    print(f"Power range: {power_init.min().item():.2f} to {power_init.max().item():.2f} MW")
    print(f"Head range: {head_init.min().item():.2f} to {head_init.max().item():.2f} m")
    print(f"Price range: {price.min().item():.2f} to {price.max().item():.2f} $/MWh")
    
    # Record start time to measure performance
    start_time = time.time()
    
    # Run the forward pass with precomputed coefficients
    print("Running pipeline with precomputed coefficients...")
    profit, p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h = pipeline.forward(
        power_init, head_init, price, price_quarter, date_str=first_date
    )
    
    # Print optimization results summary
    print(f"\nOptimization results summary:")
    print(f"Simulated Profit: {profit.item():.2f}")
    print(f"Optimized Profit (revenue - operating cost): {optimized_profit.item():.2f}")
    print(f"Optimized Objective (includes penalties): {optimized_objective.item():.2f}")

    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    
    # Print optimization results summary
    print(f"\nOptimization results summary:")
    print(f"Profit: {profit.item():.2f}")
    print(f"Optimized power range: {p_opt.min().item():.2f} to {p_opt.max().item():.2f} MW")
    print(f"Optimized flow range: {q_opt.min().item():.2f} to {q_opt.max().item():.2f} m³/s")
    print(f"Final volume: {v_low_clb[-1].item():.2f} m³")
    
    print("\nHourly schedule:")
    print("Hour  |  Power (MW)  |  Flow (m³/s)  |  Head (m)  |  Volume (m³)")
    print("-" * 65)
    for t in range(len(p_opt)):
        print(f"{t:4d}  |  {p_opt[t].item():10.2f}  |  {q_opt[t].item():11.2f}  |  {h_opt[t].item():8.2f}  |  {v_low_clb[t*60].item():10.2f}")
    
    print("\nSimulation results:")
    print("Hour  |  Power (MW)  |  Flow (m³/s)  |  Head (m)  |  Volume (m³)")
    print("-" * 65)
    for t in range(0, len(p_sim_clb), 60):  # Print every hour (every 60th minute)
        hour = t // 60
        print(f"{hour:4d}  |  {p_sim_clb[t].item():10.2f}  |  {q_sim_clb[t].item():11.2f}  |  {h_sim_clb[t].item():8.2f}  |  {v_low_clb[t].item():10.2f}")
    
    # Plot the results
    def plot_optimization_simulation_results(p_opt, q_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, date_str, max_vol_low=max_vol_low, save_path=None):
        """
        Plot optimization and simulation results comparison with upper reservoir volume and save as SVG
       
        Args:
            p_opt (torch.Tensor): Optimized power schedule (hourly, size=24)
            q_opt (torch.Tensor): Optimized flow schedule (hourly, size=24)
            p_sim_clb (torch.Tensor): Simulated power schedule (per minute, size=1440)
            q_sim_clb (torch.Tensor): Simulated flow schedule (per minute, size=1440)
            h_sim_clb (torch.Tensor): Simulated head schedule (per minute, size=1440)
            v_low_clb (torch.Tensor): Simulated lower reservoir volume (per minute, size=1440)
            date_str (str): Date string for the title
            max_vol_low (float): Maximum volume of reservoirs
            save_path (str): Path where to save the SVG file
        """

        # Ensure all tensors are on CPU and detached from computation graph
        p_opt = p_opt.detach().cpu()
        q_opt = q_opt.detach().cpu()
        p_sim_clb = p_sim_clb.detach().cpu()
        q_sim_clb = q_sim_clb.detach().cpu()
        h_sim_clb = h_sim_clb.detach().cpu()
        v_low_clb = v_low_clb.detach().cpu()

        # Create figure with 4 subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16))
        fig.suptitle(f"Hydro Pump-Storage Optimization Results - {date_str}", fontsize=16)
       
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
        plt.subplots_adjust(top=0.95)  # Make room for suptitle
        
        # Save the figure if path provided
        if save_path:
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
    
    # Plot the results
    print("\nPlotting results...")
    save_path = f"./results/optimized_{first_date}.svg"
    plot_optimization_simulation_results(
        p_opt, q_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb,
        date_str=first_date, save_path=save_path
    )
    
    # Move coefficients to CPU and print the first few for reference
    c_cpu = c.cpu().detach().numpy().round(4)
    d_cpu = d.cpu().detach().numpy().round(4)
    e_cpu = e.cpu().detach().numpy().round(4)
    a_cpu = a.cpu().detach().numpy().round(4)
    b_cpu = b.cpu().detach().numpy().round(4)

    print("\nSample of precomputed coefficients used:")
    print("c (first 3 hours):", c_cpu[:3])
    print("d (first 3 hours):", d_cpu[:3])
    print("e (first 3 hours):", e_cpu[:3])
    print("a (first 3 hours):", a_cpu[:3])
    print("b (first 3 hours):", b_cpu[:3])
    
    # Compare optimized results with original data
    print("\nOptimized vs Original Schedule Comparison:")
    for hour in range(0, 24, 4):  # Print every 4 hours for brevity
        print(f"Hour {hour}:")
        print(f"  Power: Original={power_init[hour]:.2f} MW, Optimized={p_opt[hour]:.2f} MW")
        print(f"  Flow: Original={flow_init[hour]:.2f} m³/s, Optimized={q_opt[hour]:.2f} m³/s")
        print(f"  Head: Original={head_init[hour]:.2f} m, Optimized={h_opt[hour]:.2f} m")

# %% Training 
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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class DirectWeightPipeline:
    """
    A simplified pipeline that directly uses the weight predictor without the simulator.
    It uses optimized_profit from OptiLayer as the loss function to train the weight predictor.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.weight_network = pipeline.weight_network
        self.params = pipeline.params
        self.optimizer = pipeline.optimizer
        self.regression = pipeline.regression
        self.historical_data = pipeline.historical_data
    
    def forward(self, date_str):
        """
        Run a forward pass maintaining the gradient chain, but only up to the optimization step.
        Uses optimized_profit directly instead of simulating the operation.
        """
        # Get the data for this date
        date_data = self.historical_data[date_str]
        power_init = date_data['power']
        head_init = date_data['head']
        price = date_data['price']
        
        # 1) Predict initial flow from (p,h)
        flow_init = predict_q_poly(power_init, head_init)
        
        # 2) Get input features for the weight predictor
        x = torch.stack([price, power_init, flow_init, head_init], dim=1)  # [time_horizon, 4]
        
        # 3) Run weight prediction DIRECTLY with gradient tracking
        w_p, w_q, w_h = self.weight_network(x)
        
        # 4) Get precomputed coefficients
        c, d, e, a, b = None, None, None, None, None
        if date_str in self.historical_data:
            date_data = self.historical_data[date_str]
            if 'c' in date_data:
                c = date_data['c']
                d = date_data['d']
                e = date_data['e']
                a = date_data['a']
                b = date_data['b']
        
        # Fall back to computing coefficients if not available
        if c is None:
            c, d, e, a, b = self.regression.run_regression(power_init, head_init)
        
        # 5) Run optimization and get optimized_profit and optimized_objective directly
        p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective = self.optimizer.forward(
            price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
            power_init.cpu(), head_init.cpu(), flow_init.cpu(),
            w_p.cpu(), w_h.cpu(), w_q.cpu()
        )
        
        # Return optimized_profit, optimized_objective, and other outputs
        return optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, w_p, w_q, w_h, c, d, e, a, b

def train_weight_predictor(pipeline, num_epochs=100, learning_rate=0.0001, patience=10):
    """
    Train the weight predictor neural network using the optimized_profit directly,
    without the simulation step.
    
    Args:
        pipeline: Pipeline instance with LSTM weight predictor
        num_epochs: Maximum number of epochs to train
        learning_rate: Initial learning rate
        patience: Number of epochs to wait before early stopping
    
    Returns:
        dict: Training results including losses and metrics
    """
    # # Set random seeds for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)
    
    # Create a simplified pipeline that maintains gradient flow but uses optimized_profit
    direct_pipeline = DirectWeightPipeline(pipeline)
    
    # Select the first 4 dates from the dataset
    all_dates = sorted(list(pipeline.historical_data.keys()))
    training_dates = all_dates[:4]
    
    print(f"Training on the following dates: {training_dates}")
    
    # Set up TensorBoard logging
    log_dir = Path("./runs/train_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    print("To view training progress, run: tensorboard --logdir=./runs")
    
    # Log hyperparameters to TensorBoard
    writer.add_hparams(
        {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'patience': patience,
            'num_training_dates': len(training_dates)
        },
        {'dummy': 0}  # Required placeholder metric
    )
    
    # Log model architecture details
    for name, module in pipeline.weight_network.named_children():
        writer.add_text(f'model_architecture/{name}', str(module))
        
        # Log layer parameters
        for param_name, param in module.named_parameters():
            writer.add_histogram(f'layer_params/{name}/{param_name}', param.data)
            writer.add_scalar(f'layer_stats/{name}/{param_name}/mean', param.data.mean())
            writer.add_scalar(f'layer_stats/{name}/{param_name}/std', param.data.std())
    
    # Add model summary as text
    model_summary = []
    total_params = 0
    for name, param in pipeline.weight_network.named_parameters():
        model_summary.append(f"{name}: {list(param.shape)}")
        total_params += param.numel()
    model_summary.append(f"\nTotal parameters: {total_params:,}")
    writer.add_text('model_summary', '\n'.join(model_summary))

    # Initialize optimizer
    optimizer = optim.Adam(pipeline.weight_network.parameters(), lr=learning_rate)
    
    # Initialize tracking variables
    epoch_losses = []
    best_loss = float('inf')
    best_model_state = None
    consecutive_increases = 0
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        pipeline.weight_network.train()  # Set model to training mode
        epoch_loss = 0.0
        successful_dates = 0
        
        # Process each date in the training set
        for date_idx, date in enumerate(training_dates):
            try:
                # Zero the gradient buffers
                optimizer.zero_grad()
                
                # Forward pass with the direct pipeline to get optimized_profit
                try:
                    optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, w_p, w_q, w_h, c, d, e, a, b = direct_pipeline.forward(date)
                    
                    # Compute loss (negative optimized_profit since we want to maximize profit)
                    loss = -optimized_profit
                    
                    # Backward pass and update weights
                    loss.backward()
                    
                    # Print gradient information for debugging
                    if epoch == 0 and date_idx == 0:
                        print("\nGradient information:")
                        for name, param in pipeline.weight_network.named_parameters():
                            if param.grad is not None:
                                print(f"{name} - Grad exists: Yes, Grad magnitude: {param.grad.abs().mean().item():.6f}")
                                # Log initial gradient distributions
                                writer.add_histogram(f'Initial_Gradients/{name}', param.grad, 0)
                                
                                # Add gradient flow visualization
                                writer.add_figure('gradient_flow',
                                                plot_grad_flow(pipeline.weight_network.named_parameters()),
                                                global_step=epoch)
                            else:
                                print(f"{name} - Grad exists: No")
                    
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(pipeline.weight_network.parameters(), max_norm=10.0)
                    
                    optimizer.step()
                    
                    # Update epoch loss
                    epoch_loss += loss.item()
                    successful_dates += 1
                    
                    # Log detailed metrics for each date
                    writer.add_scalars(f'Metrics/Date_{date}', {
                        'loss': loss.item(),
                        'optimized_profit': optimized_profit.item()
                    }, epoch)
                    
                    # Log weight values
                    writer.add_scalars(f'Weights/Date_{date}', {
                        'w_p_mean': w_p.mean().item(),
                        'w_q_mean': w_q.mean().item(),
                        'w_h_mean': w_h.mean().item(),
                        'w_p_std': w_p.std().item(),
                        'w_q_std': w_q.std().item(),
                        'w_h_std': w_h.std().item(),
                        'w_p_max': w_p.max().item(),
                        'w_q_max': w_q.max().item(),
                        'w_h_max': w_h.max().item(),
                    }, epoch)
                        
                except Exception as e:
                    print(f"Forward pass error for date {date} in epoch {epoch}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
            except Exception as e:
                print(f"Error processing date {date} in epoch {epoch}: {str(e)}")
                continue
        
        # Check if any dates were processed successfully
        if successful_dates == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: No dates were processed successfully. Skipping epoch.")
            continue
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / successful_dates
        epoch_losses.append(avg_epoch_loss)
        
        # Log comprehensive metrics
        writer.add_scalars('Training_Metrics', {
            'average_loss': avg_epoch_loss,
            'successful_dates': successful_dates,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, epoch)
        
        # Log network weights, gradients and weight distributions
        for name, param in pipeline.weight_network.named_parameters():
            writer.add_histogram(f'Weights/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                
            # Add weight distribution plots
            if 'weight' in name:
                fig = plt.figure()
                plt.hist(param.data.cpu().numpy().flatten(), bins=50)
                plt.title(f'Weight Distribution - {name}')
                writer.add_figure(f'weight_dist/{name}', fig, epoch)
                plt.close()
        
        # Calculate and log epoch duration
        epoch_duration = time.time() - epoch_start_time
        writer.add_scalar('Time/epoch_duration', epoch_duration, epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Optimized Profit: {-avg_epoch_loss:.4f}, Time: {epoch_duration:.2f}s, Successful dates: {successful_dates}/{len(training_dates)}")
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = {
                key: value.cpu().clone() 
                for key, value in pipeline.weight_network.state_dict().items()
            }
            consecutive_increases = 0
            print(f"New best model found with loss: {best_loss:.4f}")
            
            # Log best model metrics
            writer.add_scalar('Best/loss', best_loss, epoch)
        else:
            consecutive_increases += 1
            print(f"Loss did not improve. Patience: {consecutive_increases}/{patience}")
            
            if consecutive_increases >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model
                pipeline.weight_network.load_state_dict(best_model_state)
                break
    
    # Log final metrics
    writer.add_hparams(
        {
            'final_epochs': epoch + 1,
            'best_loss': best_loss
        },
        {'hparam/best_loss': best_loss}
    )
    
    # Close TensorBoard writer
    writer.close()
    
    # Return training results
    return {
        'epoch_losses': epoch_losses,
        'best_loss': best_loss,
        'training_dates': training_dates,
        'best_model_state': best_model_state,
        'tensorboard_log_dir': str(log_dir)
    }

def plot_training_results(training_results):
    """
    Plot learning curve and save it
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_results['epoch_losses'])
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative Profit)')
    plt.grid(True)
    
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig('./results/learning_curve.png')
    plt.show()

def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the network during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
            
    fig = plt.figure(figsize=(10, 7))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    return fig

def test_trained_model(pipeline, date_str):
    """
    Test the trained model on a specific date, showing both optimization results
    and simulation results for comparison
    """
    # Create the direct pipeline for testing
    direct_pipeline = DirectWeightPipeline(pipeline)
    
    # Get the data for this date
    date_data = pipeline.historical_data[date_str]
    power_init = date_data['power']
    head_init = date_data['head']
    price = date_data['price']
    price_quarter = hourly_to_quarterly(price)
    
    # Run through the direct pipeline to get optimized_profit
    optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, *_ = direct_pipeline.forward(date_str)
    
    print(f"Test optimized profit for {date_str}: {optimized_profit.item():.2f}")
    
    # Now run the simulation to see how well the optimized schedule performs
    p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb = pipeline.simulator.simulate_operation(
        p_opt, q_opt, h_opt
    )
    
    # Calculate simulated profit
    simulated_profit = pipeline.simulator.calc_profit(
        p_sim_clb, p_opt, v_low_clb, price_quarter
    )
    
    print(f"Test simulated profit for {date_str}: {simulated_profit.item():.2f}")
    print(f"Difference between optimized and simulated profit: {(optimized_profit - simulated_profit).item():.2f}")
    
    # Plot the results for visual inspection
    save_path = f"./results/trained_model_{date_str}.svg"
    plot_optimization_simulation_results(
        p_opt, q_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb,
        date_str=date_str, save_path=save_path
    )
    
    return {
        'optimized_profit': optimized_profit.item(),
        'simulated_profit': simulated_profit.item(),
        'profit_gap': (optimized_profit - simulated_profit).item(),
        'p_opt': p_opt,
        'q_opt': q_opt,
        'h_opt': h_opt,
        'p_sim_clb': p_sim_clb,
        'q_sim_clb': q_sim_clb,
        'h_sim_clb': h_sim_clb,
        'v_low_clb': v_low_clb
    }

if __name__ == "__main__":
    # Initialize parameters
    params = HydroParameters()
    
    # Initialize pipeline with precomputed coefficients
    print("Initializing pipeline with precomputed coefficients...")
    pipeline = Pipeline(params, use_precomputed_coefficients=True, archetype='RNN')
    
    # Check if historical data was loaded successfully
    if pipeline.historical_data is None or len(pipeline.historical_data) == 0:
        print("Error: Failed to load historical data with precomputed coefficients")
        exit(1)
    
    # Print the number of dates in the dataset
    num_dates = len(pipeline.historical_data)
    print(f"Loaded {num_dates} dates from the historical dataset")
    
    # Train the model
    print("\nStarting training...")
    training_results = train_weight_predictor(
        pipeline,
        num_epochs=100,
        learning_rate=1e-3,
        patience=50
    )
    
    # Save the trained model with TensorBoard log directory
    model_dir = Path("./models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "weight_predictor.pth"
    
    torch.save({
        'model_state_dict': training_results['best_model_state'],
        'training_dates': training_results['training_dates'],
        'best_loss': training_results['best_loss'],
        'epoch_losses': training_results['epoch_losses'],
        'tensorboard_log_dir': training_results['tensorboard_log_dir']
    }, model_path)
    
    print(f"\nTraining completed. Best model saved to {model_path}")
    print(f"TensorBoard logs saved to {training_results['tensorboard_log_dir']}")
    print("To view training visualizations, run: tensorboard --logdir=./runs")
    print("Or use the TensorBoard extension in VSCode")
    
    # Plot the learning curve
    plot_training_results(training_results)
    
    # Test the trained model on the first training date
    test_date = training_results['training_dates'][0]
    print(f"\nTesting trained model on date: {test_date}")
    test_trained_model(pipeline, test_date)

# tensorboard --logdir=./runs

# %%

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import itertools
import sys
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Import the necessary components from the pipeline module
from test_pipeline_learnable_weight_ver import (
    HydroParameters, OptiLayer, SimulationLayer, 
    load_historical_data, hourly_to_quarterly, predict_q_poly
)

def calculate_baseline_profit(price, power, operational_cost):
    """Calculate baseline profit from initial data"""
    revenue = torch.sum(price * power)
    cost = operational_cost * torch.sum(power**2)
    return (revenue - cost).item()

def calculate_baseline_ex_post(price_quarter, power_init, head_init, flow_init, simulator):
    """Calculate baseline ex-post profit from initial data"""
    try:
        # Try to simulate operation using the initial data
        p_init_sim, q_init_sim, h_init_sim, v_low_init_sim = simulator.simulate_operation(
            power_init, flow_init, head_init
        )
        
        # Calculate ex-post profit
        ex_post_profit = simulator.calc_profit(
            p_init_sim, power_init, v_low_init_sim, price_quarter
        )
        
        return ex_post_profit.item()
    except Exception as e:
        print(f"Error calculating baseline ex-post profit: {str(e)}")
        
        # Fallback method: use expanded power directly
        try:
            p_init_minute = power_init.repeat_interleave(60)
            v_low_init = torch.tensor(simulator.params.v_low_init, dtype=torch.float32).expand(len(p_init_minute))
            
            ex_post_profit = simulator.calc_profit(
                p_init_minute, power_init, v_low_init, price_quarter
            )
            
            return ex_post_profit.item()
        except Exception as e2:
            print(f"Fallback calculation also failed: {str(e2)}")
            return float('nan')

def grid_search_weights(params, historical_data, date_str, num_points=2, min_weight=1e-6, max_weight=5):
    """
    Perform grid search over weight parameters w_p, w_q, and w_h with specified range and density.
    
    Args:
        params: HydroParameters object
        historical_data: Dictionary of historical data
        date_str: Date string to use for testing
        num_points: Number of points per dimension in log space (default 10)
        min_weight: Minimum weight value (default 1e-6)
        max_weight: Maximum weight value (default 5)
        
    Returns:
        DataFrame with all test results
    """
    # Get data for the specified date
    date_data = historical_data[date_str]
    power_init = date_data['power']
    head_init = date_data['head']
    flow_init = date_data['flow']
    price = date_data['price']
    price_quarter = hourly_to_quarterly(price)
    
    # Get precomputed coefficients
    c = date_data['c']
    d = date_data['d']
    e = date_data['e']
    a = date_data['a']
    b = date_data['b']
    
    # Calculate baseline objective
    revenue = torch.sum(price * power_init)
    operating_cost = params.operational_cost * torch.sum(power_init**2)
    baseline_objective = revenue - operating_cost
    
    # Create weight grid in log space with specified range and density
    weight_values = np.logspace(np.log10(min_weight), np.log10(max_weight), num_points)
    
    # Initialize OptiLayer and SimulationLayer
    optimizer = OptiLayer(params)
    simulator = SimulationLayer(params)
    
    # Create empty list to store results
    results = []
    
    # Total combinations
    total_combinations = len(weight_values) ** 3
    
    print(f"Starting grid search with {num_points} points per dimension ({total_combinations} total combinations)")
    print(f"Weight range: {min_weight:.2e} to {max_weight:.2e}")
    
    # Loop through all combinations of weights
    for i, (w_p_val, w_h_val, w_q_val) in enumerate(itertools.product(weight_values, weight_values, weight_values)):
        # Create weight tensors of the right shape
        w_p = torch.full_like(power_init, w_p_val)
        w_h = torch.full_like(power_init, w_h_val)
        w_q = torch.full_like(power_init, w_q_val)
        
        # Print progress every 10 combinations
        if i % 10 == 0 or i == total_combinations - 1:
            print(f"Testing combination {i+1}/{total_combinations}: w_p={w_p_val:.6f}, w_h={w_h_val:.6f}, w_q={w_q_val:.6f}")
        
        try:
            # Run optimization
            p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective = optimizer.forward(
                price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                power_init.cpu(), head_init.cpu(), flow_init.cpu(),
                w_p.cpu(), w_h.cpu(), w_q.cpu()
            )
            
            # Run simulation to get ex-post profit
            p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb = simulator.simulate_operation(
                p_opt, q_opt, h_opt
            )
            
            # Calculate ex-post profit
            ex_post_profit = simulator.calc_profit(
                p_sim_clb, p_opt, v_low_clb, price_quarter
            )
            
            # Store results
            results.append({
                'w_p': w_p_val,
                'w_h': w_h_val,
                'w_q': w_q_val,
                'optimized_profit': optimized_profit.item(),
                'optimized_objective': optimized_objective.item(),
                'ex_post_profit': ex_post_profit.item(),
                'successful': True
            })
            
        except Exception as e:
            # Record the failure
            results.append({
                'w_p': w_p_val,
                'w_h': w_h_val,
                'w_q': w_q_val,
                'optimized_profit': float('nan'),
                'optimized_objective': float('nan'),
                'ex_post_profit': float('nan'),
                'successful': False
            })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Calculate baseline profits (using initial values)
    baseline_profit = calculate_baseline_profit(
        price, power_init, params.operational_cost
    )
    
    baseline_ex_post = calculate_baseline_ex_post(
        price_quarter, power_init, head_init, flow_init, simulator
    )
    
    print(f"\nBaseline profit (initial data): {baseline_profit:.2f}")
    print(f"Baseline objective (initial data): {baseline_objective.item():.2f}")
    print(f"Baseline ex-post profit (initial data): {baseline_ex_post:.2f}")
    
    # Add improvement columns
    results_df['profit_improvement'] = results_df['optimized_profit'] - baseline_profit
    results_df['objective_improvement'] = results_df['optimized_objective'] - baseline_objective.item()
    results_df['ex_post_improvement'] = results_df['ex_post_profit'] - baseline_ex_post
    
    return results_df, baseline_profit, baseline_ex_post, baseline_objective.item()

def plot_grid_search_results(results_df, baseline_profit, baseline_ex_post, baseline_objective):
    """Create visualizations of the grid search results"""
    # Filter out failed optimizations
    successful_results = results_df[results_df['successful']]
    
    # Create directory if it doesn't exist
    output_dir = Path("./results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 3D scatter plot of weights vs ex-post profit
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize profit for better color mapping
    min_profit = successful_results['ex_post_profit'].min()
    max_profit = successful_results['ex_post_profit'].max()
    norm_profit = (successful_results['ex_post_profit'] - min_profit) / (max_profit - min_profit)
    
    # Create the scatter plot
    scatter = ax.scatter(
        np.log10(successful_results['w_p']),
        np.log10(successful_results['w_h']),
        np.log10(successful_results['w_q']),
        c=successful_results['ex_post_profit'],
        cmap='viridis',
        s=100 * norm_profit + 20,  # Size based on profit
        alpha=0.8
    )
    
    # Add labels and title
    ax.set_xlabel('log10(w_p)', fontsize=12)
    ax.set_ylabel('log10(w_h)', fontsize=12)
    ax.set_zlabel('log10(w_q)', fontsize=12)
    ax.set_title('Weight Values vs Ex-Post Profit', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Ex-Post Profit', fontsize=12)
    
    # Add best point annotation
    best_idx = successful_results['ex_post_profit'].idxmax()
    best_point = successful_results.loc[best_idx]
    ax.text(
        np.log10(best_point['w_p']),
        np.log10(best_point['w_h']),
        np.log10(best_point['w_q']),
        f"Best: ({best_point['w_p']:.4f}, {best_point['w_h']:.4f}, {best_point['w_q']:.4f})",
        color='red',
        fontsize=10
    )
    
    # Save figure
    plt.savefig(output_dir / "weight_grid_search_3d.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Comparison bar chart of top 10 combinations
    top_10 = successful_results.sort_values('ex_post_profit', ascending=False).head(10)
    
    # Create labels for combinations
    top_10['weights_label'] = top_10.apply(
        lambda row: f"({row['w_p']:.4f}, {row['w_h']:.4f}, {row['w_q']:.4f})", 
        axis=1
    )
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(top_10))
    width = 0.3
    
    plt.bar(x - width, top_10['optimized_profit'], width, label='Optimized Profit')
    plt.bar(x, top_10['optimized_objective'], width, label='Optimized Objective', alpha=0.8)
    plt.bar(x + width, top_10['ex_post_profit'], width, label='Ex-Post Profit')
    
    # Add baseline lines
    plt.axhline(y=baseline_profit, color='r', linestyle='--', label='Baseline Profit')
    plt.axhline(y=baseline_objective, color='g', linestyle='--', label='Baseline Objective')
    plt.axhline(y=baseline_ex_post, color='b', linestyle='--', label='Baseline Ex-Post')
    
    plt.xlabel('Weight Combinations (w_p, w_h, w_q)', fontsize=12)
    plt.ylabel('Profit/Objective Value', fontsize=12)
    plt.title('Top 10 Weight Combinations by Ex-Post Profit', fontsize=16)
    plt.xticks(x, top_10['weights_label'], rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / "weight_grid_search_top10.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_3d_visualizations(results_df, baseline_profit, baseline_ex_post, baseline_objective):
    """
    Create six specific 3D interactive visualizations:
    1. Absolute value of optimized objective
    2. Percentage improvement of optimized objective
    3. Absolute value of optimized profit
    4. Percentage improvement of optimized profit
    5. Absolute value of ex-post profit
    6. Percentage improvement of ex-post profit
    """
    # Filter out failed optimizations
    successful_results = results_df[results_df['successful']].copy()
    
    # Create directory if it doesn't exist
    output_dir = Path("./results/interactive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate percentage improvements
    successful_results['optimized_profit_pct_improvement'] = (
        (successful_results['optimized_profit'] - baseline_profit) / abs(baseline_profit) * 100
    )
    successful_results['ex_post_profit_pct_improvement'] = (
        (successful_results['ex_post_profit'] - baseline_ex_post) / abs(baseline_ex_post) * 100
    )
    successful_results['optimized_objective_pct_improvement'] = (
        (successful_results['optimized_objective'] - baseline_objective) / abs(baseline_objective) * 100
    )
    
    # Find best combinations
    best_opt_idx = successful_results['optimized_profit'].idxmax()
    best_ex_post_idx = successful_results['ex_post_profit'].idxmax()
    best_obj_idx = successful_results['optimized_objective'].idxmax()
    best_opt_combo = successful_results.loc[best_opt_idx]
    best_ex_post_combo = successful_results.loc[best_ex_post_idx]
    best_obj_combo = successful_results.loc[best_obj_idx]
    
    # Create the 6 required plots
    
    # 1. ABSOLUTE VALUE PLOTS
    
    # 1.1 Absolute Optimized Objective Plot
    fig_obj = go.Figure(data=[
        go.Scatter3d(
            x=successful_results['w_p'],
            y=successful_results['w_h'],
            z=successful_results['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=successful_results['optimized_objective'],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Objective Value"),
                showscale=True
            ),
            text=[
                f"w_p: {row['w_p']:.6f}<br>" +
                f"w_h: {row['w_h']:.6f}<br>" +
                f"w_q: {row['w_q']:.6f}<br>" +
                f"Optimized Objective: {row['optimized_objective']:.2f}"
                for _, row in successful_results.iterrows()
            ],
            hoverinfo='text'
        )
    ])
    
    # Add best point as a different marker
    fig_obj.add_trace(go.Scatter3d(
        x=[best_obj_combo['w_p']],
        y=[best_obj_combo['w_h']],
        z=[best_obj_combo['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        name='Best Objective',
        text=[
            f"BEST OBJECTIVE<br>" +
            f"w_p: {best_obj_combo['w_p']:.6f}<br>" +
            f"w_h: {best_obj_combo['w_h']:.6f}<br>" +
            f"w_q: {best_obj_combo['w_q']:.6f}<br>" +
            f"Objective Value: {best_obj_combo['optimized_objective']:.2f}"
        ],
        hoverinfo='text'
    ))
    
    # Update layout
    fig_obj.update_layout(
        title="Absolute Optimized Objective vs. Weight Parameters",
        scene=dict(
            xaxis_title="w_p",
            yaxis_title="w_h",
            zaxis_title="w_q",
            xaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            yaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            zaxis=dict(type="log", gridcolor='rgb(230, 230, 230)')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )
    
    # Add annotation with baseline info
    fig_obj.add_annotation(
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        text=f"Baseline Objective: {baseline_objective:.2f}",
        showarrow=False,
        font=dict(color="black", size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    pio.write_html(fig_obj, file=output_dir / "1_absolute_optimized_objective_3d.html", auto_open=False)
    
    # 1.2 Absolute Optimized Profit Plot
    fig_opt = go.Figure(data=[
        go.Scatter3d(
            x=successful_results['w_p'],
            y=successful_results['w_h'],
            z=successful_results['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=successful_results['optimized_profit'],
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title="Profit Value"),
                showscale=True
            ),
            text=[
                f"w_p: {row['w_p']:.6f}<br>" +
                f"w_h: {row['w_h']:.6f}<br>" +
                f"w_q: {row['w_q']:.6f}<br>" +
                f"Optimized Profit: {row['optimized_profit']:.2f}"
                for _, row in successful_results.iterrows()
            ],
            hoverinfo='text'
        )
    ])
    
    # Add best point as a different marker
    fig_opt.add_trace(go.Scatter3d(
        x=[best_opt_combo['w_p']],
        y=[best_opt_combo['w_h']],
        z=[best_opt_combo['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        name='Best Optimized',
        text=[
            f"BEST OPTIMIZED<br>" +
            f"w_p: {best_opt_combo['w_p']:.6f}<br>" +
            f"w_h: {best_opt_combo['w_h']:.6f}<br>" +
            f"w_q: {best_opt_combo['w_q']:.6f}<br>" +
            f"Optimized Profit: {best_opt_combo['optimized_profit']:.2f}"
        ],
        hoverinfo='text'
    ))
    
    # Update layout
    fig_opt.update_layout(
        title="Absolute Optimized Profit vs. Weight Parameters",
        scene=dict(
            xaxis_title="w_p",
            yaxis_title="w_h",
            zaxis_title="w_q",
            xaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            yaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            zaxis=dict(type="log", gridcolor='rgb(230, 230, 230)')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )
    
    # Add annotation with baseline info
    fig_opt.add_annotation(
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        text=f"Baseline Profit: {baseline_profit:.2f}",
        showarrow=False,
        font=dict(color="black", size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    pio.write_html(fig_opt, file=output_dir / "2_absolute_optimized_profit_3d.html", auto_open=False)
    
    # 1.3 Absolute Ex-Post Profit Plot
    fig_ex_post = go.Figure(data=[
        go.Scatter3d(
            x=successful_results['w_p'],
            y=successful_results['w_h'],
            z=successful_results['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=successful_results['ex_post_profit'],
                colorscale='Cividis',
                opacity=0.8,
                colorbar=dict(title="Ex-Post Profit"),
                showscale=True
            ),
            text=[
                f"w_p: {row['w_p']:.6f}<br>" +
                f"w_h: {row['w_h']:.6f}<br>" +
                f"w_q: {row['w_q']:.6f}<br>" +
                f"Ex-post Profit: {row['ex_post_profit']:.2f}"
                for _, row in successful_results.iterrows()
            ],
            hoverinfo='text'
        )
    ])
    
    # Add best point as a different marker
    fig_ex_post.add_trace(go.Scatter3d(
        x=[best_ex_post_combo['w_p']],
        y=[best_ex_post_combo['w_h']],
        z=[best_ex_post_combo['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        name='Best Ex-Post',
        text=[
            f"BEST EX-POST<br>" +
            f"w_p: {best_ex_post_combo['w_p']:.6f}<br>" +
            f"w_h: {best_ex_post_combo['w_h']:.6f}<br>" +
            f"w_q: {best_ex_post_combo['w_q']:.6f}<br>" +
            f"Ex-post Profit: {best_ex_post_combo['ex_post_profit']:.2f}"
        ],
        hoverinfo='text'
    ))
    
    # Update layout
    fig_ex_post.update_layout(
        title="Absolute Ex-Post Profit vs. Weight Parameters",
        scene=dict(
            xaxis_title="w_p",
            yaxis_title="w_h",
            zaxis_title="w_q",
            xaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            yaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            zaxis=dict(type="log", gridcolor='rgb(230, 230, 230)')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )
    
    # Add annotation with baseline info
    fig_ex_post.add_annotation(
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        text=f"Baseline Ex-Post Profit: {baseline_ex_post:.2f}",
        showarrow=False,
        font=dict(color="black", size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    pio.write_html(fig_ex_post, file=output_dir / "3_absolute_ex_post_profit_3d.html", auto_open=False)
    
    # 2. PERCENTAGE IMPROVEMENT PLOTS
    
    # 2.1 Optimized Objective Percentage Improvement
    fig_obj_pct = go.Figure(data=[
        go.Scatter3d(
            x=successful_results['w_p'],
            y=successful_results['w_h'],
            z=successful_results['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=successful_results['optimized_objective_pct_improvement'],
                colorscale='RdBu',
                cmid=0,  # Set the middle of the colorscale at 0
                opacity=0.8,
                colorbar=dict(title="% Improvement"),
                showscale=True
            ),
            text=[
                f"w_p: {row['w_p']:.6f}<br>" +
                f"w_h: {row['w_h']:.6f}<br>" +
                f"w_q: {row['w_q']:.6f}<br>" +
                f"% Improvement: {row['optimized_objective_pct_improvement']:.2f}%"
                for _, row in successful_results.iterrows()
            ],
            hoverinfo='text'
        )
    ])
    
    # Add best point as a different marker
    fig_obj_pct.add_trace(go.Scatter3d(
        x=[best_obj_combo['w_p']],
        y=[best_obj_combo['w_h']],
        z=[best_obj_combo['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='black',
            symbol='diamond'
        ),
        name='Best Objective',
        text=[
            f"BEST OBJECTIVE<br>" +
            f"w_p: {best_obj_combo['w_p']:.6f}<br>" +
            f"w_h: {best_obj_combo['w_h']:.6f}<br>" +
            f"w_q: {best_obj_combo['w_q']:.6f}<br>" +
            f"% Improvement: {best_obj_combo['optimized_objective_pct_improvement']:.2f}%"
        ],
        hoverinfo='text'
    ))
    
    # Update layout
    fig_obj_pct.update_layout(
        title="Optimized Objective Percentage Improvement vs. Weight Parameters",
        scene=dict(
            xaxis_title="w_p",
            yaxis_title="w_h",
            zaxis_title="w_q",
            xaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            yaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            zaxis=dict(type="log", gridcolor='rgb(230, 230, 230)')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )
    
    # Add annotations explaining colors
    fig_obj_pct.add_annotation(
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        text="<b>Color Legend:</b><br>Blue = Positive Improvement<br>Red = Negative Improvement",
        showarrow=False,
        font=dict(color="black", size=12),
        align="left",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    pio.write_html(fig_obj_pct, file=output_dir / "4_percentage_improvement_objective_3d.html", auto_open=False)
    
    # 2.2 Optimized Profit Percentage Improvement Plot
    fig_opt_pct = go.Figure(data=[
        go.Scatter3d(
            x=successful_results['w_p'],
            y=successful_results['w_h'],
            z=successful_results['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=successful_results['optimized_profit_pct_improvement'],
                colorscale='RdBu',
                cmid=0,  # Set the middle of the colorscale at 0
                opacity=0.8,
                colorbar=dict(title="% Improvement"),
                showscale=True
            ),
            text=[
                f"w_p: {row['w_p']:.6f}<br>" +
                f"w_h: {row['w_h']:.6f}<br>" +
                f"w_q: {row['w_q']:.6f}<br>" +
                f"% Improvement: {row['optimized_profit_pct_improvement']:.2f}%"
                for _, row in successful_results.iterrows()
            ],
            hoverinfo='text'
        )
    ])
    
    # Add best point as a different marker
    fig_opt_pct.add_trace(go.Scatter3d(
        x=[best_opt_combo['w_p']],
        y=[best_opt_combo['w_h']],
        z=[best_opt_combo['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='black',
            symbol='diamond'
        ),
        name='Best Optimized',
        text=[
            f"BEST OPTIMIZED<br>" +
            f"w_p: {best_opt_combo['w_p']:.6f}<br>" +
            f"w_h: {best_opt_combo['w_h']:.6f}<br>" +
            f"w_q: {best_opt_combo['w_q']:.6f}<br>" +
            f"% Improvement: {best_opt_combo['optimized_profit_pct_improvement']:.2f}%"
        ],
        hoverinfo='text'
    ))
    
    # Update layout
    fig_opt_pct.update_layout(
        title="Optimized Profit Percentage Improvement vs. Weight Parameters",
        scene=dict(
            xaxis_title="w_p",
            yaxis_title="w_h",
            zaxis_title="w_q",
            xaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            yaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            zaxis=dict(type="log", gridcolor='rgb(230, 230, 230)')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )
    
    # Add annotations explaining colors
    fig_opt_pct.add_annotation(
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        text="<b>Color Legend:</b><br>Blue = Positive Improvement<br>Red = Negative Improvement",
        showarrow=False,
        font=dict(color="black", size=12),
        align="left",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    pio.write_html(fig_opt_pct, file=output_dir / "5_percentage_improvement_optimized_profit_3d.html", auto_open=False)
    
    # 2.3 Ex-Post Profit Percentage Improvement Plot
    fig_ex_post_pct = go.Figure(data=[
        go.Scatter3d(
            x=successful_results['w_p'],
            y=successful_results['w_h'],
            z=successful_results['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=successful_results['ex_post_profit_pct_improvement'],
                colorscale='RdBu',
                cmid=0,  # Set the middle of the colorscale at 0
                opacity=0.8,
                colorbar=dict(title="% Improvement"),
                showscale=True
            ),
            text=[
                f"w_p: {row['w_p']:.6f}<br>" +
                f"w_h: {row['w_h']:.6f}<br>" +
                f"w_q: {row['w_q']:.6f}<br>" +
                f"% Improvement: {row['ex_post_profit_pct_improvement']:.2f}%"
                for _, row in successful_results.iterrows()
            ],
            hoverinfo='text'
        )
    ])
    
    # Add best point as a different marker
    fig_ex_post_pct.add_trace(go.Scatter3d(
        x=[best_ex_post_combo['w_p']],
        y=[best_ex_post_combo['w_h']],
        z=[best_ex_post_combo['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='black',
            symbol='diamond'
        ),
        name='Best Ex-Post',
        text=[
            f"BEST EX-POST<br>" +
            f"w_p: {best_ex_post_combo['w_p']:.6f}<br>" +
            f"w_h: {best_ex_post_combo['w_h']:.6f}<br>" +
            f"w_q: {best_ex_post_combo['w_q']:.6f}<br>" +
            f"% Improvement: {best_ex_post_combo['ex_post_profit_pct_improvement']:.2f}%"
        ],
        hoverinfo='text'
    ))
    
    # Update layout
    fig_ex_post_pct.update_layout(
        title="Ex-Post Profit Percentage Improvement vs. Weight Parameters",
        scene=dict(
            xaxis_title="w_p",
            yaxis_title="w_h",
            zaxis_title="w_q",
            xaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            yaxis=dict(type="log", gridcolor='rgb(230, 230, 230)'),
            zaxis=dict(type="log", gridcolor='rgb(230, 230, 230)')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )
    
    # Add annotations explaining colors
    fig_ex_post_pct.add_annotation(
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        text="<b>Color Legend:</b><br>Blue = Positive Improvement<br>Red = Negative Improvement",
        showarrow=False,
        font=dict(color="black", size=12),
        align="left",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    pio.write_html(fig_ex_post_pct, file=output_dir / "6_percentage_improvement_ex_post_profit_3d.html", auto_open=False)
    
    # Create a dashboard with all plots and key insights
    create_summary_dashboard(successful_results, best_opt_combo, best_ex_post_combo, best_obj_combo, 
                           baseline_profit, baseline_ex_post, baseline_objective, output_dir)
    
    print(f"Created 6 interactive 3D plots in {output_dir}")

def create_summary_dashboard(df, best_opt_combo, best_ex_post_combo, best_obj_combo, 
                           baseline_profit, baseline_ex_post, baseline_objective, output_dir):
    """Create an HTML dashboard with links to all plots and key insights"""
    
    # Import required for datetime
    from datetime import datetime
    
    # Generate a simple HTML file with links to all plots
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UPHES Optimization Results Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .plot-link {{ 
                width: 45%; 
                margin: 10px; 
                padding: 15px; 
                background-color: #f8f9fa; 
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .plot-link h3 {{ color: #3498db; }}
            .insights {{ 
                background-color: #e8f4f8; 
                padding: 15px; 
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>UPHES Weight Parameter Optimization Results</h1>
        
        <div class="insights">
            <h2>Key Insights</h2>
            <p>
                <span class="metric">Best Weight Combinations:</span><br>
                <b>Best Objective:</b> w_p={best_obj_combo['w_p']:.6f}, w_h={best_obj_combo['w_h']:.6f}, w_q={best_obj_combo['w_q']:.6f}<br>
                <b>Best Optimized Profit:</b> w_p={best_opt_combo['w_p']:.6f}, w_h={best_opt_combo['w_h']:.6f}, w_q={best_opt_combo['w_q']:.6f}<br>
                <b>Best Ex-Post Profit:</b> w_p={best_ex_post_combo['w_p']:.6f}, w_h={best_ex_post_combo['w_h']:.6f}, w_q={best_ex_post_combo['w_q']:.6f}
            </p>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Best Value</th>
                    <th>Improvement</th>
                    <th>% Improvement</th>
                </tr>
                <tr>
                    <td>Optimized Objective</td>
                    <td>{baseline_objective:.2f}</td>
                    <td>{best_obj_combo['optimized_objective']:.2f}</td>
                    <td>{best_obj_combo['optimized_objective'] - baseline_objective:.2f}</td>
                    <td>{(best_obj_combo['optimized_objective'] - baseline_objective)/abs(baseline_objective)*100:.2f}%</td>
                </tr>
                <tr>
                    <td>Optimized Profit</td>
                    <td>{baseline_profit:.2f}</td>
                    <td>{best_opt_combo['optimized_profit']:.2f}</td>
                    <td>{best_opt_combo['optimized_profit'] - baseline_profit:.2f}</td>
                    <td>{(best_opt_combo['optimized_profit'] - baseline_profit)/abs(baseline_profit)*100:.2f}%</td>
                </tr>
                <tr>
                    <td>Ex-Post Profit</td>
                    <td>{baseline_ex_post:.2f}</td>
                    <td>{best_ex_post_combo['ex_post_profit']:.2f}</td>
                    <td>{best_ex_post_combo['ex_post_profit'] - baseline_ex_post:.2f}</td>
                    <td>{(best_ex_post_combo['ex_post_profit'] - baseline_ex_post)/abs(baseline_ex_post)*100:.2f}%</td>
                </tr>
            </table>
        </div>
        
        <h2>Interactive 3D Plots</h2>
        
        <div class="container">
            <div class="plot-link">
                <h3>Absolute Values</h3>
                <p><a href="1_absolute_optimized_objective_3d.html" target="_blank">1. Absolute Optimized Objective</a></p>
                <p><a href="2_absolute_optimized_profit_3d.html" target="_blank">2. Absolute Optimized Profit</a></p>
                <p><a href="3_absolute_ex_post_profit_3d.html" target="_blank">3. Absolute Ex-Post Profit</a></p>
            </div>
            
            <div class="plot-link">
                <h3>Percentage Improvements</h3>
                <p><a href="4_percentage_improvement_objective_3d.html" target="_blank">4. Optimized Objective % Improvement</a></p>
                <p><a href="5_percentage_improvement_optimized_profit_3d.html" target="_blank">5. Optimized Profit % Improvement</a></p>
                <p><a href="6_percentage_improvement_ex_post_profit_3d.html" target="_blank">6. Ex-Post Profit % Improvement</a></p>
            </div>
        </div>
        
        <div class="insights">
            <h2>Observations</h2>
            <p>This dashboard presents the results of a grid search over {len(df)} weight combinations for the UPHES optimization problem.</p>
            <ul>
                <li>The best weights for optimized objective, optimized profit, and ex-post profit are often different, suggesting different tradeoffs.</li>
                <li>Blue regions in percentage improvement plots show positive improvements over the baseline.</li>
                <li>Very small weight values (near {df['w_p'].min():.2e}) or very large values (near {df['w_p'].max():.2f}) typically result in suboptimal performance.</li>
                <li>The correlation between optimized profit and ex-post profit shows how well the optimization translates to real-world performance.</li>
            </ul>
        </div>
        
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body>
    </html>
    """
    
    # Write the HTML file
    dashboard_path = output_dir / "dashboard.html"
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created summary dashboard at {dashboard_path}")

def create_top_performers_table(df, output_dir):
    """Create an interactive table of top performers for each metric"""
    
    # Create combined table of top performers for each metric
    df_top_opt = df.sort_values('optimized_profit', ascending=False).head(5)[
        ['w_p', 'w_h', 'w_q', 'optimized_profit', 'optimized_objective', 'ex_post_profit']
    ].copy()
    df_top_opt['rank_type'] = 'Best Optimized Profit'
    
    df_top_ex_post = df.sort_values('ex_post_profit', ascending=False).head(5)[
        ['w_p', 'w_h', 'w_q', 'optimized_profit', 'optimized_objective', 'ex_post_profit']
    ].copy()
    df_top_ex_post['rank_type'] = 'Best Ex-Post Profit'
    
    df_top_obj = df.sort_values('optimized_objective', ascending=False).head(5)[
        ['w_p', 'w_h', 'w_q', 'optimized_profit', 'optimized_objective', 'ex_post_profit']
    ].copy()
    df_top_obj['rank_type'] = 'Best Optimized Objective'
    
    # Combine tables
    df_top = pd.concat([df_top_opt, df_top_ex_post, df_top_obj])
    
    # Create ranking within each group
    df_top['rank'] = df_top.groupby('rank_type').cumcount() + 1
    
    # Generate colors for each row
    row_colors = []
    for r in df_top['rank_type']:
        if r == 'Best Optimized Profit':
            row_colors.append('lightblue')
        elif r == 'Best Optimized Objective':
            row_colors.append('lightgreen')
        else:  # 'Best Ex-Post Profit'
            row_colors.append('lightcoral')
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank Type', 'Rank', 'w_p', 'w_h', 'w_q', 'Optimized Profit', 'Optimized Objective', 'Ex-Post Profit'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[
                df_top['rank_type'],
                df_top['rank'],
                df_top['w_p'].apply(lambda x: f"{x:.6f}"),
                df_top['w_h'].apply(lambda x: f"{x:.6f}"),
                df_top['w_q'].apply(lambda x: f"{x:.6f}"),
                df_top['optimized_profit'].apply(lambda x: f"{x:.2f}"),
                df_top['optimized_objective'].apply(lambda x: f"{x:.2f}"),
                df_top['ex_post_profit'].apply(lambda x: f"{x:.2f}")
            ],
            fill_color=[row_colors] * 8,  # Repeat the same colors for all columns
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title="Top Weight Combinations by Different Metrics",
        height=600,
        width=1000
    )
    
    pio.write_html(fig, file=output_dir / "top_performers_table.html", auto_open=False)
    
    # Also save as CSV for reference
    df_top.to_csv(output_dir / "top_performers.csv", index=False)

def run_dense_grid_search():
    """
    Run grid search with 10 points per dimension in range 1e-6 to 5.
    """
    # Start timing
    start_time = time.time()
    
    # Initialize parameters
    params = HydroParameters()
    
    # Load historical data
    print("Loading historical data...")
    historical_data = load_historical_data(
        file_path="./Data/database_no_piecewise_with_coeff.csv",
        with_coefficients=True
    )
    
    if historical_data is None or len(historical_data) == 0:
        print("Error: Failed to load historical data")
        return
    
    # Get the first date
    first_date = sorted(list(historical_data.keys()))[0]
    print(f"Running dense grid search for date: {first_date}")
    
    # Run grid search with requested parameters:
    # - 10 points per dimension (1000 total combinations)
    # - Range from 1e-6 to 5
    results_df, baseline_profit, baseline_ex_post, baseline_objective = grid_search_weights(
        params, historical_data, first_date, 
        num_points=30,  # 10 points per dimension
        min_weight=1e-8,  # Minimum weight 1e-6
        max_weight=5     # Maximum weight 5
    )
    
    # Save results
    output_dir = Path("./results")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "dense_grid_search_results.csv", index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_grid_search_results(results_df, baseline_profit, baseline_ex_post, baseline_objective)
    
    # Create interactive Plotly visualizations
    print("\nCreating interactive 3D visualizations...")
    create_interactive_3d_visualizations(results_df, baseline_profit, baseline_ex_post, baseline_objective)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"\nGrid search completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # Print best results summary
    print_best_results_summary(results_df, baseline_profit, baseline_ex_post, baseline_objective)
    
    return results_df, baseline_profit, baseline_ex_post, baseline_objective

def print_best_results_summary(results_df, baseline_profit, baseline_ex_post, baseline_objective):
    """Print a summary of the best results from the grid search"""
    
    # Filter successful trials
    successful_results = results_df[results_df['successful']]
    
    # Find best combinations
    best_opt_idx = successful_results['optimized_profit'].idxmax()
    best_ex_post_idx = successful_results['ex_post_profit'].idxmax()
    best_obj_idx = successful_results['optimized_objective'].idxmax()
    
    best_opt_combo = successful_results.loc[best_opt_idx]
    best_ex_post_combo = successful_results.loc[best_ex_post_idx]
    best_obj_combo = successful_results.loc[best_obj_idx]
    
    # Print summary
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nSuccessful optimizations: {successful_results.shape[0]} out of {results_df.shape[0]} " +
          f"({successful_results.shape[0]/results_df.shape[0]*100:.1f}%)")
    
    print("\nBaseline Values:")
    print(f"  Optimized Profit: {baseline_profit:.2f}")
    print(f"  Optimized Objective: {baseline_objective:.2f}")
    print(f"  Ex-Post Profit: {baseline_ex_post:.2f}")
    
    print("\nBest Optimized Profit Combination:")
    print(f"  Weights: w_p={best_opt_combo['w_p']:.6f}, w_h={best_opt_combo['w_h']:.6f}, w_q={best_opt_combo['w_q']:.6f}")
    print(f"  Optimized Profit: {best_opt_combo['optimized_profit']:.2f} (Improvement: {best_opt_combo['optimized_profit']-baseline_profit:.2f})")
    print(f"  Optimized Objective: {best_opt_combo['optimized_objective']:.2f}")
    print(f"  Ex-Post Profit: {best_opt_combo['ex_post_profit']:.2f}")
    
    print("\nBest Ex-Post Profit Combination:")
    print(f"  Weights: w_p={best_ex_post_combo['w_p']:.6f}, w_h={best_ex_post_combo['w_h']:.6f}, w_q={best_ex_post_combo['w_q']:.6f}")
    print(f"  Optimized Profit: {best_ex_post_combo['optimized_profit']:.2f}")
    print(f"  Optimized Objective: {best_ex_post_combo['optimized_objective']:.2f}")
    print(f"  Ex-Post Profit: {best_ex_post_combo['ex_post_profit']:.2f} (Improvement: {best_ex_post_combo['ex_post_profit']-baseline_ex_post:.2f})")
    
    print("\nBest Optimized Objective Combination:")
    print(f"  Weights: w_p={best_obj_combo['w_p']:.6f}, w_h={best_obj_combo['w_h']:.6f}, w_q={best_obj_combo['w_q']:.6f}")
    print(f"  Optimized Profit: {best_obj_combo['optimized_profit']:.2f}")
    print(f"  Optimized Objective: {best_obj_combo['optimized_objective']:.2f} (Improvement: {best_obj_combo['optimized_objective']-baseline_objective:.2f})")
    print(f"  Ex-Post Profit: {best_obj_combo['ex_post_profit']:.2f}")
    
    print("\nAnalysis of w_p, w_h, w_q relationships to profits:")
    
    # Calculate correlations
    correlations = successful_results[['w_p', 'w_h', 'w_q', 'optimized_profit', 'optimized_objective', 'ex_post_profit']].corr()
    print("\nCorrelation of weights with profits:")
    print(correlations.loc[['w_p', 'w_h', 'w_q'], ['optimized_profit', 'optimized_objective', 'ex_post_profit']].round(3))
    
    print("="*80)

if __name__ == "__main__":
    # Run the grid search with dense grid
    print("Running dense grid search (10 points per dimension, range 1e-6 to 5)...")
    results_df, baseline_profit, baseline_ex_post, baseline_objective = run_dense_grid_search()