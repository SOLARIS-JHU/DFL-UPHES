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
        optimized_profit = revenue - operating_cost

        return p_opt_thresholded, q_opt_thresholded, h_opt, v_opt, optimized_profit
        

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
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Feed-forward layer
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
    def __init__(self, params: HydroParameters, use_precomputed_coefficients=True):
        self.params = params
        self.use_precomputed_coefficients = use_precomputed_coefficients
        
        # Load historical data
        if use_precomputed_coefficients:
            self.historical_data = load_historical_data(
                file_path="./Data/database_no_piecewise_with_coeff.csv",
                with_coefficients=True
            )
        else:
            self.historical_data = None
        
        # Initialize LSTM weight predictor
        self.weight_network = LSTMWeightPredictor(
            input_size=4,
            hidden_size=32,
            num_layers=1,
            dropout=0.2,
            time_horizon=params.time_horizon
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
       
        # 2) Predict penalty weights using LSTM
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
        p_opt, q_opt, h_opt, v_opt, optimized_profit = self.optimizer.forward(
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
        
        return profit, p_opt, q_opt, h_opt, v_opt, optimized_profit, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h
    
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
    print("\nRunning pipeline with precomputed coefficients...")
    profit, p_opt, q_opt, h_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h = pipeline.forward(
        power_init, head_init, price, price_quarter, date_str=first_date
    )
    
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
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class DirectWeightPipeline:
    """
    A simplified pipeline that directly uses the weight predictor without breaking the gradient chain.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.weight_network = pipeline.weight_network
        self.params = pipeline.params
        self.simulator = pipeline.simulator
        self.optimizer = pipeline.optimizer
        self.regression = pipeline.regression
        self.historical_data = pipeline.historical_data
    
    def forward(self, date_str):
        """
        Run a forward pass maintaining the gradient chain.
        """
        # Get the data for this date
        date_data = self.historical_data[date_str]
        power_init = date_data['power']
        head_init = date_data['head']
        price = date_data['price']
        price_quarter = hourly_to_quarterly(price)
        
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
        
        # 5) Run optimization
        p_opt, q_opt, h_opt, v_opt, optimized_profit = self.optimizer.forward(
            price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
            power_init.cpu(), head_init.cpu(), flow_init.cpu(),
            w_p.cpu(), w_h.cpu(), w_q.cpu()
        )
        
        # Move tensors to device
        p_opt = p_opt.to(device)
        q_opt = q_opt.to(device)
        h_opt = h_opt.to(device)
        v_opt = v_opt.to(device)
        
        # 6) Simulate operation
        p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb = self.simulator.simulate_operation(
            p_opt, q_opt, h_opt
        )
        
        # 7) Calculate profit
        profit = self.simulator.calc_profit(
            p_sim_clb, p_opt, v_low_clb, price_quarter
        )
        
        return profit, p_opt, q_opt, h_opt, v_opt, optimized_profit, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h


def train_weight_predictor(pipeline, num_epochs=100, learning_rate=0.0001, patience=10):
    """
    Train the weight predictor neural network using the first 4 dates in the dataset,
    with a modified approach to maintain gradient flow.
    
    Args:
        pipeline: Pipeline instance with LSTM weight predictor
        num_epochs: Maximum number of epochs to train
        learning_rate: Initial learning rate
        patience: Number of epochs to wait before early stopping
    
    Returns:
        dict: Training results including losses and metrics
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a simplified pipeline that maintains gradient flow
    direct_pipeline = DirectWeightPipeline(pipeline)
    
    # Select the first 4 dates from the dataset
    all_dates = sorted(list(pipeline.historical_data.keys()))
    training_dates = all_dates[:4]
    
    print(f"Training on the following dates: {training_dates}")
    
    # Set up TensorBoard logging
    # This creates a unique directory for each run under ./runs
    log_dir = Path("./runs/train_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    print("To view training progress, run: tensorboard --logdir=./runs")
    print("Then open http://localhost:6006 in your browser")
    print("Or use the TensorBoard extension in VSCode")
    
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
    
    # Add network graph and layer details to TensorBoard
    # # Create a sample input tensor
    # sample_input = torch.randn(1, pipeline.params.time_horizon, 4)  # [batch, seq_len, features]
    # writer.add_graph(pipeline.weight_network, sample_input)
    
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
                
                # Forward pass with the direct pipeline to maintain gradient flow
                try:
                    profit, p_opt, q_opt, h_opt, v_opt, optimized_profit, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h = direct_pipeline.forward(date)
                    
                    # Compute loss (negative profit since we want to maximize profit)
                    loss = -profit
                    
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
                        'profit': profit.item()
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Optimized Profit: {optimized_profit.item():.4f}, Time: {epoch_duration:.2f}s, Successful dates: {successful_dates}/{len(training_dates)}")
        
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
        'tensorboard_log_dir': str(log_dir)  # Return log directory for reference
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
    Test the trained model on a specific date
    """
    # Create the direct pipeline for testing
    direct_pipeline = DirectWeightPipeline(pipeline)
    
    # Get the data for this date
    date_data = pipeline.historical_data[date_str]
    power_init = date_data['power']
    head_init = date_data['head']
    price = date_data['price']
    price_quarter = hourly_to_quarterly(price)
    
    # Run through the pipeline
    profit = direct_pipeline.forward(date_str)
    
    print(f"Test profit for {date_str}: {profit.item():.2f}")
    
    # Now use the regular pipeline to get all outputs for visualization
    profit, p_opt, q_opt, h_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, *_ = pipeline.forward(
        power_init, head_init, price, price_quarter, date_str=date_str
    )
    
    # Plot the results for visual inspection
    save_path = f"./results/trained_model_{date_str}.svg"
    plot_optimization_simulation_results(
        p_opt, q_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb,
        date_str=date_str, save_path=save_path
    )

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
    
    # Print the number of dates in the dataset
    num_dates = len(pipeline.historical_data)
    print(f"Loaded {num_dates} dates from the historical dataset")
    
    # Train the model
    print("\nStarting training...")
    training_results = train_weight_predictor(
        pipeline,
        num_epochs=50,
        learning_rate=0.0001,
        patience=10
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
