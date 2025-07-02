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
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
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
# Function to read data from CSV file 
def load_historical_data(file_path="./Data/database_no_piecewise_with_coeff.csv", with_coefficients=False):
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

class TaylorRegressionLayer:
    def __init__(self, params: HydroParameters):
        """
        A class for performing first-order Taylor approximation
        on the nonlinear functions in the hydropower optimization problem.
        """
        self.params = params

    def calculate_gradients(self, func, x, create_graph=False, retain_graph=False):
        """
        Calculate gradients of a function with respect to its inputs.
        
        Args:
            func: A callable that computes the output
            x: Input tensor with requires_grad=True
            create_graph: Whether to create a computational graph for the gradient
            retain_graph: Whether to retain the computational graph
            
        Returns:
            Gradient of func with respect to x
        """
        try:
            y = func(x)
            grad = torch.autograd.grad(
                outputs=y, 
                inputs=x, 
                create_graph=create_graph, 
                retain_graph=retain_graph,
                grad_outputs=torch.ones_like(y)
            )[0]
            return grad
        except Exception as e:
            print(f"Error in gradient calculation: {e}")
            return torch.tensor(0.0, device=x.device, requires_grad=True)

    def run_regression(self, power, head, flow=None):
        """
        Run Taylor regression to linearize the nonlinear functions
        at each operational point.
        
        Args:
            power (torch.Tensor): Power values [time_horizon]
            head (torch.Tensor): Head values [time_horizon]
            flow (torch.Tensor, optional): Flow values [time_horizon]
            
        Returns:
            tuple: (c, d, e, a, b) tensors for the linearized equations:
                  q = c*p + d*h + e
                  v_low = a*h + b
        """
        TH = self.params.time_horizon
        device = power.device
        c_list, d_list, e_list = [], [], []
        a_list, b_list = [], []

        # Process each time step individually
        for t in range(TH):
            try:
                # Get operational point
                p0 = power[t].detach().clone().requires_grad_(True)
                h0 = head[t].detach().clone().requires_grad_(True)
                
                # Skip computation for idle mode
                if abs(p0.item()) < 0.01:  # Close to zero power
                    c_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                    d_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                    e_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                else:
                    # Create function to compute q given p with fixed h0
                    def q_given_p(p):
                        return self.params.predict_q_poly(p.unsqueeze(0), h0.unsqueeze(0)).squeeze(0)
                    
                    # Create function to compute q given h with fixed p0
                    def q_given_h(h):
                        return self.params.predict_q_poly(p0.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
                    
                    # Compute gradients (partial derivatives)
                    dq_dp = self.calculate_gradients(q_given_p, p0, retain_graph=True)
                    dq_dh = self.calculate_gradients(q_given_h, h0, retain_graph=True)
                    
                    # Compute q0 at the operating point
                    q0 = self.params.predict_q_poly(p0.unsqueeze(0), h0.unsqueeze(0)).squeeze(0).detach()
                    
                    # Compute Taylor coefficients for q = c*p + d*h + e
                    # Using first-order Taylor expansion around (p0, h0):
                    # q(p,h) ≈ q(p0,h0) + (∂q/∂p)(p-p0) + (∂q/∂h)(h-h0)
                    # Rearranged as: q(p,h) ≈ (∂q/∂p)*p + (∂q/∂h)*h + [q(p0,h0) - (∂q/∂p)*p0 - (∂q/∂h)*h0]
                    c = dq_dp.detach()  # corresponds to ∂q/∂p
                    d = dq_dh.detach()  # corresponds to ∂q/∂h
                    e = q0 - c * p0.detach() - d * h0.detach()  # constant term
                    
                    c_list.append(c)
                    d_list.append(d)
                    e_list.append(e)
                
                # Create function to compute v_low given h
                def v_low_given_h(h):
                    return self.params.h_to_v_low_fitted(h)
                
                # Compute derivative dv_low/dh at h0
                dv_low_dh = self.calculate_gradients(v_low_given_h, h0, retain_graph=False)
                
                # Compute v_low0 at the operating point
                v_low0 = self.params.h_to_v_low_fitted(h0).detach()
                
                # Compute Taylor coefficients for v_low = a*h + b
                # Using first-order Taylor expansion around h0:
                # v_low(h) ≈ v_low(h0) + (dv_low/dh)(h-h0)
                # Rearranged as: v_low(h) ≈ (dv_low/dh)*h + [v_low(h0) - (dv_low/dh)*h0]
                a = dv_low_dh.detach()  # corresponds to dv_low/dh
                b = v_low0 - a * h0.detach()  # constant term
                
                a_list.append(a)
                b_list.append(b)
                
            except Exception as e:
                # On error, add default values to maintain consistent list lengths
                print(f"Error at time step {t}: {e}")
                c_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                d_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                e_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                a_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                b_list.append(torch.tensor(0.0, device=device, requires_grad=True))

        try:
            # Stack results with gradient tracking
            c_tensor = torch.stack(c_list)
            d_tensor = torch.stack(d_list)
            e_tensor = torch.stack(e_list)
            a_tensor = torch.stack(a_list)
            b_tensor = torch.stack(b_list)
        except RuntimeError as e:
            # Handle stacking errors
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
        except Exception as er:
            print(f"\n⚠️ Solver error: {er}")
            print("Problematic parameters:")
            print(f"DA_prices: {DA_prices.detach().cpu().numpy().round(2)}")
            print(f"c: {c.detach().cpu().numpy().round(2)}")
            print(f"d: {d.detach().cpu().numpy().round(2)}")
            print(f"e: {e.detach.cpu().numpy().round(2)}")
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

class LogWeightPredictor(nn.Module):
    """Modified weight predictor that works in the log domain"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2, time_horizon=24, archetype='LSTM'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        self.archetype = archetype.upper()
        
        # Same architecture as before
        if self.archetype == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True 
            )
        elif self.archetype == 'RNN':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.archetype == 'FC':
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
        
        # Output layer now produces log-weights (no activation needed)
        # We'll apply exponentiation when needed
        self.output = nn.Linear(hidden_size, 3 * time_horizon)
        
        # Add a small initial bias to prevent log(0) issues
        self.output.bias.data.fill_(-3.0)  # Initialize to log(0.0067) ≈ -5.0
        
        # Initialize remaining weights
        self._init_weights()
                
    def _init_weights(self):
        """Initialize weights based on the selected architecture"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'output' not in name:  # Skip output layer
                nn.init.xavier_normal_(param, gain=1.5)
            elif 'bias' in name and 'output' not in name:  # Skip output layer
                nn.init.constant_(param, 0.1)
                
    def forward(self, x):
        # Same forward logic as before
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        if self.archetype in ['LSTM', 'RNN']:
            if self.archetype == 'LSTM':
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                output, _ = self.rnn(x, (h0, c0))
            else:  # RNN
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                output, _ = self.rnn(x, h0)
            
            last_output = output[:, -1, :]
        else:  # FC architecture
            batch_size = x.size(0)
            x_flat = x.reshape(batch_size, -1)
            last_output = self.fc_layers(x_flat)
        
        # Get log-weights through output layer (no activation)
        log_weights = self.output(last_output)
        
        # Reshape log-weights
        log_weights = log_weights.view(-1, 3, self.time_horizon)
        log_w_p, log_w_q, log_w_h = log_weights[:, 0, :], log_weights[:, 1, :], log_weights[:, 2, :]
        
        # Remove batch dimension if it was added
        if x.size(0) == 1:
            log_w_p, log_w_q, log_w_h = log_w_p.squeeze(0), log_w_q.squeeze(0), log_w_h.squeeze(0)
        
        return log_w_p, log_w_q, log_w_h
    
    def predict_weights(self, DA_prices, power, flow, head):
        """
        Predict weights for a sequence of inputs.
        Returns both log_weights and exponentiated weights.
        """
        # Stack features into sequence
        x = torch.stack([DA_prices, power, flow, head], dim=1)  # [time_horizon, 4]
        
        with torch.no_grad():
            log_w_p, log_w_q, log_w_h = self.forward(x)
            
            # Exponentiate to get actual weights
            w_p = torch.exp(log_w_p)
            w_q = torch.exp(log_w_q)
            w_h = torch.exp(log_w_h)
        
        return (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h)

class RecursiveLinearizationPipeline:
    """
    Pipeline that recursively updates linearization coefficients using optimization results.
    Works with log-domain weights and exponentiates them for the optimizer.
    Also reinitializes the OptiLayer at each iteration with the latest optimization results.
    """
    def __init__(self, weight_network, params, optimizer, regression, historical_data, max_iterations=3):
        self.weight_network = weight_network
        self.params = params
        self.optimizer = optimizer
        self.regression = regression
        self.historical_data = historical_data
        self.max_iterations = max_iterations
    
    def forward(self, date_str, p_init=None, h_init=None, q_init=None):
        # Get the data for this date
        date_data = self.historical_data[date_str]
        power_init = date_data['power'].clone()  # Ensure we have a copy
        head_init = date_data['head'].clone()
        price = date_data['price'].clone()
        
        # Predict initial flow from (p,h) if not provided
        if q_init is None:
            flow_init = predict_q_poly(power_init, head_init)
        else:
            flow_init = q_init.clone()
        
        # Get input features for the weight predictor
        x = torch.stack([price, power_init, flow_init, head_init], dim=1)  # [time_horizon, 4]
        
        # Run weight prediction with gradient tracking
        log_w_p, log_w_q, log_w_h = self.weight_network(x)
        
        # Exponentiate to get actual weights (with gradient tracking)
        w_p = torch.exp(log_w_p)
        w_q = torch.exp(log_w_q)
        w_h = torch.exp(log_w_h)
        
        # Initialize parameters for first iteration - use provided values if available
        p_current = p_init.clone().detach() if p_init is not None else power_init.clone().detach()
        h_current = h_init.clone().detach() if h_init is not None else head_init.clone().detach()
        flow_current = q_init.clone().detach() if q_init is not None else flow_init.clone().detach()
        
        # Store results from each iteration
        iter_results = []
        
        # Recursive linearization loop
        for iteration in range(self.max_iterations):
            # Compute linearization coefficients based on current power and head
            c, d, e, a, b = self.regression.run_regression(p_current, h_current, flow_current)
            
            # Initialize the OptiLayer with current values before optimization
            # This ensures the layer uses the latest values for warm-starting
            self.optimizer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())
            
            # Run optimization with current coefficients and weights
            p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective = self.optimizer.forward(
                price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                w_p.cpu(), w_h.cpu(), w_q.cpu()
            )
            
            # Store results from this iteration
            iter_result = {
                'iteration': iteration,
                'optimized_profit': optimized_profit.item(),
                'optimized_objective': optimized_objective.item(),
                'p_opt': p_opt.detach().cpu().numpy(),
                'q_opt': q_opt.detach().cpu().numpy(),
                'h_opt': h_opt.detach().cpu().numpy(),
                'c': c.detach().cpu().numpy(),
                'd': d.detach().cpu().numpy(),
                'e': e.detach().cpu().numpy(),
                'a': a.detach().cpu().numpy(),
                'b': b.detach().cpu().numpy()
            }
            iter_results.append(iter_result)
            
            # If not the last iteration, update current power, head, and flow for next iteration
            if iteration < self.max_iterations - 1:
                p_current = torch.tensor(p_opt, device=power_init.device)
                h_current = torch.tensor(h_opt, device=head_init.device)
                flow_current = torch.tensor(q_opt, device=flow_init.device)
        
        # Return final optimization results and all iteration results
        return optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
               (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results

def train_recursive_linearization(weight_network, params, optimizer_layer, regression_layer, 
                               historical_data, num_epochs=100, learning_rate=0.001, 
                               patience=10, max_iterations=3):
    """
    Train the log-domain weight predictor with recursive linearization.
    Each epoch begins with the solution from the last iteration of the previous epoch.
    """
    # Move network to the appropriate device
    device = next(weight_network.parameters()).device
    weight_network.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(weight_network.parameters(), lr=learning_rate)
    # Exponential decay scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Create the pipeline
    pipeline = RecursiveLinearizationPipeline(
        weight_network, params, optimizer_layer, regression_layer, historical_data, max_iterations
    )
    
    # Select a single date for training (or you could iterate through multiple dates)
    train_date = list(historical_data.keys())[0]
    print(f"Training on date: {train_date}")
    
    # Get original data
    date_data = historical_data[train_date]
    power_orig = date_data['power']
    head_orig = date_data['head']
    flow_orig = predict_q_poly(power_orig, head_orig)
    
    # Initialize history tracking
    history = {
        'epoch': [],
        'loss': [],
        'profit': [],
        'log_w_p': [],
        'log_w_q': [],
        'log_w_h': [],
        'w_p': [],
        'w_q': [],
        'w_h': [],
        'p_opt': [],
        'h_opt': [],
        'q_opt': [],
        'p_orig': power_orig.cpu().numpy(),
        'h_orig': head_orig.cpu().numpy(),
        'q_orig': flow_orig.cpu().numpy(),
        'iterations': []  # Store results from each recursive iteration
    }
    
    # Initialize early stopping
    best_profit = float('-inf')
    best_weights = None
    patience_counter = 0
    
    # Initialize warm-start variables from previous epoch's last iteration
    last_p_opt = None
    last_h_opt = None
    last_q_opt = None
    
    print(f"Starting training with recursive linearization (max iterations={max_iterations})...")
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with recursive linearization
        # If we have results from the previous epoch's last iteration, use them as initial values
        if last_p_opt is not None and last_h_opt is not None and last_q_opt is not None:
            optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
            (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results = pipeline.forward(
                train_date, last_p_opt, last_h_opt, last_q_opt
            )
        else:
            # First epoch - use default initialization
            optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
            (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results = pipeline.forward(
                train_date
            )
        
        # Store the results from the last iteration for the next epoch
        last_p_opt = torch.tensor(p_opt, device=device)
        last_h_opt = torch.tensor(h_opt, device=device)
        last_q_opt = torch.tensor(q_opt, device=device)
        
        # Compute loss (negative profit to maximize)
        loss = -optimized_profit
        
        # Backward pass and optimization
        loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(weight_network.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        # Record history
        history['epoch'].append(epoch)
        history['loss'].append(loss.item())
        history['profit'].append(optimized_profit.item())
        history['log_w_p'].append(log_w_p.detach().cpu().numpy())
        history['log_w_q'].append(log_w_q.detach().cpu().numpy())
        history['log_w_h'].append(log_w_h.detach().cpu().numpy())
        history['w_p'].append(w_p.detach().cpu().numpy())
        history['w_q'].append(w_q.detach().cpu().numpy())
        history['w_h'].append(w_h.detach().cpu().numpy())
        history['p_opt'].append(p_opt.detach().numpy())
        history['h_opt'].append(h_opt.detach().numpy())
        history['q_opt'].append(q_opt.detach().numpy())
        history['iterations'].append(iter_results)
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Final Profit = {optimized_profit.item():.4f}")
            
            # Print iteration details for this epoch
            profit_progress = []
            for i, result in enumerate(iter_results):
                profit_val = result['optimized_profit']
                profit_progress.append(profit_val)
                print(f"  Iter {i}: Profit = {profit_val:.4f}")
            
            # Calculate profit improvement
            if len(profit_progress) > 1:
                improvement = profit_progress[-1] - profit_progress[0]
                percent_improvement = (improvement / abs(profit_progress[0])) * 100 if profit_progress[0] != 0 else 0
                print(f"  Profit improvement from recursion: {improvement:.4f} ({percent_improvement:.2f}%)")
            
            print(f"  Log weights ranges - w_p: [{log_w_p.min().item():.2f}, {log_w_p.max().item():.2f}], " +
                  f"w_q: [{log_w_q.min().item():.2f}, {log_w_q.max().item():.2f}], " +
                  f"w_h: [{log_w_h.min().item():.2f}, {log_w_h.max().item():.2f}]")
            print(f"  Actual weights ranges - w_p: [{w_p.min().item():.6f}, {w_p.max().item():.6f}], " +
                  f"w_q: [{w_q.min().item():.6f}, {w_q.max().item():.6f}], " +
                  f"w_h: [{w_h.min().item():.6f}, {w_h.max().item():.6f}]")
        
        # Early stopping check
        if optimized_profit.item() > best_profit:
            best_profit = optimized_profit.item()
            best_weights = weight_network.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    # Load best weights
    if best_weights is not None:
        weight_network.load_state_dict(best_weights)
    
    # Generate plots
    plot_recursive_linearization_results(history, max_iterations)
    
    return weight_network, history
def train_recursive_linearization(weight_network, params, optimizer_layer, regression_layer, 
                                historical_data, num_epochs=100, learning_rate=0.001, 
                                patience=10, max_iterations=3, optimizer=None, scheduler=None):
    """
    Train the log-domain weight predictor with recursive linearization.
    Each epoch begins with the solution from the last iteration of the previous epoch.
    
    Parameters:
        weight_network: Neural network model to train
        params: HydroParameters instance
        optimizer_layer: OptiLayer instance for power system optimization
        regression_layer: TaylorRegressionLayer instance
        historical_data: Dictionary of historical data by date
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate (used only if optimizer is not provided)
        patience: Number of epochs with no improvement before early stopping
        max_iterations: Number of linearization iterations per epoch
        optimizer: PyTorch optimizer (if None, Adam optimizer is used)
        scheduler: PyTorch learning rate scheduler (if None, no scheduler is used)
    
    Returns:
        Tuple of (trained network, training history)
    """
    # Move network to the appropriate device
    device = next(weight_network.parameters()).device
    weight_network.train()
    
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(weight_network.parameters(), lr=learning_rate)
    
    # Create the pipeline
    pipeline = RecursiveLinearizationPipeline(
        weight_network, params, optimizer_layer, regression_layer, historical_data, max_iterations
    )
    
    # Select a single date for training (or you could iterate through multiple dates)
    train_date = list(historical_data.keys())[0]
    print(f"Training on date: {train_date}")
    
    # Get original data
    date_data = historical_data[train_date]
    power_orig = date_data['power']
    head_orig = date_data['head']
    flow_orig = predict_q_poly(power_orig, head_orig)
    
    # Initialize history tracking
    history = {
        'epoch': [],
        'loss': [],
        'profit': [],
        'log_w_p': [],
        'log_w_q': [],
        'log_w_h': [],
        'w_p': [],
        'w_q': [],
        'w_h': [],
        'p_opt': [],
        'h_opt': [],
        'q_opt': [],
        'p_orig': power_orig.cpu().numpy(),
        'h_orig': head_orig.cpu().numpy(),
        'q_orig': flow_orig.cpu().numpy(),
        'iterations': [],  # Store results from each recursive iteration
        'learning_rates': []  # Track learning rates over time
    }
    
    # Initialize early stopping
    best_profit = float('-inf')
    best_weights = None
    patience_counter = 0
    
    # Initialize warm-start variables from previous epoch's last iteration
    last_p_opt = None
    last_h_opt = None
    last_q_opt = None
    
    print(f"Starting training with recursive linearization (max iterations={max_iterations})...")
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with recursive linearization
        # If we have results from the previous epoch's last iteration, use them as initial values
        if last_p_opt is not None and last_h_opt is not None and last_q_opt is not None:
            optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
            (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results = pipeline.forward(
                train_date, last_p_opt, last_h_opt, last_q_opt
            )
        else:
            # First epoch - use default initialization
            optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
            (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results = pipeline.forward(
                train_date
            )
        
        # Store the results from the last iteration for the next epoch
        last_p_opt = torch.tensor(p_opt, device=device)
        last_h_opt = torch.tensor(h_opt, device=device)
        last_q_opt = torch.tensor(q_opt, device=device)
        
        # Compute loss (negative profit to maximize)
        loss = -optimized_profit
        
        # Backward pass and optimization
        loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(weight_network.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update scheduler if provided
        if scheduler is not None:
            # Check scheduler type to determine how to step it
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(optimized_profit)
            else:
                scheduler.step()
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Record history
        history['epoch'].append(epoch)
        history['loss'].append(loss.item())
        history['profit'].append(optimized_profit.item())
        history['log_w_p'].append(log_w_p.detach().cpu().numpy())
        history['log_w_q'].append(log_w_q.detach().cpu().numpy())
        history['log_w_h'].append(log_w_h.detach().cpu().numpy())
        history['w_p'].append(w_p.detach().cpu().numpy())
        history['w_q'].append(w_q.detach().cpu().numpy())
        history['w_h'].append(w_h.detach().cpu().numpy())
        history['p_opt'].append(p_opt.detach().numpy())
        history['h_opt'].append(h_opt.detach().numpy())
        history['q_opt'].append(q_opt.detach().numpy())
        history['iterations'].append(iter_results)
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Final Profit = {optimized_profit.item():.4f}, LR = {current_lr:.6f}")
            
            # Print iteration details for this epoch
            profit_progress = []
            for i, result in enumerate(iter_results):
                profit_val = result['optimized_profit']
                profit_progress.append(profit_val)
                print(f"  Iter {i}: Profit = {profit_val:.4f}")
            
            # Calculate profit improvement
            if len(profit_progress) > 1:
                improvement = profit_progress[-1] - profit_progress[0]
                percent_improvement = (improvement / abs(profit_progress[0])) * 100 if profit_progress[0] != 0 else 0
                print(f"  Profit improvement from recursion: {improvement:.4f} ({percent_improvement:.2f}%)")
            
            print(f"  Log weights ranges - w_p: [{log_w_p.min().item():.2f}, {log_w_p.max().item():.2f}], " +
                  f"w_q: [{log_w_q.min().item():.2f}, {log_w_q.max().item():.2f}], " +
                  f"w_h: [{log_w_h.min().item():.2f}, {log_w_h.max().item():.2f}]")
            print(f"  Actual weights ranges - w_p: [{w_p.min().item():.6f}, {w_p.max().item():.6f}], " +
                  f"w_q: [{w_q.min().item():.6f}, {w_q.max().item():.6f}], " +
                  f"w_h: [{w_h.min().item():.6f}, {w_h.max().item():.6f}]")
        
        # Early stopping check
        if optimized_profit.item() > best_profit:
            best_profit = optimized_profit.item()
            best_weights = weight_network.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    # Load best weights
    if best_weights is not None:
        weight_network.load_state_dict(best_weights)
    
    # Generate plots
    plot_recursive_linearization_results(history, max_iterations)
    
    return weight_network, history

def main():
    """Main execution function for training with recursive linearization."""
    # Load historical data
    historical_data = load_historical_data(
        file_path="./Data/database_no_piecewise_with_coeff.csv",
        with_coefficients=True
    )
    
    if not historical_data:
        print("Error: Could not load historical data")
        return
    
    # Initialize parameters
    params = HydroParameters()
    
    # Initialize layers
    regression_layer = TaylorRegressionLayer(params)
    optimizer_layer = OptiLayer(params)
    
    # Initialize log-domain weight predictor network
    weight_network = LogWeightPredictor(
        input_size=4,
        hidden_size=128,
        num_layers=1,
        dropout=0.2,
        time_horizon=params.time_horizon,
        archetype='LSTM'
    ).to(device)
    
    # Configure optimizer
    optimizer = torch.optim.Adam(weight_network.parameters(), lr=1e-5)
    
    # Configure scheduler
    # Option 1: ReduceLROnPlateau - reduces learning rate when a metric stops improving
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=5, verbose=True
    # )
    
    # Option 2: ExponentialLR - exponential decay of learning rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # Option 3: CosineAnnealingLR - cosine annealing learning rate
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=20, eta_min=1e-6
    # )
    
    # Train the network with recursive linearization
    trained_network, history = train_recursive_linearization(
        weight_network=weight_network,
        params=params,
        optimizer_layer=optimizer_layer,
        regression_layer=regression_layer,
        historical_data=historical_data,
        num_epochs=100,
        learning_rate=1e-5,  # Only used if optimizer not provided
        patience=5,
        max_iterations=50,  # Number of linearization recursions
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Save the trained model
    torch.save(trained_network.state_dict(), 'trained_recursive_linearization_model.pth')
    print(f"Training complete. Model saved to 'trained_recursive_linearization_model.pth'")
    print(f"Final profit: {history['profit'][-1]:.2f}")

# Modified plot function to include learning rate graph
def plot_recursive_linearization_results(history, max_iterations):
    """
    Generate plots for recursive linearization training results.
    Focus on weight evolution and iteration improvements.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))  # Increased height for additional plot
    
    # 1. Learning curve (profit over epochs)
    ax1 = fig.add_subplot(5, 2, 1)  # Changed from 4,2 to 5,2
    ax1.plot(history['epoch'], history['profit'], 'b-')
    ax1.set_title('Profit vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Profit')
    ax1.grid(True)
    
    # 2. Exponentiated weights
    ax2 = fig.add_subplot(5, 2, 2)  # Changed from 4,2 to 5,2
    last_idx = len(history['epoch']) - 1
    time_horizon = len(history['w_p'][0])
    x = range(time_horizon)
    
    ax2.plot(x, history['w_p'][last_idx], 'r-', label='w_p')
    ax2.plot(x, history['w_q'][last_idx], 'g-', label='w_q')
    ax2.plot(x, history['w_h'][last_idx], 'b-', label='w_h')
    ax2.set_title('Final Weights (exponentiated)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Weight Value')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')  # Log scale for small values
    
    # 3. Power comparison
    ax3 = fig.add_subplot(5, 2, 3)  # Changed from 4,2 to 5,2
    ax3.plot(x, history['p_orig'], 'k--', label='Original')
    ax3.plot(x, history['p_opt'][last_idx], 'r-', label='Optimized')
    ax3.set_title('Power Comparison')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Power (MW)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Recursive iteration profit improvements
    ax4 = fig.add_subplot(5, 2, 4)  # Changed from 4,2 to 5,2
    
    # Prepare data for selected epochs to avoid overcrowding
    num_epochs = len(history['epoch'])
    num_samples = min(5, num_epochs)  # Show 5 evenly spaced epochs
    
    # Calculate evenly spaced indices
    if num_epochs <= num_samples:
        indices = list(range(num_epochs))
    else:
        indices = [int(i * (num_epochs-1) / (num_samples-1)) for i in range(num_samples)]
    
    # Plot profit improvement for each iteration across selected epochs
    for i, epoch_idx in enumerate(indices):
        epoch = history['epoch'][epoch_idx]
        iter_results = history['iterations'][epoch_idx]
        profits = [result['optimized_profit'] for result in iter_results]
        iterations = list(range(len(profits)))
        
        ax4.plot(iterations, profits, marker='o', label=f'Epoch {epoch}')
    
    ax4.set_title('Profit Improvement Across Recursive Iterations')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Profit')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Learning rate over epochs (NEW PLOT)
    ax5 = fig.add_subplot(5, 2, 5)
    ax5.plot(history['epoch'], history['learning_rates'], 'g-')
    ax5.set_title('Learning Rate vs Epoch')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.grid(True)
    if min(history['learning_rates']) > 0:
        ax5.set_yscale('log')  # Log scale if appropriate
    
    # 6. Weight evolution with color mapping for w_p
    ax6 = fig.add_subplot(5, 2, 6)  # Changed from 4,2,5 to 5,2,6
    
    # Create colormap
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(0, len(indices)-1)
    
    # Plot evolution with color gradient
    for i, epoch_idx in enumerate(indices):
        color = cmap(norm(i))
        ax6.plot(x, history['w_p'][epoch_idx], color=color, alpha=0.8)
    
    # Add a colorbar for epoch reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax6)
    cbar.set_label('Training Progress (Epochs)')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'Epoch {history["epoch"][indices[0]]}', 
                         f'Epoch {history["epoch"][indices[len(indices)//2]]}',
                         f'Epoch {history["epoch"][indices[-1]]}'])
    
    ax6.set_title('w_p Evolution Over Training')
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('w_p Value')
    ax6.set_yscale('log')  # Log scale for small values
    ax6.grid(True)
    
    # 7. Linearization coefficients evolution (final epoch)
    ax7 = fig.add_subplot(5, 2, 7)  # Changed from 4,2,6 to 5,2,7
    
    # Plot c, d, e coefficients for the final epoch across iterations
    last_epoch_results = history['iterations'][last_idx]
    num_iter = len(last_epoch_results)
    iter_x = range(num_iter)
    
    c_values = np.array([result['c'].mean() for result in last_epoch_results])
    d_values = np.array([result['d'].mean() for result in last_epoch_results])
    e_values = np.array([result['e'].mean() for result in last_epoch_results])
    
    ax7.plot(iter_x, c_values, 'r-o', label='c (mean)')
    ax7.plot(iter_x, d_values, 'g-o', label='d (mean)')
    ax7.plot(iter_x, e_values, 'b-o', label='e (mean)')
    
    ax7.set_title('Flow Linearization Coefficients Evolution (Final Epoch)')
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Coefficient Value')
    ax7.legend()
    ax7.grid(True)
    
    # 8. Power evolution across iterations (final epoch)
    ax8 = fig.add_subplot(5, 2, 8)  # Changed from 4,2,7 to 5,2,8
    
    # For the last epoch, show how power evolves across iterations
    # Sample a few time steps to avoid cluttering
    sample_steps = [0, time_horizon//4, time_horizon//2, 3*time_horizon//4, time_horizon-1]
    markers = ['o', 's', '^', 'd', 'x']
    
    for i, step in enumerate(sample_steps):
        p_values = [result['p_opt'][step] for result in last_epoch_results]
        ax8.plot(iter_x, p_values, marker=markers[i], label=f'Hour {step}')
    
    ax8.set_title('Power Evolution Across Iterations (Final Epoch)')
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Power (MW)')
    ax8.legend()
    ax8.grid(True)
    
    # 9. Head evolution across iterations (final epoch)
    ax9 = fig.add_subplot(5, 2, 9)  # Changed from 4,2,8 to 5,2,9
    
    # For the last epoch, show how head evolves across iterations
    for i, step in enumerate(sample_steps):
        h_values = [result['h_opt'][step] for result in last_epoch_results]
        ax9.plot(iter_x, h_values, marker=markers[i], label=f'Hour {step}')
    
    ax9.set_title('Head Evolution Across Iterations (Final Epoch)')
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Head (m)')
    ax9.legend()
    ax9.grid(True)
    
    # 10. Loss curve over epochs
    ax10 = fig.add_subplot(5, 2, 10)
    ax10.plot(history['epoch'], history['loss'], 'r-')
    ax10.set_title('Loss vs Epoch')
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('Loss')
    ax10.grid(True)
    
    plt.tight_layout()
    plt.savefig('recursive_linearization_results.png')
    plt.show()
    
    # Create a new figure for iteration convergence analysis
    # Rest of the plotting code...
    # (omitted for brevity but would include the convergence analysis plots)

# Execute the main function
if __name__ == "__main__":
    main()