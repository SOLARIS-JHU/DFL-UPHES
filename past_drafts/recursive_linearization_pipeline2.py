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
        self.head_init = head_init.clone().detach().to(device=device, dtype=torch.float32)
        self.v_low_init = v_low_init.clone().detach().to(device=device, dtype=torch.float32)

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
        A simplified class for hourly simulation of the operation,
        using the same parameters object as the other modules.
        """
        self.params = params

    def simulate_operation(self, p, q, h):
        """
        Simulate hourly operation with physical constraints.
        
        Args:
            p (torch.Tensor): Hourly power schedule [time_horizon]
            q (torch.Tensor): Hourly flow schedule [time_horizon]
            h (torch.Tensor): Hourly head schedule [time_horizon]
        
        Returns:
            tuple: Calibrated hourly (p, q, h, v_low) schedules.
        """
        TH = self.params.time_horizon
        
        # Initialize lists for each state
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        # Start states
        v_current = self.params.v_low_init  # user-chosen initial reservoir volume
        v_list.append(v_current)

        for i in range(TH):
            # Current state values
            h_current = h[i]
            p_current = p[i]
            
            # a) Base: idle => q=0
            q_candidate = torch.zeros_like(p_current)

            # b) For turbine mode (p_current>0), clamp p between pos_min(h) and pos_max(h)
            #    then get q via polynomial
            if p_current > 0.5:  # Turbine mode
                p_min_turb = self.params.pos_min(h_current)
                p_max_turb = self.params.pos_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            
            # c) For pump mode (p_current<0), clamp p between neg_min(h) and neg_max(h)
            elif p_current < -0.5:  # Pump mode
                p_min_pump = self.params.neg_min(h_current)
                p_max_pump = self.params.neg_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_pump, max=p_max_pump)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            
            # Update volume: v_next = v_current + q * 3600 (seconds in an hour)
            v_next = v_current + q_candidate * 3600
            
            # Check if volume is within bounds
            out_of_bounds = (v_next > self.params.max_vol_up) | (v_next < self.params.min_vol_low)
            
            # If out of bounds, set to idle mode
            if out_of_bounds:
                p_final = torch.zeros_like(p_current)
                q_final = torch.zeros_like(q_candidate)
                v_next = v_current  # No change to volume
                h_next = h_current  # No change to head
            else:
                p_final = p_clamped if p_current != 0 else torch.zeros_like(p_current)
                q_final = q_candidate
                # Update head based on new volume
                h_next = self.params.v_low_to_h_fitted(v_next)
            
            # Append states for this hour
            p_list.append(p_final)
            q_list.append(q_final)
            h_list.append(h_next)
            v_list.append(v_next.item())
            
            # Update current volume for next iteration
            v_current = v_next
        
        # Convert lists to tensors
        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list[:-1])  # Remove the extra head value
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32)
        
        return p_sim, q_sim, h_sim, v_low_sim

    def calc_profit(self, p_sim, p_opt, v_low_sim, DA_price):
        """
        Calculate the daily profit from the hourly simulation.
        """
        # Calculate energy per hour (MWh)
        e_sim = p_sim  # Already in MW, and we're using hourly intervals

        # Calculate revenue
        revenue = torch.sum(DA_price * e_sim)

        # Determine the System Imbalance (SI) price
        surplus_penalty_multiplier = -0.5
        shortage_penalty_multiplier = -2.0

        SI_price = torch.where(
            e_sim < p_opt,  # Shortage in simulation
            shortage_penalty_multiplier * DA_price,  # Lower output penalty
            surplus_penalty_multiplier * DA_price  # Higher output penalty
        )
        
        # Calculate imbalance penalty
        imbalance = e_sim - p_opt
        penalty = imbalance * SI_price
        SI_penalty = penalty.sum()

        # Volume penalty - if final volume exceeds target
        volume_deficit = max(0, v_low_sim[-1] - self.params.target_vol_low)
        energy_loss = self.params.rho * volume_deficit * self.params.g * self.params.target_head * self.params.mu / 3.6e9  # Convert J to MWh
        volume_penalty = energy_loss * torch.median(DA_price)

        # Operating cost
        operating_cost = self.params.operational_cost * torch.sum(p_sim**2)

        # Total profit
        total_profit = revenue - operating_cost - SI_penalty - volume_penalty
        
        return total_profit, SI_penalty, volume_penalty, operating_cost

class BoundedLogWeightPredictor(nn.Module):
    """Modified weight predictor with bounded log-domain weights"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2, 
                 time_horizon=24, archetype='LSTM', 
                 init_w_p=0.05, init_w_q=0.05, init_w_h=0.05,
                 w_p_min=0.01, w_p_max=10.0,
                 w_q_min=0.01, w_q_max=5.0,
                 w_h_min=0.01, w_h_max=5.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        self.archetype = archetype.upper()
        
        # Process initial weights
        self.init_w_p = init_w_p
        self.init_w_q = init_w_q
        self.init_w_h = init_w_h
        
        # Store bounds for each weight type
        self.w_p_min = w_p_min
        self.w_p_max = w_p_max
        self.w_q_min = w_q_min
        self.w_q_max = w_q_max
        self.w_h_min = w_h_min
        self.w_h_max = w_h_max
        
        # Compute log-domain bounds
        self.log_w_p_min = torch.log(torch.tensor(w_p_min))
        self.log_w_p_max = torch.log(torch.tensor(w_p_max))
        self.log_w_q_min = torch.log(torch.tensor(w_q_min))
        self.log_w_q_max = torch.log(torch.tensor(w_q_max))
        self.log_w_h_min = torch.log(torch.tensor(w_h_min))
        self.log_w_h_max = torch.log(torch.tensor(w_h_max))
        
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
        
        # Output layer for log-weights
        self.output = nn.Linear(hidden_size, 3 * time_horizon)
        
        # Initialize remaining weights
        self._init_weights()
        
        # Set initial biases using the provided weight values
        self._set_initial_weights()
                
    def _init_weights(self):
        """Initialize weights based on the selected architecture"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'output' not in name:
                nn.init.xavier_normal_(param, gain=1.5)
            elif 'bias' in name and 'output' not in name:
                nn.init.constant_(param, 0.1)
    
    def _set_initial_weights(self):
        """Set the output bias to initialize log-weights to desired values"""
        # Convert weight values to log domain
        log_w_p = torch.log(torch.tensor(self.init_w_p))
        log_w_q = torch.log(torch.tensor(self.init_w_q))
        log_w_h = torch.log(torch.tensor(self.init_w_h))
        
        # The bias has shape [3 * time_horizon]
        # We need to set segments of it for each weight type
        bias = self.output.bias.data
        
        # Set the first third to log_w_p
        bias[0:self.time_horizon] = log_w_p
        
        # Set the middle third to log_w_q
        bias[self.time_horizon:2*self.time_horizon] = log_w_q
        
        # Set the last third to log_w_h
        bias[2*self.time_horizon:3*self.time_horizon] = log_w_h
                
    def _clamp_log_weights(self, log_w_p, log_w_q, log_w_h):
        """Clamp log weights to ensure they stay within bounds"""
        # Move bounds to the appropriate device
        device = log_w_p.device
        log_w_p_min = self.log_w_p_min.to(device)
        log_w_p_max = self.log_w_p_max.to(device)
        log_w_q_min = self.log_w_q_min.to(device)
        log_w_q_max = self.log_w_q_max.to(device)
        log_w_h_min = self.log_w_h_min.to(device)
        log_w_h_max = self.log_w_h_max.to(device)
        
        # Apply clamping
        log_w_p = torch.clamp(log_w_p, min=log_w_p_min, max=log_w_p_max)
        log_w_q = torch.clamp(log_w_q, min=log_w_q_min, max=log_w_q_max)
        log_w_h = torch.clamp(log_w_h, min=log_w_h_min, max=log_w_h_max)
        
        return log_w_p, log_w_q, log_w_h
    
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
        
        # Apply bounds to log weights
        log_w_p, log_w_q, log_w_h = self._clamp_log_weights(log_w_p, log_w_q, log_w_h)
        
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
    Also includes simulation to calculate realistic profit.
    """
    def __init__(self, weight_network, params, optimizer, regression, historical_data, 
                 max_iterations=3, penalty_growth_rate=1.2):
        self.weight_network = weight_network
        self.params = params
        self.optimizer = optimizer
        self.regression = regression
        self.historical_data = historical_data
        self.max_iterations = max_iterations
        self.simulator = SimulationLayer(params)  # Add simulation layer
        self.penalty_growth_rate = penalty_growth_rate  # New hyperparameter to control weight growth
    
    def forward(self, date_str):
        # Get the data for this date
        date_data = self.historical_data[date_str]
        power_init = date_data['power'].clone()  # Ensure we have a copy
        head_init = date_data['head'].clone()
        price = date_data['price'].clone()
        
        # Predict initial flow from (p,h)
        flow_init = predict_q_poly(power_init, head_init)
        
        # Get input features for the weight predictor
        x = torch.stack([price, power_init, flow_init, head_init], dim=1)  # [time_horizon, 4]
        
        # Run weight prediction with gradient tracking
        log_w_p, log_w_q, log_w_h = self.weight_network(x)
        
        # Exponentiate to get initial weights (with gradient tracking)
        w_p_initial = torch.exp(log_w_p)
        w_q_initial = torch.exp(log_w_q)
        w_h_initial = torch.exp(log_w_h)
        
        # Initialize parameters for first iteration
        p_current = power_init.clone().detach()  # Detach to start fresh
        h_current = head_init.clone().detach()
        flow_current = flow_init.clone().detach()
        
        # Store results from each iteration
        iter_results = []
        
        # Recursive linearization loop
        for iteration in range(self.max_iterations):
            # Apply growth to penalty weights based on iteration number
            # For the first iteration, use initial weights
            # For subsequent iterations, increase by the growth rate
            growth_factor = self.penalty_growth_rate ** iteration
            w_p = w_p_initial * growth_factor
            w_q = w_q_initial * growth_factor
            w_h = w_h_initial * growth_factor
            
            # Compute linearization coefficients based on current power and head
            c, d, e, a, b = self.regression.run_regression(p_current, h_current, flow_current)
            
            # Initialize the OptiLayer with current values before optimization
            # This ensures the layer uses the latest values for warm-starting
            self.optimizer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())
            
            # Run optimization with current coefficients and growing weights
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
                'b': b.detach().cpu().numpy(),
                'growth_factor': growth_factor,  # Store the growth factor used
                'w_p': w_p.detach().cpu().numpy(),  # Store the actual weights used
                'w_q': w_q.detach().cpu().numpy(),
                'w_h': w_h.detach().cpu().numpy()
            }
            iter_results.append(iter_result)
            
            # If not the last iteration, update current power, head, and flow for next iteration
            if iteration < self.max_iterations - 1:
                p_current = p_opt.clone().detach().to(device=power_init.device) 
                h_current = h_opt.clone().detach().to(device=head_init.device)
                flow_current = q_opt.clone().detach().to(device=flow_init.device)
        
        # After optimization, run simulation with the final p_opt, q_opt, h_opt
        p_sim, q_sim, h_sim, v_low_sim = self.simulator.simulate_operation(
            p_opt.to(device), q_opt.to(device), h_opt.to(device)
        )
        
        # Calculate the simulated profit
        simulated_profit, SI_penalty, volume_penalty, operating_cost = self.simulator.calc_profit(
            p_sim, p_opt.to(device), v_low_sim, price.to(device)
        )
        
        # Return both optimized and simulated results
        return simulated_profit, optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
               p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty, operating_cost, \
               (log_w_p, log_w_q, log_w_h), (w_p_initial, w_q_initial, w_h_initial), c, d, e, a, b, iter_results

def train_recursive_linearization(weight_network, params, optimizer_layer, regression_layer, 
                               historical_data, num_epochs=100, learning_rate=0.001, 
                               patience=10, max_iterations=3, penalty_growth_rate=1.2):
    """
    Train the log-domain weight predictor with recursive linearization.
    Uses simulated profit as the loss function.
    """
    # Move network to the appropriate device
    device = next(weight_network.parameters()).device
    weight_network.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(weight_network.parameters(), lr=learning_rate)
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5
    )

    # Create the pipeline with growth rate parameter
    pipeline = RecursiveLinearizationPipeline(
        weight_network, params, optimizer_layer, regression_layer, historical_data, 
        max_iterations=max_iterations, penalty_growth_rate=penalty_growth_rate
    )
    
    # Select a single date for training (or you could iterate through multiple dates)
    train_date = list(historical_data.keys())[0]
    print(f"Training on date: {train_date}")
    
    # Get original data
    date_data = historical_data[train_date]
    power_orig = date_data['power']
    head_orig = date_data['head']
    flow_orig = predict_q_poly(power_orig, head_orig)
    
    # Initialize history tracking with simulation results
    history = {
        'epoch': [],
        'loss': [],
        'profit': [],
        'simulated_profit': [],
        'SI_penalty': [],
        'volume_penalty': [],
        'operating_cost': [],
        'log_w_p': [],
        'log_w_q': [],
        'log_w_h': [],
        'w_p': [],
        'w_q': [],
        'w_h': [],
        'p_opt': [],
        'h_opt': [],
        'q_opt': [],
        'p_sim': [],
        'q_sim': [],
        'h_sim': [],
        'v_sim': [],
        'p_orig': power_orig.cpu().numpy(),
        'h_orig': head_orig.cpu().numpy(),
        'q_orig': flow_orig.cpu().numpy(),
        'iterations': []  # Store results from each recursive iteration
    }
    
    # Initialize early stopping
    best_profit = float('-inf')
    best_weights = None
    patience_counter = 0
    
    print(f"Starting training with recursive linearization (max iterations={max_iterations}, penalty growth rate={penalty_growth_rate})...")
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with recursive linearization and simulation
        simulated_profit, optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
        p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty, operating_cost, \
        (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results = pipeline.forward(train_date)
        
        # Record iteration details including growing weights
        history['iterations'].append(iter_results)
        
        # Compute loss using negative simulated profit (to maximize simulated profit)
        loss = -simulated_profit
        
        # Backward pass and optimization
        loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(weight_network.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update learning rate scheduler using simulated profit
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(simulated_profit)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Adjust learning rate if it has changed
        if new_lr != old_lr:
            print(f"Learning rate adjusted from {old_lr:.6f} to {new_lr:.6f}")
        
        # Record history
        history['epoch'].append(epoch)
        history['loss'].append(loss.item())
        history['profit'].append(optimized_profit.item())
        history['simulated_profit'].append(simulated_profit.item())
        history['SI_penalty'].append(SI_penalty.item())
        history['volume_penalty'].append(volume_penalty.item())
        history['operating_cost'].append(operating_cost.item())
        history['log_w_p'].append(log_w_p.detach().cpu().numpy())
        history['log_w_q'].append(log_w_q.detach().cpu().numpy())
        history['log_w_h'].append(log_w_h.detach().cpu().numpy())
        history['w_p'].append(w_p.detach().cpu().numpy())
        history['w_q'].append(w_q.detach().cpu().numpy())
        history['w_h'].append(w_h.detach().cpu().numpy())
        history['p_opt'].append(p_opt.detach().numpy())
        history['h_opt'].append(h_opt.detach().numpy())
        history['q_opt'].append(q_opt.detach().numpy())
        history['p_sim'].append(p_sim.detach().cpu().numpy())
        history['q_sim'].append(q_sim.detach().cpu().numpy())
        history['h_sim'].append(h_sim.detach().cpu().numpy())
        history['v_sim'].append(v_low_sim.detach().cpu().numpy())
        
        # Enhanced printing of results and penalties - print for every epoch
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Optimized Profit = {optimized_profit.item():.4f}, Simulated Profit = {simulated_profit.item():.4f}")
        
        # Print detailed simulation results
        print(f"  Simulation Results:")
        print(f"    - Simulated Profit: {simulated_profit.item():.4f}")
        print(f"    - System Imbalance Penalty: {SI_penalty.item():.4f}")
        print(f"    - Volume Penalty: {volume_penalty.item():.4f}")
        print(f"    - Operating Cost: {operating_cost.item():.4f}")
        print(f"    - Final Volume: {v_low_sim[-1].item():.2f} (Target: {params.target_vol_low:.2f})")
        
        # Print detailed power and flow comparison
        print(f"  Power Comparison:")
        print(f"    - Original Power (MW): {power_orig.cpu().numpy()}")
        print(f"    - Optimized Power (MW): {p_opt.detach().cpu().numpy()}")
        print(f"    - Simulated Power (MW): {p_sim.detach().cpu().numpy()}")
        print(f"  Flow Comparison:")
        print(f"    - Original Flow (m³/s): {flow_orig.cpu().numpy()}")
        print(f"    - Optimized Flow (m³/s): {q_opt.detach().cpu().numpy()}")
        print(f"    - Simulated Flow (m³/s): {q_sim.detach().cpu().numpy()}")
        
        # Print information for each recursion iteration
        print(f"  Recursive Iterations:")
        for i, result in enumerate(iter_results):
            print(f"    Iteration {i}:")
            print(f"      - Optimized Profit: {result['optimized_profit']:.4f}")
            print(f"      - Penalty Growth Factor: {result['growth_factor']:.2f}")
            print(f"      - Average Penalty Weights: w_p={np.mean(result['w_p']):.4f}, w_q={np.mean(result['w_q']):.4f}, w_h={np.mean(result['w_h']):.4f}")
        
        # Print weights information
        print(f"  Weight Information:")
        print(f"    - w_p range: [{w_p.min().item():.6f}, {w_p.max().item():.6f}], mean: {w_p.mean().item():.6f}")
        print(f"    - w_q range: [{w_q.min().item():.6f}, {w_q.max().item():.6f}], mean: {w_q.mean().item():.6f}")
        print(f"    - w_h range: [{w_h.min().item():.6f}, {w_h.max().item():.6f}], mean: {w_h.mean().item():.6f}")
        
        # Early stopping check based on simulated profit
        if simulated_profit.item() > best_profit:
            best_profit = simulated_profit.item()
            best_weights = weight_network.state_dict().copy()
            patience_counter = 0
            print(f"  New best simulated profit: {best_profit:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    # Load best weights
    if best_weights is not None:
        weight_network.load_state_dict(best_weights)
    
    # Print final results
    print("\nTraining Complete")
    print(f"Best Simulated Profit: {best_profit:.4f}")
    
    return weight_network, history

def plot_recursive_linearization_results(history, max_iterations):
    """
    Generate plots for recursive linearization training results.
    Include simulation results in the plots.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))  # Increased size to accommodate more plots
    
    # 1. Learning curve (profit over epochs) - now includes simulated profit
    ax1 = fig.add_subplot(6, 2, 1)
    ax1.plot(history['epoch'], history['profit'], 'b-', label='Optimized Profit')
    ax1.plot(history['epoch'], history['simulated_profit'], 'r-', label='Simulated Profit')
    ax1.set_title('Profit vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Profit')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Exponentiated weights
    ax2 = fig.add_subplot(6, 2, 2)
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
    
    # 3. Power comparison (now includes simulated power)
    ax3 = fig.add_subplot(6, 2, 3)
    ax3.plot(x, history['p_orig'], 'k--', label='Original')
    ax3.plot(x, history['p_opt'][last_idx], 'b-', label='Optimized')
    ax3.plot(x, history['p_sim'][last_idx], 'r-', label='Simulated')
    ax3.set_title('Power Comparison')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Power (MW)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Recursive iteration profit improvements
    ax4 = fig.add_subplot(6, 2, 4)
    
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
    
    # 5. Weight evolution with color mapping for w_p
    ax5 = fig.add_subplot(6, 2, 5)
    
    # Create colormap
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(0, len(indices)-1)
    
    # Plot evolution with color gradient
    for i, epoch_idx in enumerate(indices):
        color = cmap(norm(i))
        ax5.plot(x, history['w_p'][epoch_idx], color=color, alpha=0.8)
    
    # Add a colorbar for epoch reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax5)
    cbar.set_label('Training Progress (Epochs)')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'Epoch {history["epoch"][indices[0]]}', 
                         f'Epoch {history["epoch"][indices[len(indices)//2]]}',
                         f'Epoch {history["epoch"][indices[-1]]}'])
    
    ax5.set_title('w_p Evolution Over Training')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('w_p Value')
    ax5.set_yscale('log')  # Log scale for small values
    ax5.grid(True)
    
    # 6. Weight evolution with color mapping for w_q (REPLACED)
    ax6 = fig.add_subplot(6, 2, 6)
    
    # Plot evolution with color gradient
    for i, epoch_idx in enumerate(indices):
        color = cmap(norm(i))
        ax6.plot(x, history['w_q'][epoch_idx], color=color, alpha=0.8)
    
    # Add a colorbar for epoch reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax6)
    cbar.set_label('Training Progress (Epochs)')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'Epoch {history["epoch"][indices[0]]}', 
                         f'Epoch {history["epoch"][indices[len(indices)//2]]}',
                         f'Epoch {history["epoch"][indices[-1]]}'])
    
    ax6.set_title('w_q Evolution Over Training')
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('w_q Value')
    ax6.set_yscale('log')  # Log scale for small values
    ax6.grid(True)
    
    # Get data for iteration plots (needed for plots 7)
    last_epoch_results = history['iterations'][last_idx]
    num_iter = len(last_epoch_results)
    iter_x = range(num_iter)
    
    # 7. Power evolution across iterations (final epoch)
    ax7 = fig.add_subplot(6, 2, 7)
    
    # For the last epoch, show how power evolves across iterations
    # Sample a few time steps to avoid cluttering
    sample_steps = [0, time_horizon//4, time_horizon//2, 3*time_horizon//4, time_horizon-1]
    markers = ['o', 's', '^', 'd', 'x']
    
    for i, step in enumerate(sample_steps):
        p_values = [result['p_opt'][step] for result in last_epoch_results]
        ax7.plot(iter_x, p_values, marker=markers[i], label=f'Hour {step}')
    
    ax7.set_title('Power Evolution Across Iterations (Final Epoch)')
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Power (MW)')
    ax7.legend()
    ax7.grid(True)
    
    # 8. Weight evolution with color mapping for w_h (REPLACED)
    ax8 = fig.add_subplot(6, 2, 8)
    
    # Plot evolution with color gradient
    for i, epoch_idx in enumerate(indices):
        color = cmap(norm(i))
        ax8.plot(x, history['w_h'][epoch_idx], color=color, alpha=0.8)
    
    # Add a colorbar for epoch reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax8)
    cbar.set_label('Training Progress (Epochs)')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'Epoch {history["epoch"][indices[0]]}', 
                         f'Epoch {history["epoch"][indices[len(indices)//2]]}',
                         f'Epoch {history["epoch"][indices[-1]]}'])
    
    ax8.set_title('w_h Evolution Over Training')
    ax8.set_xlabel('Time Step')
    ax8.set_ylabel('w_h Value')
    ax8.set_yscale('log')  # Log scale for small values
    ax8.grid(True)
    
    # 9. Simulation penalties over training
    ax9 = fig.add_subplot(6, 2, 9)
    ax9.plot(history['epoch'], history['SI_penalty'], 'r-', label='System Imbalance Penalty')
    ax9.plot(history['epoch'], history['volume_penalty'], 'g-', label='Volume Penalty')
    ax9.plot(history['epoch'], history['operating_cost'], 'b-', label='Operating Cost')
    ax9.set_title('Simulation Penalties Over Training')
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Penalty Value')
    ax9.legend()
    ax9.grid(True)
    
    # 10. Optimized vs. Simulated Power (Final Epoch)
    ax10 = fig.add_subplot(6, 2, 10)
    p_diff = history['p_opt'][last_idx] - history['p_sim'][last_idx]
    ax10.bar(x, p_diff, color='r', alpha=0.7)
    ax10.set_title('Power Difference (Optimized - Simulated) Final Epoch')
    ax10.set_xlabel('Time Step')
    ax10.set_ylabel('Power Difference (MW)')
    ax10.grid(True)
    
    # 11. Optimized vs. Simulated Profit Evolution
    ax11 = fig.add_subplot(6, 2, 11)
    profit_ratio = [sim/opt if opt != 0 else 0 for sim, opt in zip(history['simulated_profit'], history['profit'])]
    ax11.plot(history['epoch'], profit_ratio, 'g-')
    ax11.axhline(y=1.0, color='r', linestyle='--')
    ax11.set_title('Simulated/Optimized Profit Ratio')
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('Ratio')
    ax11.grid(True)
    
    # 12. Simulated vs Scheduled Power for Final Epoch (CHANGED)
    ax12 = fig.add_subplot(6, 2, 12)
    ax12.plot(x, history['p_sim'][last_idx], 'b-', label='Simulated Power')
    ax12.plot(x, history['p_opt'][last_idx], 'r-', label='Scheduled Power')
    ax12.set_title('Simulated vs Scheduled Power (Final Epoch)')
    ax12.set_xlabel('Time Step')
    ax12.set_ylabel('Power (MW)')
    ax12.legend()
    ax12.grid(True)
    
    plt.tight_layout()
    plt.savefig('recursive_linearization_with_simulation_results.png')
    plt.show()

def plot_weight_growth(history):
    """
    Generate an independent graph showing how weights grow across iterations.
    This focuses specifically on the penalty weight growth pattern.
    """
    # Get data for final epoch
    last_idx = len(history['epoch']) - 1
    last_epoch_results = history['iterations'][last_idx]
    
    # Extract weight data from iterations
    iterations = list(range(len(last_epoch_results)))
    growth_factors = [result['growth_factor'] for result in last_epoch_results]
    
    # Extract weights for each time step and take average
    avg_w_p = [np.mean(result['w_p']) for result in last_epoch_results]
    avg_w_q = [np.mean(result['w_q']) for result in last_epoch_results]
    avg_w_h = [np.mean(result['w_h']) for result in last_epoch_results]
    
    # Also get max values to show range
    max_w_p = [np.max(result['w_p']) for result in last_epoch_results]
    max_w_q = [np.max(result['w_q']) for result in last_epoch_results]
    max_w_h = [np.max(result['w_h']) for result in last_epoch_results]
    
    # Create a new figure for weight growth
    plt.figure(figsize=(12, 8))
    
    # Create primary axis for weights
    ax1 = plt.gca()
    
    # Plot average weights with solid lines
    ax1.plot(iterations, avg_w_p, 'r-', marker='o', linewidth=2, label='avg_w_p')
    ax1.plot(iterations, avg_w_q, 'g-', marker='s', linewidth=2, label='avg_w_q')
    ax1.plot(iterations, avg_w_h, 'b-', marker='^', linewidth=2, label='avg_w_h')
    
    # Plot max weights with dashed lines
    ax1.plot(iterations, max_w_p, 'r--', alpha=0.6, label='max_w_p')
    ax1.plot(iterations, max_w_q, 'g--', alpha=0.6, label='max_w_q')
    ax1.plot(iterations, max_w_h, 'b--', alpha=0.6, label='max_w_h')
    
    # Add secondary y-axis for growth factor
    ax2 = ax1.twinx()
    ax2.plot(iterations, growth_factors, 'k-', marker='x', linewidth=2.5, label='Growth Factor')
    
    # Set scales and labels
    ax1.set_yscale('log')  # Log scale to show all weights clearly
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Weight Value (log scale)', fontsize=12)
    ax2.set_ylabel('Growth Factor', fontsize=12, color='k')
    
    # Set title
    plt.title('Penalty Weight Growth Across Iterations (Final Epoch)', fontsize=14)
    
    # Add legends for both axes
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add gridlines
    ax1.grid(True, alpha=0.3)
    
    # Add annotation explaining the growth pattern
    growth_rate = growth_factors[1] / growth_factors[0]
    plt.figtext(0.5, 0.01, 
                f"Weights grow by a factor of {growth_rate:.2f} each iteration, reflecting increasing confidence in the solution.",
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='aliceblue', alpha=0.5))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('weight_growth_pattern.png')
    plt.close()

def test():
    """Main execution test function for training with recursive linearization and simulation."""
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
    
    # Initialize with custom weight values and bounds
    weight_network = BoundedLogWeightPredictor(
        input_size=4,
        hidden_size=128,
        num_layers=1,
        dropout=0.2,
        time_horizon=params.time_horizon,
        archetype='LSTM',
        init_w_p=0.6,   # Initial values
        init_w_q=0.02,
        init_w_h=0.1,
    
        w_p_min=0.1,  
        w_p_max=3.0,   
        
        w_q_min=0.001,
        w_q_max=0.2,
        
        w_h_min=0.01,
        w_h_max=5.0     
    ).to(device)
    
    # Train the network with recursive linearization and simulation
    trained_network, history = train_recursive_linearization(
        weight_network=weight_network,
        params=params,
        optimizer_layer=optimizer_layer,
        regression_layer=regression_layer,
        historical_data=historical_data,
        num_epochs=1000,
        learning_rate=1e-3,
        patience=20,
        max_iterations=10,  # Number of linearization recursions
        penalty_growth_rate=1.2  # Growth rate for penalty weights per iteration
    )
    
    print(f"Final optimized profit: {history['profit'][-1]:.2f}")
    print(f"Final simulated profit: {history['simulated_profit'][-1]:.2f}")

# Execute the test function
# if __name__ == "__main__":
#     test()

# %% Pretraining and Saving Models with Multiple Configurations
# This code is designed to perform pretraining of a model using grid search over different architectures, layers, and max iterations.
import os
import csv
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import itertools
import json
import traceback

def load_data_for_pretraining(file_path, source_name):
    """
    Load data from a specific file for pretraining.
    
    Args:
        file_path: Path to the data file
        source_name: Name of the source (for logging purposes)
        
    Returns:
        dict: Dictionary with data grouped by date
    """
    try:
        # Read the file
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ['Time', 'Power', 'Head', 'Flow', 'Price', 'Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert Date to datetime
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
            
            data_by_date[date_str] = date_data
        
        print(f"Successfully loaded {source_name} data for {len(data_by_date)} days.")
        return data_by_date
    
    except Exception as e:
        print(f"Error loading {source_name} data: {e}")
        return None

def pretraining_with_grid_search():
    """
    Perform pretraining with grid search over architectures, layers, and max iterations.
    Train models separately for SOS2 and Global Linearization databases.
    """
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting pretraining with grid search at {start_timestamp}...")
    
    # Define database sources
    database_sources = {
        'SOS2': './Data/piecewise_opreation_data_SOS2.csv',
        'GlobalLinear': './Data/database_no_piecewise.csv'
    }
    
    # Define grid search parameters
    architectures = ['LSTM','RNN']
    num_layers_list = [1, 2, 3]
    max_iterations_list = [3, 5, 10, 15]
    
    # Initialize parameters
    params = HydroParameters()
    
    # Process each database source separately
    for source_name, file_path in database_sources.items():
        print(f"\n{'='*80}")
        print(f"Processing {source_name} database from {file_path}")
        print(f"{'='*80}")
        
        # Load data for this source
        historical_data = load_data_for_pretraining(file_path, source_name)
        
        if not historical_data:
            print(f"Error: Could not load data for {source_name}. Skipping...")
            continue
        
        # Create root directory for saving models for this source
        root_dir = Path(f"./trained_models_wide_bounds/{source_name}")
        root_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a benchmark CSV file for this source
        benchmark_file = root_dir / "pretraining_benchmarks.csv"
        with open(benchmark_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Architecture', 'Num_Layers', 'Max_Iterations', 'Date',
                'Training_Time_Seconds', 'Epochs_Trained', 'Best_Epoch',
                'Optimized_Profit', 'Simulated_Profit', 'SI_Penalty', 
                'Volume_Penalty', 'Operating_Cost', 'Timestamp'
            ])
        
        # Initialize layers (common to all configurations)
        regression_layer = TaylorRegressionLayer(params)
        optimizer_layer = OptiLayer(params)
        
        # Total configurations to train
        total_configs = len(architectures) * len(num_layers_list) * len(max_iterations_list) * len(historical_data)
        config_counter = 0
        
        # Loop through all configurations
        for architecture, num_layers, max_iterations in itertools.product(
                architectures, num_layers_list, max_iterations_list):
            
            # Create directory for this configuration
            config_name = f"{architecture}_{num_layers}layer_{max_iterations}iter"
            config_dir = root_dir / config_name
            config_dir.mkdir(exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Starting training for configuration: {config_name} with {source_name} data")
            print(f"{'='*60}")
            
            # Train on each date in the database for this configuration
            for date_idx, (date_str, date_data) in enumerate(historical_data.items()):
                config_counter += 1
                print(f"\n[{config_counter}/{total_configs}] Training {source_name} data for date: {date_str} with {config_name}")
                
                # Create directory for this date within the current configuration
                date_dir = config_dir / date_str
                date_dir.mkdir(exist_ok=True)
                
                try:
                    # Initialize network with the current configuration
                    weight_network = BoundedLogWeightPredictor(
                        input_size=4,
                        hidden_size=128,
                        num_layers=num_layers,
                        dropout=0.2,
                        time_horizon=params.time_horizon,
                        archetype=architecture,

                        init_w_p=0.6,
                        init_w_q=0.02,
                        init_w_h=0.1,
                        
                        w_p_min=0.0,  
                        w_p_max=3.0*1e6,   
                        w_q_min=0.0,
                        w_q_max=0.5*1e6,
                        w_h_min=0.0,
                        w_h_max=5.0*1e6    
                    ).to(device)
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Train the network with recursive linearization
                    trained_network, history = train_recursive_linearization(
                        weight_network=weight_network,
                        params=params,
                        optimizer_layer=optimizer_layer,
                        regression_layer=regression_layer,
                        historical_data={date_str: date_data},
                        num_epochs=500,  # Adjust as needed
                        learning_rate=1e-3,
                        patience=20,
                        max_iterations=max_iterations,
                        penalty_growth_rate=1.2
                    )
                    
                    # Calculate training time
                    training_time = time.time() - start_time
                    
                    # Save trained model
                    torch.save(trained_network.state_dict(), date_dir / "model.pt")
                    
                    # Save training history
                    with open(date_dir / "training_history.json", 'w') as f:
                        # Convert tensor values to Python native types for JSON serialization
                        simplified_history = {
                            'epoch': history['epoch'],
                            'loss': [float(x) for x in history['loss']],
                            'profit': [float(x) for x in history['profit']],
                            'simulated_profit': [float(x) for x in history['simulated_profit']],
                            'SI_penalty': [float(x) if hasattr(x, 'item') else x for x in history['SI_penalty']],
                            'volume_penalty': [float(x) if hasattr(x, 'item') else x for x in history['volume_penalty']],
                            'operating_cost': [float(x) if hasattr(x, 'item') else x for x in history['operating_cost']],
                        }
                        json.dump(simplified_history, f, indent=4)
                    
                    # Generate plots
                    plot_recursive_linearization_results(history, max_iterations)
                    plt.savefig(date_dir / "training_results.png")
                    plt.close()
                    
                    # Get final metrics from the last epoch
                    last_idx = len(history['epoch']) - 1
                    final_optimized_profit = float(history['profit'][last_idx])
                    final_simulated_profit = float(history['simulated_profit'][last_idx])
                    final_si_penalty = float(history['SI_penalty'][last_idx])
                    final_volume_penalty = float(history['volume_penalty'][last_idx])
                    final_operating_cost = float(history['operating_cost'][last_idx])
                    
                    # Find best epoch (maximum simulated profit)
                    best_epoch_idx = np.argmax(history['simulated_profit'])
                    best_epoch = history['epoch'][best_epoch_idx]
                    
                    # Save best model separately
                    torch.save(trained_network.state_dict(), date_dir / "best_model.pt")
                    
                    # Append benchmark data
                    with open(benchmark_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            architecture, num_layers, max_iterations, date_str,
                            f"{training_time:.2f}", last_idx+1, best_epoch,
                            f"{final_optimized_profit:.2f}", f"{final_simulated_profit:.2f}",
                            f"{final_si_penalty:.2f}", f"{final_volume_penalty:.2f}",
                            f"{final_operating_cost:.2f}", start_timestamp
                        ])
                    
                    print(f"Training completed for {date_str} with {config_name}:")
                    print(f"  Training time: {training_time:.2f} seconds")
                    print(f"  Final optimized profit: {final_optimized_profit:.2f}")
                    print(f"  Final simulated profit: {final_simulated_profit:.2f}")
                    print(f"  Results saved to: {date_dir}")
                    
                except Exception as e:
                    print(f"Error training date {date_str} with {config_name}: {e}")
                    print(traceback.format_exc())
                    
                    # Log the error
                    with open(root_dir / "error_log.txt", 'a') as f:
                        f.write(f"\n[{datetime.now()}] Error training {date_str} with {config_name}:\n")
                        f.write(traceback.format_exc())
                        f.write("\n" + "-"*50 + "\n")
        
        # Generate summary analysis of the benchmarks for this source
        generate_benchmark_summary(benchmark_file, source_name)
    
    # Generate cross-database comparison
    generate_cross_database_comparison()
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_duration = datetime.strptime(end_timestamp, "%Y%m%d_%H%M%S") - datetime.strptime(start_timestamp, "%Y%m%d_%H%M%S")
    
    print(f"\nPretraining completed!")
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Total duration: {total_duration}")

def generate_benchmark_summary(benchmark_file, source_name):
    """Generate summary analysis and visualizations of the benchmark data."""
    try:
        # Read benchmark data
        df = pd.read_csv(benchmark_file)
        
        # Create output directory for summary
        summary_dir = Path(f"./trained_models/{source_name}/summary")
        summary_dir.mkdir(exist_ok=True)
        
        # 1. Training time by configuration
        plt.figure(figsize=(12, 8))
        avg_time = df.groupby(['Architecture', 'Num_Layers', 'Max_Iterations'])['Training_Time_Seconds'].mean().reset_index()
        
        # Create configuration labels for plotting
        avg_time['Config'] = avg_time.apply(
            lambda x: f"{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", axis=1
        )
        
        # Plot bar chart
        plt.bar(avg_time['Config'], avg_time['Training_Time_Seconds'], color='skyblue')
        plt.title(f'Average Training Time by Configuration ({source_name} Database)')
        plt.xlabel('Configuration')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(summary_dir / "training_time_by_config.png")
        plt.close()
        
        # 2. Profit comparison by configuration
        plt.figure(figsize=(12, 8))
        avg_profit = df.groupby(['Architecture', 'Num_Layers', 'Max_Iterations'])[
            ['Optimized_Profit', 'Simulated_Profit']
        ].mean().reset_index()
        
        # Create configuration labels
        avg_profit['Config'] = avg_profit.apply(
            lambda x: f"{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", axis=1
        )
        
        # Plot grouped bar chart
        x = np.arange(len(avg_profit['Config']))
        width = 0.35
        plt.bar(x - width/2, avg_profit['Optimized_Profit'], width, label='Optimized Profit', color='blue')
        plt.bar(x + width/2, avg_profit['Simulated_Profit'], width, label='Simulated Profit', color='red')
        plt.title(f'Average Profit by Configuration ({source_name} Database)')
        plt.xlabel('Configuration')
        plt.ylabel('Profit')
        plt.xticks(x, avg_profit['Config'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(summary_dir / "profit_by_config.png")
        plt.close()
        
        # 3. Find best configurations
        best_optimized = avg_profit.loc[avg_profit['Optimized_Profit'].idxmax()]
        best_simulated = avg_profit.loc[avg_profit['Simulated_Profit'].idxmax()]
        
        # Save summary report
        with open(summary_dir / "benchmark_summary.txt", 'w') as f:
            f.write(f"Pretraining Benchmark Summary for {source_name} Database\n")
            f.write("===========================\n\n")
            
            f.write(f"Total configurations tested: {len(avg_profit)}\n")
            f.write(f"Total dates processed: {len(df['Date'].unique())}\n\n")
            
            f.write("Best Configuration for Optimized Profit:\n")
            f.write(f"  Architecture: {best_optimized['Architecture']}\n")
            f.write(f"  Number of Layers: {best_optimized['Num_Layers']}\n")
            f.write(f"  Max Iterations: {best_optimized['Max_Iterations']}\n")
            f.write(f"  Average Optimized Profit: {best_optimized['Optimized_Profit']:.2f}\n\n")
            
            f.write("Best Configuration for Simulated Profit:\n")
            f.write(f"  Architecture: {best_simulated['Architecture']}\n")
            f.write(f"  Number of Layers: {best_simulated['Num_Layers']}\n")
            f.write(f"  Max Iterations: {best_simulated['Max_Iterations']}\n")
            f.write(f"  Average Simulated Profit: {best_simulated['Simulated_Profit']:.2f}\n\n")
            
            f.write("Average training time: {:.2f} seconds\n".format(df['Training_Time_Seconds'].mean()))
            f.write("Maximum training time: {:.2f} seconds\n".format(df['Training_Time_Seconds'].max()))
            f.write("Minimum training time: {:.2f} seconds\n".format(df['Training_Time_Seconds'].min()))
        
        # Save best configuration as a JSON for easy retrieval during validation
        best_config = {
            'architecture': best_simulated['Architecture'],
            'num_layers': int(best_simulated['Num_Layers']),
            'max_iterations': int(best_simulated['Max_Iterations']),
            'average_simulated_profit': float(best_simulated['Simulated_Profit'])
        }
        
        with open(summary_dir / "best_configuration.json", 'w') as f:
            json.dump(best_config, f, indent=4)
        
        print(f"Benchmark summary generated for {source_name} in {summary_dir}")
        
    except Exception as e:
        print(f"Error generating benchmark summary for {source_name}: {e}")

def generate_cross_database_comparison():
    """Generate comparative analysis between SOS2 and Global Linearization databases."""
    try:
        # Check if both benchmark files exist
        sos2_benchmark = Path("./trained_models/SOS2/pretraining_benchmarks.csv")
        global_benchmark = Path("./trained_models/GlobalLinear/pretraining_benchmarks.csv")
        
        if not sos2_benchmark.exists() or not global_benchmark.exists():
            print("Cannot generate cross-database comparison: missing benchmark files")
            return
        
        # Read benchmark data
        df_sos2 = pd.read_csv(sos2_benchmark)
        df_global = pd.read_csv(global_benchmark)
        
        # Add source column
        df_sos2['Source'] = 'SOS2'
        df_global['Source'] = 'GlobalLinear'
        
        # Combine data
        df_combined = pd.concat([df_sos2, df_global])
        
        # Create output directory
        comparison_dir = Path("./trained_models/comparison")
        comparison_dir.mkdir(exist_ok=True)
        
        # 1. Compare average simulated profit between databases
        plt.figure(figsize=(12, 8))
        
        # Compute average profit by configuration and source
        avg_profit = df_combined.groupby(['Source', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
            'Simulated_Profit'
        ].mean().reset_index()
        
        # Create configuration labels
        avg_profit['Config'] = avg_profit.apply(
            lambda x: f"{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", axis=1
        )
        
        # Plot grouped by source
        configs = avg_profit['Config'].unique()
        x = np.arange(len(configs))
        width = 0.35
        
        # Filter for each source
        sos2_data = avg_profit[avg_profit['Source'] == 'SOS2']
        global_data = avg_profit[avg_profit['Source'] == 'GlobalLinear']
        
        # Create mapping from config to index
        config_to_idx = {config: i for i, config in enumerate(configs)}
        
        # Get data points aligned by config
        sos2_y = [sos2_data[sos2_data['Config'] == config]['Simulated_Profit'].values[0] 
                  if len(sos2_data[sos2_data['Config'] == config]) > 0 else 0 
                  for config in configs]
        
        global_y = [global_data[global_data['Config'] == config]['Simulated_Profit'].values[0]
                    if len(global_data[global_data['Config'] == config]) > 0 else 0
                    for config in configs]
        
        # Plot bars
        plt.bar(x - width/2, sos2_y, width, label='SOS2', color='blue')
        plt.bar(x + width/2, global_y, width, label='GlobalLinear', color='green')
        
        plt.title('Average Simulated Profit by Configuration and Database')
        plt.xlabel('Configuration')
        plt.ylabel('Average Simulated Profit')
        plt.xticks(x, configs, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(comparison_dir / "profit_comparison_by_database.png")
        plt.close()
        
        # 2. Compare training time between databases
        plt.figure(figsize=(12, 8))
        
        # Compute average training time by configuration and source
        avg_time = df_combined.groupby(['Source', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
            'Training_Time_Seconds'
        ].mean().reset_index()
        
        # Create configuration labels
        avg_time['Config'] = avg_time.apply(
            lambda x: f"{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", axis=1
        )
        
        # Filter for each source
        sos2_time = avg_time[avg_time['Source'] == 'SOS2']
        global_time = avg_time[avg_time['Source'] == 'GlobalLinear']
        
        # Get data points aligned by config
        sos2_time_y = [sos2_time[sos2_time['Config'] == config]['Training_Time_Seconds'].values[0]
                       if len(sos2_time[sos2_time['Config'] == config]) > 0 else 0
                       for config in configs]
        
        global_time_y = [global_time[global_time['Config'] == config]['Training_Time_Seconds'].values[0]
                         if len(global_time[global_time['Config'] == config]) > 0 else 0
                         for config in configs]
        
        # Plot bars
        plt.bar(x - width/2, sos2_time_y, width, label='SOS2', color='blue')
        plt.bar(x + width/2, global_time_y, width, label='GlobalLinear', color='green')
        
        plt.title('Average Training Time by Configuration and Database')
        plt.xlabel('Configuration')
        plt.ylabel('Average Training Time (seconds)')
        plt.xticks(x, configs, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(comparison_dir / "training_time_comparison.png")
        plt.close()
        
        # 3. Find the best configuration for each database
        sos2_best = avg_profit[avg_profit['Source'] == 'SOS2'].loc[
            avg_profit[avg_profit['Source'] == 'SOS2']['Simulated_Profit'].idxmax()
        ]
        
        global_best = avg_profit[avg_profit['Source'] == 'GlobalLinear'].loc[
            avg_profit[avg_profit['Source'] == 'GlobalLinear']['Simulated_Profit'].idxmax()
        ]
        
        # Save comparison summary
        with open(comparison_dir / "database_comparison.txt", 'w') as f:
            f.write("Database Comparison Summary\n")
            f.write("=========================\n\n")
            
            f.write("SOS2 Database:\n")
            f.write(f"  Best Configuration: {sos2_best['Config']}\n")
            f.write(f"  Average Simulated Profit: {sos2_best['Simulated_Profit']:.2f}\n")
            f.write(f"  Average Training Time: {sos2_time[sos2_time['Config'] == sos2_best['Config']]['Training_Time_Seconds'].values[0]:.2f} seconds\n\n")
            
            f.write("Global Linearization Database:\n")
            f.write(f"  Best Configuration: {global_best['Config']}\n")
            f.write(f"  Average Simulated Profit: {global_best['Simulated_Profit']:.2f}\n")
            f.write(f"  Average Training Time: {global_time[global_time['Config'] == global_best['Config']]['Training_Time_Seconds'].values[0]:.2f} seconds\n\n")
            
            # Determine overall best
            if sos2_best['Simulated_Profit'] > global_best['Simulated_Profit']:
                overall_best = "SOS2"
                profit_diff = sos2_best['Simulated_Profit'] - global_best['Simulated_Profit']
                percent_diff = (profit_diff / global_best['Simulated_Profit']) * 100
            else:
                overall_best = "GlobalLinear"
                profit_diff = global_best['Simulated_Profit'] - sos2_best['Simulated_Profit']
                percent_diff = (profit_diff / sos2_best['Simulated_Profit']) * 100
            
            f.write(f"Overall Best Database: {overall_best}\n")
            f.write(f"  Profit Difference: {profit_diff:.2f} ({percent_diff:.2f}%)\n")
        
        # Save overall best configuration for use in validation
        overall_best_config = {
            'database': overall_best,
            'architecture': sos2_best['Architecture'] if overall_best == 'SOS2' else global_best['Architecture'],
            'num_layers': int(sos2_best['Num_Layers']) if overall_best == 'SOS2' else int(global_best['Num_Layers']),
            'max_iterations': int(sos2_best['Max_Iterations']) if overall_best == 'SOS2' else int(global_best['Max_Iterations']),
            'average_simulated_profit': float(sos2_best['Simulated_Profit']) if overall_best == 'SOS2' else float(global_best['Simulated_Profit'])
        }
        
        with open(comparison_dir / "overall_best_configuration.json", 'w') as f:
            json.dump(overall_best_config, f, indent=4)
        
        print(f"Cross-database comparison generated in {comparison_dir}")
        
    except Exception as e:
        print(f"Error generating cross-database comparison: {e}")

# Run the pretraining if this file is executed directly
if __name__ == "__main__":
    pretraining_with_grid_search()
    print("Pretraining and grid search completed.")
    
# %% Retraining Failed Models
# This code is designed to retrain specific models that failed during the pretraining phase.
def retrain_failed_models():
    """
    Retrain specifically the failed models from the error log:
    1. LSTM_3layer_10iter for date 2023-11-01
    2. RNN_1layer_3iter for date 2023-11-01
    """
    # Load historical data specifically for SOS2 database
    historical_data = load_data_for_pretraining('./Data/piecewise_opreation_data_SOS2.csv', 'SOS2')
    
    if not historical_data or '2023-11-01' not in historical_data:
        print("Error: Could not load data for 2023-11-01. Exiting...")
        return
    
    # Extract just the data for the date we need
    date_str = '2023-11-01'
    date_data = historical_data[date_str]
    date_historical_data = {date_str: date_data}
    
    # Initialize parameters
    params = HydroParameters()
    
    # Initialize layers (common to all configurations)
    regression_layer = TaylorRegressionLayer(params)
    optimizer_layer = OptiLayer(params)
    
    # Define the configurations to retrain
    configurations = [
        {'name': 'LSTM_3layer_10iter', 'architecture': 'LSTM', 'num_layers': 3, 'max_iterations': 10},
        {'name': 'RNN_1layer_3iter', 'architecture': 'RNN', 'num_layers': 1, 'max_iterations': 3}
    ]
    
    # Root directory for saving models
    root_dir = Path("./trained_models/SOS2")
    
    # Benchmark file for updating
    benchmark_file = root_dir / "pretraining_benchmarks.csv"
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"Retraining: {config['name']} for date: {date_str}")
        print(f"{'='*60}")
        
        # Create directory for this configuration
        config_dir = root_dir / config['name']
        config_dir.mkdir(exist_ok=True)
        
        # Create directory for this date within the current configuration
        date_dir = config_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        try:
            # Initialize network with the exact same configuration
            weight_network = BoundedLogWeightPredictor(
                input_size=4,
                hidden_size=128,
                num_layers=config['num_layers'],
                dropout=0.2,
                time_horizon=params.time_horizon,
                archetype=config['architecture'],
                init_w_p=0.6,
                init_w_q=0.02,
                init_w_h=0.1,
                w_p_min=0.01,  
                w_p_max=3.0,   
                w_q_min=0.001,
                w_q_max=0.5,
                w_h_min=0.01,
                w_h_max=5.0     
            ).to(device)
            
            # Start timing
            start_time = time.time()
            
            # Train the network with same parameters
            trained_network, history = train_recursive_linearization(
                weight_network=weight_network,
                params=params,
                optimizer_layer=optimizer_layer,
                regression_layer=regression_layer,
                historical_data=date_historical_data,
                num_epochs=500,
                learning_rate=1e-3,
                patience=20,
                max_iterations=config['max_iterations'],
                penalty_growth_rate=1.2
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Save trained model
            torch.save(trained_network.state_dict(), date_dir / "model.pt")
            
            # Save training history
            with open(date_dir / "training_history.json", 'w') as f:
                # Convert tensor values to Python native types for JSON serialization
                simplified_history = {
                    'epoch': history['epoch'],
                    'loss': [float(x) for x in history['loss']],
                    'profit': [float(x) for x in history['profit']],
                    'simulated_profit': [float(x) for x in history['simulated_profit']],
                    'SI_penalty': [float(x) if hasattr(x, 'item') else x for x in history['SI_penalty']],
                    'volume_penalty': [float(x) if hasattr(x, 'item') else x for x in history['volume_penalty']],
                    'operating_cost': [float(x) if hasattr(x, 'item') else x for x in history['operating_cost']],
                }
                json.dump(simplified_history, f, indent=4)
            
            # Generate plots
            plot_recursive_linearization_results(history, config['max_iterations'])
            plt.savefig(date_dir / "training_results.png")
            plt.close()
            
            # Get final metrics from the last epoch
            last_idx = len(history['epoch']) - 1
            final_optimized_profit = float(history['profit'][last_idx])
            final_simulated_profit = float(history['simulated_profit'][last_idx])
            final_si_penalty = float(history['SI_penalty'][last_idx])
            final_volume_penalty = float(history['volume_penalty'][last_idx])
            final_operating_cost = float(history['operating_cost'][last_idx])
            
            # Find best epoch (maximum simulated profit)
            best_epoch_idx = np.argmax(history['simulated_profit'])
            best_epoch = history['epoch'][best_epoch_idx]
            
            # Save best model separately
            torch.save(trained_network.state_dict(), date_dir / "best_model.pt")
            
            # Append benchmark data
            with open(benchmark_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    config['architecture'], config['num_layers'], config['max_iterations'], date_str,
                    f"{training_time:.2f}", last_idx+1, best_epoch,
                    f"{final_optimized_profit:.2f}", f"{final_simulated_profit:.2f}",
                    f"{final_si_penalty:.2f}", f"{final_volume_penalty:.2f}",
                    f"{final_operating_cost:.2f}", datetime.now().strftime("%Y%m%d_%H%M%S")
                ])
            
            print(f"Retraining completed for {date_str} with {config['name']}:")
            print(f"  Training time: {training_time:.2f} seconds")
            print(f"  Final optimized profit: {final_optimized_profit:.2f}")
            print(f"  Final simulated profit: {final_simulated_profit:.2f}")
            print(f"  Results saved to: {date_dir}")
            
        except Exception as e:
            print(f"Error retraining {date_str} with {config['name']}: {e}")
            print(traceback.format_exc())
            
            # Log the error
            with open(root_dir / "error_log.txt", 'a') as f:
                f.write(f"\n[{datetime.now()}] Error retraining {date_str} with {config['name']}:\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-"*50 + "\n")
            
    print("\nRetraining completed for the failed models.")


# Call the function to retrain the failed models
if __name__ == "__main__":
    retrain_failed_models()

# %% Scheduling Validation with New Price Data
# This code is designed to validate the scheduling of a hydroelectric power plant using new price data.
import os
import csv
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
import traceback
import itertools

def load_new_price_data(file_path="./Data/price_data_2024.csv"):
    """
    Load new price data for scheduling validation.
    
    Args:
        file_path: Path to the CSV file with new price data
        
    Returns:
        dict: Dictionary with date strings as keys and price tensors as values
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check column names from the first line
        if 'date' not in df.columns or 'cluster_index' not in df.columns or 'prices_hourly' not in df.columns:
            # Try to handle the case where column headers might be different
            if len(df.columns) >= 3:
                # Assume first column is date, third column has hourly prices
                df.columns = ['date', 'cluster_index', 'prices_hourly']
            else:
                raise ValueError(f"Expected columns 'date', 'cluster_index', 'prices_hourly' but got {df.columns}")
        
        # Dictionary to store price data by date
        price_data = {}
        
        # Process each row
        for _, row in df.iterrows():
            date_str = row['date']
            prices_str = row['prices_hourly']
            
            # Parse the prices (attempting different delimiter formats)
            try:
                # First try splitting by comma
                prices = [float(p) for p in prices_str.split(',')]
            except:
                try:
                    # If that fails, try splitting by semicolon
                    prices = [float(p) for p in prices_str.split(';')]
                except:
                    # If that fails too, try to interpret as a list-like string
                    prices_str = prices_str.strip('[]')
                    prices = [float(p) for p in prices_str.split()]
            
            # Ensure we have 24 hours of data
            if len(prices) != 24:
                print(f"Warning: Date {date_str} has {len(prices)} price values instead of 24")
                # Pad or truncate as needed
                if len(prices) < 24:
                    prices.extend([prices[-1]] * (24 - len(prices)))  # Pad with last value
                else:
                    prices = prices[:24]  # Truncate
            
            # Convert to tensor
            price_tensor = torch.tensor(prices, dtype=torch.float32, device=device)
            
            # Add to dictionary
            price_data[date_str] = price_tensor
        
        print(f"Successfully loaded price data for {len(price_data)} days.")
        return price_data
    
    except Exception as e:
        print(f"Error loading new price data: {e}")
        return None

def load_data_for_validation(file_path, source_name):
    """
    Load historical data for finding similar price profiles.
    
    Args:
        file_path: Path to the data file
        source_name: Name of the source (for logging purposes)
        
    Returns:
        dict: Dictionary with data grouped by date
    """
    try:
        # Read the file
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ['Time', 'Power', 'Head', 'Flow', 'Price', 'Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert Date to datetime
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
            
            data_by_date[date_str] = date_data
        
        print(f"Successfully loaded {source_name} data for {len(data_by_date)} days.")
        return data_by_date
    
    except Exception as e:
        print(f"Error loading {source_name} data: {e}")
        return None

def find_closest_date(new_price, historical_data):
    """
    Find the date in historical data with the most similar price signal.
    
    Args:
        new_price: Tensor of shape [24] with hourly prices
        historical_data: Dictionary of historical data
        
    Returns:
        str: Date string of the closest match
        float: Distance metric value
    """
    closest_date = None
    min_distance = float('inf')
    
    for date_str, date_data in historical_data.items():
        historical_price = date_data['price'][:24]  # Ensure we use only 24 hours
        
        # Calculate Euclidean distance between price profiles
        distance = torch.norm(new_price - historical_price).item()
        
        if distance < min_distance:
            min_distance = distance
            closest_date = date_str
    
    return closest_date, min_distance

def comprehensive_validation():
    """
    Perform validation across all model configurations from pretraining.
    
    Tests all combinations of:
    - Database sources: SOS2, GlobalLinear
    - Architectures: LSTM, RNN
    - Number of layers: 1, 2, 3
    - Max iterations: 3, 5, 10, 15
    """
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting comprehensive validation at {start_timestamp}...")
    
    # Define database sources
    database_sources = {
        'SOS2': './Data/piecewise_opreation_data_SOS2.csv',
        'GlobalLinear': './Data/database_no_piecewise.csv'
    }
    
    # Define grid search parameters
    architectures = ['LSTM', 'RNN']
    num_layers_list = [1, 2, 3]
    max_iterations_list = [3, 5, 10, 15]
    
    # Load new price data (common to all validations)
    new_price_data = load_new_price_data()
    if not new_price_data:
        print("Error: Could not load new price data")
        return
    
    # Initialize parameters object (common to all validations)
    params = HydroParameters()
    
    # Create master benchmark file for all configurations
    master_dir = Path("./validation_results/comprehensive")
    master_dir.mkdir(exist_ok=True, parents=True)
    
    master_benchmark_file = master_dir / "master_validation_benchmarks.csv"
    with open(master_benchmark_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Database', 'Architecture', 'Num_Layers', 'Max_Iterations',
            'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
            'Optimized_Profit', 'Simulated_Profit', 'SI_Penalty',
            'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
            'Timestamp'
        ])
    
    # Create a tracking dict for best configurations per date
    best_configs = {}
    
    # Total configurations to test
    total_configs = len(database_sources) * len(architectures) * len(num_layers_list) * len(max_iterations_list)
    config_counter = 0
    
    # Iterate through all configurations
    for db_name, arch, num_layers, max_iter in itertools.product(
            database_sources.keys(), architectures, num_layers_list, max_iterations_list):
        
        config_counter += 1
        config_name = f"{arch}_{num_layers}layer_{max_iter}iter"
        
        print(f"\n{'='*80}")
        print(f"[{config_counter}/{total_configs}] Validating with configuration: {db_name}/{config_name}")
        print(f"{'='*80}")
        
        # Create output directory for this configuration
        config_dir = Path(f"./custom_validation_results/{db_name}/{config_name}")
        config_dir.mkdir(exist_ok=True, parents=True)
        
        # Create benchmark CSV file for this configuration
        benchmark_file = config_dir / "scheduling_benchmarks.csv"
        with open(benchmark_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
                'Optimized_Profit', 'Simulated_Profit', 'SI_Penalty',
                'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
                'Timestamp'
            ])
        
        try:
            # Load historical data for this database
            historical_data = load_data_for_validation(database_sources[db_name], db_name)
            if not historical_data:
                print(f"Error: Could not load historical data for {db_name}")
                continue
            
            # Initialize layers
            regression_layer = TaylorRegressionLayer(params)
            optimizer_layer = OptiLayer(params)
            
            # Process each new date
            for date_idx, (new_date, new_price) in enumerate(new_price_data.items()):
                print(f"\n[{date_idx+1}/{len(new_price_data)}] Processing date: {new_date} with {db_name}/{config_name}")
                
                # Create directory for this date
                date_dir = config_dir / new_date
                date_dir.mkdir(exist_ok=True)
                
                try:
                    # Start timing
                    start_time = time.time()
                    
                    # 1. Find the closest historical date
                    closest_date, distance = find_closest_date(new_price, historical_data)
                    print(f"Closest historical date: {closest_date} (distance: {distance:.2f})")
                    
                    # 2. Look for the pretrained model at this path
                    model_path = Path(f"./trained_models_wide_bounds/{db_name}/{config_name}/{closest_date}/best_model.pt")
                    
                    if not model_path.exists():
                        # Try model.pt if best_model.pt doesn't exist
                        model_path = Path(f"./trained_models_wide_bounds/{db_name}/{config_name}/{closest_date}/model.pt")
                        
                        if not model_path.exists():
                            print(f"Warning: No model found at {model_path}. Skipping this date.")
                            continue
                    
                    # 3. Initialize weight network with the configuration
                    weight_network = BoundedLogWeightPredictor(
                        input_size=4,
                        hidden_size=128,
                        num_layers=num_layers,
                        dropout=0.2,
                        time_horizon=params.time_horizon,
                        archetype=arch,

                        init_w_p=0.6,
                        init_w_q=0.02,
                        init_w_h=0.1,
                        
                        w_p_min=0.0,  
                        w_p_max=3.0*1e6,   
                        w_q_min=0.0,
                        w_q_max=0.5*1e6,
                        w_h_min=0.0,
                        w_h_max=5.0*1e6    
                    ).to(device)
                    
                    # 4. Load the pretrained weights
                    weight_network.load_state_dict(torch.load(model_path, map_location=device))
                    weight_network.eval()
                    
                    # 5. Get the power, head, and flow from the closest date
                    closest_data = historical_data[closest_date]
                    power_init = closest_data['power'][:24].clone()
                    head_init = closest_data['head'][:24].clone()
                    flow_init = predict_q_poly(power_init, head_init)
                    
                    # 6. Create pipeline for recursive linearization
                    pipeline = RecursiveLinearizationPipeline(
                        weight_network, params, optimizer_layer, regression_layer, 
                        {closest_date: closest_data}, max_iterations=max_iter, 
                        penalty_growth_rate=1.2
                    )
                    
                    # 7. Prepare input for weight prediction
                    x = torch.stack([new_price, power_init, flow_init, head_init], dim=1)
                    
                    # 8. Predict weights
                    with torch.no_grad():
                        log_w_p, log_w_q, log_w_h = weight_network(x)
                        w_p = torch.exp(log_w_p)
                        w_q = torch.exp(log_w_q)
                        w_h = torch.exp(log_w_h)
                    
                    # 9. Initialize and run the recursive linearization
                    p_current = power_init.clone().detach()
                    h_current = head_init.clone().detach()
                    flow_current = flow_init.clone().detach()
                    
                    # Track iteration results
                    iter_results = []
                    
                    for iteration in range(max_iter):
                        # Apply growth to weights
                        growth_factor = pipeline.penalty_growth_rate ** iteration
                        w_p_iter = w_p * growth_factor
                        w_q_iter = w_q * growth_factor
                        w_h_iter = w_h * growth_factor
                        
                        # Compute linearization coefficients
                        c, d, e, a, b = regression_layer.run_regression(p_current, h_current, flow_current)
                        
                        # Initialize OptiLayer
                        optimizer_layer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())
                        
                        # Run optimization
                        p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective = optimizer_layer.forward(
                            new_price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                            p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                            w_p_iter.cpu(), w_h_iter.cpu(), w_q_iter.cpu()
                        )
                        
                        # Store iteration results
                        iter_results.append({
                            'iteration': iteration,
                            'p_opt': p_opt.detach().cpu().numpy(),
                            'q_opt': q_opt.detach().cpu().numpy(),
                            'h_opt': h_opt.detach().cpu().numpy(),
                            'optimized_profit': optimized_profit.item()
                        })
                        
                        # Update for next iteration
                        if iteration < max_iter - 1:
                            p_current = p_opt.clone().detach().to(device=power_init.device) 
                            h_current = h_opt.clone().detach().to(device=head_init.device)
                            flow_current = q_opt.clone().detach().to(device=flow_init.device)
                    
                    # 10. Run simulation
                    simulator = SimulationLayer(params)
                    p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(
                        p_opt.to(device), q_opt.to(device), h_opt.to(device)
                    )
                    
                    # 11. Calculate simulated profit
                    simulated_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
                        p_sim, p_opt.to(device), v_low_sim, new_price.to(device)
                    )
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # 12. Save results
                    results = {
                        'p_opt': p_opt.detach().cpu().numpy(),
                        'q_opt': q_opt.detach().cpu().numpy(),
                        'h_opt': h_opt.detach().cpu().numpy(),
                        'v_opt': v_opt.detach().cpu().numpy(),
                        'p_sim': p_sim.detach().cpu().numpy(),
                        'q_sim': q_sim.detach().cpu().numpy(),
                        'h_sim': h_sim.detach().cpu().numpy(),
                        'v_low_sim': v_low_sim.detach().cpu().numpy(),
                        'new_price': new_price.detach().cpu().numpy(),
                        'closest_price': closest_data['price'][:24].detach().cpu().numpy(),
                        'closest_power': closest_data['power'][:24].detach().cpu().numpy(),
                        'optimized_profit': optimized_profit.item(),
                        'simulated_profit': simulated_profit.item(),
                        'SI_penalty': SI_penalty.item(),
                        'volume_penalty': volume_penalty.item(),
                        'operating_cost': operating_cost.item(),
                        'iter_results': iter_results,
                        'database': db_name,
                        'architecture': arch,
                        'num_layers': num_layers,
                        'max_iterations': max_iter
                    }
                    
                    # Save as numpy arrays
                    np.save(date_dir / "results.npy", results)
                    
                    # 13. Generate plots
                    plt.figure(figsize=(18, 12))
                    
                    # Plot price comparison
                    plt.subplot(3, 2, 1)
                    plt.plot(range(24), results['new_price'], 'b-', label='New Price')
                    plt.plot(range(24), results['closest_price'], 'r--', label=f'Closest ({closest_date})')
                    plt.title('Price Comparison')
                    plt.xlabel('Hour')
                    plt.ylabel('Price (EUR/MWh)')
                    plt.legend()
                    plt.grid(True)
                    
                    # Plot power comparison
                    plt.subplot(3, 2, 2)
                    plt.plot(range(24), results['p_opt'], 'g-', label='Optimized Power')
                    plt.plot(range(24), results['p_sim'], 'b-', label='Simulated Power')
                    plt.plot(range(24), results['closest_power'], 'r--', label=f'Historical ({closest_date})')
                    plt.title('Power Schedule')
                    plt.xlabel('Hour')
                    plt.ylabel('Power (MW)')
                    plt.legend()
                    plt.grid(True)
                    
                    # Plot flow
                    plt.subplot(3, 2, 3)
                    plt.plot(range(24), results['q_opt'], 'b-')
                    plt.title('Optimized Flow')
                    plt.xlabel('Hour')
                    plt.ylabel('Flow (m³/s)')
                    plt.grid(True)
                    
                    # Plot head
                    plt.subplot(3, 2, 4)
                    plt.plot(range(24), results['h_opt'], 'g-')
                    plt.title('Optimized Head')
                    plt.xlabel('Hour')
                    plt.ylabel('Head (m)')
                    plt.grid(True)
                    
                    # Plot iteration power evolution
                    plt.subplot(3, 2, 5)
                    sample_hours = [0, 6, 12, 18, 23]
                    for hour in sample_hours:
                        hour_values = [iter_result['p_opt'][hour] for iter_result in iter_results]
                        plt.plot(range(len(hour_values)), hour_values, marker='o', label=f'Hour {hour}')
                    
                    plt.title('Power Evolution Across Iterations')
                    plt.xlabel('Iteration')
                    plt.ylabel('Power (MW)')
                    plt.legend()
                    plt.grid(True)
                    
                    # Add result statistics as text
                    plt.subplot(3, 2, 6)
                    plt.axis('off')
                    stats_text = (
                        f"Date: {new_date}\n"
                        f"Closest historical date: {closest_date}\n"
                        f"Distance metric: {distance:.2f}\n\n"
                        f"Configuration: {db_name}, {arch}, {num_layers} layers, {max_iter} iterations\n\n"
                        f"Optimized profit: {optimized_profit.item():.2f}\n"
                        f"Simulated profit: {simulated_profit.item():.2f}\n"
                        f"SI penalty: {SI_penalty.item():.2f}\n"
                        f"Volume penalty: {volume_penalty.item():.2f}\n"
                        f"Operating cost: {operating_cost.item():.2f}\n\n"
                        f"Processing time: {processing_time:.2f} seconds"
                    )
                    plt.text(0.1, 0.5, stats_text, fontsize=10, va='center')
                    
                    plt.suptitle(f"Validation Results for {new_date} using {db_name}/{config_name}", fontsize=16)
                    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
                    plt.savefig(date_dir / "validation_results.png")
                    plt.close()
                    
                    # 14. Append to configuration benchmark CSV
                    with open(benchmark_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            new_date, closest_date, f"{distance:.2f}",
                            f"{optimized_profit.item():.2f}", f"{simulated_profit.item():.2f}",
                            f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                            f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                            start_timestamp
                        ])
                    
                    # 15. Append to master benchmark CSV
                    with open(master_benchmark_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            db_name, arch, num_layers, max_iter,
                            new_date, closest_date, f"{distance:.2f}",
                            f"{optimized_profit.item():.2f}", f"{simulated_profit.item():.2f}",
                            f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                            f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                            start_timestamp
                        ])
                    
                    # 16. Track the best configuration for this date
                    config_key = (db_name, arch, num_layers, max_iter)
                    if new_date not in best_configs:
                        best_configs[new_date] = {'config': config_key, 'profit': simulated_profit.item()}
                    elif simulated_profit.item() > best_configs[new_date]['profit']:
                        best_configs[new_date] = {'config': config_key, 'profit': simulated_profit.item()}
                    
                    print(f"Validation for {new_date} completed:")
                    print(f"  Configuration: {db_name}/{config_name}")
                    print(f"  Processing time: {processing_time:.2f} seconds")
                    print(f"  Optimized profit: {optimized_profit.item():.2f}")
                    print(f"  Simulated profit: {simulated_profit.item():.2f}")
                    print(f"  Results saved to: {date_dir}")
                    
                except Exception as e:
                    print(f"Error processing date {new_date} with {db_name}/{config_name}: {e}")
                    print(traceback.format_exc())
                    
                    # Log the error
                    with open(config_dir / "error_log.txt", 'a') as f:
                        f.write(f"\n[{datetime.now()}] Error processing {new_date}:\n")
                        f.write(traceback.format_exc())
                        f.write("\n" + "-"*50 + "\n")
        
        except Exception as e:
            print(f"Error with configuration {db_name}/{config_name}: {e}")
            print(traceback.format_exc())
            
            # Log the error
            with open(master_dir / "error_log.txt", 'a') as f:
                f.write(f"\n[{datetime.now()}] Error with configuration {db_name}/{config_name}:\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-"*50 + "\n")
    
    # Save best configuration for each date
    with open(master_dir / "best_configurations.json", 'w') as f:
        best_configs_serializable = {}
        for date, data in best_configs.items():
            config = data['config']
            best_configs_serializable[date] = {
                'database': config[0],
                'architecture': config[1],
                'num_layers': int(config[2]),
                'max_iterations': int(config[3]),
                'profit': float(data['profit'])
            }
        json.dump(best_configs_serializable, f, indent=4)
    
    # Generate comprehensive summary analysis
    generate_comprehensive_summary(master_benchmark_file)
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_duration = datetime.strptime(end_timestamp, "%Y%m%d_%H%M%S") - datetime.strptime(start_timestamp, "%Y%m%d_%H%M%S")
    
    print(f"\nComprehensive validation completed!")
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Total duration: {total_duration}")
    print(f"Master benchmark saved to: {master_benchmark_file}")
    print(f"Best configurations saved to: {master_dir / 'best_configurations.json'}")

def generate_comprehensive_summary(master_benchmark_file):
    """Generate comprehensive analysis of all validation results."""
    try:
        # Read master benchmark data
        df = pd.read_csv(master_benchmark_file)
        
        # Create output directory
        summary_dir = Path("./validation_results/comprehensive/summary")
        summary_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Compute average performance by configuration
        avg_by_config = df.groupby(['Database', 'Architecture', 'Num_Layers', 'Max_Iterations'])[
            ['Optimized_Profit', 'Simulated_Profit', 'SI_Penalty', 
             'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds']
        ].mean().reset_index()
        
        # Add a Config column for easier plotting
        avg_by_config['Config'] = avg_by_config.apply(
            lambda x: f"{x['Database']}-{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter", 
            axis=1
        )
        
        # Find the best configuration based on average simulated profit
        best_config_row = avg_by_config.loc[avg_by_config['Simulated_Profit'].idxmax()]
        best_config = best_config_row['Config']
        
        # 2. Plot average simulated profit by configuration
        plt.figure(figsize=(15, 8))
        avg_by_config = avg_by_config.sort_values('Simulated_Profit', ascending=False)
        bar_colors = ['green' if config == best_config else 'skyblue' for config in avg_by_config['Config']]
        
        plt.bar(avg_by_config['Config'], avg_by_config['Simulated_Profit'], color=bar_colors)
        plt.title('Average Simulated Profit by Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Average Simulated Profit')
        plt.xticks(rotation=90)
        plt.axhline(y=avg_by_config['Simulated_Profit'].mean(), color='r', linestyle='--', 
                   label=f'Mean: {avg_by_config["Simulated_Profit"].mean():.2f}')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(summary_dir / "avg_profit_by_config.png")
        plt.close()
        
        # 3. Plot processing time by configuration
        plt.figure(figsize=(15, 8))
        avg_by_config_time = avg_by_config.sort_values('Processing_Time_Seconds')
        
        plt.bar(avg_by_config_time['Config'], avg_by_config_time['Processing_Time_Seconds'], color='orange')
        plt.title('Average Processing Time by Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Average Processing Time (seconds)')
        plt.xticks(rotation=90)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(summary_dir / "avg_processing_time.png")
        plt.close()
        
        # 4. Analysis by database
        db_analysis = df.groupby('Database')['Simulated_Profit'].agg(['mean', 'max', 'min', 'std']).reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(db_analysis['Database'], db_analysis['mean'], yerr=db_analysis['std'], 
                capsize=10, color=['blue', 'green'])
        plt.title('Average Simulated Profit by Database')
        plt.xlabel('Database')
        plt.ylabel('Average Simulated Profit')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(summary_dir / "profit_by_database.png")
        plt.close()
        
        # 5. Analysis by architecture
        arch_analysis = df.groupby('Architecture')['Simulated_Profit'].agg(['mean', 'max', 'min', 'std']).reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(arch_analysis['Architecture'], arch_analysis['mean'], yerr=arch_analysis['std'], 
                capsize=10, color=['purple', 'orange'])
        plt.title('Average Simulated Profit by Architecture')
        plt.xlabel('Architecture')
        plt.ylabel('Average Simulated Profit')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(summary_dir / "profit_by_architecture.png")
        plt.close()
        
        # 6. Effect of number of layers and max iterations
        layer_iter_analysis = df.groupby(['Num_Layers', 'Max_Iterations'])['Simulated_Profit'].mean().reset_index()
        
        # Convert to pivot for heatmap
        pivot_layer_iter = layer_iter_analysis.pivot(
            index='Num_Layers', columns='Max_Iterations', values='Simulated_Profit'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_layer_iter, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title('Average Simulated Profit by Layers and Max Iterations')
        plt.ylabel('Number of Layers')
        plt.xlabel('Max Iterations')
        plt.tight_layout()
        plt.savefig(summary_dir / "layers_iterations_heatmap.png")
        plt.close()
        
        # 7. Save summary statistics
        summary_stats = {
            'best_overall_config': {
                'database': best_config_row['Database'],
                'architecture': best_config_row['Architecture'],
                'num_layers': int(best_config_row['Num_Layers']),
                'max_iterations': int(best_config_row['Max_Iterations']),
                'avg_simulated_profit': float(best_config_row['Simulated_Profit']),
                'avg_optimized_profit': float(best_config_row['Optimized_Profit']),
                'avg_processing_time': float(best_config_row['Processing_Time_Seconds'])
            },
            'best_by_database': {},
            'best_by_architecture': {},
            'overall_stats': {
                'total_configurations': len(avg_by_config),
                'total_dates_processed': len(df['New_Date'].unique()),
                'avg_simulated_profit_all': float(df['Simulated_Profit'].mean()),
                'avg_processing_time_all': float(df['Processing_Time_Seconds'].mean())
            }
        }
        
        # Add best by database
        for db in df['Database'].unique():
            db_df = df[df['Database'] == db]
            db_avg = db_df.groupby(['Architecture', 'Num_Layers', 'Max_Iterations'])['Simulated_Profit'].mean()
            best_idx = db_avg.idxmax()
            
            summary_stats['best_by_database'][db] = {
                'architecture': best_idx[0],
                'num_layers': int(best_idx[1]),
                'max_iterations': int(best_idx[2]),
                'avg_simulated_profit': float(db_avg.max())
            }
        
        # Add best by architecture
        for arch in df['Architecture'].unique():
            arch_df = df[df['Architecture'] == arch]
            arch_avg = arch_df.groupby(['Database', 'Num_Layers', 'Max_Iterations'])['Simulated_Profit'].mean()
            best_idx = arch_avg.idxmax()
            
            summary_stats['best_by_architecture'][arch] = {
                'database': best_idx[0],
                'num_layers': int(best_idx[1]),
                'max_iterations': int(best_idx[2]),
                'avg_simulated_profit': float(arch_avg.max())
            }
        
        # Save as JSON
        with open(summary_dir / "comprehensive_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=4)
        
        # 8. Create a text summary report
        with open(summary_dir / "comprehensive_summary.txt", 'w') as f:
            f.write("Comprehensive Validation Summary\n")
            f.write("===============================\n\n")
            
            f.write(f"Total configurations tested: {len(avg_by_config)}\n")
            f.write(f"Total dates processed: {len(df['New_Date'].unique())}\n\n")
            
            f.write("Best Overall Configuration:\n")
            f.write(f"  {best_config}\n")
            f.write(f"  Database: {best_config_row['Database']}\n")
            f.write(f"  Architecture: {best_config_row['Architecture']}\n")
            f.write(f"  Number of Layers: {best_config_row['Num_Layers']}\n")
            f.write(f"  Max Iterations: {best_config_row['Max_Iterations']}\n")
            f.write(f"  Average Simulated Profit: {best_config_row['Simulated_Profit']:.2f}\n")
            f.write(f"  Average Processing Time: {best_config_row['Processing_Time_Seconds']:.2f} seconds\n\n")
            
            f.write("Performance by Database:\n")
            for _, row in db_analysis.iterrows():
                f.write(f"  {row['Database']}:\n")
                f.write(f"    Average Profit: {row['mean']:.2f}\n")
                f.write(f"    Max Profit: {row['max']:.2f}\n")
                f.write(f"    Min Profit: {row['min']:.2f}\n")
                f.write(f"    Standard Deviation: {row['std']:.2f}\n")
                
                best_db_config = summary_stats['best_by_database'][row['Database']]
                f.write(f"    Best Configuration: {best_db_config['architecture']}, "
                        f"{best_db_config['num_layers']} layers, "
                        f"{best_db_config['max_iterations']} iterations "
                        f"(Profit: {best_db_config['avg_simulated_profit']:.2f})\n\n")
            
            f.write("Performance by Architecture:\n")
            for _, row in arch_analysis.iterrows():
                f.write(f"  {row['Architecture']}:\n")
                f.write(f"    Average Profit: {row['mean']:.2f}\n")
                f.write(f"    Max Profit: {row['max']:.2f}\n")
                f.write(f"    Min Profit: {row['min']:.2f}\n")
                f.write(f"    Standard Deviation: {row['std']:.2f}\n")
                
                best_arch_config = summary_stats['best_by_architecture'][row['Architecture']]
                f.write(f"    Best Configuration: {best_arch_config['database']}, "
                        f"{best_arch_config['num_layers']} layers, "
                        f"{best_arch_config['max_iterations']} iterations "
                        f"(Profit: {best_arch_config['avg_simulated_profit']:.2f})\n\n")
        
        print(f"Comprehensive summary generated in {summary_dir}")
        
    except Exception as e:
        print(f"Error generating comprehensive summary: {e}")
        print(traceback.format_exc())

# Run the comprehensive validation if this file is executed directly
if __name__ == "__main__":
    # Make sure seaborn is imported for heatmaps
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn not found. Installing it for better visualization...")
        # !pip install seaborn
        import seaborn as sns
    
    comprehensive_validation()