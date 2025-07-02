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
            elif p_current < 0.5:  # Pump mode
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
        surplus_penalty_multiplier = -0.75
        shortage_penalty_multiplier = -1.5

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

class LogWeightPredictor(nn.Module):
    """Modified weight predictor that works in the log domain with custom initialization"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2, 
                 time_horizon=24, archetype='LSTM', 
                 init_w_p=0.05, init_w_q=0.05, init_w_h=0.05):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        self.archetype = archetype.upper()
        
        # Process initial weights
        self.init_w_p = init_w_p
        self.init_w_q = init_w_q
        self.init_w_h = init_w_h
        
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
    Also includes simulation to calculate realistic profit.
    """
    def __init__(self, weight_network, params, optimizer, regression, historical_data, max_iterations=3):
        self.weight_network = weight_network
        self.params = params
        self.optimizer = optimizer
        self.regression = regression
        self.historical_data = historical_data
        self.max_iterations = max_iterations
        self.simulator = SimulationLayer(params)  # Add simulation layer
    
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
        
        # Exponentiate to get actual weights (with gradient tracking)
        w_p = torch.exp(log_w_p)
        w_q = torch.exp(log_w_q)
        w_h = torch.exp(log_w_h)
        
        # Initialize parameters for first iteration
        p_current = power_init.clone().detach()  # Detach to start fresh
        h_current = head_init.clone().detach()
        flow_current = flow_init.clone().detach()
        
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
               (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results

def train_recursive_linearization(weight_network, params, optimizer_layer, regression_layer, 
                               historical_data, num_epochs=100, learning_rate=0.001, 
                               patience=10, max_iterations=3):
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
    
    print(f"Starting training with recursive linearization (max iterations={max_iterations})...")
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with recursive linearization and simulation
        simulated_profit, optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
        p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty, operating_cost, \
        (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results = pipeline.forward(train_date)
        
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
        history['iterations'].append(iter_results)
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Optimized Profit = {optimized_profit.item():.4f}, Simulated Profit = {simulated_profit.item():.4f}")
            
            # Print the power schedule for this epoch
            print(f"  Power schedule (p_opt): {p_opt.detach().cpu().numpy()}")
            print(f"  Simulated power (p_sim): {p_sim.detach().cpu().numpy()}")

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
            
            # Print simulation penalties
            print(f"  Simulation penalties - SI: {SI_penalty.item():.2f}, Volume: {volume_penalty.item():.2f}, Operating: {operating_cost.item():.2f}")
            
            print(f"  Log weights ranges - w_p: [{log_w_p.min().item():.2f}, {log_w_p.max().item():.2f}], " +
                  f"w_q: [{log_w_q.min().item():.2f}, {log_w_q.max().item():.2f}], " +
                  f"w_h: [{log_w_h.min().item():.2f}, {log_w_h.max().item():.2f}]")
            print(f"  Actual weights ranges - w_p: [{w_p.min().item():.6f}, {w_p.max().item():.6f}], " +
                  f"w_q: [{w_q.min().item():.6f}, {w_q.max().item():.6f}], " +
                  f"w_h: [{w_h.min().item():.6f}, {w_h.max().item():.6f}]")
        
        # Early stopping check based on simulated profit
        if simulated_profit.item() > best_profit:
            best_profit = simulated_profit.item()
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
    
    # 6. Linearization coefficients evolution (final epoch)
    ax6 = fig.add_subplot(6, 2, 6)
    
    # Plot c, d, e coefficients for the final epoch across iterations
    last_epoch_results = history['iterations'][last_idx]
    num_iter = len(last_epoch_results)
    iter_x = range(num_iter)
    
    c_values = np.array([result['c'].mean() for result in last_epoch_results])
    d_values = np.array([result['d'].mean() for result in last_epoch_results])
    e_values = np.array([result['e'].mean() for result in last_epoch_results])
    
    ax6.plot(iter_x, c_values, 'r-o', label='c (mean)')
    ax6.plot(iter_x, d_values, 'g-o', label='d (mean)')
    ax6.plot(iter_x, e_values, 'b-o', label='e (mean)')
    
    ax6.set_title('Flow Linearization Coefficients Evolution (Final Epoch)')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Coefficient Value')
    ax6.legend()
    ax6.grid(True)
    
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
    
    # 8. Head evolution across iterations (final epoch)
    ax8 = fig.add_subplot(6, 2, 8)
    
    # For the last epoch, show how head evolves across iterations
    for i, step in enumerate(sample_steps):
        h_values = [result['h_opt'][step] for result in last_epoch_results]
        ax8.plot(iter_x, h_values, marker=markers[i], label=f'Hour {step}')
    
    ax8.set_title('Head Evolution Across Iterations (Final Epoch)')
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Head (m)')
    ax8.legend()
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
    
    # 12. Simulated Power and Flow for Final Epoch
    ax12 = fig.add_subplot(6, 2, 12)
    ax12.plot(x, history['p_sim'][last_idx], 'b-', label='Simulated Power')
    ax12.plot(x, history['q_sim'][last_idx], 'r-', label='Simulated Flow')
    ax12.set_title('Simulated Power and Flow (Final Epoch)')
    ax12.set_xlabel('Time Step')
    ax12.set_ylabel('Value')
    ax12.legend()
    ax12.grid(True)
    
    plt.tight_layout()
    plt.savefig('recursive_linearization_with_simulation_results.png')
    plt.show()

def main():
    """Main execution function for training with recursive linearization and simulation."""
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
    
    # Initialize with custom weight values
    weight_network = LogWeightPredictor(
        input_size=4,
        hidden_size=128,
        num_layers=1,
        dropout=0.2,
        time_horizon=params.time_horizon,
        archetype='LSTM',
        init_w_p=0.6,    # Your desired initial w_p value
        init_w_q=0.02,    # Your desired initial w_q value
        init_w_h=0.1     # Your desired initial w_h value
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
        max_iterations=10  # Number of linearization recursions
    )
    
    # Save the trained model
    torch.save(trained_network.state_dict(), 'trained_recursive_linearization_with_simulation_model.pth')
    print(f"Training complete. Model saved to 'trained_recursive_linearization_with_simulation_model.pth'")
    print(f"Final optimized profit: {history['profit'][-1]:.2f}")
    print(f"Final simulated profit: {history['simulated_profit'][-1]:.2f}")
    
    print(f"Evaluation complete. Results saved to 'recursive_linearization_with_simulation_results.png'")

# Execute the main function
if __name__ == "__main__":
    main()

# %% 
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
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import time
import itertools
from datetime import datetime

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def grid_search_weights(historical_data, params, max_iterations=3):
    """
    Perform a grid search of weight values to find the combinations
    that produce the best optimization metrics.
    
    Args:
        historical_data (dict): Historical data by date
        params (HydroParameters): Parameters for the model
        max_iterations (int): Number of recursive linearization iterations
        
    Returns:
        tuple: (results, baseline_profit, baseline_ex_post, baseline_objective)
    """
    # Create grid of weight values to test (5×5×5)
    w_p_values = np.logspace(-3, 2, 10)  # 15 points from 1e-3 to 100
    w_q_values = np.logspace(-5, 1, 10)  # 15 points from 1e-5 to 10
    w_h_values = np.logspace(-5, 1, 10)  # 15 points from 1e-5 to 10
    
    # Select a date for testing
    test_date = list(historical_data.keys())[0]
    print(f"Grid search using date: {test_date}")
    
    # Get date data
    date_data = historical_data[test_date]
    power_init = date_data['power']
    head_init = date_data['head']
    flow_init = predict_q_poly(power_init, head_init)
    price = date_data['price']
    
    # Calculate baseline profit
    baseline_profit = (torch.sum(price * power_init) - params.operational_cost * torch.sum(power_init**2)).item()
    baseline_objective = baseline_profit  # Without penalties
    
    # Create regression and optimizer layers
    regression_layer = TaylorRegressionLayer(params)
    optimizer_layer = OptiLayer(params)
    simulator = SimulationLayer(params)
    
    # Initialize results tracking
    results = []
    
    # Total number of combinations to test
    total_combinations = len(w_p_values) * len(w_q_values) * len(w_h_values)
    current_combination = 0
    
    print(f"Starting grid search with {total_combinations} combinations (max_iterations={max_iterations})...")
    
    # Calculate baseline ex-post profit
    p_sim_baseline, q_sim_baseline, h_sim_baseline, v_low_sim_baseline = simulator.simulate_operation(
        power_init, flow_init, head_init
    )
    baseline_ex_post, baseline_SI_penalty, baseline_volume_penalty, baseline_operating_cost = simulator.calc_profit(
        p_sim_baseline, power_init, v_low_sim_baseline, price
    )
    baseline_ex_post = baseline_ex_post.item()
    
    # Iterate through all weight combinations
    for w_p_val in w_p_values:
        for w_q_val in w_q_values:
            for w_h_val in w_h_values:
                current_combination += 1
                print(f"Testing combination {current_combination}/{total_combinations}: "
                      f"w_p={w_p_val:.6f}, w_q={w_q_val:.6f}, w_h={w_h_val:.6f}")
                
                try:
                    # Initialize fixed weights
                    w_p = torch.full_like(power_init, w_p_val)
                    w_q = torch.full_like(power_init, w_q_val)
                    w_h = torch.full_like(power_init, w_h_val)
                    
                    # Initialize current values
                    p_current = power_init.clone().detach()
                    h_current = head_init.clone().detach()
                    flow_current = flow_init.clone().detach()
                    
                    # Store iteration results
                    iter_results = []
                    
                    # Perform recursive linearization
                    for iteration in range(max_iterations):
                        # Compute linearization coefficients
                        c, d, e, a, b = regression_layer.run_regression(p_current, h_current, flow_current)
                        
                        # Run optimization
                        p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective = optimizer_layer.forward(
                            price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                            p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                            w_p.cpu(), w_h.cpu(), w_q.cpu()
                        )
                        
                        # Store iteration result
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
                        
                        # Update for next iteration if not the last one
                        if iteration < max_iterations - 1:
                            p_current = p_opt.clone().detach().to(device=power_init.device)
                            h_current = h_opt.clone().detach().to(device=head_init.device)
                            flow_current = q_opt.clone().detach().to(device=flow_init.device)
                    
                    # Run simulation with final values
                    p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(
                        p_opt.to(device), q_opt.to(device), h_opt.to(device)
                    )
                    
                    # Calculate simulated profit
                    simulated_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
                        p_sim, p_opt.to(device), v_low_sim, price.to(device)
                    )
                    
                    # Store result
                    result = {
                        'w_p': w_p_val,
                        'w_q': w_q_val,
                        'w_h': w_h_val,
                        'simulated_profit': simulated_profit.item(),
                        'optimized_profit': optimized_profit.item(),
                        'optimized_objective': optimized_objective.item(),
                        'SI_penalty': SI_penalty.item(),
                        'volume_penalty': volume_penalty.item(),
                        'operating_cost': operating_cost.item(),
                        'iterations': iter_results,
                        'p_opt': p_opt.detach().cpu().numpy(),
                        'q_opt': q_opt.detach().cpu().numpy(),
                        'h_opt': h_opt.detach().cpu().numpy(),
                        'v_opt': v_opt.detach().cpu().numpy(),
                        'p_sim': p_sim.detach().cpu().numpy(),
                        'q_sim': q_sim.detach().cpu().numpy(),
                        'h_sim': h_sim.detach().cpu().numpy(),
                        'v_sim': v_low_sim.detach().cpu().numpy(),
                        'successful': True,
                        'max_iterations': max_iterations
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"  Error with weights (w_p={w_p_val}, w_q={w_q_val}, w_h={w_h_val}): {str(e)}")
                    # Add failed result
                    results.append({
                        'w_p': w_p_val,
                        'w_q': w_q_val,
                        'w_h': w_h_val,
                        'simulated_profit': float('nan'),
                        'optimized_profit': float('nan'),
                        'optimized_objective': float('nan'),
                        'SI_penalty': float('nan'),
                        'volume_penalty': float('nan'),
                        'operating_cost': float('nan'),
                        'successful': False,
                        'error': str(e),
                        'max_iterations': max_iterations
                    })
    
    return results, baseline_profit, baseline_ex_post, baseline_objective

def create_interactive_3d_visualizations(results, baseline_profit, baseline_ex_post, baseline_objective, output_dir, max_iterations):
    """
    Create interactive 3D visualizations for the grid search results and save as HTML files.
    """
    # Create output directory if it doesn't exist
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter successful results
    successful_results = [r for r in results if r.get('successful', False)]
    
    if not successful_results:
        print("No successful optimizations to visualize")
        return
    
    # Extract data for visualization
    data_for_viz = []
    for r in successful_results:
        if 'iterations' in r and r['iterations']:
            # Get the last iteration
            final_iter = r['iterations'][-1]
            data_for_viz.append({
                'w_p': r['w_p'],
                'w_q': r['w_q'],
                'w_h': r['w_h'],
                'optimized_profit': final_iter['optimized_profit'],
                'optimized_objective': final_iter['optimized_objective'],
                'ex_post_profit': r['simulated_profit'],
                'SI_penalty': r['SI_penalty'],
                'volume_penalty': r['volume_penalty'],
                'operating_cost': r['operating_cost'],
                'max_iterations': max_iterations
            })
    
    # Convert to DataFrame for easier filtering
    df = pd.DataFrame(data_for_viz)
    
    # Find best combinations
    best_opt_idx = df['optimized_profit'].idxmax()
    best_obj_idx = df['optimized_objective'].idxmax()
    best_ex_post_idx = df['ex_post_profit'].idxmax()
    min_si_idx = df['SI_penalty'].idxmin()  # Lowest SI penalty
    min_vol_idx = df['volume_penalty'].idxmin()  # Lowest volume penalty
    
    best_opt = df.iloc[best_opt_idx]
    best_obj = df.iloc[best_obj_idx]
    best_ex_post = df.iloc[best_ex_post_idx]
    min_si = df.iloc[min_si_idx]
    min_vol = df.iloc[min_vol_idx]
    
    # Create the 5 requested visualizations with max_iterations in the filename
    
    # 1. Objective from the final iteration
    fig_obj = go.Figure(data=[
        go.Scatter3d(
            x=df['w_p'],
            y=df['w_h'],
            z=df['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['optimized_objective'],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Optimized Objective"),
                showscale=True
            ),
            text=[
                f"w_p: {w_p:.6f}<br>" +
                f"w_h: {w_h:.6f}<br>" +
                f"w_q: {w_q:.6f}<br>" +
                f"Objective: {obj:.2f}"
                for w_p, w_h, w_q, obj in zip(df['w_p'], df['w_h'], df['w_q'], df['optimized_objective'])
            ],
            hoverinfo='text'
        )
    ])
    
    # Add best point marker
    fig_obj.add_trace(go.Scatter3d(
        x=[best_obj['w_p']],
        y=[best_obj['w_h']],
        z=[best_obj['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        name='Best Objective',
        text=[
            f"BEST OBJECTIVE<br>" +
            f"w_p: {best_obj['w_p']:.6f}<br>" +
            f"w_h: {best_obj['w_h']:.6f}<br>" +
            f"w_q: {best_obj['w_q']:.6f}<br>" +
            f"Objective: {best_obj['optimized_objective']:.2f}"
        ],
        hoverinfo='text'
    ))
    
    fig_obj.update_layout(
        title=f"Optimized Objective (Max Iterations: {max_iterations})",
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
    
    # Add baseline annotation
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
    
    pio.write_html(fig_obj, file=output_dir / f"objective_iter{max_iterations}.html", auto_open=False)
    
    # 2. Expected profit from the final iteration
    fig_profit = go.Figure(data=[
        go.Scatter3d(
            x=df['w_p'],
            y=df['w_h'],
            z=df['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['optimized_profit'],
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title="Expected Profit"),
                showscale=True
            ),
            text=[
                f"w_p: {w_p:.6f}<br>" +
                f"w_h: {w_h:.6f}<br>" +
                f"w_q: {w_q:.6f}<br>" +
                f"Expected Profit: {profit:.2f}"
                for w_p, w_h, w_q, profit in zip(df['w_p'], df['w_h'], df['w_q'], df['optimized_profit'])
            ],
            hoverinfo='text'
        )
    ])
    
    # Add best point marker
    fig_profit.add_trace(go.Scatter3d(
        x=[best_opt['w_p']],
        y=[best_opt['w_h']],
        z=[best_opt['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        name='Best Expected Profit',
        text=[
            f"BEST EXPECTED PROFIT<br>" +
            f"w_p: {best_opt['w_p']:.6f}<br>" +
            f"w_h: {best_opt['w_h']:.6f}<br>" +
            f"w_q: {best_opt['w_q']:.6f}<br>" +
            f"Expected Profit: {best_opt['optimized_profit']:.2f}"
        ],
        hoverinfo='text'
    ))
    
    fig_profit.update_layout(
        title=f"Expected Profit (Max Iterations: {max_iterations})",
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
    
    # Add baseline annotation
    fig_profit.add_annotation(
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
    
    pio.write_html(fig_profit, file=output_dir / f"expected_profit_iter{max_iterations}.html", auto_open=False)
    
    # 3. Ex-post profit
    fig_ex_post = go.Figure(data=[
        go.Scatter3d(
            x=df['w_p'],
            y=df['w_h'],
            z=df['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['ex_post_profit'],
                colorscale='Cividis',
                opacity=0.8,
                colorbar=dict(title="Ex-Post Profit"),
                showscale=True
            ),
            text=[
                f"w_p: {w_p:.6f}<br>" +
                f"w_h: {w_h:.6f}<br>" +
                f"w_q: {w_q:.6f}<br>" +
                f"Ex-Post Profit: {profit:.2f}"
                for w_p, w_h, w_q, profit in zip(df['w_p'], df['w_h'], df['w_q'], df['ex_post_profit'])
            ],
            hoverinfo='text'
        )
    ])
    
    # Add best point marker
    fig_ex_post.add_trace(go.Scatter3d(
        x=[best_ex_post['w_p']],
        y=[best_ex_post['w_h']],
        z=[best_ex_post['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        name='Best Ex-Post Profit',
        text=[
            f"BEST EX-POST PROFIT<br>" +
            f"w_p: {best_ex_post['w_p']:.6f}<br>" +
            f"w_h: {best_ex_post['w_h']:.6f}<br>" +
            f"w_q: {best_ex_post['w_q']:.6f}<br>" +
            f"Ex-Post Profit: {best_ex_post['ex_post_profit']:.2f}"
        ],
        hoverinfo='text'
    ))
    
    fig_ex_post.update_layout(
        title=f"Ex-Post Profit (Max Iterations: {max_iterations})",
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
    
    # Add baseline annotation
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
    
    pio.write_html(fig_ex_post, file=output_dir / f"ex_post_profit_iter{max_iterations}.html", auto_open=False)
    
    # 4. System imbalance penalty
    fig_si = go.Figure(data=[
        go.Scatter3d(
            x=df['w_p'],
            y=df['w_h'],
            z=df['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['SI_penalty'],
                colorscale='RdBu_r',  # Reversed so blue = low penalty
                opacity=0.8,
                colorbar=dict(title="SI Penalty"),
                showscale=True
            ),
            text=[
                f"w_p: {w_p:.6f}<br>" +
                f"w_h: {w_h:.6f}<br>" +
                f"w_q: {w_q:.6f}<br>" +
                f"SI Penalty: {penalty:.2f}"
                for w_p, w_h, w_q, penalty in zip(df['w_p'], df['w_h'], df['w_q'], df['SI_penalty'])
            ],
            hoverinfo='text'
        )
    ])
    
    # Add lowest penalty point marker
    fig_si.add_trace(go.Scatter3d(
        x=[min_si['w_p']],
        y=[min_si['w_h']],
        z=[min_si['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='green',  # Green for lowest penalty
            symbol='diamond'
        ),
        name='Lowest SI Penalty',
        text=[
            f"LOWEST SI PENALTY<br>" +
            f"w_p: {min_si['w_p']:.6f}<br>" +
            f"w_h: {min_si['w_h']:.6f}<br>" +
            f"w_q: {min_si['w_q']:.6f}<br>" +
            f"SI Penalty: {min_si['SI_penalty']:.2f}"
        ],
        hoverinfo='text'
    ))
    
    fig_si.update_layout(
        title=f"System Imbalance Penalty (Max Iterations: {max_iterations})",
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
    
    pio.write_html(fig_si, file=output_dir / f"system_imbalance_penalty_iter{max_iterations}.html", auto_open=False)
    
    # 5. Volume penalty
    fig_vol = go.Figure(data=[
        go.Scatter3d(
            x=df['w_p'],
            y=df['w_h'],
            z=df['w_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['volume_penalty'],
                colorscale='RdBu_r',  # Reversed so blue = low penalty
                opacity=0.8,
                colorbar=dict(title="Volume Penalty"),
                showscale=True
            ),
            text=[
                f"w_p: {w_p:.6f}<br>" +
                f"w_h: {w_h:.6f}<br>" +
                f"w_q: {w_q:.6f}<br>" +
                f"Volume Penalty: {penalty:.2f}"
                for w_p, w_h, w_q, penalty in zip(df['w_p'], df['w_h'], df['w_q'], df['volume_penalty'])
            ],
            hoverinfo='text'
        )
    ])
    
    # Add lowest penalty point marker
    fig_vol.add_trace(go.Scatter3d(
        x=[min_vol['w_p']],
        y=[min_vol['w_h']],
        z=[min_vol['w_q']],
        mode='markers',
        marker=dict(
            size=15,
            color='green',  # Green for lowest penalty
            symbol='diamond'
        ),
        name='Lowest Volume Penalty',
        text=[
            f"LOWEST VOLUME PENALTY<br>" +
            f"w_p: {min_vol['w_p']:.6f}<br>" +
            f"w_h: {min_vol['w_h']:.6f}<br>" +
            f"w_q: {min_vol['w_q']:.6f}<br>" +
            f"Volume Penalty: {min_vol['volume_penalty']:.2f}"
        ],
        hoverinfo='text'
    ))
    
    fig_vol.update_layout(
        title=f"Volume Penalty (Max Iterations: {max_iterations})",
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
    
    pio.write_html(fig_vol, file=output_dir / f"volume_penalty_iter{max_iterations}.html", auto_open=False)
    
    # Create interactive table of top performers
    create_top_performers_table(df, output_dir, max_iterations)
    
    print(f"Created 5 interactive 3D visualizations for max_iterations={max_iterations} in {output_dir}")
    
    # Return best results for combined dashboard
    return {
        'df': df,
        'best_obj': best_obj,
        'best_opt': best_opt,
        'best_ex_post': best_ex_post,
        'min_si': min_si,
        'min_vol': min_vol,
        'max_iterations': max_iterations
    }

def create_top_performers_table(df, output_dir, max_iterations):
    """Create an interactive table of top performers for each metric"""
    
    # Create combined table of top performers for each metric
    df_top_opt = df.nlargest(5, 'optimized_profit')[
        ['w_p', 'w_h', 'w_q', 'optimized_profit', 'optimized_objective', 'ex_post_profit']
    ].copy()
    df_top_opt['rank_type'] = 'Best Expected Profit'
    
    df_top_ex_post = df.nlargest(5, 'ex_post_profit')[
        ['w_p', 'w_h', 'w_q', 'optimized_profit', 'optimized_objective', 'ex_post_profit']
    ].copy()
    df_top_ex_post['rank_type'] = 'Best Ex-Post Profit'
    
    df_top_obj = df.nlargest(5, 'optimized_objective')[
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
        if r == 'Best Expected Profit':
            row_colors.append('lightblue')
        elif r == 'Best Optimized Objective':
            row_colors.append('lightgreen')
        else:  # 'Best Ex-Post Profit'
            row_colors.append('lightcoral')
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank Type', 'Rank', 'w_p', 'w_h', 'w_q', 'Expected Profit', 'Optimized Objective', 'Ex-Post Profit'],
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
        title=f"Top Weight Combinations by Different Metrics (Max Iterations: {max_iterations})",
        height=600,
        width=1000
    )
    
    pio.write_html(fig, file=output_dir / f"top_performers_table_iter{max_iterations}.html", auto_open=False)
    
    # Also save as CSV for reference
    df_top.to_csv(output_dir / f"top_performers_iter{max_iterations}.csv", index=False)

def create_combined_dashboard(all_results, baseline_profit, baseline_ex_post, baseline_objective, output_dir):
    """Create a comprehensive dashboard that compares results across different max_iterations values"""
    
    iteration_values = sorted([result['max_iterations'] for result in all_results])
    
    # Extract best values for each metric across different max_iterations
    best_objective_by_iter = {}
    best_expected_profit_by_iter = {}
    best_ex_post_profit_by_iter = {}
    lowest_si_by_iter = {}
    lowest_vol_by_iter = {}
    
    for result in all_results:
        iter_val = result['max_iterations']
        best_objective_by_iter[iter_val] = result['best_obj']['optimized_objective']
        best_expected_profit_by_iter[iter_val] = result['best_opt']['optimized_profit']
        best_ex_post_profit_by_iter[iter_val] = result['best_ex_post']['ex_post_profit']
        lowest_si_by_iter[iter_val] = result['min_si']['SI_penalty']
        lowest_vol_by_iter[iter_val] = result['min_vol']['volume_penalty']
    
    # Find the overall best configurations
    best_objective_iter = max(best_objective_by_iter.items(), key=lambda x: x[1])[0]
    best_expected_profit_iter = max(best_expected_profit_by_iter.items(), key=lambda x: x[1])[0]
    best_ex_post_profit_iter = max(best_ex_post_profit_by_iter.items(), key=lambda x: x[1])[0]
    lowest_si_iter = min(lowest_si_by_iter.items(), key=lambda x: x[1])[0]
    lowest_vol_iter = min(lowest_vol_by_iter.items(), key=lambda x: x[1])[0]
    
    # Get best configurations from each iteration
    best_obj_configs = {r['max_iterations']: r['best_obj'] for r in all_results}
    best_profit_configs = {r['max_iterations']: r['best_opt'] for r in all_results}
    best_ex_post_configs = {r['max_iterations']: r['best_ex_post'] for r in all_results}
    
    # Create comparison plots
    # 1. Bar chart comparing best values across iterations
    fig_comparison = go.Figure()
    
    # Add expected profit comparison
    fig_comparison.add_trace(go.Bar(
        x=iteration_values,
        y=[best_expected_profit_by_iter[i] for i in iteration_values],
        name='Best Expected Profit',
        marker_color='royalblue'
    ))
    
    # Add ex-post profit comparison
    fig_comparison.add_trace(go.Bar(
        x=iteration_values,
        y=[best_ex_post_profit_by_iter[i] for i in iteration_values],
        name='Best Ex-Post Profit',
        marker_color='indianred'
    ))
    
    # Add objective comparison
    fig_comparison.add_trace(go.Bar(
        x=iteration_values,
        y=[best_objective_by_iter[i] for i in iteration_values],
        name='Best Objective',
        marker_color='seagreen'
    ))
    
    fig_comparison.update_layout(
        title="Comparison of Best Results Across Max Iterations",
        xaxis_title="Max Iterations",
        yaxis_title="Value",
        barmode='group',
        height=500,
        width=900
    )
    
    # Save comparison plot
    pio.write_html(fig_comparison, file=output_dir / "metric_comparison_across_iterations.html", auto_open=False)
    
    # 2. Create penalty comparison plot
    fig_penalties = go.Figure()
    
    # Add SI penalty comparison
    fig_penalties.add_trace(go.Bar(
        x=iteration_values,
        y=[lowest_si_by_iter[i] for i in iteration_values],
        name='Lowest SI Penalty',
        marker_color='darkred'
    ))
    
    # Add volume penalty comparison
    fig_penalties.add_trace(go.Bar(
        x=iteration_values,
        y=[lowest_vol_by_iter[i] for i in iteration_values],
        name='Lowest Volume Penalty',
        marker_color='darkblue'
    ))
    
    fig_penalties.update_layout(
        title="Comparison of Lowest Penalties Across Max Iterations",
        xaxis_title="Max Iterations",
        yaxis_title="Penalty Value",
        barmode='group',
        height=500,
        width=900
    )
    
    # Save penalty comparison plot
    pio.write_html(fig_penalties, file=output_dir / "penalty_comparison_across_iterations.html", auto_open=False)
    
    # Create tab content for each iteration
    tab_contents = []
    for i in iteration_values:
        # Create tab content for iteration i
        tab_content = f'''
            <div id="tab{i}" class="tab-content">
                <h3>Results for {i} Iterations</h3>
                
                <div class="container">
                    <div class="half-section">
                        <h4>Best Configurations</h4>
                        <table class="metrics-table">
                            <tr>
                                <th>Metric</th>
                                <th>w_p</th>
                                <th>w_h</th>
                                <th>w_q</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Best Objective</td>
                                <td>{best_obj_configs[i]["w_p"]:.6f}</td>
                                <td>{best_obj_configs[i]["w_h"]:.6f}</td>
                                <td>{best_obj_configs[i]["w_q"]:.6f}</td>
                                <td>{best_obj_configs[i]["optimized_objective"]:.2f}</td>
                            </tr>
                            <tr>
                                <td>Best Expected Profit</td>
                                <td>{best_profit_configs[i]["w_p"]:.6f}</td>
                                <td>{best_profit_configs[i]["w_h"]:.6f}</td>
                                <td>{best_profit_configs[i]["w_q"]:.6f}</td>
                                <td>{best_profit_configs[i]["optimized_profit"]:.2f}</td>
                            </tr>
                            <tr>
                                <td>Best Ex-Post Profit</td>
                                <td>{best_ex_post_configs[i]["w_p"]:.6f}</td>
                                <td>{best_ex_post_configs[i]["w_h"]:.6f}</td>
                                <td>{best_ex_post_configs[i]["w_q"]:.6f}</td>
                                <td>{best_ex_post_configs[i]["ex_post_profit"]:.2f}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="half-section">
                        <h4>Detailed Analysis</h4>
                        <p><a href="top_performers_table_iter{i}.html" target="_blank">View Top Performers for {i} Iterations</a></p>
                        <p>Baseline values for comparison:</p>
                        <ul>
                            <li>Baseline Expected Profit: {baseline_profit:.2f}</li>
                            <li>Baseline Ex-Post Profit: {baseline_ex_post:.2f}</li>
                            <li>Baseline Objective: {baseline_objective:.2f}</li>
                        </ul>
                        <p>Using {i} iterations {'provides better results than the baseline' if best_ex_post_configs[i]['ex_post_profit'] > baseline_ex_post else 'does not significantly improve over the baseline'} for ex-post profit.</p>
                    </div>
                </div>
                
                <h3>Interactive 3D Visualizations ({i} Iterations)</h3>
                <div class="container">
                    <div class="viz-section">
                        <h4>Profit & Objective</h4>
                        <div class="viz-link">
                            <a href="objective_iter{i}.html" target="_blank">Optimized Objective</a>
                        </div>
                        <div class="viz-link">
                            <a href="expected_profit_iter{i}.html" target="_blank">Expected Profit</a>
                        </div>
                        <div class="viz-link">
                            <a href="ex_post_profit_iter{i}.html" target="_blank">Ex-Post Profit</a>
                        </div>
                    </div>
                    
                    <div class="viz-section">
                        <h4>Penalties</h4>
                        <div class="viz-link">
                            <a href="system_imbalance_penalty_iter{i}.html" target="_blank">System Imbalance Penalty</a>
                        </div>
                        <div class="viz-link">
                            <a href="volume_penalty_iter{i}.html" target="_blank">Volume Penalty</a>
                        </div>
                    </div>
                </div>
            </div>
        '''
        tab_contents.append(tab_content)
    
    # Join all tab contents
    all_tab_contents = '\n'.join(tab_contents)
    
    # Create tabs HTML
    tabs_html = ' '.join([f'<div class="tab" onclick="openTab(event, \'tab{i}\')">{i} Iterations</div>' for i in iteration_values])
    
    # Create main dashboard HTML - note we're being careful with f-strings
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UPHES Grid Search Results Across Multiple Iterations</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .section {{ 
                width: 100%; 
                margin-bottom: 20px; 
                background-color: white; 
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
            }}
            .half-section {{
                width: 48%;
                margin-bottom: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
            }}
            .viz-section {{ 
                width: 23%; 
                margin-bottom: 20px; 
                background-color: white; 
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 15px;
            }}
            .viz-link {{ 
                margin: 10px 0;
                padding: 8px;
                background-color: #f8f9fa;
                border-radius: 5px;
                transition: background-color 0.2s;
            }}
            .viz-link:hover {{ 
                background-color: #e9ecef;
            }}
            .viz-link a {{ 
                color: #3498db; 
                text-decoration: none; 
                font-weight: bold;
            }}
            .viz-link a:hover {{ 
                text-decoration: underline; 
            }}
            .metrics-table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metrics-table th, .metrics-table td {{ 
                padding: 10px; 
                text-align: left; 
                border-bottom: 1px solid #ddd; 
            }}
            .metrics-table th {{ 
                background-color: #f2f2f2; 
                font-weight: bold;
            }}
            .metrics-table tr:nth-child(even) {{ 
                background-color: #f9f9f9; 
            }}
            .metrics-table tr:hover {{ 
                background-color: #eef5ff; 
            }}
            .best {{ background-color: #d4edda; }}
            .metric-value {{ 
                font-weight: bold; 
                color: #2980b9;
            }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .header-section {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}
            .summary-box {{
                background-color: #eef5ff;
                border-left: 5px solid #3498db;
                padding: 15px;
                margin: 15px 0;
            }}
            .tabs {{
                display: flex;
                margin-bottom: 20px;
            }}
            .tab {{
                padding: 10px 20px;
                background-color: #f8f9fa;
                cursor: pointer;
                margin-right: 5px;
                border-radius: 5px 5px 0 0;
                border: 1px solid #dee2e6;
                border-bottom: none;
            }}
            .tab.active {{
                background-color: white;
                border-bottom: 1px solid white;
                margin-bottom: -1px;
                font-weight: bold;
            }}
            .tab-content {{
                display: none;
                padding: 20px;
                border: 1px solid #dee2e6;
                border-radius: 0 0 5px 5px;
                background-color: white;
            }}
            .tab-content.active {{
                display: block;
            }}
            iframe {{
                border: none;
                width: 100%;
                height: 600px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                }}
                tablinks = document.getElementsByClassName("tab");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(tabName).className += " active";
                evt.currentTarget.className += " active";
            }}
            
            // Auto-open first tab on load
            window.onload = function() {{
                document.getElementsByClassName("tab")[0].click();
            }};
        </script>
    </head>
    <body>
        <div class="header-section">
            <h1>UPHES Grid Search Results Across Multiple Max Iterations</h1>
            <div>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Summary of Results</h2>
            
            <div class="summary-box">
                <p>This dashboard presents the results of grid searches with different maximum iterations ({', '.join(map(str, iteration_values))}).
                The grid search explored 125 combinations of weights (w_p, w_h, w_q) ranging from 1e-3 to 10 for each max_iterations setting.</p>
            </div>
            
            <h3>Interactive Comparisons Across Iterations</h3>
            <div class="container">
                <div class="half-section">
                    <h4>Profit & Objective Metrics</h4>
                    <iframe src="metric_comparison_across_iterations.html"></iframe>
                </div>
                
                <div class="half-section">
                    <h4>Penalty Metrics</h4>
                    <iframe src="penalty_comparison_across_iterations.html"></iframe>
                </div>
            </div>
            
            <h3>Best Results by Metric Across All Iterations</h3>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Best Value</th>
                    <th>Max Iterations</th>
                    <th>w_p</th>
                    <th>w_h</th>
                    <th>w_q</th>
                    <th>Improvement from Baseline</th>
                </tr>
                <tr class="best">
                    <td>Best Objective</td>
                    <td class="metric-value">{best_objective_by_iter[best_objective_iter]:.2f}</td>
                    <td>{best_objective_iter}</td>
                    <td>{best_obj_configs[best_objective_iter]['w_p']:.6f}</td>
                    <td>{best_obj_configs[best_objective_iter]['w_h']:.6f}</td>
                    <td>{best_obj_configs[best_objective_iter]['w_q']:.6f}</td>
                    <td class="{('positive' if best_objective_by_iter[best_objective_iter] > baseline_objective else 'negative')}">
                        {best_objective_by_iter[best_objective_iter] - baseline_objective:.2f} 
                        ({(best_objective_by_iter[best_objective_iter] - baseline_objective)/abs(baseline_objective)*100 if baseline_objective != 0 else 0:.2f}%)
                    </td>
                </tr>
                <tr class="best">
                    <td>Best Expected Profit</td>
                    <td class="metric-value">{best_expected_profit_by_iter[best_expected_profit_iter]:.2f}</td>
                    <td>{best_expected_profit_iter}</td>
                    <td>{best_profit_configs[best_expected_profit_iter]['w_p']:.6f}</td>
                    <td>{best_profit_configs[best_expected_profit_iter]['w_h']:.6f}</td>
                    <td>{best_profit_configs[best_expected_profit_iter]['w_q']:.6f}</td>
                    <td class="{('positive' if best_expected_profit_by_iter[best_expected_profit_iter] > baseline_profit else 'negative')}">
                        {best_expected_profit_by_iter[best_expected_profit_iter] - baseline_profit:.2f}
                        ({(best_expected_profit_by_iter[best_expected_profit_iter] - baseline_profit)/abs(baseline_profit)*100 if baseline_profit != 0 else 0:.2f}%)
                    </td>
                </tr>
                <tr class="best">
                    <td>Best Ex-Post Profit</td>
                    <td class="metric-value">{best_ex_post_profit_by_iter[best_ex_post_profit_iter]:.2f}</td>
                    <td>{best_ex_post_profit_iter}</td>
                    <td>{best_ex_post_configs[best_ex_post_profit_iter]['w_p']:.6f}</td>
                    <td>{best_ex_post_configs[best_ex_post_profit_iter]['w_h']:.6f}</td>
                    <td>{best_ex_post_configs[best_ex_post_profit_iter]['w_q']:.6f}</td>
                    <td class="{('positive' if best_ex_post_profit_by_iter[best_ex_post_profit_iter] > baseline_ex_post else 'negative')}">
                        {best_ex_post_profit_by_iter[best_ex_post_profit_iter] - baseline_ex_post:.2f}
                        ({(best_ex_post_profit_by_iter[best_ex_post_profit_iter] - baseline_ex_post)/abs(baseline_ex_post)*100 if baseline_ex_post != 0 else 0:.2f}%)
                    </td>
                </tr>
                <tr>
                    <td>Lowest SI Penalty</td>
                    <td class="metric-value">{lowest_si_by_iter[lowest_si_iter]:.2f}</td>
                    <td>{lowest_si_iter}</td>
                    <td colspan="4">Lower values are better (represents lower system imbalance)</td>
                </tr>
                <tr>
                    <td>Lowest Volume Penalty</td>
                    <td class="metric-value">{lowest_vol_by_iter[lowest_vol_iter]:.2f}</td>
                    <td>{lowest_vol_iter}</td>
                    <td colspan="4">Lower values are better (represents better adherence to volume constraints)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Detailed Results by Max Iterations</h2>
            
            <div class="tabs">
                {tabs_html}
            </div>
            
            {all_tab_contents}
        </div>
        
        <div class="section">
            <h2>Conclusions</h2>
            
            <div class="summary-box">
                <h3>Effect of Max Iterations on Results</h3>
                <p>Based on the grid search results across different maximum iterations:</p>
                <ul>
                    <li>The best overall objective value was found with {best_objective_iter} iterations.</li>
                    <li>The best expected profit was achieved with {best_expected_profit_iter} iterations.</li>
                    <li>The best ex-post profit was observed with {best_ex_post_profit_iter} iterations.</li>
                    <li>More iterations {"generally lead to better results" if best_ex_post_profit_iter > min(iteration_values) else "do not necessarily lead to better results"} for ex-post profit.</li>
                    <li>The lowest system imbalance penalty was found with {lowest_si_iter} iterations.</li>
                    <li>The lowest volume penalty was achieved with {lowest_vol_iter} iterations.</li>
                </ul>
            </div>
            
            <h3>Recommended Weight Configuration</h3>
            <p>Based on all results, the recommended weight configuration for best overall performance is:</p>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Max Iterations</th>
                    <th>w_p</th>
                    <th>w_h</th>
                    <th>w_q</th>
                    <th>Value</th>
                </tr>
                <tr class="best">
                    <td>Best Ex-Post Profit</td>
                    <td>{best_ex_post_profit_iter}</td>
                    <td>{best_ex_post_configs[best_ex_post_profit_iter]['w_p']:.6f}</td>
                    <td>{best_ex_post_configs[best_ex_post_profit_iter]['w_h']:.6f}</td>
                    <td>{best_ex_post_configs[best_ex_post_profit_iter]['w_q']:.6f}</td>
                    <td>{best_ex_post_profit_by_iter[best_ex_post_profit_iter]:.2f}</td>
                </tr>
            </table>
            
            <p>This configuration provides the best real-world performance after simulation.</p>
        </div>
        
        <div class="footer">
            <p>UPHES Weight Parameter Grid Search © {datetime.now().year} | Total configurations tested: {len(iteration_values) * 125}</p>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    dashboard_path = output_dir / "grid_search_dashboard_combined.html"
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created comprehensive combined dashboard at {dashboard_path}")

def run_multi_iteration_grid_search():
    """
    Run grid search with multiple max iterations values (3, 5, 10, 15).
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
    
    # Define max_iterations values to test
    max_iterations_values = [3, 5, 10, 15]
    
    # Create main output directory
    output_dir = Path("./uphes_grid_search_multi_iter")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all results for the combined dashboard
    all_viz_results = []
    
    # For baseline values (same for all runs)
    baseline_profit = None
    baseline_ex_post = None
    baseline_objective = None
    
    # Run grid search for each max_iterations value
    for max_iter in max_iterations_values:
        print(f"\n{'='*80}")
        print(f"Running grid search with max_iterations = {max_iter}")
        print(f"{'='*80}")
        
        # Run grid search with current max_iterations value
        results, baseline_profit, baseline_ex_post, baseline_objective = grid_search_weights(
            historical_data=historical_data,
            params=params,
            max_iterations=max_iter
        )
        
        # Create directory for this specific max_iterations value
        iter_dir = output_dir / f"iter_{max_iter}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results for this max_iterations
        with open(iter_dir / f"grid_search_results_iter{max_iter}.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Create DataFrame for CSV export
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if not isinstance(v, (list, dict, np.ndarray, torch.Tensor))} 
            for r in results
        ])
        
        # Save as CSV
        results_df.to_csv(iter_dir / f"grid_search_results_iter{max_iter}.csv", index=False)
        
        # Create visualizations for this max_iterations
        print(f"\nCreating visualizations for max_iterations = {max_iter}...")
        viz_results = create_interactive_3d_visualizations(
            results, baseline_profit, baseline_ex_post, baseline_objective, iter_dir, max_iter
        )
        
        # Track results for combined dashboard
        all_viz_results.append(viz_results)
    
    # Create a single combined dashboard
    print("\nCreating combined dashboard across all max_iterations values...")
    create_combined_dashboard(all_viz_results, baseline_profit, baseline_ex_post, baseline_objective, output_dir)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"\nMulti-iteration grid search completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"Results saved to {output_dir}")
    
    return all_viz_results, baseline_profit, baseline_ex_post, baseline_objective

if __name__ == "__main__":
    # Run the grid search with multiple max_iterations values
    all_results, baseline_profit, baseline_ex_post, baseline_objective = run_multi_iteration_grid_search()