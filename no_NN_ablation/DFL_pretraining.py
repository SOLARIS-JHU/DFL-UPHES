# %% Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import sys
# from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from joblib import Parallel, delayed
import multiprocessing
# torch.autograd.set_detect_anomaly(True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# load portfolio data
sys.path.append('../Library')
from V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('../preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = torch.tensor(77.0, device=device)  # Initial head value
print(f"Initial head: {head_init.item()}")
v_low_init = torch.tensor(h_to_v_low_fitted(head_init), device=device)  # Initial lower reservoir volume
print(f"Initial head: {head_init.item()}, Initial v_low: {v_low_init.item()}")
def hourly_to_quarterly(tensor_data):
    return tensor_data.repeat_interleave(4)

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
        operational_cost=0.4, 
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
                    # q(p,h) â‰ˆ q(p0,h0) + (âˆ‚q/âˆ‚p)(p-p0) + (âˆ‚q/âˆ‚h)(h-h0)
                    # Rearranged as: q(p,h) â‰ˆ (âˆ‚q/âˆ‚p)*p + (âˆ‚q/âˆ‚h)*h + [q(p0,h0) - (âˆ‚q/âˆ‚p)*p0 - (âˆ‚q/âˆ‚h)*h0]
                    c = dq_dp.detach()  # corresponds to âˆ‚q/âˆ‚p
                    d = dq_dh.detach()  # corresponds to âˆ‚q/âˆ‚h
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
                # v_low(h) â‰ˆ v_low(h0) + (dv_low/dh)(h-h0)
                # Rearranged as: v_low(h) â‰ˆ (dv_low/dh)*h + [v_low(h0) - (dv_low/dh)*h0]
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
            print(f"\nâš ï¸ Solver error: {er}")
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
            print("\nâŒ NaN detected in solution. Parameters:")
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
            q (torch.Tensor): Hourly flow schedule [time_horizon] (not directly used, recalculated)
            h (torch.Tensor): Hourly head schedule [time_horizon] (from optimization, for reference)
        
        Returns:
            tuple: Calibrated hourly (p, q, h, v_low) schedules.
        """
        TH = self.params.time_horizon
        
        # Initialize lists for each state
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        # Start states - use initial conditions
        v_current = self.params.v_low_init  # Initial reservoir volume
        h_current = self.params.head_init   # Initial head value
        
        v_list.append(v_current)
        h_list.append(h_current)  # Store initial head

        for i in range(TH):
            p_current = p[i]
            
            # a) Base: idle => q=0
            q_candidate = torch.zeros_like(p_current)
            p_clamped = p_current

            # b) For turbine mode (p_current>0), clamp p between pos_min(h) and pos_max(h)
            #    then get q via polynomial using CURRENT head (not optimized head)
            if p_current > 0.5:  # Turbine mode
                p_min_turb = self.params.pos_min(h_current)  # Use current head
                p_max_turb = self.params.pos_max(h_current)  # Use current head
                p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            
            # c) For pump mode (p_current<0), clamp p between neg_min(h) and neg_max(h)
            elif p_current < -0.5:  # Pump mode
                p_min_pump = self.params.neg_min(h_current)  # Use current head
                p_max_pump = self.params.neg_max(h_current)  # Use current head
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
            
            # Update current states for next iteration
            v_current = v_next
            h_current = h_next  # Important: update h_current for next iteration
            
            v_list.append(v_current.item())
            h_list.append(h_current)
        
        # Convert lists to tensors
        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list[:-1])  # Remove the extra head value (we have TH+1 heads)
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32)  # Remove extra volume
        
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
                 max_iterations=3, penalty_growth_rate=1.5):
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
                               patience=10, max_iterations=3, penalty_growth_rate=1.5):
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
        print(f"    - Original Flow (mÂ³/s): {flow_orig.cpu().numpy()}")
        print(f"    - Optimized Flow (mÂ³/s): {q_opt.detach().cpu().numpy()}")
        print(f"    - Simulated Flow (mÂ³/s): {q_sim.detach().cpu().numpy()}")
        
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
    try:
        # Force comma separator - the files are comma-separated, not tab-separated
        df = pd.read_csv(file_path, sep=',', header=0)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        print(f"Actual columns in {source_name}: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
        print(f"First few rows:\n{df.head(3)}")
        
        # Check for required columns
        required_columns = ['date', 'hour', 'power', 'head', 'flow']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check if price column already exists in the file
        if 'price' in df.columns:
            print(f"Using price data from {source_name} file.")
            # Check for missing price values
            if df['price'].isna().any():
                raise ValueError("Price values are missing. Synthetic prices have been removed - please provide complete price data.")
        else:
            print(f"No price column found in {source_name}. Trying to load from original MIQP file...")
            
            # Load original MIQP data to get price information
            original_miqp_file = "../MIQP/MIQP_piecewise/MIQP_piecewise_results.csv"
            if os.path.exists(original_miqp_file):
                print(f"Loading price data from {original_miqp_file}...")
                price_df = pd.read_csv(original_miqp_file)
                price_df.columns = price_df.columns.str.strip()
                
                # Convert date formats to match for merging
                # Handle different date formats
                try:
                    df['date_normalized'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    price_df['date_normalized'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')
                    
                    # Merge price data with noisy data on normalized date and hour
                    df = df.merge(price_df[['date_normalized', 'hour', 'price']], 
                                 left_on=['date_normalized', 'hour'], 
                                 right_on=['date_normalized', 'hour'], 
                                 how='left')
                    df.drop('date_normalized', axis=1, inplace=True)
                    
                except Exception as e:
                    print(f"Date format conversion failed: {e}")
                    df['price'] = None
                
                if 'price' not in df.columns or df['price'].isna().all():
                    raise ValueError("Price merge failed and synthetic prices have been removed. Please provide price data directly in the source file.")
            else:
                raise ValueError(f"Original MIQP file {original_miqp_file} not found and synthetic prices have been removed. Please provide price data.")
        
        # Convert date column - handle different formats
        try:
            df['Date'] = pd.to_datetime(df['date'])
        except:
            # Try different date formats
            try:
                df['Date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            except:
                df['Date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        
        df['Time'] = df['hour']
        
        # Rename columns to match expected format (capitalize first letter)
        df = df.rename(columns={
            'power': 'Power',
            'head': 'Head', 
            'flow': 'Flow',
            'price': 'Price'
        })
        
        # Add 'Mode' column based on power values
        conditions = [
            (abs(df['Power']) < 0.01),  # Idle mode
            (df['Power'] > 0),          # Turbine mode  
            (df['Power'] < 0)           # Pump mode
        ]
        choices = ['Idle', 'Turbine', 'Pump']
        df['Mode'] = np.select(conditions, choices, default='Unknown')
        
        # Verify price data
        print(f"Price data statistics:")
        print(f"  Min: {df['Price'].min():.2f}")
        print(f"  Max: {df['Price'].max():.2f}")
        print(f"  Mean: {df['Price'].mean():.2f}")
        print(f"  Missing values: {df['Price'].isna().sum()}")
        
        # Group data by date
        data_by_date = {}
        for date, group in df.groupby('Date'):
            # Sort by hour to ensure correct order
            group = group.sort_values('Time')
            
            # Ensure we have 24 hours of data
            if len(group) != 24:
                print(f"Warning: Date {date.strftime('%Y-%m-%d')} has {len(group)} hours instead of 24. Skipping.")
                continue
            
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
        import traceback
        traceback.print_exc()
        return None

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
        avg_time = df.groupby(['Architecture', 'Num_Layers', 'Max_Iterations', 'Noise_Level'])['Training_Time_Seconds'].mean().reset_index()
        
        # Create configuration labels for plotting
        avg_time['Config'] = avg_time.apply(
            lambda x: f"{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter-noise{x['Noise_Level']}", axis=1
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
        avg_profit = df.groupby(['Architecture', 'Num_Layers', 'Max_Iterations', 'Noise_Level'])[
            ['Optimized_Profit', 'Simulated_Profit']
        ].mean().reset_index()
        
        # Create configuration labels
        avg_profit['Config'] = avg_profit.apply(
            lambda x: f"{x['Architecture']}-{x['Num_Layers']}L-{x['Max_Iterations']}iter-noise{x['Noise_Level']}", axis=1
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
            f.write(f"  Noise Level: {best_optimized['Noise_Level']}\n")
            f.write(f"  Average Optimized Profit: {best_optimized['Optimized_Profit']:.2f}\n\n")
            
            f.write("Best Configuration for Simulated Profit:\n")
            f.write(f"  Architecture: {best_simulated['Architecture']}\n")
            f.write(f"  Number of Layers: {best_simulated['Num_Layers']}\n")
            f.write(f"  Max Iterations: {best_simulated['Max_Iterations']}\n")
            f.write(f"  Noise Level: {best_simulated['Noise_Level']}\n")
            f.write(f"  Average Simulated Profit: {best_simulated['Simulated_Profit']:.2f}\n\n")
            
            f.write("Average training time: {:.2f} seconds\n".format(df['Training_Time_Seconds'].mean()))
            f.write("Maximum training time: {:.2f} seconds\n".format(df['Training_Time_Seconds'].max()))
            f.write("Minimum training time: {:.2f} seconds\n".format(df['Training_Time_Seconds'].min()))
        
        # Save best configuration as a JSON for easy retrieval during validation
        best_config = {
            'architecture': best_simulated['Architecture'],
            'num_layers': int(best_simulated['Num_Layers']),
            'max_iterations': int(best_simulated['Max_Iterations']),
            'noise_level': float(best_simulated['Noise_Level']),
            'average_simulated_profit': float(best_simulated['Simulated_Profit'])
        }
        
        with open(summary_dir / "best_configuration.json", 'w') as f:
            json.dump(best_config, f, indent=4)
        
        print(f"Benchmark summary generated for {source_name} in {summary_dir}")
        
    except Exception as e:
        print(f"Error generating benchmark summary for {source_name}: {e}")

def generate_cross_database_comparison():
    """Generate comparative analysis between databases."""
    try:
        # Define noise levels for analysis - updated to match relative noise script output
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Check if benchmark files exist for noise levels
        benchmark_files = {}
        for noise_level in noise_levels:
            source_name = f"MIQP_piecewise_results_relative_noise_{int(noise_level*100)}pct"
            benchmark_file = Path(f"./trained_models/{source_name}/pretraining_benchmarks.csv")
            if benchmark_file.exists():
                benchmark_files[noise_level] = benchmark_file
        
        if not benchmark_files:
            print("Cannot generate database analysis: no benchmark files found")
            return
        
        # Create output directory
        analysis_dir = Path("./trained_models/analysis")
        analysis_dir.mkdir(exist_ok=True)
        
        # Combine all data
        all_data = []
        for noise_level, benchmark_file in benchmark_files.items():
            df = pd.read_csv(benchmark_file)
            df['Noise_Level'] = noise_level
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 1. Generate profit analysis by noise level
        plt.figure(figsize=(15, 10))
        
        # Compute average profit by noise level and configuration
        avg_profit = combined_df.groupby(['Noise_Level', 'Max_Iterations'])[
            'Simulated_Profit'
        ].mean().reset_index()
        
        # Plot by noise level
        for noise_level in noise_levels:
            if noise_level in benchmark_files:
                data = avg_profit[avg_profit['Noise_Level'] == noise_level]
                plt.plot(data['Max_Iterations'], data['Simulated_Profit'], 
                        marker='o', label=f'Noise {int(noise_level*100)}%')
        
        plt.title('Average Simulated Profit by Noise Level and Max Iterations')
        plt.xlabel('Max Iterations')
        plt.ylabel('Average Simulated Profit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(analysis_dir / "profit_by_noise_level.png")
        plt.close()
        
        # 2. Find the best configuration overall
        best_config = combined_df.loc[combined_df['Simulated_Profit'].idxmax()]
        
        # Save analysis summary
        with open(analysis_dir / "database_analysis.txt", 'w') as f:
            f.write("Multi-Noise Level Database Analysis Summary\n")
            f.write("==========================================\n\n")
            
            f.write(f"Noise levels processed: {list(benchmark_files.keys())}\n")
            f.write(f"Total configurations tested: {len(combined_df)}\n\n")
            
            f.write("Best Configuration Overall:\n")
            f.write(f"  Architecture: {best_config['Architecture']}\n")
            f.write(f"  Number of Layers: {best_config['Num_Layers']}\n")
            f.write(f"  Max Iterations: {best_config['Max_Iterations']}\n")
            f.write(f"  Noise Level: {best_config['Noise_Level']}\n")
            f.write(f"  Date: {best_config['Date']}\n")
            f.write(f"  Simulated Profit: {best_config['Simulated_Profit']:.2f}\n\n")
            
            # Statistics by noise level
            f.write("Statistics by Noise Level:\n")
            for noise_level in sorted(benchmark_files.keys()):
                data = combined_df[combined_df['Noise_Level'] == noise_level]
                f.write(f"  Noise {int(noise_level*100)}%:\n")
                f.write(f"    Configurations: {len(data)}\n")
                f.write(f"    Average Simulated Profit: {data['Simulated_Profit'].mean():.2f}\n")
                f.write(f"    Best Simulated Profit: {data['Simulated_Profit'].max():.2f}\n")
                f.write(f"    Average Training Time: {data['Training_Time_Seconds'].mean():.2f}s\n\n")
        
        # Save best configuration for use in validation
        best_config_dict = {
            'architecture': best_config['Architecture'],
            'num_layers': int(best_config['Num_Layers']),
            'max_iterations': int(best_config['Max_Iterations']),
            'noise_level': float(best_config['Noise_Level']),
            'date': best_config['Date'],
            'simulated_profit': float(best_config['Simulated_Profit'])
        }
        
        with open(analysis_dir / "best_configuration.json", 'w') as f:
            json.dump(best_config_dict, f, indent=4)
        
        print(f"Database analysis generated in {analysis_dir}")
        print(f"Best configuration: LSTM-3L-{best_config['Max_Iterations']}iter-noise{int(best_config['Noise_Level']*100)}%")
        print(f"Best simulated profit: {best_config['Simulated_Profit']:.2f}")
        
    except Exception as e:
        print(f"Error generating database analysis: {e}")
        import traceback
        traceback.print_exc()

# %% Train models in parallel for each configuration and date
def train_single_model(data_type, noise_level, file_path, architecture, num_layers, max_iterations, date_str, date_data, start_timestamp):
    """Train a single model configuration for a specific date and database type."""
    try:
        # Initialize parameters (each process needs its own)
        params = HydroParameters()
        regression_layer = TaylorRegressionLayer(params)
        optimizer_layer = OptiLayer(params)
        
        # Create directory structure based on data type
        if data_type == 'random_samples':
            source_name = "MIQP_piecewise_results_random_samples"
        else:  # relative_noise
            source_name = f"MIQP_piecewise_results_relative_noise_{int(noise_level*100)}pct"
        
        root_dir = Path(f"./trained_models/{source_name}")
        config_name = f"{architecture}_{num_layers}layer_{max_iterations}iter"
        config_dir = root_dir / config_name
        date_dir = config_dir / date_str
        date_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize network
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
            w_p_min=0.1,  
            w_p_max=3.0,   
            w_q_min=0.001,
            w_q_max=0.2,
            w_h_min=0.01,
            w_h_max=5.0
        ).to(device)
        
        # Train
        start_time = time.time()
        trained_network, history = train_recursive_linearization(
            weight_network=weight_network,
            params=params,
            optimizer_layer=optimizer_layer,
            regression_layer=regression_layer,
            historical_data={date_str: date_data},
            num_epochs=500,
            learning_rate=1e-3,
            patience=20,
            max_iterations=max_iterations,
            penalty_growth_rate=1.5
        )
        training_time = time.time() - start_time
        
        # Save results
        torch.save(trained_network.state_dict(), date_dir / "model.pt")
        torch.save(trained_network.state_dict(), date_dir / "best_model.pt")
        
        # Save history
        simplified_history = {
            'epoch': history['epoch'],
            'loss': [float(x) for x in history['loss']],
            'profit': [float(x) for x in history['profit']],
            'simulated_profit': [float(x) for x in history['simulated_profit']],
            'SI_penalty': [float(x) if hasattr(x, 'item') else x for x in history['SI_penalty']],
            'volume_penalty': [float(x) if hasattr(x, 'item') else x for x in history['volume_penalty']],
            'operating_cost': [float(x) if hasattr(x, 'item') else x for x in history['operating_cost']],
        }
        
        with open(date_dir / "training_history.json", 'w') as f:
            json.dump(simplified_history, f, indent=4)
        
        # Return results for benchmark
        last_idx = len(history['epoch']) - 1
        best_epoch = history['epoch'][np.argmax(history['simulated_profit'])]
        
        return {
            'data_type': data_type,
            'noise_level': noise_level if data_type == 'relative_noise' else None,
            'source_name': source_name,
            'architecture': architecture,
            'num_layers': num_layers,
            'max_iterations': max_iterations,
            'date_str': date_str,
            'training_time': training_time,
            'epochs_trained': last_idx + 1,
            'best_epoch': best_epoch,
            'optimized_profit': float(history['profit'][last_idx]),
            'simulated_profit': float(history['simulated_profit'][last_idx]),
            'SI_penalty': float(history['SI_penalty'][last_idx]),
            'volume_penalty': float(history['volume_penalty'][last_idx]),
            'operating_cost': float(history['operating_cost'][last_idx]),
            'timestamp': start_timestamp,
            'success': True
        }
        
    except Exception as e:
        if data_type == 'random_samples':
            source_name = "MIQP_piecewise_results_random_samples"
        else:
            source_name = f"MIQP_piecewise_results_relative_noise_{int(noise_level*100)}pct"
        
        print(f"Error training {source_name} {architecture}_{num_layers}layer_{max_iterations}iter for {date_str}: {e}")
        return {
            'data_type': data_type,
            'noise_level': noise_level if data_type == 'relative_noise' else None,
            'source_name': source_name,
            'architecture': architecture,
            'num_layers': num_layers,
            'max_iterations': max_iterations,
            'date_str': date_str,
            'error': str(e),
            'success': False
        }

def pretraining_with_grid_search():
    """Perform pretraining with parallel execution and noise level grid search."""
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting parallel pretraining with noise levels at {start_timestamp}...")
    
    # Define noise levels and generate corresponding file paths
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Grid search parameters - simplified as requested
    architectures = ['LSTM']
    num_layers_list = [3]
    max_iterations_list = list(range(1, 21))  # from 1 to 20

    # Prepare all training jobs
    all_jobs = []
    
    # Process noise level databases
    for noise_level in noise_levels:
        file_path = f"MIQP_piecewise_results_relative_noise_{int(noise_level*100)}pct.csv"
        source_name = f"MIQP_piecewise_results_relative_noise_{int(noise_level*100)}pct"
        
        print(f"Loading {source_name} database...")
        historical_data = load_data_for_pretraining(file_path, source_name)
        
        if not historical_data:
            print(f"Skipping {source_name} due to loading error")
            continue
            
        # Create benchmark file
        root_dir = Path(f"./trained_models/{source_name}")
        root_dir.mkdir(exist_ok=True, parents=True)
        
        # Add all combinations to job list
        for architecture, num_layers, max_iterations in itertools.product(
                architectures, num_layers_list, max_iterations_list):
            for date_str, date_data in historical_data.items():
                all_jobs.append((
                    'relative_noise', noise_level, file_path, architecture, num_layers, 
                    max_iterations, date_str, date_data, start_timestamp
                ))
    
    # Process random samples database
    random_samples_file = "MIQP_piecewise_results_random_samples.csv"
    random_samples_source = "MIQP_piecewise_results_random_samples"
    
    print(f"Loading {random_samples_source} database...")
    random_samples_data = load_data_for_pretraining(random_samples_file, random_samples_source)
    
    if random_samples_data:
        # Create benchmark file
        root_dir = Path(f"./trained_models/{random_samples_source}")
        root_dir.mkdir(exist_ok=True, parents=True)
        
        # Add all combinations to job list
        for architecture, num_layers, max_iterations in itertools.product(
                architectures, num_layers_list, max_iterations_list):
            for date_str, date_data in random_samples_data.items():
                all_jobs.append((
                    'random_samples', None, random_samples_file, architecture, num_layers, 
                    max_iterations, date_str, date_data, start_timestamp
                ))
    else:
        print(f"Skipping {random_samples_source} due to loading error")
    
    print(f"Total jobs to run: {len(all_jobs)}")
    print(f"Using {min(20, multiprocessing.cpu_count())} parallel processes")
    
    # Run in parallel (use 20 cores to leave 4 for system)
    results = Parallel(n_jobs=20, verbose=1)(
        delayed(train_single_model)(*job) for job in all_jobs
    )
    
    # Process results and write benchmark files
    benchmark_data = {}
    
    for result in results:
        if result['success']:
            source = result['source_name']
            if source not in benchmark_data:
                benchmark_data[source] = []
            benchmark_data[source].append(result)
        else:
            print(f"Failed job: {result}")
    
    # Write benchmark files for each database
    benchmark_files = {}
    for source_name, results_list in benchmark_data.items():
        root_dir = Path(f"./trained_models/{source_name}")
        benchmark_file = root_dir / "pretraining_benchmarks.csv"
        benchmark_files[source_name] = benchmark_file
        
        with open(benchmark_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Architecture', 'Num_Layers', 'Max_Iterations', 'Data_Type', 'Noise_Level', 'Date',
                'Training_Time_Seconds', 'Epochs_Trained', 'Best_Epoch',
                'Optimized_Profit', 'Simulated_Profit', 'SI_Penalty', 
                'Volume_Penalty', 'Operating_Cost', 'Timestamp'
            ])
            
            for result in results_list:
                noise_val = f"{result['noise_level']}" if result['noise_level'] is not None else 'N/A'
                writer.writerow([
                    result['architecture'], result['num_layers'], result['max_iterations'], 
                    result['data_type'], noise_val, result['date_str'], f"{result['training_time']:.2f}", 
                    result['epochs_trained'], result['best_epoch'],
                    f"{result['optimized_profit']:.2f}", f"{result['simulated_profit']:.2f}",
                    f"{result['SI_penalty']:.2f}", f"{result['volume_penalty']:.2f}",
                    f"{result['operating_cost']:.2f}", result['timestamp']
                ])
    
    # Generate summary analysis for each database
    print("\nGenerating benchmark summaries...")
    for source_name, benchmark_file in benchmark_files.items():
        print(f"Generating summary for {source_name}...")
        generate_benchmark_summary(benchmark_file, source_name)
    
    # Generate cross-database analysis
    print("\nGenerating cross-database analysis...")
    generate_cross_database_comparison()
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_duration = datetime.strptime(end_timestamp, "%Y%m%d_%H%M%S") - datetime.strptime(start_timestamp, "%Y%m%d_%H%M%S")
    
    print(f"\nParallel pretraining completed!")
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Total duration: {total_duration}")
    
if __name__ == "__main__":
    pretraining_with_grid_search()
    print("Pretraining and grid search completed.")