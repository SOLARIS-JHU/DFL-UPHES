# %% Import libraries and setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from joblib import Parallel, delayed
import multiprocessing
import os
import csv
import time
from datetime import datetime
from pathlib import Path
import json
import traceback
import itertools

device = torch.device("cpu")

def initialize_hydro_system():
    """Initialize the hydro system parameters and functions."""
    # load portfolio data
    sys.path.append('../Library')
    from V_H_relations import load_portfolio_data, gross_head, get_v_low
    load_portfolio_data()
    from V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

    # load preprocessed functions & data
    with open('../preprocess.pkl', 'rb') as f:
        v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

    head_init = torch.tensor(77.0, device=device)
    v_low_init = torch.tensor(h_to_v_low_fitted(head_init), device=device)
    
    return {
        'head_init': head_init,
        'v_low_init': v_low_init,
        'predict_q_poly': predict_q_poly,
        'h_to_v_low_fitted': h_to_v_low_fitted,
        'v_low_to_h_fitted': v_low_to_h_fitted,
        'neg_min_fit': neg_min_fit,
        'neg_max_fit': neg_max_fit,
        'pos_min_fit': pos_min_fit,
        'pos_max_fit': pos_max_fit,
        'neg_min': neg_min,
        'neg_max': neg_max,
        'pos_min': pos_min,
        'pos_max': pos_max,
        'head_max': head_max,
        'head_min': head_min,
        'max_vol_up': max_vol_up,
        'min_vol_low': min_vol_low,
        'target_vol_low': target_vol_low,
        'target_head': target_head,
        'ramp_up': ramp_up,
        'ramp_down': ramp_down
    }

# Initialize hydro system globally
HYDRO_SYSTEM = initialize_hydro_system()

def hourly_to_quarterly(tensor_data):
    return tensor_data.repeat_interleave(4)

# Standalone classes (copied from DFL_pretraining to avoid import issues)
class HydroParameters:
    def __init__(self):
        self.time_horizon = 24
        self.sampling_rate = 50
        self.operational_cost = 0.4
        
        self.δ_p = torch.tensor(0.5, dtype=torch.float32, device=device)
        self.δ_h = torch.tensor(1, dtype=torch.float32, device=device)
        self.δ_q = torch.tensor(0.5, dtype=torch.float32, device=device)
        self.rho = torch.tensor(1000, dtype=torch.float32, device=device)
        self.g = torch.tensor(9.81, dtype=torch.float32, device=device)
        self.mu = torch.tensor(0.9, dtype=torch.float32, device=device)

        self.head_min = torch.tensor(HYDRO_SYSTEM['head_min'], dtype=torch.float32, device=device)
        self.head_max = torch.tensor(HYDRO_SYSTEM['head_max'], dtype=torch.float32, device=device)
        self.max_vol_up = torch.tensor(HYDRO_SYSTEM['max_vol_up'], dtype=torch.float32, device=device)
        self.min_vol_low = torch.tensor(HYDRO_SYSTEM['min_vol_low'], dtype=torch.float32, device=device)
        self.ramp_up = torch.tensor(HYDRO_SYSTEM['ramp_up'], dtype=torch.float32, device=device)
        self.ramp_down = torch.tensor(HYDRO_SYSTEM['ramp_down'], dtype=torch.float32, device=device)

        self.target_head = torch.tensor(HYDRO_SYSTEM['target_head'], dtype=torch.float32, device=device)
        self.target_vol_low = torch.tensor(HYDRO_SYSTEM['target_vol_low'], dtype=torch.float32, device=device)
        self.head_init = HYDRO_SYSTEM['head_init'].clone().detach().to(device=device, dtype=torch.float32)
        self.v_low_init = HYDRO_SYSTEM['v_low_init'].clone().detach().to(device=device, dtype=torch.float32)

        self.neg_min_fit = torch.tensor(HYDRO_SYSTEM['neg_min_fit'], dtype=torch.float32, device=device)
        self.neg_max_fit = torch.tensor(HYDRO_SYSTEM['neg_max_fit'], dtype=torch.float32, device=device)
        self.pos_min_fit = torch.tensor(HYDRO_SYSTEM['pos_min_fit'], dtype=torch.float32, device=device)
        self.pos_max_fit = torch.tensor(HYDRO_SYSTEM['pos_max_fit'], dtype=torch.float32, device=device)

        self.neg_min = HYDRO_SYSTEM['neg_min']
        self.neg_max = HYDRO_SYSTEM['neg_max']
        self.pos_min = HYDRO_SYSTEM['pos_min']
        self.pos_max = HYDRO_SYSTEM['pos_max']

        self.predict_q_poly = HYDRO_SYSTEM['predict_q_poly']
        self.h_to_v_low_fitted = HYDRO_SYSTEM['h_to_v_low_fitted']
        self.v_low_to_h_fitted = HYDRO_SYSTEM['v_low_to_h_fitted']

class TaylorRegressionLayer:
    def __init__(self, params: HydroParameters):
        self.params = params

    def calculate_gradients(self, func, x, create_graph=False, retain_graph=False):
        try:
            y = func(x)
            grad = torch.autograd.grad(
                outputs=y, inputs=x, create_graph=create_graph, 
                retain_graph=retain_graph, grad_outputs=torch.ones_like(y)
            )[0]
            return grad
        except Exception as e:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

    def run_regression(self, power, head, flow=None):
        TH = self.params.time_horizon
        device = power.device
        c_list, d_list, e_list = [], [], []
        a_list, b_list = [], []

        for t in range(TH):
            try:
                p0 = power[t].detach().clone().requires_grad_(True)
                h0 = head[t].detach().clone().requires_grad_(True)
                
                if abs(p0.item()) < 0.01:
                    c_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                    d_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                    e_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                else:
                    def q_given_p(p):
                        return self.params.predict_q_poly(p.unsqueeze(0), h0.unsqueeze(0)).squeeze(0)
                    
                    def q_given_h(h):
                        return self.params.predict_q_poly(p0.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
                    
                    dq_dp = self.calculate_gradients(q_given_p, p0, retain_graph=True)
                    dq_dh = self.calculate_gradients(q_given_h, h0, retain_graph=True)
                    
                    q0 = self.params.predict_q_poly(p0.unsqueeze(0), h0.unsqueeze(0)).squeeze(0).detach()
                    
                    c = dq_dp.detach()
                    d = dq_dh.detach()
                    e = q0 - c * p0.detach() - d * h0.detach()
                    
                    c_list.append(c)
                    d_list.append(d)
                    e_list.append(e)
                
                def v_low_given_h(h):
                    return self.params.h_to_v_low_fitted(h)
                
                dv_low_dh = self.calculate_gradients(v_low_given_h, h0, retain_graph=False)
                v_low0 = self.params.h_to_v_low_fitted(h0).detach()
                
                a = dv_low_dh.detach()
                b = v_low0 - a * h0.detach()
                
                a_list.append(a)
                b_list.append(b)
                
            except Exception as e:
                c_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                d_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                e_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                a_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                b_list.append(torch.tensor(0.0, device=device, requires_grad=True))

        try:
            c_tensor = torch.stack(c_list)
            d_tensor = torch.stack(d_list)
            e_tensor = torch.stack(e_list)
            a_tensor = torch.stack(a_list)
            b_tensor = torch.stack(b_list)
        except RuntimeError:
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
        self.params = params.to_cpu() if hasattr(params, 'to_cpu') else params
        self.layer = None
        self.power_init = None
        self.head_init = None
        self.flow_init = None
    
    def initialize_layer(self, power, head, flow):
        if (self.layer is None 
            or not torch.allclose(self.power_init, power) 
            or not torch.allclose(self.head_init, head)):
            
            self.power_init = power.detach().cpu()
            self.head_init = head.detach().cpu()
            self.flow_init = flow.detach().cpu()
            self.layer = self._build_cvxpy()

    def _build_cvxpy(self):
        TH = self.params.time_horizon
        p_var = cp.Variable(TH)
        q_var = cp.Variable(TH)
        h_var = cp.Variable(TH)
        v_low_var = cp.Variable(TH)

        DA_price_param = cp.Parameter(TH)
        c_param = cp.Parameter(TH)
        d_param = cp.Parameter(TH)
        e_param = cp.Parameter(TH)
        a_param = cp.Parameter(TH)
        b_param = cp.Parameter(TH)
        w_p_param = cp.Parameter(TH, nonneg=True)
        w_h_param = cp.Parameter(TH, nonneg=True)
        w_q_param = cp.Parameter(TH, nonneg=True)

        p_var.value = self.power_init.tolist()
        h_var.value = self.head_init.tolist()

        revenue = DA_price_param @ p_var
        cost = self.params.operational_cost * cp.sum_squares(p_var)
        power_dev_pen = cp.sum(w_p_param @ cp.square(p_var - self.power_init))
        head_dev_pen = cp.sum(w_h_param @ cp.square(h_var - self.head_init))
        flow_dev_pen = cp.sum(w_q_param @ cp.square(q_var - self.flow_init))

        objective = cp.Maximize(revenue - cost - power_dev_pen - head_dev_pen - flow_dev_pen)

        constraints = []
        for t in range(TH):
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

            constraints += [
                h_var[t] >= self.params.head_min,
                h_var[t] <= self.params.head_max,
                v_low_var[t] == a_param[t] * h_var[t] + b_param[t],
            ]

            if t == 0:
                constraints += [v_low_var[0] == self.params.v_low_init + q_var[0] * 3600]
            else:
                constraints += [v_low_var[t] == v_low_var[t-1] + q_var[t] * 3600]

        constraints += [v_low_var[TH-1] <= self.params.target_vol_low]

        problem = cp.Problem(objective, constraints)
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
            (p_opt, q_opt, h_opt, v_opt) = self.layer(
                DA_prices_cpu, c_cpu, d_cpu, e_cpu, a_cpu, b_cpu, 
                w_p_cpu, w_h_cpu, w_q_cpu,
                solver_args={
                    "solve_method": "ECOS",
                    "max_iters": 200000,
                    "reltol": 1e-5,
                    "abstol": 1e-5,
                    "feastol": 1e-5,
                    "verbose": False
                }
            )
        except Exception as er:
            print(f"Solver error: {er}")
            raise

        threshold = 0.1
        p_opt_thresholded = torch.where(torch.abs(p_opt) < threshold, torch.zeros_like(p_opt), p_opt)
        q_opt_thresholded = torch.where(torch.abs(q_opt) < threshold, torch.zeros_like(q_opt), q_opt)

        revenue = torch.sum(DA_prices_cpu * p_opt_thresholded)
        operating_cost = self.params.operational_cost * torch.sum(p_opt_thresholded**2)
        
        power_dev_pen = torch.sum(w_p_cpu * torch.square(p_opt_thresholded - self.power_init))
        head_dev_pen = torch.sum(w_h_cpu * torch.square(h_opt - self.head_init))
        flow_dev_pen = torch.sum(w_q_cpu * torch.square(q_opt_thresholded - self.flow_init))
        
        optimized_objective = revenue - operating_cost - power_dev_pen - head_dev_pen - flow_dev_pen
        optimized_profit = revenue - operating_cost

        return p_opt_thresholded, q_opt_thresholded, h_opt, v_opt, optimized_profit, optimized_objective

class SimulationLayer:
    def __init__(self, params):
        self.params = params

    def simulate_operation(self, p, q, h):
        TH = self.params.time_horizon
        
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        v_current = self.params.v_low_init
        v_list.append(v_current)

        for i in range(TH):
            h_current = h[i]
            p_current = p[i]
            
            q_candidate = torch.zeros_like(p_current)
            p_clamped = p_current

            if p_current > 0.5:  # Turbine mode
                p_min_turb = self.params.pos_min(h_current)
                p_max_turb = self.params.pos_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            
            elif p_current < -0.5:  # Pump mode
                p_min_pump = self.params.neg_min(h_current)
                p_max_pump = self.params.neg_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_pump, max=p_max_pump)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            
            v_next = v_current + q_candidate * 3600
            
            out_of_bounds = (v_next > self.params.max_vol_up) | (v_next < self.params.min_vol_low)
            
            if out_of_bounds:
                p_final = torch.zeros_like(p_current)
                q_final = torch.zeros_like(q_candidate)
                v_next = v_current
                h_next = h_current
            else:
                p_final = p_clamped if p_current != 0 else torch.zeros_like(p_current)
                q_final = q_candidate
                h_next = self.params.v_low_to_h_fitted(v_next)
            
            p_list.append(p_final)
            q_list.append(q_final)
            h_list.append(h_next)
            v_list.append(v_next.item())
            
            v_current = v_next
        
        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list[:-1])
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32)
        
        return p_sim, q_sim, h_sim, v_low_sim

    def calc_profit(self, p_sim, p_opt, v_low_sim, DA_price):
        e_sim = p_sim
        revenue = torch.sum(DA_price * e_sim)

        surplus_penalty_multiplier = -0.5
        shortage_penalty_multiplier = -2.0

        SI_price = torch.where(
            e_sim < p_opt,
            shortage_penalty_multiplier * DA_price,
            surplus_penalty_multiplier * DA_price
        )
        
        imbalance = e_sim - p_opt
        penalty = imbalance * SI_price
        SI_penalty = penalty.sum()

        volume_deficit = max(0, v_low_sim[-1] - self.params.target_vol_low)
        energy_loss = self.params.rho * volume_deficit * self.params.g * self.params.target_head * self.params.mu / 3.6e9
        volume_penalty = energy_loss * torch.median(DA_price)

        operating_cost = self.params.operational_cost * torch.sum(p_sim**2)

        total_profit = revenue - operating_cost - SI_penalty - volume_penalty
        
        return total_profit, SI_penalty, volume_penalty, operating_cost

class BoundedLogWeightPredictor(nn.Module):
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
        
        self.init_w_p = init_w_p
        self.init_w_q = init_w_q
        self.init_w_h = init_w_h
        
        self.w_p_min = w_p_min
        self.w_p_max = w_p_max
        self.w_q_min = w_q_min
        self.w_q_max = w_q_max
        self.w_h_min = w_h_min
        self.w_h_max = w_h_max
        
        self.log_w_p_min = torch.log(torch.tensor(w_p_min))
        self.log_w_p_max = torch.log(torch.tensor(w_p_max))
        self.log_w_q_min = torch.log(torch.tensor(w_q_min))
        self.log_w_q_max = torch.log(torch.tensor(w_q_max))
        self.log_w_h_min = torch.log(torch.tensor(w_h_min))
        self.log_w_h_max = torch.log(torch.tensor(w_h_max))
        
        if self.archetype == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
                batch_first=True 
            )
        elif self.archetype == 'RNN':
            self.rnn = nn.RNN(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.archetype == 'FC':
            self.fc_layers = nn.Sequential(
                nn.Linear(input_size * time_horizon, hidden_size * 2),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unsupported archetype: {archetype}")
        
        self.output = nn.Linear(hidden_size, 3 * time_horizon)
        
        self._init_weights()
        self._set_initial_weights()
                
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'output' not in name:
                nn.init.xavier_normal_(param, gain=1.5)
            elif 'bias' in name and 'output' not in name:
                nn.init.constant_(param, 0.1)
    
    def _set_initial_weights(self):
        log_w_p = torch.log(torch.tensor(self.init_w_p))
        log_w_q = torch.log(torch.tensor(self.init_w_q))
        log_w_h = torch.log(torch.tensor(self.init_w_h))
        
        bias = self.output.bias.data
        bias[0:self.time_horizon] = log_w_p
        bias[self.time_horizon:2*self.time_horizon] = log_w_q
        bias[2*self.time_horizon:3*self.time_horizon] = log_w_h
                
    def _clamp_log_weights(self, log_w_p, log_w_q, log_w_h):
        device = log_w_p.device
        log_w_p_min = self.log_w_p_min.to(device)
        log_w_p_max = self.log_w_p_max.to(device)
        log_w_q_min = self.log_w_q_min.to(device)
        log_w_q_max = self.log_w_q_max.to(device)
        log_w_h_min = self.log_w_h_min.to(device)
        log_w_h_max = self.log_w_h_max.to(device)
        
        log_w_p = torch.clamp(log_w_p, min=log_w_p_min, max=log_w_p_max)
        log_w_q = torch.clamp(log_w_q, min=log_w_q_min, max=log_w_q_max)
        log_w_h = torch.clamp(log_w_h, min=log_w_h_min, max=log_w_h_max)
        
        return log_w_p, log_w_q, log_w_h
    
    def forward(self, x):
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
        
        log_weights = self.output(last_output)
        
        log_weights = log_weights.view(-1, 3, self.time_horizon)
        log_w_p, log_w_q, log_w_h = log_weights[:, 0, :], log_weights[:, 1, :], log_weights[:, 2, :]
        
        log_w_p, log_w_q, log_w_h = self._clamp_log_weights(log_w_p, log_w_q, log_w_h)
        
        if x.size(0) == 1:
            log_w_p, log_w_q, log_w_h = log_w_p.squeeze(0), log_w_q.squeeze(0), log_w_h.squeeze(0)
        
        return log_w_p, log_w_q, log_w_h

def load_data_for_pretraining(file_path, source_name):
    """Load data for pretraining with proper format handling."""
    try:
        df = pd.read_csv(file_path, sep=',', header=0)
        df.columns = df.columns.str.strip()
        
        required_columns = ['date', 'hour', 'power', 'head', 'flow', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df['Date'] = pd.to_datetime(df['date'])
        df['Time'] = df['hour']
        df = df.rename(columns={
            'power': 'Power', 'head': 'Head', 'flow': 'Flow', 'price': 'Price'
        })
        
        conditions = [
            (abs(df['Power']) < 0.01),
            (df['Power'] > 0),
            (df['Power'] < 0)
        ]
        choices = ['Idle', 'Turbine', 'Pump']
        df['Mode'] = np.select(conditions, choices, default='Unknown')
        
        data_by_date = {}
        for date, group in df.groupby('Date'):
            group = group.sort_values('Time')
            date_str = date.strftime('%Y-%m-%d')
            
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

def train_recursive_linearization_with_penalties(
    weight_network, params, optimizer_layer, regression_layer, 
    historical_data, num_epochs=100, learning_rate=0.001, 
    patience=10, max_iterations=3, penalty_growth_rate=1.5,
    si_penalty_factor=0.0, volume_penalty_factor=0.0):
    """Modified training function that includes penalty factors in the loss."""
    device = next(weight_network.parameters()).device
    weight_network.train()
    
    optimizer = torch.optim.Adam(weight_network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Simple pipeline implementation
    train_date = list(historical_data.keys())[0]
    date_data = historical_data[train_date]
    power_orig = date_data['power']
    head_orig = date_data['head']
    flow_orig = HYDRO_SYSTEM['predict_q_poly'](power_orig, head_orig)
    
    history = {
        'epoch': [], 'loss': [], 'profit': [], 'simulated_profit': [],
        'SI_penalty': [], 'volume_penalty': [], 'operating_cost': []
    }
    
    best_profit = float('-inf')
    best_weights = None
    patience_counter = 0
    
    print(f"Training with SI penalty factor: {si_penalty_factor}, Volume penalty factor: {volume_penalty_factor}")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Get input features for the weight predictor
        x = torch.stack([date_data['price'], power_orig, flow_orig, head_orig], dim=1)
        
        # Run weight prediction
        log_w_p, log_w_q, log_w_h = weight_network(x)
        w_p_initial = torch.exp(log_w_p)
        w_q_initial = torch.exp(log_w_q)
        w_h_initial = torch.exp(log_w_h)
        
        # Initialize for iterations
        p_current = power_orig.clone().detach()
        h_current = head_orig.clone().detach()
        flow_current = flow_orig.clone().detach()
        
        # Recursive linearization loop
        for iteration in range(max_iterations):
            growth_factor = penalty_growth_rate ** iteration
            w_p = w_p_initial * growth_factor
            w_q = w_q_initial * growth_factor
            w_h = w_h_initial * growth_factor
            
            c, d, e, a, b = regression_layer.run_regression(p_current, h_current, flow_current)
            
            optimizer_layer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())
            
            p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective = optimizer_layer.forward(
                date_data['price'].cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                w_p.cpu(), w_h.cpu(), w_q.cpu()
            )
            
            if iteration < max_iterations - 1:
                p_current = p_opt.clone().detach().to(device=power_orig.device) 
                h_current = h_opt.clone().detach().to(device=head_orig.device)
                flow_current = q_opt.clone().detach().to(device=flow_orig.device)
        
        # Run simulation
        simulator = SimulationLayer(params)
        p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(
            p_opt.to(device), q_opt.to(device), h_opt.to(device)
        )
        
        simulated_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
            p_sim, p_opt.to(device), v_low_sim, date_data['price'].to(device)
        )
        
        # Modified loss function with penalty factors
        loss = -simulated_profit + si_penalty_factor * SI_penalty + volume_penalty_factor * volume_penalty
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(weight_network.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler.step(simulated_profit)
        
        # Record history
        history['epoch'].append(epoch)
        history['loss'].append(loss.item())
        history['profit'].append(optimized_profit.item())
        history['simulated_profit'].append(simulated_profit.item())
        history['SI_penalty'].append(SI_penalty.item())
        history['volume_penalty'].append(volume_penalty.item())
        history['operating_cost'].append(operating_cost.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Profit={simulated_profit.item():.4f}")
        
        # Early stopping
        if simulated_profit.item() > best_profit:
            best_profit = simulated_profit.item()
            best_weights = weight_network.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    if best_weights is not None:
        weight_network.load_state_dict(best_weights)
    
    print(f"Training complete. Best simulated profit: {best_profit:.4f}")
    return weight_network, history

def train_single_penalty_model(source_name, file_path, date_str, date_data, 
                              si_penalty_factor, volume_penalty_factor, start_timestamp):
    """Train a single model with specific penalty factors."""
    try:
        params = HydroParameters()
        regression_layer = TaylorRegressionLayer(params)
        optimizer_layer = OptiLayer(params)
        
        # Create directory structure
        penalty_name = f"SI{si_penalty_factor:.1f}_Vol{volume_penalty_factor:.1f}"
        root_dir = Path(f"./penalty_ablation/{source_name}")
        penalty_dir = root_dir / penalty_name
        date_dir = penalty_dir / date_str
        date_dir.mkdir(exist_ok=True, parents=True)
        
        # Fixed configuration: 3-layer RNN
        weight_network = BoundedLogWeightPredictor(
            input_size=4, hidden_size=128, num_layers=3, dropout=0.2,
            time_horizon=params.time_horizon, archetype='RNN',
            init_w_p=0.6, init_w_q=0.02, init_w_h=0.1,
            w_p_min=0.1, w_p_max=3.0, w_q_min=0.001, w_q_max=0.2,
            w_h_min=0.01, w_h_max=5.0
        ).to(device)
        
        start_time = time.time()
        trained_network, history = train_recursive_linearization_with_penalties(
            weight_network=weight_network, params=params,
            optimizer_layer=optimizer_layer, regression_layer=regression_layer,
            historical_data={date_str: date_data}, num_epochs=500, learning_rate=1e-3,
            patience=20, max_iterations=10, penalty_growth_rate=1.5,
            si_penalty_factor=si_penalty_factor, volume_penalty_factor=volume_penalty_factor
        )
        training_time = time.time() - start_time
        
        # Save model and history
        torch.save(trained_network.state_dict(), date_dir / "model.pt")
        torch.save(trained_network.state_dict(), date_dir / "best_model.pt")
        
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
        
        # Return results
        last_idx = len(history['epoch']) - 1
        return {
            'si_penalty_factor': si_penalty_factor,
            'volume_penalty_factor': volume_penalty_factor,
            'date_str': date_str,
            'training_time': training_time,
            'epochs_trained': last_idx + 1,
            'optimized_profit': float(history['profit'][last_idx]),
            'simulated_profit': float(history['simulated_profit'][last_idx]),
            'SI_penalty': float(history['SI_penalty'][last_idx]),
            'volume_penalty': float(history['volume_penalty'][last_idx]),
            'operating_cost': float(history['operating_cost'][last_idx]),
            'timestamp': start_timestamp,
            'success': True
        }
        
    except Exception as e:
        print(f"Error training penalty model SI{si_penalty_factor:.1f}_Vol{volume_penalty_factor:.1f} for {date_str}: {e}")
        return {
            'si_penalty_factor': si_penalty_factor,
            'volume_penalty_factor': volume_penalty_factor,
            'date_str': date_str,
            'error': str(e),
            'success': False
        }

# %% Penalty Ablation Pretraining
def penalty_ablation_pretraining():
    """Perform penalty ablation study with parallel training."""
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting penalty ablation pretraining at {start_timestamp}...")
    
    # Load euclidean_piecewise data
    source_name = 'euclidean_piecewise'
    file_path = '../MIQP/historical_operation_solver/euclidean_piecewise/detailed_results.csv'
    
    historical_data = load_data_for_pretraining(file_path, source_name)
    if not historical_data:
        print(f"Error: Could not load data for {source_name}")
        return
    
    # Define penalty factor combinations
    penalty_factors = []
    
    # Baseline: no additional penalties
    penalty_factors.append((0.0, 0.0))
    
    # SI penalty study: A ∈ {0.1, 0.2, ..., 1.0}, B = 0
    for a in np.arange(0.1, 1.1, 0.1):
        penalty_factors.append((round(a, 1), 0.0))
    
    # Volume penalty study: A = 0, B ∈ {0.1, 0.2, ..., 1.0}
    for b in np.arange(0.1, 1.1, 0.1):
        penalty_factors.append((0.0, round(b, 1)))
    
    print(f"Total penalty combinations: {len(penalty_factors)}")
    print(f"Penalty factors: {penalty_factors}")
    
    # Prepare all training jobs
    all_jobs = []
    for si_factor, vol_factor in penalty_factors:
        for date_str, date_data in historical_data.items():
            all_jobs.append((
                source_name, file_path, date_str, date_data, 
                si_factor, vol_factor, start_timestamp
            ))
    
    print(f"Total jobs to run: {len(all_jobs)}")
    
    # Run in parallel (reduce number of jobs to avoid memory issues)
    results = Parallel(n_jobs=min(10, multiprocessing.cpu_count()), verbose=1)(
        delayed(train_single_penalty_model)(*job) for job in all_jobs
    )
    
    # Process results and create benchmark file
    root_dir = Path(f"./penalty_ablation/{source_name}")
    benchmark_file = root_dir / "penalty_ablation_benchmarks.csv"
    
    with open(benchmark_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'SI_Penalty_Factor', 'Volume_Penalty_Factor', 'Date',
            'Training_Time_Seconds', 'Epochs_Trained',
            'Optimized_Profit', 'Simulated_Profit', 'SI_Penalty', 
            'Volume_Penalty', 'Operating_Cost', 'Timestamp'
        ])
        
        for result in results:
            if result['success']:
                writer.writerow([
                    result['si_penalty_factor'], result['volume_penalty_factor'], result['date_str'],
                    f"{result['training_time']:.2f}", result['epochs_trained'],
                    f"{result['optimized_profit']:.2f}", f"{result['simulated_profit']:.2f}",
                    f"{result['SI_penalty']:.2f}", f"{result['volume_penalty']:.2f}",
                    f"{result['operating_cost']:.2f}", result['timestamp']
                ])
    
    print(f"Penalty ablation pretraining completed!")
    print(f"Benchmark file saved: {benchmark_file}")

# %% Penalty Ablation Validation and Results
def load_new_price_data(file_path="../Data/price_data_2024.csv"):
    """Load new price data for scheduling validation."""
    try:
        df = pd.read_csv(file_path)
        
        if 'date' not in df.columns or 'cluster_index' not in df.columns or 'prices_hourly' not in df.columns:
            if len(df.columns) >= 3:
                df.columns = ['date', 'cluster_index', 'prices_hourly']
            else:
                raise ValueError(f"Expected columns 'date', 'cluster_index', 'prices_hourly' but got {df.columns}")
        
        price_data = {}
        
        for _, row in df.iterrows():
            date_str = row['date']
            prices_str = row['prices_hourly']
            
            try:
                prices = [float(p) for p in prices_str.split(',')]
            except:
                try:
                    prices = [float(p) for p in prices_str.split(';')]
                except:
                    prices_str = prices_str.strip('[]')
                    prices = [float(p) for p in prices_str.split()]
            
            if len(prices) != 24:
                print(f"Warning: Date {date_str} has {len(prices)} price values instead of 24")
                if len(prices) < 24:
                    prices.extend([prices[-1]] * (24 - len(prices)))
                else:
                    prices = prices[:24]
            
            price_tensor = torch.tensor(prices, dtype=torch.float32, device=device)
            price_data[date_str] = price_tensor
        
        print(f"Successfully loaded price data for {len(price_data)} days.")
        return price_data
    
    except Exception as e:
        print(f"Error loading new price data: {e}")
        print("Using sample price data instead...")
        return {
            '2024-01-15': torch.tensor([45.2, 42.1, 40.3, 38.9, 41.2, 48.5, 55.7, 62.3, 
                                       58.4, 52.1, 49.8, 47.6, 46.2, 48.9, 52.3, 57.8, 
                                       61.2, 65.4, 59.7, 54.3, 51.6, 49.2, 47.8, 44.6], device=device),
            '2024-02-20': torch.tensor([42.8, 39.7, 37.2, 35.6, 38.9, 45.3, 52.1, 58.7, 
                                       55.2, 49.8, 47.1, 44.9, 43.5, 46.2, 49.6, 54.3, 
                                       57.8, 61.9, 56.4, 51.2, 48.7, 46.3, 45.1, 41.9], device=device),
            '2024-03-10': torch.tensor([38.5, 36.2, 34.8, 33.1, 35.7, 42.8, 49.3, 55.1, 
                                       51.8, 46.9, 44.2, 42.0, 40.8, 43.5, 46.9, 51.2, 
                                       54.6, 58.2, 53.1, 48.4, 45.9, 43.7, 42.3, 39.1], device=device)
        }

def find_closest_date(new_price, historical_data):
    """Find the date in historical data with the most similar price signal."""
    closest_date = None
    min_distance = float('inf')
    
    for date_str, date_data in historical_data.items():
        historical_price = date_data['price'][:24]
        distance = torch.norm(new_price - historical_price).item()
        
        if distance < min_distance:
            min_distance = distance
            closest_date = date_str
    
    return closest_date, min_distance

def comprehensive_penalty_ablation_validation():
    """
    Perform comprehensive validation across all penalty configurations.
    Similar to DFL_validation.py but for penalty ablation study.
    """
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting comprehensive penalty ablation validation at {start_timestamp}...")
    
    source_name = 'euclidean_piecewise'
    
    # Load new price data (same as DFL_validation.py)
    new_price_data = load_new_price_data()
    if not new_price_data:
        print("Error: Could not load new price data")
        return
    
    # Load historical data for finding closest matches
    file_path = '../MIQP/historical_operation_solver/euclidean_piecewise/detailed_results.csv'
    historical_data = load_data_for_pretraining(file_path, source_name)
    
    if not historical_data:
        print("Error: Could not load historical data")
        return
    
    # Initialize parameters
    params = HydroParameters()
    
    # Define penalty factors (same as in pretraining)
    penalty_factors = [(0.0, 0.0)]  # Baseline
    for a in np.arange(0.1, 1.1, 0.1):
        penalty_factors.append((round(a, 1), 0.0))
    for b in np.arange(0.1, 1.1, 0.1):
        penalty_factors.append((0.0, round(b, 1)))
    
    # Create master validation directory
    validation_dir = Path(f"./penalty_ablation/{source_name}/comprehensive_validation")
    validation_dir.mkdir(exist_ok=True, parents=True)
    
    # Create master benchmark file
    master_benchmark_file = validation_dir / "penalty_ablation_validation_benchmarks.csv"
    with open(master_benchmark_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'SI_Penalty_Factor', 'Volume_Penalty_Factor', 'Method_Name',
            'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
            'Expected_Profit', 'Ex_post_Profit', 'SI_Penalty',
            'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
            'Timestamp'
        ])
    
    # Track best configurations per date
    best_configs = {}
    
    # Total configurations to test
    total_configs = len(penalty_factors) * len(new_price_data)
    config_counter = 0
    
    # Initialize layers (reuse for all configurations)
    regression_layer = TaylorRegressionLayer(params)
    optimizer_layer = OptiLayer(params)
    
    # Iterate through all penalty configurations
    for si_factor, vol_factor in penalty_factors:
        penalty_name = f"SI{si_factor:.1f}_Vol{vol_factor:.1f}"
        
        # Determine method name for reporting
        if si_factor == 0.0 and vol_factor == 0.0:
            method_name = "Baseline"
        elif vol_factor == 0.0:
            method_name = f"SI-{si_factor:.1f}"
        else:
            method_name = f"Vol-{vol_factor:.1f}"
        
        print(f"\n{'='*80}")
        print(f"Validating penalty configuration: {penalty_name} ({method_name})")
        print(f"{'='*80}")
        
        # Create output directory for this configuration
        config_dir = validation_dir / penalty_name
        config_dir.mkdir(exist_ok=True)
        
        # Create benchmark CSV file for this configuration
        config_benchmark_file = config_dir / "validation_benchmarks.csv"
        with open(config_benchmark_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
                'Expected_Profit', 'Ex_post_Profit', 'SI_Penalty',
                'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
                'Timestamp'
            ])
        
        # Process each new date
        for date_idx, (new_date, new_price) in enumerate(new_price_data.items()):
            config_counter += 1
            print(f"\n[{config_counter}/{total_configs}] Processing {new_date} with {penalty_name}")
            
            # Create directory for this date
            safe_date = new_date.replace('/', '-')
            date_dir = config_dir / safe_date
            date_dir.mkdir(exist_ok=True)
            
            try:
                # Start timing
                start_time = time.time()
                
                # Find the closest historical date
                closest_date, distance = find_closest_date(new_price, historical_data)
                print(f"Closest historical date: {closest_date} (distance: {distance:.2f})")
                
                # Look for the pretrained model
                model_path = Path(f"./penalty_ablation/{source_name}/{penalty_name}/{closest_date}/best_model.pt")
                
                if not model_path.exists():
                    model_path = Path(f"./penalty_ablation/{source_name}/{penalty_name}/{closest_date}/model.pt")
                    
                    if not model_path.exists():
                        print(f"Warning: No model found at {model_path}. Skipping this date.")
                        continue
                
                # Initialize weight network
                weight_network = BoundedLogWeightPredictor(
                    input_size=4, hidden_size=128, num_layers=3, dropout=0.2,
                    time_horizon=params.time_horizon, archetype='RNN',
                    init_w_p=0.6, init_w_q=0.02, init_w_h=0.1,
                    w_p_min=0.1, w_p_max=3.0, w_q_min=0.001, w_q_max=0.2,
                    w_h_min=0.01, w_h_max=5.0
                ).to(device)
                
                # Load the pretrained weights
                weight_network.load_state_dict(torch.load(model_path, map_location=device))
                weight_network.eval()
                
                # Get the power, head, and flow from the closest date
                closest_data = historical_data[closest_date]
                power_init = closest_data['power'][:24].clone()
                head_init = closest_data['head'][:24].clone()
                flow_init = HYDRO_SYSTEM['predict_q_poly'](power_init, head_init)
                
                # Prepare input for weight prediction
                x = torch.stack([new_price, power_init, flow_init, head_init], dim=1)
                
                # Predict weights
                with torch.no_grad():
                    log_w_p, log_w_q, log_w_h = weight_network(x)
                    w_p = torch.exp(log_w_p)
                    w_q = torch.exp(log_w_q)
                    w_h = torch.exp(log_w_h)
                
                # Run recursive linearization (10 iterations, fixed)
                p_current = power_init.clone().detach()
                h_current = head_init.clone().detach()
                flow_current = flow_init.clone().detach()
                
                # Track iteration results
                iter_results = []
                
                for iteration in range(10):
                    # Apply growth to weights
                    growth_factor = 1.5 ** iteration
                    w_p_iter = w_p * growth_factor
                    w_q_iter = w_q * growth_factor
                    w_h_iter = w_h * growth_factor
                    
                    # Compute linearization coefficients
                    c, d, e, a, b = regression_layer.run_regression(p_current, h_current, flow_current)
                    
                    # Initialize OptiLayer
                    optimizer_layer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())
                    
                    # Run optimization
                    p_opt, q_opt, h_opt, v_opt, expected_profit, optimized_objective = optimizer_layer.forward(
                        new_price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                        p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                        w_p_iter.cpu(), w_h_iter.cpu(), w_q_iter.cpu()
                    )
                    
                    # Store iteration results
                    iter_results.append({
                        'iteration': iteration,
                        'p_opt': p_opt.detach().cpu().numpy(),
                        'expected_profit': expected_profit.item()
                    })
                    
                    # Update for next iteration
                    if iteration < 9:
                        p_current = p_opt.clone().detach().to(device=power_init.device) 
                        h_current = h_opt.clone().detach().to(device=head_init.device)
                        flow_current = q_opt.clone().detach().to(device=flow_init.device)
                
                # Run simulation
                simulator = SimulationLayer(params)
                p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(
                    p_opt.to(device), q_opt.to(device), h_opt.to(device)
                )
                
                # Calculate ex-post profit
                ex_post_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
                    p_sim, p_opt.to(device), v_low_sim, new_price.to(device)
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Save results for this date and configuration
                results = {
                    'penalty_name': penalty_name,
                    'method_name': method_name,
                    'si_factor': si_factor,
                    'vol_factor': vol_factor,
                    'new_date': new_date,
                    'closest_date': closest_date,
                    'distance': distance,
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
                    'expected_profit': expected_profit.item(),
                    'ex_post_profit': ex_post_profit.item(),
                    'SI_penalty': SI_penalty.item(),
                    'volume_penalty': volume_penalty.item(),
                    'operating_cost': operating_cost.item(),
                    'processing_time': processing_time,
                    'iter_results': iter_results
                }
                
                # Save as numpy file
                np.save(date_dir / "results.npy", results)
                
                # Generate plots (similar to DFL_validation.py)
                plt.figure(figsize=(18, 12))
                
                # Price comparison
                plt.subplot(3, 2, 1)
                plt.plot(range(24), results['new_price'], 'b-', label='New Price')
                plt.plot(range(24), results['closest_price'], 'r--', label=f'Closest ({closest_date})')
                plt.title('Price Comparison')
                plt.xlabel('Hour')
                plt.ylabel('Price (EUR/MWh)')
                plt.legend()
                plt.grid(True)
                
                # Power comparison
                plt.subplot(3, 2, 2)
                plt.plot(range(24), results['p_opt'], 'g-', label='Optimized Power')
                plt.plot(range(24), results['p_sim'], 'b-', label='Simulated Power')
                plt.title('Power Schedule')
                plt.xlabel('Hour')
                plt.ylabel('Power (MW)')
                plt.legend()
                plt.grid(True)
                
                # Flow
                plt.subplot(3, 2, 3)
                plt.plot(range(24), results['q_opt'], 'b-')
                plt.title('Optimized Flow')
                plt.xlabel('Hour')
                plt.ylabel('Flow (m³/s)')
                plt.grid(True)
                
                # Head
                plt.subplot(3, 2, 4)
                plt.plot(range(24), results['h_opt'], 'g-')
                plt.title('Optimized Head')
                plt.xlabel('Hour')
                plt.ylabel('Head (m)')
                plt.grid(True)
                
                # Iteration evolution
                plt.subplot(3, 2, 5)
                expected_profits = [iter_result['expected_profit'] for iter_result in iter_results]
                plt.plot(range(len(expected_profits)), expected_profits, 'ro-')
                plt.title('Expected Profit Evolution')
                plt.xlabel('Iteration')
                plt.ylabel('Expected Profit')
                plt.grid(True)
                
                # Results summary
                plt.subplot(3, 2, 6)
                plt.axis('off')
                stats_text = (
                    f"Configuration: {method_name}\n"
                    f"Date: {new_date}\n"
                    f"Closest: {closest_date} (dist: {distance:.2f})\n\n"
                    f"Expected profit: {expected_profit.item():.2f}\n"
                    f"Ex-post profit: {ex_post_profit.item():.2f}\n"
                    f"SI penalty: {SI_penalty.item():.2f}\n"
                    f"Volume penalty: {volume_penalty.item():.2f}\n"
                    f"Operating cost: {operating_cost.item():.2f}\n\n"
                    f"Processing time: {processing_time:.2f} seconds"
                )
                plt.text(0.1, 0.5, stats_text, fontsize=10, va='center')
                
                plt.suptitle(f"Penalty Ablation Results: {method_name} for {new_date}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig(date_dir / "validation_results.png")
                plt.close()
                
                # Append to configuration benchmark CSV
                with open(config_benchmark_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        safe_date, closest_date, f"{distance:.2f}",
                        f"{expected_profit.item():.2f}", f"{ex_post_profit.item():.2f}",
                        f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                        f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                        start_timestamp
                    ])
                
                # Append to master benchmark CSV
                with open(master_benchmark_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        si_factor, vol_factor, method_name,
                        safe_date, closest_date, f"{distance:.2f}",
                        f"{expected_profit.item():.2f}", f"{ex_post_profit.item():.2f}",
                        f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                        f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                        start_timestamp
                    ])
                
                # Track best configuration for this date
                if safe_date not in best_configs:
                    best_configs[safe_date] = {'method': method_name, 'profit': ex_post_profit.item()}
                elif ex_post_profit.item() > best_configs[safe_date]['profit']:
                    best_configs[safe_date] = {'method': method_name, 'profit': ex_post_profit.item()}
                
                print(f"Validation for {new_date} completed:")
                print(f"  Method: {method_name}")
                print(f"  Processing time: {processing_time:.2f} seconds")
                print(f"  Expected profit: {expected_profit.item():.2f}")
                print(f"  Ex-post profit: {ex_post_profit.item():.2f}")
                print(f"  Results saved to: {date_dir}")
                
            except Exception as e:
                print(f"Error processing {new_date} with {penalty_name}: {e}")
                print(traceback.format_exc())
                
                # Log error
                with open(config_dir / "error_log.txt", 'a') as f:
                    f.write(f"\n[{datetime.now()}] Error processing {new_date}:\n")
                    f.write(traceback.format_exc())
                    f.write("\n" + "-"*50 + "\n")
    
    # Save best configurations
    with open(validation_dir / "best_configurations.json", 'w') as f:
        json.dump(best_configs, f, indent=4)
    
    # Generate comprehensive summary and LaTeX table
    generate_penalty_ablation_summary(master_benchmark_file, validation_dir)
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_duration = datetime.strptime(end_timestamp, "%Y%m%d_%H%M%S") - datetime.strptime(start_timestamp, "%Y%m%d_%H%M%S")
    
    print(f"\nComprehensive penalty ablation validation completed!")
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Total duration: {total_duration}")
    print(f"Master benchmark saved to: {master_benchmark_file}")
    print(f"Best configurations saved to: {validation_dir / 'best_configurations.json'}")

def generate_penalty_ablation_summary(master_benchmark_file, validation_dir):
    """Generate comprehensive summary and LaTeX table."""
    try:
        # Read master benchmark data
        df = pd.read_csv(master_benchmark_file)
        
        # Create summary directory
        summary_dir = validation_dir / "summary"
        summary_dir.mkdir(exist_ok=True, parents=True)
        
        # Compute average performance by method
        avg_by_method = df.groupby('Method_Name')[
            ['Expected_Profit', 'Ex_post_Profit', 'SI_Penalty', 
             'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds']
        ].mean().reset_index()
        
        # Sort by ex-post profit (descending)
        avg_by_method = avg_by_method.sort_values('Ex_post_Profit', ascending=False)
        
        # Generate LaTeX table
        latex_table = generate_latex_table(avg_by_method)
        
        # Save LaTeX table
        with open(summary_dir / "penalty_ablation_table.tex", 'w') as f:
            f.write(latex_table)
        
        # Generate plots
        plt.figure(figsize=(15, 10))
        
        # Ex-post profit by method
        plt.subplot(2, 2, 1)
        bars = plt.bar(avg_by_method['Method_Name'], avg_by_method['Ex_post_Profit'])
        bars[0].set_color('green')  # Highlight best method
        plt.title('Average Ex-post Profit by Method')
        plt.xlabel('Method')
        plt.ylabel('Average Ex-post Profit')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # SI Penalty by method
        plt.subplot(2, 2, 2)
        plt.bar(avg_by_method['Method_Name'], avg_by_method['SI_Penalty'], color='red', alpha=0.7)
        plt.title('Average SI Penalty by Method')
        plt.xlabel('Method')
        plt.ylabel('Average SI Penalty')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Volume Penalty by method
        plt.subplot(2, 2, 3)
        plt.bar(avg_by_method['Method_Name'], avg_by_method['Volume_Penalty'], color='orange', alpha=0.7)
        plt.title('Average Volume Penalty by Method')
        plt.xlabel('Method')
        plt.ylabel('Average Volume Penalty')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Processing time by method
        plt.subplot(2, 2, 4)
        plt.bar(avg_by_method['Method_Name'], avg_by_method['Processing_Time_Seconds'], color='purple', alpha=0.7)
        plt.title('Average Processing Time by Method')
        plt.xlabel('Method')
        plt.ylabel('Average Processing Time (seconds)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(summary_dir / "penalty_ablation_summary.png")
        plt.close()
        
        # Save summary statistics
        best_method = avg_by_method.iloc[0]
        summary_stats = {
            'best_method': {
                'name': best_method['Method_Name'],
                'ex_post_profit': float(best_method['Ex_post_Profit']),
                'expected_profit': float(best_method['Expected_Profit']),
                'si_penalty': float(best_method['SI_Penalty']),
                'volume_penalty': float(best_method['Volume_Penalty']),
                'operating_cost': float(best_method['Operating_Cost']),
                'processing_time': float(best_method['Processing_Time_Seconds'])
            },
            'overall_stats': {
                'total_methods': len(avg_by_method),
                'total_validations': len(df),
                'avg_ex_post_profit_all': float(df['Ex_post_Profit'].mean()),
                'avg_processing_time_all': float(df['Processing_Time_Seconds'].mean())
            }
        }
        
        # Save as JSON
        with open(summary_dir / "penalty_ablation_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=4)
        
        # Generate text summary
        with open(summary_dir / "penalty_ablation_summary.txt", 'w') as f:
            f.write("Penalty Ablation Study Summary\n")
            f.write("=============================\n\n")
            
            f.write(f"Total methods tested: {len(avg_by_method)}\n")
            f.write(f"Total validations: {len(df)}\n\n")
            
            f.write("Best Method Overall:\n")
            f.write(f"  Method: {best_method['Method_Name']}\n")
            f.write(f"  Average Ex-post Profit: {best_method['Ex_post_Profit']:.2f}\n")
            f.write(f"  Average Expected Profit: {best_method['Expected_Profit']:.2f}\n")
            f.write(f"  Average SI Penalty: {best_method['SI_Penalty']:.2f}\n")
            f.write(f"  Average Volume Penalty: {best_method['Volume_Penalty']:.2f}\n")
            f.write(f"  Average Operating Cost: {best_method['Operating_Cost']:.2f}\n")
            f.write(f"  Average Processing Time: {best_method['Processing_Time_Seconds']:.2f} seconds\n\n")
            
            f.write("All Methods (ranked by ex-post profit):\n")
            for _, row in avg_by_method.iterrows():
                f.write(f"  {row['Method_Name']}: {row['Ex_post_Profit']:.2f}\n")
        
        print(f"Penalty ablation summary generated in {summary_dir}")
        print("\nLaTeX Table:")
        print(latex_table)
        
        return avg_by_method, latex_table
        
    except Exception as e:
        print(f"Error generating penalty ablation summary: {e}")
        print(traceback.format_exc())

def generate_latex_table(avg_by_method):
    """Generate LaTeX table with results (updated for pandas DataFrame)."""
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Penalty Ablation Study Results}
\begin{tabular}{lcccccc}
\toprule
Method & Ex-post Profit & Expected Profit & SI Penalty & Vol Penalty & Op Cost & Time (s) \\
\midrule
"""
    
    for _, row in avg_by_method.iterrows():
        method = row['Method_Name']
        latex += f"{method} & "
        latex += f"{row['Ex_post_Profit']:.2f} & "
        latex += f"{row['Expected_Profit']:.2f} & "
        latex += f"{row['SI_Penalty']:.2f} & "
        latex += f"{row['Volume_Penalty']:.2f} & "
        latex += f"{row['Operating_Cost']:.2f} & "
        latex += f"{row['Processing_Time_Seconds']:.2f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\label{tab:penalty_ablation}
\end{table}
"""
    
    return latex

def generate_latex_table(validation_results):
    """Generate LaTeX table with results."""
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Penalty Ablation Study Results}
\begin{tabular}{lcccccc}
\toprule
Method & Ex-post Profit & Expected Profit & SI Penalty & Vol Penalty & Op Cost & Time (s) \\
\midrule
"""
    
    for result in validation_results:
        si_f = result['si_factor']
        vol_f = result['vol_factor']
        
        # Determine method name
        if si_f == 0.0 and vol_f == 0.0:
            method = "Baseline"
        elif vol_f == 0.0:
            method = f"SI-{si_f:.1f}"
        else:
            method = f"Vol-{vol_f:.1f}"
        
        latex += f"{method} & "
        latex += f"{result['ex_post_profit']:.2f} & "
        latex += f"{result['expected_profit']:.2f} & "
        latex += f"{result['SI_penalty']:.2f} & "
        latex += f"{result['volume_penalty']:.2f} & "
        latex += f"{result['operating_cost']:.2f} & "
        latex += f"{result['processing_time']:.2f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\label{tab:penalty_ablation}
\end{table}
"""
    
    return latex

# Run the penalty ablation study
if __name__ == "__main__":
    # Run pretraining
    penalty_ablation_pretraining()
    
    # Run comprehensive validation
    comprehensive_penalty_ablation_validation()