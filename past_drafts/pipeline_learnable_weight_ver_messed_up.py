# %% Import libraries
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import sys
# torch.autograd.set_detect_anomaly(True)

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly,neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

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

# Example usage:
sample_date = "2022-01-01"
DA_price_hour = read_da_price(sample_date)
DA_price_quarter = hourly_to_quarterly(DA_price_hour)
print(DA_price_hour)
print(DA_price_quarter)

# %% Define test data
# Create sample input data
power = torch.tensor([-6.77, -7.01, -7.32, -7.63, -7.95, -8.26, -8.19, 4.27, 4.11, 4.43, 4.23, 4.01, 3.78, 3.55, 
                        3.37, 3.3, 3.23, 4.17, 4.8, 4.55, 3.91, 3.66, 2.64, 2.57], dtype=torch.float32, requires_grad=True)
head = torch.tensor([76.96, 79.39, 81.82, 84.25, 86.67, 89.12, 91.47, 90.13, 88.82, 87.34, 85.89, 84.48, 83.13, 81.85, 
                        80.6, 79.35, 78.09, 76.37, 74.23, 72.18, 70.49, 68.9, 67.77, 66.67], dtype=torch.float32, requires_grad=True)
flow = torch.tensor([-10.24, -10.14, -10.12, -10.11, -10.11, -10.18, -9.79, 5.55, 5.48, 6.16, 6.06, 5.87, 5.61, 5.36, 
                        5.19, 5.21, 5.24, 7.15, 8.95, 8.53, 7.04, 6.64, 4.68, 4.61], dtype=torch.float32, requires_grad=True)
head_init = 77.0 # Initial head value
v_low_init = h_to_v_low_fitted(head_init) # Initial lower reservoir volume

# %% 
class HydroParameters:
    def __init__(
        self,
        time_horizon=24, # number of time periods
        sampling_rate=100, # number of samples for regression
        δ_p=5,
        δ_h=20,
        δ_q=7,
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
        self.δ_p = δ_p
        self.δ_h = δ_h
        self.δ_q = δ_q
        self.operational_cost = operational_cost
        self.rho = rho
        self.g = g
        self.mu = mu

        self.head_min = head_min
        self.head_max = head_max
        self.max_vol_up = max_vol_up
        self.min_vol_low = min_vol_low
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down

        self.target_head = target_head
        self.target_vol_low = target_vol_low
        self.head_init = head_init
        self.v_low_init = v_low_init

        self.neg_min_fit = neg_min_fit
        self.neg_max_fit = neg_max_fit
        self.pos_min_fit = pos_min_fit
        self.pos_max_fit = pos_max_fit

        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.predict_q_poly = predict_q_poly
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.gross_head = gross_head

class RegressionLayer:
    def __init__(self, params: HydroParameters):
        self.params = params

    @staticmethod 
    def least_squares_UPC_torch(p_samples, h_samples, q_values):
        """
        Perform least squares for q = c*p + d*h + e.

        p_samples: Tensor of shape [N]
        h_samples: Tensor of shape [N]
        q_values:  Tensor of shape [N]
        """
        # Construct design matrix X = [p, h, 1]
        ones = torch.ones_like(p_samples)
        X = torch.stack([p_samples, h_samples, ones], dim=1)  # shape: [N, 3]
        y = q_values.unsqueeze(1)                             # shape: [N, 1]
        
        # Solve least squares: beta = (X^T X)^{-1} X^T y
        XTX = X.transpose(0, 1) @ X       # shape: [3, 3]
        XTy = X.transpose(0, 1) @ y       # shape: [3, 1]
        
        # Use torch.inverse or torch.linalg.inv
        XTX_inv = torch.inverse(XTX)      # shape: [3, 3]
        beta = XTX_inv @ XTy             # shape: [3, 1]
        
        # Squeeze to shape [3]
        return beta.squeeze(-1)

    @staticmethod
    def least_squares_v_low_torch(h_samples, v_low_samples):
        """
        Perform least squares for v_low = a*h + b.
        
        h_samples: Tensor [N]
        v_low_samples: Tensor [N]
        """
        ones = torch.ones_like(h_samples)
        X = torch.stack([h_samples, ones], dim=1)  # shape: [N, 2]
        y = v_low_samples.unsqueeze(1)             # shape: [N, 1]

        XTX = X.transpose(0, 1) @ X  # [2,2]
        XTy = X.transpose(0, 1) @ y  # [2,1]
        XTX_inv = torch.inverse(XTX)
        beta = XTX_inv @ XTy         # shape: [2,1]
        return beta.squeeze(-1)      # shape: [2]

    def run_regression(self, power, head):
        """
        For each hour t in [0..time_horizon-1], 
        compute local linear models for q = c*p + d*h + e and v_low = a*h + b.
        Returns c, d, e, a, b as Tensors of size [time_horizon].
        """
        TH = self.params.time_horizon
        
        c_list, d_list, e_list = [], [], []
        a_list, b_list = [], []

        # Convert scalar params to tensors once (same dtype/device as `power`)
        device = power.device
        dtype  = power.dtype

        head_min_tensor = torch.tensor(self.params.head_min, dtype=dtype, device=device)
        head_max_tensor = torch.tensor(self.params.head_max, dtype=dtype, device=device)
        delta_h_tensor  = torch.tensor(self.params.δ_h,       dtype=dtype, device=device)
        delta_p_tensor  = torch.tensor(self.params.δ_p,       dtype=dtype, device=device)

        for t in range(TH):
            # 1) p_center, h_center are Tensors
            p_center = power[t]
            h_center = head[t]

            # 2) Build local region around (p_center, h_center) using Tensors
            p_lo = p_center - delta_p_tensor
            p_hi = p_center + delta_p_tensor
            
            # Torch max/min must get Tensors, not ints
            h_lo = torch.maximum(head_min_tensor, h_center - delta_h_tensor)
            h_hi = torch.minimum(head_max_tensor, h_center + delta_h_tensor)

            # 3) Sample in a differentiable manner
            p_samples = torch.linspace(p_lo, p_hi, steps=self.params.sampling_rate)
            h_samples = torch.linspace(h_lo, h_hi, steps=self.params.sampling_rate)

            # Meshgrid, flatten
            p_mesh, h_mesh = torch.meshgrid(p_samples, h_samples, indexing="ij")
            p_flat = p_mesh.reshape(-1)
            h_flat = h_mesh.reshape(-1)

            # 4) Determine valid region
            mask_turbine = (
                (p_flat >= self.params.pos_min_fit[0]*h_flat + self.params.pos_min_fit[1]) &
                (p_flat <= self.params.pos_max_fit[0]*h_flat + self.params.pos_max_fit[1])
            )
            mask_pump = (
                (p_flat >= self.params.neg_min_fit[0]*h_flat + self.params.neg_min_fit[1]) &
                (p_flat <= self.params.neg_max_fit[0]*h_flat + self.params.neg_max_fit[1])
            )
            mask_valid = mask_turbine | mask_pump

            p_valid = p_flat[mask_valid]
            h_valid = h_flat[mask_valid]

            # 5) If no valid points => fallback
            if p_valid.numel() < 2:
                c_list.append(torch.tensor(0.0, device=device, dtype=dtype))
                d_list.append(torch.tensor(0.0, device=device, dtype=dtype))
                e_list.append(torch.tensor(0.0, device=device, dtype=dtype))
            else:
                # Evaluate q in vectorized manner
                q_valid = self.params.predict_q_poly(p_valid, h_valid)
                # Solve for c, d, e
                beta_q = self.least_squares_UPC_torch(p_valid, h_valid, q_valid)
                c_list.append(beta_q[0])
                d_list.append(beta_q[1])
                e_list.append(beta_q[2])

            # 6) For v_low = a*h + b
            h_samples_2 = torch.linspace(h_lo, h_hi, steps=self.params.sampling_rate)
            v_low_vals = self.params.h_to_v_low_fitted(h_samples_2)
            beta_v = self.least_squares_v_low_torch(h_samples_2, v_low_vals)
            a_list.append(beta_v[0])
            b_list.append(beta_v[1])

        # 7) Stack
        c_tensor = torch.stack(c_list, dim=0)
        d_tensor = torch.stack(d_list, dim=0)
        e_tensor = torch.stack(e_list, dim=0)
        a_tensor = torch.stack(a_list, dim=0)
        b_tensor = torch.stack(b_list, dim=0)

        return c_tensor, d_tensor, e_tensor, a_tensor, b_tensor



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

    def forward(self, 
                DA_prices, c, d, e, a, b, 
                power, head, flow, 
                w_p, w_h, w_q):
        """
        Solve the optimization for new parameter values.
        """
        self.initialize_layer(power, head, flow)

        (p_opt, q_opt, h_opt, v_opt) = self.layer(
            DA_prices, c, d, e, a, b, 
            w_p, w_h, w_q,
            solver_args={"solve_method": "ECOS"}
        )
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
            p_prev = p_list[-1]
            q_prev = q_list[-1]
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
            q_turb = self.params.predict_q_poly(p_new_turb, h_prev)

            # c) For pump mode (p_new<0), clamp p between neg_min(h) and neg_max(h)
            p_min_pump = self.params.neg_min(h_prev)
            p_max_pump = self.params.neg_max(h_prev)
            p_new_pump = torch.clamp(p_new, min=p_min_pump, max=p_max_pump)
            q_pump = self.params.predict_q_poly(p_new_pump, h_prev)

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

            # Debug print if desired
            print(f"Minute {i}: v_low={v_next.item():.3f}")

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

class Pipeline:
    def __init__(self, params: HydroParameters):
        self.params = params

        # Sub-modules
        self.regression = RegressionLayer(params)
        self.optimizer = OptiLayer(params)
        self.simulator = SimulationLayer(params)

        # Weight-prediction network
        TH = self.params.time_horizon
        self.weight_network = nn.Sequential(
            nn.Linear(4 * TH, 10), # Input: concatenated DA_prices, power, flow, head
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 3 * TH), # Output: w_p, w_q, w_h for each timestep
            nn.Softplus() # Ensure positive weights
        )

    def predict_weights(self, DA_prices, power, flow, head):
        x = torch.cat([DA_prices, power, flow, head]) # Concatenate inputs
        output = self.weight_network(x)

        # Split output into three weight vectors
        TH = self.params.time_horizon
        w_p = output[:TH]
        w_q = output[TH:2*TH]
        w_h = output[2*TH:]

        return w_p, w_q, w_h

    def forward(self, 
                power_init, head_init, 
                DA_prices, DA_price_quarter):
        """
        Orchestrate the steps:
         1) Predict flow & weights
         2) Regression to get c,d,e,a,b
         3) Solve optimization
         4) Simulate + profit
        """

        # 1) Predict initial flow from (p,h)
        flow_init = torch.tensor([
            self.params.predict_q_poly(p.item(), h.item()) 
            for p, h in zip(power_init, head_init)
        ], dtype=torch.float32)

        # 2) Predict penalty weights
        w_p, w_q, w_h = self.predict_weights(DA_prices, power_init, flow_init, head_init)

        '''
        # Check the values of w_p, w_q, w_h
        print("w_p: ", w_p)
        print("w_q: ", w_q)
        print("w_h: ", w_h)
        '''

        # 3) Run regression layer
        c, d, e, a, b = self.regression.run_regression(power_init, head_init)
        
        """
        # check the values of c, d, e, a, b
        print("c: ", c)
        print("d: ", d)
        print("e: ", e)
        print("a: ", a)
        print("b: ", b)
        """

        # 4) Solve optimization
        p_opt, q_opt, h_opt, v_opt = self.optimizer.forward(
            DA_prices, c, d, e, a, b,
            power_init, head_init, flow_init,
            w_p, w_h, w_q
        )

        """ 
        # print optimal power, flow, head, and volume
        print("Optimized Power Schedule:")
        print(p_opt.detach().numpy())
        print("\nOptimized Flow Schedule:")
        print(q_opt.detach().numpy())
        print("\nOptimized Head Schedule:")
        print(h_opt.detach().numpy())
        print("\nOptimized Lower Reservoir Volume Schedule:")
        print(v_opt.detach().numpy()) 
        """

        # 5) Simulate actual operation
        p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb = self.simulator.simulate_operation(
            p_opt, q_opt, h_opt
        )

        # 6) Calculate profit
        profit = self.simulator.calc_profit(
            p_sim_clb, p_opt, v_low_clb, DA_price_quarter
        )

        return profit, p_opt, q_opt, h_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h

# %% Test the pipeline
params = HydroParameters()
pipeline = Pipeline(params)

profit, p_opt, q_opt, h_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb, c, d, e, a, b, w_p, w_q, w_h = pipeline.forward(power, head, DA_price_hour, DA_price_quarter)

# %% Test backpropagation of SimulationLayer

# Wrap p_opt, q_opt, h_opt with requires_grad for gradient computation
p_opt_test = p_opt.detach().clone().requires_grad_(True)
q_opt_test = q_opt.detach().clone().requires_grad_(True)
h_opt_test = h_opt.detach().clone().requires_grad_(True)

# Create SimulationLayer instance for forward simulation
sim_layer = SimulationLayer(params)

# Run forward simulation
p_sim_clb_test, q_sim_clb_test, h_sim_clb_test, v_low_clb_test = sim_layer.simulate_operation(
    p_opt_test, q_opt_test, h_opt_test
)

# Compute profit scalar for differentiation
test_profit = -sim_layer.calc_profit(
    p_sim_clb_test, p_opt_test, v_low_clb_test, DA_price_quarter
)

# Run backpropagation
print(">>> Attempting backprop on test_profit:")
test_profit.backward()

# Display gradients
print("Gradient wrt p_opt_test:\n", p_opt_test.grad)
print("Gradient wrt q_opt_test:\n", q_opt_test.grad)
print("Gradient wrt h_opt_test:\n", h_opt_test.grad)

# %% Test backpropagation of OptiLayer
params = HydroParameters()
opti_layer = OptiLayer(params)

# Create test tensors with gradients
c_test = c.detach().clone().requires_grad_(False)
d_test = d.detach().clone().requires_grad_(False)
e_test = e.detach().clone().requires_grad_(False)
a_test = a.detach().clone().requires_grad_(False)
b_test = b.detach().clone().requires_grad_(False)
w_p_test = w_p.detach().clone().requires_grad_(True)
w_h_test = w_h.detach().clone().requires_grad_(True)
w_q_test = w_q.detach().clone().requires_grad_(True)

# Run optimization
p_opt_test, q_opt_test, h_opt_test, v_opt_test = opti_layer.forward(
    DA_price_hour, c_test, d_test, e_test, a_test, b_test,
    power, head, flow,
    w_p_test, w_h_test, w_q_test
)

# Calculate test loss and backpropagate
test_loss = -torch.mean(p_opt_test)
print(">>> Attempting backprop on test_loss:")
test_loss.backward()

# Display gradients
print("\nGradients for regression coefficients:")
print("c gradient:", c_test.grad)
print("d gradient:", d_test.grad)
print("e gradient:", e_test.grad)
print("a gradient:", a_test.grad)
print("b gradient:", b_test.grad)

print("\nGradients for weights:")
print("w_p gradient:", w_p_test.grad)
print("w_h gradient:", w_h_test.grad)
print("w_q gradient:", w_q_test.grad)

# %% Test backpropagation of RegressionLayer
power_test = power.detach().clone().requires_grad_(True)
head_test = head.detach().clone().requires_grad_(True)

regression_layer = RegressionLayer(params)
c_test, d_test, e_test, a_test, b_test = regression_layer.run_regression(power_test, head_test)

# Example scalar loss
loss = c_test.sum() + d_test.sum() + e_test.sum() + a_test.sum() + b_test.sum()
print("Attempting backprop on regression loss")
loss.backward()

print("Grad wrt power_test:", power_test.grad)
print("Grad wrt head_test:", head_test.grad)

# %%
'''
1. back propagation
2. epocs on 10 days of decisions (double for loop: database of 10 days; epocs)
'''
# %% Plot and print results
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

# %%
