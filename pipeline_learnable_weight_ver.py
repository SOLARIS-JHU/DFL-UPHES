# %% Import libraries
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import dill as pickle
import pandas as pd
import sys

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
        self.params = params  # Store the hydro parameters

    def least_squares_UPC_torch(self, p_samples, h_samples, q_values):
        """Perform least squares for q = c*p + d*h + e."""
        X = torch.stack([p_samples, h_samples, torch.ones_like(p_samples)], dim=1)
        y = q_values.unsqueeze(1)
        XTX = X.t() @ X
        XTX_inv = torch.inverse(XTX)
        XTy = X.t() @ y
        beta = XTX_inv @ XTy
        return beta.squeeze()

    def least_squares_v_low_torch(self, h_samples, v_low_samples):
        """Perform least squares for v_low = a*h + b."""
        X = torch.stack([h_samples, torch.ones_like(h_samples)], dim=1)
        y = v_low_samples.unsqueeze(1)
        XTX = X.t() @ X
        XTX_inv = torch.inverse(XTX)
        XTy = X.t() @ y
        beta = XTX_inv @ XTy
        return beta.squeeze()

    def run_regression(self, power, head):
        """
        For each hour t in [0..time_horizon-1], 
        compute local linear models for q and v_low.
        Returns c, d, e, a, b as Tensors of size [time_horizon].
        """
        TH = self.params.time_horizon
        c_list, d_list, e_list = [], [], []
        a_list, b_list = [], []

        for t in range(TH):
            # 1) Build samples for (p, h) around current operating point
            p_center = power[t].item()
            h_center = head[t].item()

            p_lo = p_center - self.params.δ_p
            p_hi = p_center + self.params.δ_p
            p_samples = torch.linspace(p_lo, p_hi, self.params.sampling_rate)  # or self.p.UPC_sampling_rate, etc.

            h_lo = max(self.params.head_min, h_center - self.params.δ_h)
            h_hi = min(self.params.head_max, h_center + self.params.δ_h)
            h_samples = torch.linspace(h_lo, h_hi, self.params.sampling_rate)  # or self.p.UPC_sampling_rate

            # Create meshgrid
            p_mesh, h_mesh = torch.meshgrid(p_samples, h_samples, indexing="ij")
            p_flat = p_mesh.flatten()
            h_flat = h_mesh.flatten()

            # 2) Filter by valid region (pump or turbine)
            mask_turbine = (
                (p_flat >= self.params.pos_min_fit[0]*h_flat + self.params.pos_min_fit[1]) &
                (p_flat <= self.params.pos_max_fit[0]*h_flat + self.params.pos_max_fit[1])
            )
            mask_pump = (
                (p_flat >= self.params.neg_min_fit[0]*h_flat + self.params.neg_min_fit[1]) &
                (p_flat <= self.params.neg_max_fit[0]*h_flat + self.params.neg_max_fit[1])
            )
            mask = mask_turbine | mask_pump
            p_valid = p_flat[mask]
            h_valid = h_flat[mask]

            # 3) Evaluate q = predict_q_poly(...) for valid points
            if p_valid.numel() == 0:
                # Fallback if no valid points
                c_list.append(0.0)
                d_list.append(0.0)
                e_list.append(0.0)
            else:
                q_values = torch.tensor([
                    self.params.predict_q_poly(pv.item(), hv.item())
                    for pv, hv in zip(p_valid, h_valid)
                ], dtype=torch.float32)
                beta = self.least_squares_UPC_torch(p_valid, h_valid, q_values)
                c_list.append(beta[0].item())
                d_list.append(beta[1].item())
                e_list.append(beta[2].item())

            # 4) Regression for v_low = a*h + b
            h_samples_2 = torch.linspace(h_lo, h_hi, self.params.sampling_rate)  # e.g. 20 points around the head
            v_low_values = torch.tensor([
                self.params.h_to_v_low_fitted(hh.item()) for hh in h_samples_2
            ], dtype=torch.float32)
            beta_v = self.least_squares_v_low_torch(h_samples_2, v_low_values)
            a_list.append(beta_v[0].item())
            b_list.append(beta_v[1].item())

        # Convert lists to Tensors
        c_tensor = torch.tensor(c_list, dtype=torch.float32)
        d_tensor = torch.tensor(d_list, dtype=torch.float32)
        e_tensor = torch.tensor(e_list, dtype=torch.float32)
        a_tensor = torch.tensor(a_list, dtype=torch.float32)
        b_tensor = torch.tensor(b_list, dtype=torch.float32)

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
        import torch
        
        # Repeat schedules to minute resolution
        p_sim = p.repeat_interleave(60) 
        q_sim = q.repeat_interleave(60)
        h_sim = h.repeat_interleave(60)
        
        # Initialize arrays
        p_sim_clb = p_sim.clone()
        q_sim_clb = torch.zeros(len(p_sim) + 1)  # +1 to allow final appended state
        h_sim_clb = torch.zeros(len(p_sim) + 1)
        v_low_clb = torch.zeros(len(p_sim) + 1)
        
        # Add end-of-day state for continuity
        p_sim_clb = torch.cat([p_sim_clb, p_sim_clb[-1].unsqueeze(0)])
        
        # Add idle minutes between mode changes (sign changes)
        for i in range(len(p_sim_clb) - 1, 0, -1):
            if p_sim_clb[i] * p_sim_clb[i - 1] < 0:
                p_sim_clb[i - 1] = 0
                
        # Backward ramping adjustment
        for hour in range(self.params.time_horizon - 1, -1, -1):
            hour_start = hour * 60
            hour_end = hour_start + 60
            
            # Set the first minute to match the original hourly schedule
            p_sim_clb[hour_start] = p[hour]
            
            # Adjust remaining minutes within that hour window
            for i in range(hour_end - 1, hour_start, -1):
                if p_sim_clb[i] - p_sim_clb[i - 1] > self.params.ramp_down:
                    p_sim_clb[i - 1] = p_sim_clb[i] - self.params.ramp_down
                elif p_sim_clb[i] - p_sim_clb[i - 1] < -self.params.ramp_up:
                    p_sim_clb[i - 1] = p_sim_clb[i] + self.params.ramp_up
        
        # Initialize first state
        v_low_clb[0] = self.params.v_low_init
        q_sim_clb[0] = q_sim[0]
        h_sim_clb[0] = h_sim[0]
        
        # Forward simulation with physical constraints
        for i in range(len(p_sim_clb) - 1):
            # Turbine mode
            if p_sim_clb[i] > 0:
                if (p_sim_clb[i] > self.params.pos_min(h_sim_clb[i]) and 
                    p_sim_clb[i] < self.params.pos_max(h_sim_clb[i])):
                    q_sim_clb[i] = self.params.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
                elif p_sim_clb[i] < self.params.pos_min(h_sim_clb[i]):
                    p_sim_clb[i] = self.params.pos_min(h_sim_clb[i])
                    q_sim_clb[i] = self.params.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
                elif p_sim_clb[i] > self.params.pos_max(h_sim_clb[i]):
                    p_sim_clb[i] = self.params.pos_max(h_sim_clb[i])
                    q_sim_clb[i] = self.params.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
            
            # Pump mode
            elif p_sim_clb[i] < 0:
                if (p_sim_clb[i] > self.params.neg_min(h_sim_clb[i]) and 
                    p_sim_clb[i] < self.params.neg_max(h_sim_clb[i])):
                    q_sim_clb[i] = self.params.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
                elif p_sim_clb[i] < self.params.neg_min(h_sim_clb[i]):
                    p_sim_clb[i] = self.params.neg_min(h_sim_clb[i])
                    q_sim_clb[i] = self.params.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
                elif p_sim_clb[i] > self.params.neg_max(h_sim_clb[i]):
                    p_sim_clb[i] = self.params.neg_max(h_sim_clb[i])
                    q_sim_clb[i] = self.params.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
            else:
                # Idle mode
                q_sim_clb[i] = 0
                
            # Update volumes and check bounds
            v_low_clb[i + 1] = v_low_clb[i] + q_sim_clb[i] * 60

            # Print the values of v_low_clb for each time step for debugging
            print(f"Time {i}: {v_low_clb[i + 1].item():.3f}")

            if (v_low_clb[i + 1] > self.params.max_vol_up or 
                v_low_clb[i + 1] < self.params.min_vol_low):
                p_sim_clb[i] = 0
                q_sim_clb[i] = 0
                h_sim_clb[i + 1] = h_sim_clb[i]
                v_low_clb[i + 1] = v_low_clb[i]
            else:
                h_sim_clb[i + 1] = self.params.gross_head(v_low=v_low_clb[i + 1])
        
        # Return calibrated schedules without the final appended state
        return p_sim_clb[:-1], q_sim_clb[:-1], h_sim_clb[:-1], v_low_clb[:-1]

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

        # Check the values of w_p, w_q, w_h
        print("w_p: ", w_p)
        print("w_q: ", w_q)
        print("w_h: ", w_h)

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

        return profit, p_opt, q_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb

# %% Test the pipeline
params = HydroParameters()
pipeline = Pipeline(params)

profit, p_opt, q_opt, p_sim_clb, q_sim_clb, h_sim_clb, v_low_clb = pipeline.forward(power, head, DA_price_hour, DA_price_quarter)

# %%
'''
1. back propagation
2. epocs on 10 days of decisions (double for loop: database of 10 days; epocs)
'''
# %%
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

# %%
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
