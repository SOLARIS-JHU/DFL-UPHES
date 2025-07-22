# %% ---------------------------------------------------------------------------
# Imports and utility helpers
# ---------------------------------------------------------------------------
import sys
import dill as pickle
import torch
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import time

# Helper – tight linear bounds for an affine expression
def affine_bounds(W_row, lb_prev, ub_prev, b_val):
    """Return (lower, upper) bounds of  w·x + b  with x in [lb_prev, ub_prev]."""
    lower = sum(w * (lb if w >= 0 else ub)
                for w, lb, ub in zip(W_row, lb_prev, ub_prev)) + b_val
    upper = sum(w * (ub if w >= 0 else lb)
                for w, lb, ub in zip(W_row, lb_prev, ub_prev)) + b_val
    return lower, upper

# %% ---------------------------------------------------------------------------
# 1.  Load portfolio data and preprocessing artefacts
# ---------------------------------------------------------------------------
# Set device
device = torch.device("cpu")
print(f"Using device: {device}")

# load portfolio data - now 2 levels up
sys.path.append('../../Library')
from V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head
 
# load preprocessed functions & data - now 2 levels up
with open('../../preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

try:
    head_init = 77.0  # Initial head value
    v_low_init = float(h_to_v_low_fitted(torch.tensor(head_init, device=device)))  # Initial lower reservoir volume
    head_min_val = float(head_min)  # Minimum head value
    head_max_val = float(head_max)  # Maximum head value
    
    print(f"Success! head_init: {head_init}, v_low_init: {v_low_init}")
    print(f"head_min_val: {head_min_val}, head_max_val: {head_max_val}")
except Exception as e:
    print(f"Error in function recreation: {e}")
    raise

# %% ---------------------------------------------------------------------------
# 2.  Load weight matrices directly from the saved .pt files
#     (handles either 'layers.*' or 'model.*' prefixes automatically)
# ---------------------------------------------------------------------------
import torch, os
device = torch.device("cpu")

def extract_mlp_weights(pt_path):
    """
    Returns (W1,b1,W2,b2,W3,b3) for a 2‑hidden‑layer (H1=16, H2=16) ReLU MLP 
    saved via nn.Sequential either as 'layers.*' or 'model.*'.
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"File not found: {pt_path}")

    sd = torch.load(pt_path, map_location=device)

    # Detect which prefix the file uses
    prefix = "layers." if any(k.startswith("layers.0") for k in sd.keys()) else "model."
    W1 = sd[f"{prefix}0.weight"].numpy();  b1 = sd[f"{prefix}0.bias"].numpy()
    W2 = sd[f"{prefix}2.weight"].numpy();  b2 = sd[f"{prefix}2.bias"].numpy()
    W3 = sd[f"{prefix}4.weight"].numpy();  b3 = sd[f"{prefix}4.bias"].numpy()
    return W1, b1, W2, b2, W3, b3

def extract_vh_weights(pt_path):
    """
    Returns (W1,b1,W2,b2,W3,b3) for the 1‑32‑16‑1 Volume→Head network.
    """
    sd = torch.load(pt_path, map_location=device)
    prefix = "layers." if any(k.startswith("layers.0") for k in sd.keys()) else "model."
    W1 = sd[f"{prefix}0.weight"].numpy();  b1 = sd[f"{prefix}0.bias"].numpy()
    W2 = sd[f"{prefix}2.weight"].numpy();  b2 = sd[f"{prefix}2.bias"].numpy()
    W3 = sd[f"{prefix}4.weight"].numpy();  b3 = sd[f"{prefix}4.bias"].numpy()
    return W1, b1, W2, b2, W3, b3

# --- 2.1  UPC networks --------------------------------------------------------
W1_T, b1_T, W2_T, b2_T, W3_T, b3_T = extract_mlp_weights(
    "models/turbine_2layers_16_16_best.pt"
)
W1_P, b1_P, W2_P, b2_P, W3_P, b3_P = extract_mlp_weights(
    "models/pump_2layers_16_16_best.pt"
)

# --- 2.2  Volume→Head network -------------------------------------------------
W1_VH, b1_VH, W2_VH, b2_VH, W3_VH, b3_VH = extract_vh_weights(
    "models/v_to_h_model_32-16.pt"
)

with open("models/norm_params.pkl", "rb") as f:
    norm_params = pickle.load(f)
v_mean, v_std = norm_params["v_low_mean"], norm_params["v_low_std"]
h_mean, h_std = norm_params["h_mean"],  norm_params["h_std"]

# %% ---------------------------------------------------------------------------
# 3.  Helper to embed a two‑hidden‑layer ReLU net in Pyomo with Big‑M
# ---------------------------------------------------------------------------
def add_two_layer_relu_network(
        model, name, Tset, input_vars_fun, output_var_fun,
        W1, b1, W2, b2, W3, b3, x_lbs, x_ubs):
    """
    Embed a  (n_in)‑16‑16‑1  ReLU network into Pyomo model 'model'.
    """
    n1 = W1.shape[0]      # 16
    n2 = W2.shape[0]      # 16
    # --- Pre‑compute neuron bounds ------------------------------------------
    # Layer 1
    L1, U1 = [], []
    for i in range(n1):
        lb, ub = affine_bounds(W1[i], x_lbs, x_ubs, b1[i])
        L1.append(lb);  U1.append(ub)
    # Layer‑1 outputs pass ReLU ⇒ y1 in [0, max(0,U1)]
    y1_ubs = [max(0, u) for u in U1]
    # Layer 2
    L2, U2 = [], []
    for j in range(n2):
        lb, ub = affine_bounds(W2[j], [0]*n1, y1_ubs, b2[j])
        L2.append(lb);  U2.append(ub)
    y2_ubs = [max(0, u) for u in U2]

    # --- Add variables ------------------------------------------------------
    model.add_component(f"{name}_z1", pyo.Var(Tset, range(n1)))
    model.add_component(f"{name}_y1", pyo.Var(Tset, range(n1)))
    model.add_component(f"{name}_b1", pyo.Var(Tset, range(n1), domain=pyo.Binary))
    model.add_component(f"{name}_z2", pyo.Var(Tset, range(n2)))
    model.add_component(f"{name}_y2", pyo.Var(Tset, range(n2)))
    model.add_component(f"{name}_b2", pyo.Var(Tset, range(n2), domain=pyo.Binary))

    z1 = getattr(model, f"{name}_z1")
    y1 = getattr(model, f"{name}_y1")
    b1_bin = getattr(model, f"{name}_b1")
    z2 = getattr(model, f"{name}_z2")
    y2 = getattr(model, f"{name}_y2")
    b2_bin = getattr(model, f"{name}_b2")

    # --- Layer 1 constraints ------------------------------------------------
    def z1_eq(m, t, i):
        inp = input_vars_fun(m, t)
        return z1[t, i] == sum(W1[i, k] * inp[k] for k in range(len(inp))) + b1[i]
    model.add_component(f"{name}_z1_eq", pyo.Constraint(Tset, range(n1), rule=z1_eq))

    def relu1_a(m, t, i):   return y1[t, i] >= z1[t, i]
    def relu1_b(m, t, i):   return y1[t, i] >= 0
    def relu1_c(m, t, i):   return y1[t, i] <= z1[t, i] - L1[i]*(1 - b1_bin[t, i])
    def relu1_d(m, t, i):   return y1[t, i] <= U1[i]*b1_bin[t, i]
    model.add_component(f"{name}_relu1_a", pyo.Constraint(Tset, range(n1), rule=relu1_a))
    model.add_component(f"{name}_relu1_b", pyo.Constraint(Tset, range(n1), rule=relu1_b))
    model.add_component(f"{name}_relu1_c", pyo.Constraint(Tset, range(n1), rule=relu1_c))
    model.add_component(f"{name}_relu1_d", pyo.Constraint(Tset, range(n1), rule=relu1_d))

    # --- Layer 2 constraints ------------------------------------------------
    def z2_eq(m, t, j):
        return z2[t, j] == sum(W2[j, i]*y1[t, i] for i in range(n1)) + b2[j]
    model.add_component(f"{name}_z2_eq", pyo.Constraint(Tset, range(n2), rule=z2_eq))

    def relu2_a(m, t, j):   return y2[t, j] >= z2[t, j]
    def relu2_b(m, t, j):   return y2[t, j] >= 0
    def relu2_c(m, t, j):   return y2[t, j] <= z2[t, j] - L2[j]*(1 - b2_bin[t, j])
    def relu2_d(m, t, j):   return y2[t, j] <= U2[j]*b2_bin[t, j]
    model.add_component(f"{name}_relu2_a", pyo.Constraint(Tset, range(n2), rule=relu2_a))
    model.add_component(f"{name}_relu2_b", pyo.Constraint(Tset, range(n2), rule=relu2_b))
    model.add_component(f"{name}_relu2_c", pyo.Constraint(Tset, range(n2), rule=relu2_c))
    model.add_component(f"{name}_relu2_d", pyo.Constraint(Tset, range(n2), rule=relu2_d))

    # --- Output equation ----------------------------------------------------
    def out_eq(m, t):
        return output_var_fun(m, t) == sum(W3[0, j]*y2[t, j] for j in range(n2)) + float(b3[0])
    model.add_component(f"{name}_out_eq", pyo.Constraint(Tset, rule=out_eq))

# %% ---------------------------------------------------------------------------
# 4.  Build the Pyomo MIQP model (24‑hour horizon)
# ---------------------------------------------------------------------------
def create_uphes_miqp_model(T, DA_prices):
    """
    Build and return a Pyomo ConcreteModel for a 24‑step MIQP formulation.
    DA_prices : list/array of 24 floats (€/MWh)
    """
    head_init = 77.0                       # starting head (m)

    # ------------------------------------------------------------------
    # 1.  Pyomo model & basic sets
    # ------------------------------------------------------------------
    model = pyo.ConcreteModel()
    model.T = range(T)

    # ------------------------------------------------------------------
    # 2.  Decision variables
    # ------------------------------------------------------------------
    model.p_T   = pyo.Var(model.T, bounds=(0, None))
    model.p_P   = pyo.Var(model.T, bounds=(None, 0))
    model.q     = pyo.Var(model.T)
    model.h     = pyo.Var(model.T, bounds=(head_min_val, head_max_val))
    model.v_low = pyo.Var(model.T, bounds=(0, max_vol_low))

    # mode indicators
    model.z_I = pyo.Var(model.T, domain=pyo.Binary)   # idle
    model.z_T = pyo.Var(model.T, domain=pyo.Binary)   # turbine
    model.z_P = pyo.Var(model.T, domain=pyo.Binary)   # pump

    # ------------------------------------------------------------------
    # 3.  Scaling vars for Volume→Head network
    # ------------------------------------------------------------------
    model.v_norm = pyo.Var(model.T)            # scaled volume
    model.h_pred = pyo.Var(model.T)            # raw NN output (ŷ)

    def vnorm_def(m, t):
        return m.v_norm[t] * v_std == m.v_low[t] - v_mean
    model.vnorm_def = pyo.Constraint(model.T, rule=vnorm_def)

    def h_def(m, t):
        return m.h[t] == h_mean + h_std * m.h_pred[t]
    model.h_def = pyo.Constraint(model.T, rule=h_def)

    # ---------- Volume→Head neural net (scaled) ------------------------
    add_two_layer_relu_network(
        model, name="vh",
        Tset=model.T,
        input_vars_fun=lambda m, t: [m.v_norm[t]],
        output_var_fun=lambda m, t: m.h_pred[t],
        W1=W1_VH, b1=b1_VH, W2=W2_VH, b2=b2_VH, W3=W3_VH, b3=b3_VH,
        x_lbs=[(0.0 - v_mean)/v_std],
        x_ubs=[(max_vol_low - v_mean)/v_std],
    )

    # ------------------------------------------------------------------
    # 5.  Mode‑selection constraint
    # ------------------------------------------------------------------
    model.one_mode = pyo.Constraint(
        model.T, rule=lambda m, t: m.z_I[t] + m.z_T[t] + m.z_P[t] == 1
    )

    # ------------------------------------------------------------------
    # 6.  Idle‑mode zero‑output constraints
    # ------------------------------------------------------------------
    M_p = 1e4
    model.idle_pT = pyo.Constraint(
        model.T, rule=lambda m, t: m.p_T[t] <=  M_p * (1 - m.z_I[t])
    )
    model.idle_pP = pyo.Constraint(
        model.T, rule=lambda m, t: m.p_P[t] >= -M_p * (1 - m.z_I[t])
    )
    model.idle_q  = pyo.Constraint(
        model.T, rule=lambda m, t: m.q[t]   <=  M_p * (1 - m.z_I[t])
    )

    # ---------- Power bounds (head‑dependent linear fits) -------------------
    model.turb_min = pyo.Constraint(
        model.T,
        rule=lambda m,t: m.p_T[t] >= (pos_min_fit[0]*m.h[t] + pos_min_fit[1]) * m.z_T[t]
    )
    model.turb_max = pyo.Constraint(
        model.T,
        rule=lambda m,t: m.p_T[t] <= (pos_max_fit[0]*m.h[t] + pos_max_fit[1]) * m.z_T[t]
    )
    model.pump_min = pyo.Constraint(
        model.T,
        rule=lambda m,t: m.p_P[t] >= (neg_min_fit[0]*m.h[t] + neg_min_fit[1]) * m.z_P[t]
    )
    model.pump_max = pyo.Constraint(
        model.T,
        rule=lambda m,t: m.p_P[t] <= (neg_max_fit[0]*m.h[t] + neg_max_fit[1]) * m.z_P[t]
    )

    # ---------- NN outputs (flow) ------------------------------------------
    model.q_tur  = pyo.Var(model.T)        # turbine‑mode flow prediction
    model.q_pump = pyo.Var(model.T)        # pump‑mode flow prediction

    # Input bounds for UPC nets
    pT_lb  = 0.0
    pT_ub  = pos_max_fit[0]*head_max_val + pos_max_fit[1]
    pP_lb  = neg_min_fit[0]*head_max_val + neg_min_fit[1]   # most negative
    pP_ub  = 0.0
    upc_h_lb, upc_h_ub = head_min_val, head_max_val

    # ---- Turbine UPC network ----------------------------------------------
    add_two_layer_relu_network(
        model, name="turb",
        Tset=model.T,
        input_vars_fun=lambda m,t: [m.p_T[t], m.h[t]],
        output_var_fun=lambda m,t: m.q_tur[t],
        W1=W1_T, b1=b1_T, W2=W2_T, b2=b2_T, W3=W3_T, b3=b3_T,
        x_lbs=[pT_lb, upc_h_lb],
        x_ubs=[pT_ub, upc_h_ub]
    )
    # ---- Pump UPC network --------------------------------------------------
    add_two_layer_relu_network(
        model, name="pump",
        Tset=model.T,
        input_vars_fun=lambda m,t: [m.p_P[t], m.h[t]],
        output_var_fun=lambda m,t: m.q_pump[t],
        W1=W1_P, b1=b1_P, W2=W2_P, b2=b2_P, W3=W3_P, b3=b3_P,
        x_lbs=[pP_lb, upc_h_lb],
        x_ubs=[pP_ub, upc_h_ub]
    )

    # ---------- Flow = turbine/pump output depending on mode ---------------
    model.flow_switch = pyo.Constraint(
        model.T,
        rule=lambda m,t: m.q[t] == m.q_tur[t]*m.z_T[t] + m.q_pump[t]*m.z_P[t]
    )

    # ---------- Reservoir dynamics -----------------------------------------
    def vol_dyn_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip          # already fixed
        return m.v_low[t] == m.v_low[t-1] + 3600*m.q[t]
    model.vol_dyn = pyo.Constraint(model.T, rule=vol_dyn_rule)

    # ---------- Terminal constraints ---------------------------------------
    model.term_vol  = pyo.Constraint(expr = model.v_low[T-1] <= float(target_vol_low))
    model.term_head = pyo.Constraint(expr = model.h[T-1]   >= float(target_head))
    model.init_head = pyo.Constraint(expr = model.h[0]     == float(head_init))
    
    # ---------- Objective (MIQP) -------------------------------------------
    C_op = 3.8
    model.obj = pyo.Objective(
        expr = sum(
            (model.p_T[t] + model.p_P[t]) * DA_prices[t]
            - C_op * (model.p_T[t] + model.p_P[t])**2
            for t in model.T
        ),
        sense = pyo.maximize
    )
    return model

# %% ---------------------------------------------------------------------------
# 5.  Read price data from new CSV format
# ---------------------------------------------------------------------------
def read_price_data(file_path="../../Data/price_data_2024.csv"):
    """Read price data from the new CSV format."""
    df = pd.read_csv(file_path, dtype={'prices_hourly': str})
    price_data = {}
    
    for _, row in df.iterrows():
        date = row['date']
        prices_str = row['prices_hourly']
        
        # Handle potential NaN or float values
        if pd.isna(prices_str) or isinstance(prices_str, float):
            print(f"Skipping {date} - invalid price data")
            continue
            
        try:
            prices = [float(x.strip()) for x in prices_str.split(',')]
            if len(prices) != 24:
                print(f"Skipping {date} - expected 24 prices, got {len(prices)}")
                continue
            price_data[date] = prices
        except Exception as e:
            print(f"Error parsing prices for {date}: {e}")
            continue
    
    return price_data

# test function read_price_data
if __name__ == "__main__":
    price_data = read_price_data()
    if price_data:
        print("Price data loaded successfully:")
        for date, prices in price_data.items():
            print(f"{date}: {prices}")
    else:
        print("No valid price data found.")

# %% ---------------------------------------------------------------------------
# 6.  Simulation Layer (unchanged)
# ---------------------------------------------------------------------------
class HydroParameters:
    def __init__(
        self,
        time_horizon=24,
        operational_cost=3.8,
        rho=1000,
        g=9.81,
        mu=0.9,
        head_init=head_init,
        v_low_init=v_low_init,
        target_head=target_head,
        target_vol_low=target_vol_low,
        max_vol_up=max_vol_up,
        min_vol_low=min_vol_low,
        neg_min=neg_min,
        neg_max=neg_max,
        pos_min=pos_min,
        pos_max=pos_max,
        predict_q_poly=predict_q_poly,
        h_to_v_low_fitted=h_to_v_low_fitted,
        v_low_to_h_fitted=v_low_to_h_fitted,
    ):
        self.time_horizon = time_horizon
        self.operational_cost = operational_cost
        self.rho = torch.tensor(rho, dtype=torch.float32, device=device)
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)
        self.head_init = head_init.clone().detach().to(device=device, dtype=torch.float32)
        self.v_low_init = v_low_init.clone().detach().to(device=device, dtype=torch.float32)
        self.target_head = torch.tensor(target_head, dtype=torch.float32, device=device)
        self.target_vol_low = torch.tensor(target_vol_low, dtype=torch.float32, device=device)
        self.max_vol_up = torch.tensor(max_vol_up, dtype=torch.float32, device=device)
        self.min_vol_low = torch.tensor(min_vol_low, dtype=torch.float32, device=device)
        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.predict_q_poly = predict_q_poly
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.v_low_to_h_fitted = v_low_to_h_fitted

class SimulationLayer:
    def __init__(self, params):
        self.params = params

    def simulate_operation(self, p, q, h):
        """Simulate hourly operation with physical constraints."""
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
        """Calculate the daily profit from the hourly simulation."""
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

# %% ---------------------------------------------------------------------------
# 7.  Main execution function
# ---------------------------------------------------------------------------
def run_miqp_optimization():
    """Run MIQP optimization for all days in price database."""
    print("Loading price data...")
    price_data = read_price_data()
    
    # Initialize result lists
    detailed_results = []
    benchmark_results = []

    # Get total number of dates
    total_dates = len(price_data)

    # Process each day
    for idx, (date_str, prices_24h) in enumerate(price_data.items(), start=1):
        print(f"\nProcessing {date_str} ({idx}/{total_dates})...")
        
        try:
            start_time = time.time()
            
            # Create and solve model
            model = create_uphes_miqp_model(24, prices_24h)
            solver = pyo.SolverFactory("gurobi")
            solver.options["TimeLimit"] = 3600  # 1 hour time limit
            solver.options["MIPGap"] = 0.01  # 1% MIP gap
            results = solver.solve(model, tee=False)
            
            solution_time = time.time() - start_time
            
            # Check solution status
            term_condition = str(results.solver.termination_condition).lower()
            if term_condition in ("infeasible", "infeasibleorunbounded"):
                print(f"Infeasible solution for {date_str}")
                continue
            
            # Extract optimization results
            expected_profit = pyo.value(model.obj)
            
            # Run simulation
            head_init_val = torch.tensor(77.0, dtype=torch.float32, device=device)
            v_low_init_val = h_to_v_low_fitted(head_init_val)
            
            params = HydroParameters(
                head_init=head_init_val,
                v_low_init=v_low_init_val,
                neg_min=neg_min, neg_max=neg_max,
                pos_min=pos_min, pos_max=pos_max,
                predict_q_poly=predict_q_poly,
                h_to_v_low_fitted=h_to_v_low_fitted,
                v_low_to_h_fitted=v_low_to_h_fitted
            )
            
            simulator = SimulationLayer(params)
            
            # Extract schedules
            p_total_schedule = [model.p_T[t]() + model.p_P[t]() for t in range(24)]
            q_schedule = [model.q[t]() for t in range(24)]
            h_schedule = [model.h[t]() for t in range(24)]
            v_low_schedule = [model.v_low[t]() for t in range(24)]
            
            # Convert to tensors
            p_tensor = torch.tensor(p_total_schedule, dtype=torch.float32, device=device)
            q_tensor = torch.tensor(q_schedule, dtype=torch.float32, device=device)
            h_tensor = torch.tensor(h_schedule, dtype=torch.float32, device=device)
            
            # Run simulation
            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_tensor, q_tensor, h_tensor)
            
            # Calculate simulation profit
            da_prices_tensor = torch.tensor(prices_24h, dtype=torch.float32, device=device)
            profit, si_penalty, vol_penalty, op_cost = simulator.calc_profit(
                p_sim, p_tensor[:len(p_sim)], v_low_sim, da_prices_tensor[:len(p_sim)]
            )
            
            # Store detailed results
            for hour in range(24):
                detailed_results.append({
                    'date': date_str,
                    'hour': hour,
                    'power': p_total_schedule[hour],
                    'head': h_schedule[hour],
                    'volume': v_low_schedule[hour],
                    'flow': q_schedule[hour],
                    'price': prices_24h[hour]
                })
            
            # Store benchmark results
            benchmark_results.append({
                'Date': date_str,
                'Solving Time (s)': solution_time,
                'Expected Profit (€)': expected_profit,
                'SI Penalty (€)': si_penalty.item(),
                'Vol Penalty (€)': vol_penalty.item(),
                'Op Cost (€)': op_cost.item(),
                'Ex-post Profit (€)': profit.item()
            })
            
            print(f"Expected profit: {expected_profit:.2f} €, Ex-post profit: {profit.item():.2f} €")
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            continue
    
    # Save results
    detailed_df = pd.DataFrame(detailed_results)
    benchmark_df = pd.DataFrame(benchmark_results)
    
    detailed_df.to_csv("MIQP_nn_results.csv", index=False)
    benchmark_df.to_csv("MIQP_nn_benchmark.csv", index=False)
    
    print(f"\nProcessing complete!")
    print(f"Detailed results saved to MIQP_nn_results.csv ({len(detailed_results)} rows)")
    print(f"Benchmark results saved to MIQP_nn_benchmark.csv ({len(benchmark_results)} rows)")

# %% ---------------------------------------------------------------------------
# 8.  Execute
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_miqp_optimization()