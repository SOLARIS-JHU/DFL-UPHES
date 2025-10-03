# PPO_pumped_hydro1_tuned.py
# Tuned for stability & anti-collapse regularization.

# %% Imports and setup (mirrors DFL_pretraining.py style)
import torch
import torch.nn as nn
import torch.nn.functional as F
import dill as pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# load portfolio data
sys.path.append('../Library')
from V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from V_H_relations import (
    r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n,
    h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up,
    min_vol_low, target_vol_up, target_vol_low, target_head
)

# load preprocessed functions & data
with open('../preprocess.pkl', 'rb') as f:
    (v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin,
     coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin,
     predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, 
     neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, 
     DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, 
     neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound) = pickle.load(f)

head_init = torch.tensor(77.0, device=device)  # Initial head value
v_low_init = torch.tensor(h_to_v_low_fitted(head_init), device=device)  # Initial lower reservoir volume

def hourly_to_quarterly(tensor_data: torch.Tensor) -> torch.Tensor:
    return tensor_data.repeat_interleave(4)

# %% HydroParameters (aligned with DFL_pretraining.py)
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

# %% SimulationLayer (aligned with DFL_pretraining.py; tweak to project to bounds not force idle)
class SimulationLayer:
    def __init__(self, params: HydroParameters):
        self.params = params

    def simulate_operation(self, p_commit: torch.Tensor, q_hint: torch.Tensor = None, h_hint: torch.Tensor = None
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        TH = self.params.time_horizon
        p_list, q_list, h_list, v_list = [], [], [], []

        v_current = self.params.v_low_init.clone()
        h_current = self.params.head_init.clone()

        v_list.append(v_current)
        h_list.append(h_current)

        for t in range(TH):
            p_t = p_commit[t]
            # Default
            q_candidate = torch.tensor(0.0, dtype=torch.float32, device=device)
            if p_t > 0.5 or p_t < -0.5:
                q_candidate = self.params.predict_q_poly(p_t.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)

            v_next = v_current + q_candidate * 3600.0

            if v_next > self.params.max_vol_up:
                if q_candidate > 0:
                    low_p = torch.tensor(0.0, dtype=torch.float32, device=device)
                    high_p = p_t.clone().detach()
                    for _ in range(12):
                        mid_p = 0.5 * (low_p + high_p)
                        mid_q = self.params.predict_q_poly(mid_p.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
                        if v_current + mid_q * 3600.0 > self.params.max_vol_up:
                            high_p = mid_p
                        else:
                            low_p = mid_p
                    p_exec = low_p
                    q_exec = self.params.predict_q_poly(p_exec.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
                    v_next = self.params.max_vol_up.clone()
                else:
                    p_exec = p_t
                    q_exec = q_candidate
            elif v_next < self.params.min_vol_low:
                if q_candidate < 0:
                    low_p = p_t.clone().detach()
                    high_p = torch.tensor(0.0, dtype=torch.float32, device=device)
                    for _ in range(12):
                        mid_p = 0.5 * (low_p + high_p)
                        mid_q = self.params.predict_q_poly(mid_p.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
                        if v_current + mid_q * 3600.0 < self.params.min_vol_low:
                            high_p = mid_p
                        else:
                            low_p = mid_p
                    p_exec = high_p
                    q_exec = self.params.predict_q_poly(p_exec.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
                    v_next = self.params.min_vol_low.clone()
                else:
                    p_exec = p_t
                    q_exec = q_candidate
            else:
                p_exec = p_t
                q_exec = q_candidate

            h_next = self.params.v_low_to_h_fitted(v_next)

            p_list.append(p_exec)
            q_list.append(q_exec)
            v_current = v_next
            h_current = h_next
            v_list.append(v_current)
            h_list.append(h_current)

        p_exec = torch.stack(p_list)
        q_exec = torch.stack(q_list)
        h_exec = torch.stack(h_list[:-1])
        v_low_sim = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32) for v in v_list[:-1]])
        return p_exec, q_exec, h_exec, v_low_sim

    def calc_profit(self, p_exec: torch.Tensor, p_commit: torch.Tensor, v_low_sim: torch.Tensor, DA_price: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e_sim = p_exec
        revenue = torch.sum(DA_price * e_sim)

        surplus_penalty_multiplier = -0.5
        shortage_penalty_multiplier = -2.0

        SI_price = torch.where(
            e_sim < p_commit,
            shortage_penalty_multiplier * DA_price,
            surplus_penalty_multiplier * DA_price
        )
        imbalance = e_sim - p_commit
        SI_penalty = torch.sum(imbalance * SI_price)

        volume_deficit = torch.clamp(v_low_sim[-1] - self.params.target_vol_low, min=0.0)
        energy_loss = self.params.rho * volume_deficit * self.params.g * self.params.target_head * self.params.mu / 3.6e9
        volume_penalty = energy_loss * torch.median(DA_price)

        operating_cost = self.params.operational_cost * torch.sum(e_sim**2)
        total_profit = revenue - operating_cost - SI_penalty - volume_penalty
        return total_profit, SI_penalty, volume_penalty, operating_cost

# %% Data Loading
def load_miqp_day_data(file_path: str, source_name: str, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(file_path):
        if verbose:
            print(f"[{source_name}] File not found: {file_path}")
        return {}
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    required = ['date', 'hour', 'power', 'head', 'flow']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"[{source_name}] Missing required column: {col}")

    if 'price' not in df.columns:
        miqp_orig = "../MIQP/MIQP_piecewise/MIQP_piecewise_results.csv"
        if os.path.exists(miqp_orig):
            price_df = pd.read_csv(miqp_orig)
            price_df.columns = price_df.columns.str.strip()
            try:
                df['date_norm'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                price_df['date_norm'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')
                df = df.merge(price_df[['date_norm', 'hour', 'price']], left_on=['date_norm', 'hour'], right_on=['date_norm', 'hour'], how='left')
                df.drop(columns=['date_norm'], inplace=True)
            except Exception as e:
                if verbose:
                    print(f"[{source_name}] Price merge failed: {e}; synthesizing price.")
                df['price'] = 50 + 20 * np.sin(2 * np.pi * df['hour'] / 24) + 5 * np.random.randn(len(df))
        else:
            if verbose:
                print(f"[{source_name}] No price in file and MIQP baseline not found; synthesizing price.")
            df['price'] = 50 + 20 * np.sin(2 * np.pi * df['hour'] / 24) + 5 * np.random.randn(len(df))

    df['Date'] = pd.to_datetime(df['date'], errors='coerce')
    data_by_date: Dict[str, Dict[str, Any]] = {}
    for date, group in df.groupby('Date'):
        group = group.sort_values('hour')
        if len(group) != 24:
            continue
        date_str = date.strftime('%Y-%m-%d')
        if 'Mode' in group.columns:
            mode_arr = group['Mode'].values
        else:
            power_vals = group['power'].values
            mode_arr = np.where(np.abs(power_vals) < 0.01, 'Idle', np.where(power_vals > 0.0, 'Turbine', 'Pump'))
        data_by_date[date_str] = {
            'power': torch.tensor(group['power'].values, dtype=torch.float32, device=device),
            'head': torch.tensor(group['head'].values, dtype=torch.float32, device=device),
            'flow': torch.tensor(group['flow'].values, dtype=torch.float32, device=device),
            'price': torch.tensor(group['price'].values, dtype=torch.float32, device=device),
            'mode': mode_arr
        }
    if verbose:
        print(f"[{source_name}] Loaded {len(data_by_date)} days from {file_path}")
    return data_by_date

def load_all_sources(noise_levels: List[int] = [10,20,30,40,50,60,70,80],
                     base_dir: str = ".", verbose: bool = True
                     ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    sources: Dict[str, Dict[str, Dict[str, Any]]] = {}
    rs_name = "MIQP_piecewise_results_random_samples"
    rs_file = os.path.join(base_dir, f"{rs_name}.csv")
    sources[rs_name] = load_miqp_day_data(rs_file, rs_name, verbose=verbose)
    for nl in noise_levels:
        name = f"MIQP_piecewise_results_relative_noise_{nl}pct"
        fpath = os.path.join(base_dir, f"{name}.csv")
        sources[name] = load_miqp_day_data(fpath, name, verbose=verbose)
    return sources

# %% PPO Components
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128, init_log_std: float = -1.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std_param = nn.Parameter(torch.full((action_dim,), init_log_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.actor_mean(x)
        value = self.critic(x)
        return mean, value

    def get_action_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std_param)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std_param)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.squeeze(-1)

# %% Environment 
class PumpHydroEnv:
    def __init__(self, params: HydroParameters, day_data: Dict[str, Any]):
        self.params = params
        self.price_profile = day_data['price'].clone().to(device)
        self.mode_schedule = day_data['mode']
        self.time_horizon = len(self.price_profile)
        self.sim = SimulationLayer(params)

        self.current_hour = 0
        self.current_head = self.params.head_init.clone()
        self.current_vol = self.params.v_low_init.clone()
        self.DA_price = self.price_profile

    def reset(self) -> np.ndarray:
        self.current_hour = 0
        self.current_head = self.params.head_init.clone()
        self.current_vol = self.params.v_low_init.clone()
        return self._obs()

    def _obs(self) -> np.ndarray:
        hour = self.current_hour
        price_now = float(self.price_profile[hour].item())
        price_next = float(self.price_profile[hour+1].item()) if hour < self.time_horizon-1 else 0.0
        vol_frac = float((self.current_vol - self.params.min_vol_low) / (self.params.max_vol_up - self.params.min_vol_low))
        vol_frac = max(0.0, min(1.0, vol_frac))
        mode = self.mode_schedule[hour]
        mode_onehot = [1.0, 0.0, 0.0] if mode == "Idle" else ([0.0, 1.0, 0.0] if mode == "Turbine" else [0.0, 0.0, 1.0])
        return np.array([hour, float(self.current_head), vol_frac, price_now, price_next] + mode_onehot, dtype=np.float32)

    def _project_to_mode_and_upc(self, a_raw: float, mode: str) -> torch.Tensor:
        p_raw = torch.tensor(float(a_raw), dtype=torch.float32, device=device)
        thr = 0.5
        if mode == "Idle":
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        if mode == "Turbine":
            if p_raw < thr:
                return torch.tensor(0.0, dtype=torch.float32, device=device)
            p_min = self.params.pos_min(self.current_head)
            p_max = self.params.pos_max(self.current_head)
            p_min_v = max(thr, float(p_min.item() if isinstance(p_min, torch.Tensor) else p_min))
            p_max_v = float(p_max.item() if isinstance(p_max, torch.Tensor) else p_max)
            return torch.tensor(max(p_min_v, min(p_max_v, float(p_raw.item()))), dtype=torch.float32, device=device)
        if mode == "Pump":
            if p_raw > -thr:
                return torch.tensor(0.0, dtype=torch.float32, device=device)
            p_min = self.params.neg_min(self.current_head)
            p_max = self.params.neg_max(self.current_head)
            p_min_v = float(p_min.item() if isinstance(p_min, torch.Tensor) else p_min)
            p_max_v = float(p_max.item() if isinstance(p_max, torch.Tensor) else p_max)
            return torch.tensor(min(p_min_v, max(p_max_v, float(p_raw.item()))), dtype=torch.float32, device=device)
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        mode = self.mode_schedule[self.current_hour]
        p_commit_t = self._project_to_mode_and_upc(action, mode)

        p_commit_vec = torch.zeros(self.params.time_horizon, dtype=torch.float32, device=device)
        p_commit_vec[0] = p_commit_t

        orig_v_init = self.params.v_low_init.clone()
        orig_h_init = self.params.head_init.clone()
        self.params.v_low_init = self.current_vol.clone()
        self.params.head_init = self.current_head.clone()

        p_exec_vec, q_exec_vec, h_exec_vec, v_low_vec = self.sim.simulate_operation(p_commit_vec)

        self.params.v_low_init = orig_v_init
        self.params.head_init = orig_h_init

        p_exec_t = p_exec_vec[0]
        DA_price_t = self.DA_price[self.current_hour]
        revenue = DA_price_t * p_exec_t
        operating_cost = self.params.operational_cost * (p_exec_t ** 2)
        SI_price_t = (-2.0 * DA_price_t) if (p_exec_t < p_commit_t) else (-0.5 * DA_price_t)
        SI_penalty_t = (p_exec_t - p_commit_t) * SI_price_t
        reward_t = revenue - operating_cost - SI_penalty_t

        self.current_vol = v_low_vec[0].clone()
        self.current_head = self.params.v_low_to_h_fitted(self.current_vol)

        done = (self.current_hour == self.time_horizon - 1)
        if done:
            volume_deficit = torch.clamp(self.current_vol - self.params.target_vol_low, min=0.0)
            energy_loss = self.params.rho * volume_deficit * self.params.g * self.params.target_head * self.params.mu / 3.6e9
            volume_penalty_final = energy_loss * torch.median(self.DA_price)
            reward_t = reward_t - volume_penalty_final

        reward_value = float(reward_t.item())
        self.current_hour += 1
        next_obs = None if done else self._obs()
        info = {}
        return (np.zeros(8, dtype=np.float32) if next_obs is None else next_obs, reward_value, done, info)

# %% PPO Trainer (TUNED: hyperparams + regularization, no architecture changes)
class PPOTrainer:
    def __init__(self, policy: ActorCritic, env: PumpHydroEnv, 
                 learning_rate=1e-4,              # === TUNING: smaller LR for stability
                 gamma=0.99, lambda_gae=0.95,
                 clip_coef=0.1,                   # === TUNING: tighter clip
                 value_coef=0.7,                  # === TUNING: slightly higher VF weight
                 entropy_coef=0.01, 
                 batch_size=5, update_epochs=3,   # === TUNING: fewer epochs per batch
                 target_kl=0.015,                 # === TUNING: KL early stopping
                 vf_clip=0.2,                     # === TUNING: value function clipping range
                 adv_clip=5.0,                    # === TUNING: clip large advantages
                 max_grad_norm=0.5,               # keep grad clipping
                 # === STAY-CLOSE REG ===
                 expert_power: torch.Tensor = None,
                 stay_close_coef: float = 0.2,
                 stay_close_anneal: Tuple[float, float] = None,  # (start, end) to linearly decay across updates
                 device: torch.device = device):
        self.policy = policy.to(device)
        self.env = env
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.target_kl = target_kl
        self.vf_clip = vf_clip
        self.adv_clip = adv_clip
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        # === STAY-CLOSE REG ===
        self.expert_power = expert_power.detach().to(device) if expert_power is not None else None
        self.stay_close_coef = stay_close_coef
        self.stay_close_anneal = stay_close_anneal
        self._updates_done = 0

    def _current_stay_close_coef(self):
        if self.stay_close_anneal is None:
            return self.stay_close_coef
        start, end = self.stay_close_anneal
        # Linear decay over first 200 updates; clamp to [end, start]
        T = 200
        t = min(self._updates_done, T)
        return float(start + (end - start) * (t / T))

    def warm_start_from_expert(self, expert_power: torch.Tensor, steps: int = 800, lr: float = 1e-3):
        states = []
        actions = []
        obs = self.env.reset()
        t = 0
        done = False
        while not done and t < len(expert_power):
            states.append(obs)
            actions.append(float(expert_power[t].item()))
            obs, _, done, _ = self.env.step(actions[-1])
            t += 1
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        expert_actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.policy.train()
        opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        for _ in range(steps):
            mean, _ = self.policy.forward(states_t)
            loss = ((mean - expert_actions_t) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.optimizer.param_groups[0]['lr'])

    def train(self, num_episodes: int) -> List[float]:
        episode_count = 0
        ep_rewards_all: List[float] = []
        while episode_count < num_episodes:
            batch_states = []
            batch_actions = []
            batch_logprobs = []
            batch_values = []
            batch_rewards = []
            batch_dones = []
            episodes_to_run = min(self.batch_size, num_episodes - episode_count)
            for _ in range(episodes_to_run):
                obs = self.env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action_t, logprob_t, value_t = self.policy.get_action_and_value(obs_t)
                    action = float(action_t.cpu().item())
                    logprob = float(logprob_t.cpu().item())
                    value = float(value_t.cpu().item())
                    next_obs, reward, done, _ = self.env.step(action)
                    ep_reward += reward
                    batch_states.append(obs)
                    batch_actions.append(action)
                    batch_logprobs.append(logprob)
                    batch_values.append(value)
                    batch_rewards.append(reward)
                    batch_dones.append(done)
                    obs = next_obs
                ep_rewards_all.append(ep_reward)
                episode_count += 1

            # Tensors
            S = torch.tensor(np.array(batch_states), dtype=torch.float32, device=self.device)
            A = torch.tensor(batch_actions, dtype=torch.float32, device=self.device).unsqueeze(1)
            old_logp = torch.tensor(batch_logprobs, dtype=torch.float32, device=self.device)
            V_old = torch.tensor(batch_values, dtype=torch.float32, device=self.device)

            # GAE (with episode resets)
            advantages = []
            returns = []
            gae = 0.0
            last_value = 0.0
            for i in reversed(range(len(batch_rewards))):
                if batch_dones[i]:
                    last_value = 0.0
                    gae = 0.0
                delta = batch_rewards[i] + self.gamma * last_value - V_old[i].item()
                gae = delta + self.gamma * self.lambda_gae * gae
                advantages.insert(0, gae)
                last_value = V_old[i].item()
                returns.insert(0, gae + V_old[i].item())
            Adv = torch.tensor(np.array(advantages), dtype=torch.float32, device=self.device)
            Ret = torch.tensor(np.array(returns), dtype=torch.float32, device=self.device)

            # Normalize & clip advantages (regularization)
            Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)
            if self.adv_clip is not None:
                Adv = torch.clamp(Adv, -self.adv_clip, self.adv_clip)

            # PPO updates
            self.policy.train()
            for epoch in range(self.update_epochs):
                new_logp, entropy, V = self.policy.evaluate_actions(S, A)
                ratios = torch.exp(new_logp - old_logp)

                # Surrogate objective
                surr1 = ratios * Adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * Adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Value function clipping (regularization)
                if self.vf_clip is not None:
                    V_clipped = V_old + torch.clamp(V - V_old, -self.vf_clip, self.vf_clip)
                    v_loss_unclipped = (V - Ret) ** 2
                    v_loss_clipped = (V_clipped - Ret) ** 2
                    value_loss = torch.mean(torch.maximum(v_loss_unclipped, v_loss_clipped))
                else:
                    value_loss = torch.mean((V - Ret) ** 2)

                entropy_loss = -torch.mean(entropy)

                # === STAY-CLOSE REG === L2(mean action vs expert action by hour)
                demo_reg = torch.tensor(0.0, device=self.device)
                if self.expert_power is not None and self.stay_close_coef is not None and self.stay_close_coef > 0.0:
                    with torch.no_grad():
                        hour_idx = S[:, 0].round().long().clamp(0, self.env.time_horizon - 1)
                        expert_targets = self.expert_power[hour_idx].unsqueeze(1)  # (N,1)
                    mean_actions, _ = self.policy.forward(S)  # (N,1)
                    demo_l2 = torch.mean((mean_actions - expert_targets) ** 2)
                    coef = self._current_stay_close_coef()
                    demo_reg = coef * demo_l2

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss + demo_reg

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Diagnostics & KL early stopping (regularization)
                with torch.no_grad():
                    approx_kl = torch.mean(old_logp - new_logp).item()
                    clipfrac = torch.mean((torch.abs(ratios - 1.0) > self.clip_coef).float()).item()
                # Optional: print(f"KL={approx_kl:.4f} clipfrac={clipfrac:.2f} demo_reg={demo_reg.item():.4f}")
                self._updates_done += 1
                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

        return ep_rewards_all

# %% Data Utilities for bulk training (optional)
def train_per_date_and_save(sources: Dict[str, Dict[str, Dict[str, Any]]],
                            out_dir: str = "./ppo_models",
                            episodes_per_date: int = 200,
                            seed: int = 42):
    # === TUNING: seed for stability ===
    torch.manual_seed(seed); np.random.seed(seed)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    params = HydroParameters()

    for source_name, date_dict in sources.items():
        if len(date_dict) == 0:
            print(f"[{source_name}] No data; skipping.")
            continue
        for date_str, day_data in date_dict.items():
            print(f"\n=== Training PPO for {source_name} / {date_str} ===")
            env = PumpHydroEnv(params, day_data)
            obs_dim = env.reset().shape[0]
            policy = ActorCritic(obs_dim, action_dim=1, hidden_size=128, init_log_std=-1.0)
            trainer = PPOTrainer(
                policy, env,
                learning_rate=1e-4, gamma=0.99, lambda_gae=0.95,
                clip_coef=0.1, value_coef=0.7, entropy_coef=0.01,
                batch_size=5, update_epochs=3,
                target_kl=0.015, vf_clip=0.2, adv_clip=5.0, max_grad_norm=0.5,
                # === STAY-CLOSE REG ===
                expert_power=day_data['power'],
                stay_close_coef=0.3,
                device=device
            )
            trainer.warm_start_from_expert(day_data['power'], steps=800, lr=1e-3)
            rewards = trainer.train(num_episodes=episodes_per_date)
            print(f"[{source_name} / {date_str}] Mean ep reward over last 10 eps: {np.mean(rewards[-10:]) if len(rewards)>=10 else np.mean(rewards):.2f}")
            date_dir = out_root / source_name / date_str
            date_dir.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), date_dir / "ppo_policy.pt")
            try:
                plt.figure(figsize=(8,4))
                plt.plot(rewards)
                plt.xlabel("Episode")
                plt.ylabel("Episode Profit (reward)")
                plt.title(f"PPO Training Rewards: {source_name} / {date_str}")
                plt.tight_layout()
                plt.savefig(date_dir / "training_rewards.png")
                plt.close()
            except Exception as e:
                print(f"Plotting failed for {source_name} / {date_str}: {e}")

# %% Evaluation Helpers (unchanged)
def evaluate_policy_schedule(params: HydroParameters, day_data: Dict[str, Any], policy: ActorCritic, device: torch.device = device
                            ) -> Dict[str, Any]:
    env_eval = PumpHydroEnv(params, day_data)
    obs = env_eval.reset()
    TH = params.time_horizon
    p_commit = torch.zeros(TH, dtype=torch.float32, device=device)

    for _ in range(TH):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _ = policy.forward(obs_t)
        a_raw = float(mean.squeeze().item())
        mode = env_eval.mode_schedule[env_eval.current_hour]
        p_commit_t = env_eval._project_to_mode_and_upc(a_raw, mode)
        p_commit[env_eval.current_hour] = p_commit_t
        obs, _, done, _ = env_eval.step(a_raw)
        if done:
            break

    sim = SimulationLayer(params)
    p_exec, q_exec, h_exec, v_low_sim = sim.simulate_operation(p_commit)
    total_profit, SI_penalty, volume_penalty, operating_cost = sim.calc_profit(
        p_exec, p_commit, v_low_sim, day_data['price']
    )

    return {
        'p_commit': p_commit.detach().cpu().numpy().tolist(),
        'p_exec': p_exec.detach().cpu().numpy().tolist(),
        'profit': float(total_profit.item()),
        'SI_penalty': float(SI_penalty.item()),
        'volume_penalty': float(volume_penalty.item()),
        'operating_cost': float(operating_cost.item())
    }

def evaluate_miqp_schedule(params: HydroParameters, day_data: Dict[str, Any]) -> Dict[str, Any]:
    p_commit = day_data['power']
    sim = SimulationLayer(params)
    p_exec, q_exec, h_exec, v_low_sim = sim.simulate_operation(p_commit)
    total_profit, SI_penalty, volume_penalty, operating_cost = sim.calc_profit(
        p_exec, p_commit, v_low_sim, day_data['price']
    )
    return {
        'p_commit': p_commit.detach().cpu().numpy().tolist(),
        'p_exec': p_exec.detach().cpu().numpy().tolist(),
        'profit': float(total_profit.item()),
        'SI_penalty': float(SI_penalty.item()),
        'volume_penalty': float(volume_penalty.item()),
        'operating_cost': float(operating_cost.item())
    }

# %% Test & Evaluation block (runs when executing the script)
if __name__ == "__main__":
    sources = load_all_sources(noise_levels=[10,20,30,40,50,60,70,80], base_dir=".", verbose=True)

    chosen_source = None
    chosen_date = None
    for s, dd in sources.items():
        if len(dd) > 0:
            chosen_source = s
            chosen_date = list(dd.keys())[0]
            break
    if chosen_source is None:
        print("No sources/dates found. Please ensure the CSV files are present in the working directory.")
        sys.exit(0)

    params = HydroParameters()
    day_data = sources[chosen_source][chosen_date]

    # === TUNING: hyperparameters for stability ===
    learning_rate = 1e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_coef = 0.1
    value_coef = 0.7
    entropy_coef = 0.01
    batch_size = 5
    update_epochs = 3
    episodes_per_date = 50
    seed = 123
    target_kl = 0.015
    vf_clip = 0.2
    adv_clip = 5.0
    max_grad_norm = 0.5

    # === TUNING: seed for reproducibility ===
    torch.manual_seed(seed); np.random.seed(seed)

    print("\n================= PPO Training Details =================")
    print(f"Source/Date         : {chosen_source} / {chosen_date}")
    print(f"Episodes per date   : {episodes_per_date}")
    print(f"Learning rate       : {learning_rate}")
    print(f"Gamma               : {gamma}")
    print(f"GAE Lambda          : {lambda_gae}")
    print(f"Clip Coef (eps)     : {clip_coef}")
    print(f"Value Coef          : {value_coef}")
    print(f"Entropy Coef        : {entropy_coef}")
    print(f"Batch size          : {batch_size}")
    print(f"Update epochs       : {update_epochs}")
    print(f"Target KL           : {target_kl}")
    print(f"VF Clip             : {vf_clip}")
    print(f"Adv Clip            : {adv_clip}")
    print(f"Seed                : {seed}")
    print("========================================================\n")

    env = PumpHydroEnv(params, day_data)
    obs_dim = env.reset().shape[0]
    policy = ActorCritic(obs_dim, action_dim=1, hidden_size=128, init_log_std=-1.0)
    trainer = PPOTrainer(
        policy, env,
        learning_rate=learning_rate, gamma=gamma, lambda_gae=lambda_gae,
        clip_coef=clip_coef, value_coef=value_coef, entropy_coef=entropy_coef,
        batch_size=batch_size, update_epochs=update_epochs,
        target_kl=target_kl, vf_clip=vf_clip, adv_clip=adv_clip,
        max_grad_norm=max_grad_norm,
        # === STAY-CLOSE REG ===
        expert_power=day_data['power'],
        stay_close_coef=0.3,           # tighten to e.g. 0.15–0.2 for even closer tracking
        # stay_close_anneal=(0.15, 0.05),  # optional: enable linear decay over ~200 updates
        device=device
    )

    print("Warm-starting from MIQP expert trajectory...")
    trainer.warm_start_from_expert(day_data['power'], steps=800, lr=1e-3)
    print("Warm-start complete.\n")

    print("Starting PPO training...")
    rewards = trainer.train(num_episodes=episodes_per_date)
    print("Training complete.")
    if len(rewards) > 0:
        print(f"Episode reward (first 5): {[round(r,2) for r in rewards[:5]]}")
        tail = rewards[-10:] if len(rewards) >= 10 else rewards
        print(f"Episode reward (last 5) : {[round(r,2) for r in rewards[-5:]]}")
        print(f"Mean of last 10         : {np.mean(tail):.2f}")
        print(f"Best episode reward     : {np.max(rewards):.2f}")
        print(f"Worst episode reward    : {np.min(rewards):.2f}")

    out_dir = Path("./ppo_models_test") / chosen_source / chosen_date
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_dir / "ppo_policy.pt")

    print("\n================= Evaluation (Ex-Post) =================")
    miqp_eval = evaluate_miqp_schedule(params, day_data)
    ppo_eval = evaluate_policy_schedule(params, day_data, policy)

    print(f"MIQP Ex-Post Profit     : {miqp_eval['profit']:.2f}")
    print(f"  SI Penalty            : {miqp_eval['SI_penalty']:.2f}")
    print(f"  Volume Penalty        : {miqp_eval['volume_penalty']:.2f}")
    print(f"  Operating Cost        : {miqp_eval['operating_cost']:.2f}")

    print(f"\nPPO  Ex-Post Profit     : {ppo_eval['profit']:.2f}")
    print(f"  SI Penalty            : {ppo_eval['SI_penalty']:.2f}")
    print(f"  Volume Penalty        : {ppo_eval['volume_penalty']:.2f}")
    print(f"  Operating Cost        : {ppo_eval['operating_cost']:.2f}")

    diff = ppo_eval['profit'] - miqp_eval['profit']
    print(f"\nProfit Difference (PPO - MIQP): {diff:.2f}")
    print("========================================================\n")

    def fmt_sched(lst): 
        return "[" + ", ".join(f"{x:.2f}" for x in lst) + "]"

    print("---- Power Schedules (Committed, 24h) ----")
    print("MIQP:", fmt_sched(miqp_eval['p_commit']))
    print("PPO :", fmt_sched(ppo_eval['p_commit']))

    print("\n---- Power Schedules (Executed after Simulation, 24h) ----")
    print("MIQP:", fmt_sched(miqp_eval['p_exec']))
    print("PPO :", fmt_sched(ppo_eval['p_exec']))
    print("\nDone.")

    
    
# %% Benchmark SimulationLayer (exact copy from document 2)
class BenchmarkSimulationLayer:
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
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32, device=device)  # Remove extra volume
        
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

# %% Benchmark Evaluation Function
def evaluate_with_benchmark_sim(params: HydroParameters, day_data: Dict[str, Any], 
                                policy: ActorCritic = None, device: torch.device = device
                                ) -> Dict[str, Any]:
    """
    Evaluate either MIQP or PPO using the exact BenchmarkSimulationLayer.
    If policy is None, evaluates MIQP schedule. Otherwise evaluates PPO policy.
    """
    if policy is not None:
        # Generate PPO schedule
        env_eval = PumpHydroEnv(params, day_data)
        obs = env_eval.reset()
        TH = params.time_horizon
        p_commit = torch.zeros(TH, dtype=torch.float32, device=device)

        for _ in range(TH):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = policy.forward(obs_t)
            a_raw = float(mean.squeeze().item())
            mode = env_eval.mode_schedule[env_eval.current_hour]
            p_commit_t = env_eval._project_to_mode_and_upc(a_raw, mode)
            p_commit[env_eval.current_hour] = p_commit_t
            obs, _, done, _ = env_eval.step(a_raw)
            if done:
                break
    else:
        # Use MIQP schedule
        p_commit = day_data['power']

    # Use BenchmarkSimulationLayer for evaluation
    benchmark_sim = BenchmarkSimulationLayer(params)
    q_dummy = torch.zeros_like(p_commit)
    h_dummy = torch.zeros_like(p_commit)
    
    p_sim, q_sim, h_sim, v_low_sim = benchmark_sim.simulate_operation(p_commit, q_dummy, h_dummy)
    total_profit, SI_penalty, volume_penalty, operating_cost = benchmark_sim.calc_profit(
        p_sim, p_commit, v_low_sim, day_data['price']
    )

    return {
        'p_commit': p_commit.detach().cpu().numpy(),
        'p_exec': p_sim.detach().cpu().numpy(),
        'profit': float(total_profit.item()),
        'SI_penalty': float(SI_penalty.item()),
        'volume_penalty': float(volume_penalty.item()),
        'operating_cost': float(operating_cost.item())
    }

# %% Calculate Distance Metric
def calculate_distance_metric(p_ppo: np.ndarray, p_miqp: np.ndarray) -> float:
    """Calculate normalized Euclidean distance between PPO and MIQP schedules."""
    return float(np.linalg.norm(p_ppo - p_miqp) / np.sqrt(len(p_ppo)))

# %% Parse Database Info
def parse_database_info(database_name: str) -> Tuple[str, float]:
    """
    Parse database name to extract data type and noise level.
    Returns: (data_type, noise_level)
    """
    if "random_samples" in database_name:
        return "random_samples", 0.0
    elif "relative_noise" in database_name:
        # Extract noise percentage (e.g., "10pct" -> 0.1)
        parts = database_name.split("_")
        for part in parts:
            if "pct" in part:
                noise_pct = int(part.replace("pct", ""))
                return "relative_noise", noise_pct / 100.0
    return "unknown", 0.0

# %% Main Benchmark Runner
def run_comprehensive_benchmark(
    sources: Dict[str, Dict[str, Dict[str, Any]]],
    out_csv: str = "./ppo_comprehensive_benchmark.csv",
    out_models_dir: str = "./ppo_models_benchmark",
    episodes_per_date: int = 200,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run PPO training on all dates in all databases and generate comprehensive benchmark.
    """
    import time
    from datetime import datetime
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prepare output directory
    out_root = Path(out_models_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize results list
    results = []
    
    params = HydroParameters()
    
    # Loop through all sources
    for source_idx, (source_name, date_dict) in enumerate(sources.items()):
        if len(date_dict) == 0:
            if verbose:
                print(f"[{source_name}] No data; skipping.")
            continue
        
        data_type, noise_level = parse_database_info(source_name)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing Database [{source_idx+1}/{len(sources)}]: {source_name}")
            print(f"  Data Type: {data_type}, Noise Level: {noise_level:.2f}")
            print(f"  Total Dates: {len(date_dict)}")
            print(f"{'='*70}")
        
        # Loop through all dates
        for date_idx, (date_str, day_data) in enumerate(date_dict.items()):
            if verbose:
                print(f"\n[{date_idx+1}/{len(date_dict)}] Training for {date_str}...")
            
            start_time = time.time()
            
            # Create environment and policy
            env = PumpHydroEnv(params, day_data)
            obs_dim = env.reset().shape[0]
            policy = ActorCritic(obs_dim, action_dim=1, hidden_size=128, init_log_std=-1.0)
            
            # Create trainer with tuned hyperparameters
            trainer = PPOTrainer(
                policy, env,
                learning_rate=1e-4, gamma=0.99, lambda_gae=0.95,
                clip_coef=0.1, value_coef=0.7, entropy_coef=0.01,
                batch_size=5, update_epochs=3,
                target_kl=0.015, vf_clip=0.2, adv_clip=5.0, max_grad_norm=0.5,
                # === STAY-CLOSE REG ===
                expert_power=day_data['power'],
                stay_close_coef=0.3,
                device=device
            )
            
            # Warm start from expert
            trainer.warm_start_from_expert(day_data['power'], steps=800, lr=1e-3)
            
            # Train PPO
            rewards = trainer.train(num_episodes=episodes_per_date)
            
            training_time = time.time() - start_time
            
            # Save model
            date_dir = out_root / source_name / date_str
            date_dir.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), date_dir / "ppo_policy.pt")
            
            # Evaluate MIQP with BenchmarkSimulationLayer
            miqp_results = evaluate_with_benchmark_sim(params, day_data, policy=None)
            
            # Evaluate PPO with BenchmarkSimulationLayer
            ppo_results = evaluate_with_benchmark_sim(params, day_data, policy=policy)
            
            # Calculate distance metric
            distance = calculate_distance_metric(ppo_results['p_commit'], miqp_results['p_commit'])
            
            # Get timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare result dictionary for MIQP
            miqp_row = {
                'Database': source_name,
                'Data_Type': data_type,
                'Noise_Level': noise_level,
                'Date': date_str,
                'Method': 'MIQP',
                'Distance_Metric': 0.00,  # MIQP compared to itself
                'Expected_Profit': miqp_results['profit'],  # Before simulation (commitment)
                'Ex_post_Profit': miqp_results['profit'],  # After simulation
                'SI_Penalty': miqp_results['SI_penalty'],
                'Volume_Penalty': miqp_results['volume_penalty'],
                'Operating_Cost': miqp_results['operating_cost'],
                'Processing_Time_Seconds': 0.0,  # MIQP is pre-computed
                'Training_Episodes': 0,
                'Mean_Episode_Reward': 0.0,
                'Timestamp': timestamp
            }
            
            # Prepare result dictionary for PPO
            ppo_row = {
                'Database': source_name,
                'Data_Type': data_type,
                'Noise_Level': noise_level,
                'Date': date_str,
                'Method': 'PPO',
                'Distance_Metric': distance,
                'Expected_Profit': miqp_results['profit'],  # MIQP as expected
                'Ex_post_Profit': ppo_results['profit'],
                'SI_Penalty': ppo_results['SI_penalty'],
                'Volume_Penalty': ppo_results['volume_penalty'],
                'Operating_Cost': ppo_results['operating_cost'],
                'Processing_Time_Seconds': training_time,
                'Training_Episodes': episodes_per_date,
                'Mean_Episode_Reward': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
                'Timestamp': timestamp
            }
            
            results.append(miqp_row)
            results.append(ppo_row)
            
            if verbose:
                print(f"  MIQP Ex-post Profit: {miqp_results['profit']:.2f}")
                print(f"  PPO  Ex-post Profit: {ppo_results['profit']:.2f}")
                print(f"  Profit Difference: {ppo_results['profit'] - miqp_results['profit']:.2f}")
                print(f"  Distance Metric: {distance:.4f}")
                print(f"  Training Time: {training_time:.2f}s")
    
    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_csv, index=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Benchmark complete! Results saved to: {out_csv}")
        print(f"Total rows: {len(df_results)}")
        print(f"{'='*70}\n")
    
    return df_results

# %% Execute Comprehensive Benchmark
if __name__ == "__main__":
    # Load all sources
    print("\n" + "="*70)
    print("LOADING DATA SOURCES")
    print("="*70)
    sources = load_all_sources(
        noise_levels=[10, 20, 30, 40, 50, 60, 70, 80],
        base_dir=".",
        verbose=True
    )
    
    # Run comprehensive benchmark
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE BENCHMARK")
    print("="*70)
    
    benchmark_df = run_comprehensive_benchmark(
        sources=sources,
        out_csv="./ppo_comprehensive_benchmark.csv",
        out_models_dir="./ppo_models_benchmark",
        episodes_per_date=200,  # Adjust as needed
        seed=42,
        verbose=True
    )
    
    # Display summary statistics
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\nTotal evaluations: {len(benchmark_df)}")
    print(f"Databases processed: {benchmark_df['Database'].nunique()}")
    print(f"Dates processed: {benchmark_df['Date'].nunique()}")
    
    # Summary by method
    print("\n--- Summary by Method ---")
    summary = benchmark_df.groupby('Method').agg({
        'Ex_post_Profit': ['mean', 'std', 'min', 'max'],
        'SI_Penalty': 'mean',
        'Volume_Penalty': 'mean',
        'Processing_Time_Seconds': 'mean'
    }).round(2)
    print(summary)
    
    # Show first few rows
    print("\n--- First 5 Rows of Benchmark ---")
    print(benchmark_df.head().to_string())
    
    print("\n" + "="*70)
    print("ALL PROCESSING COMPLETE")
    print("="*70)

# %%
