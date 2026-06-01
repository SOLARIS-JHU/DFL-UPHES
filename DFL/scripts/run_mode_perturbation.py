#!/usr/bin/env python3
"""
Mode-Perturbation Experiment (Reviewer 3, Round 2, Issue 1)
===========================================================

Tests DFL robustness when the warm-start mode sequence is INCORRECT.

The discrete operating mode at each hour is fixed by the sign of the warm-start
power (see DFL/core/layers.py OptiLayer): p==0 -> idle, p>0 -> turbine,
p<0 -> pump. DFL locks this mode for all K iterations and cannot correct it.
Here we deliberately corrupt the warm-start mode sequence and measure the
resulting ex-post profit, using the clean MIQP-PW reference schedule of each of
the 19 representative days as the (correct) starting point.

Flips are chosen by "most uncertain commitment", NOT uniformly at random:

  * idle2active   : activate the rho most temptingly-priced idle hours into the
                    economically-consistent mode (turbine at high price, pump at
                    low price), with power at that mode's head-dependent minimum.
  * active2idle   : de-commit the rho smallest-|p| active hours (closest to the
                    p_min floor = most marginal commitments) to idle.
  * sign          : reverse the sign (turbine<->pump) of the rho lowest-|p|
                    active hours (least-bad sign reversals).

Everything runs SOLVER-FREE (no Gurobi): it reuses each date's pretrained LSTM
penalty predictor and the differentiable QP / simulation layers.

Usage:
    python DFL/scripts/run_mode_perturbation.py --flip-type idle2active
    python DFL/scripts/run_mode_perturbation.py --flip-type idle2active --rhos 1,2,3,4,5 --seeds 3
"""

import sys
import os
import argparse
import csv
import contextlib
import numpy as np
import pandas as pd
import torch


@contextlib.contextmanager
def suppress_stdout_fd():
    """Silence file-descriptor-level stdout (ECOS solver is hard-coded verbose)."""
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pathlib import Path

from DFL.config.pw_config import PWConfig
from DFL.utils.helpers import (
    setup_device, load_portfolio_data,
    load_preprocessed_data, initialize_head_and_volume,
)
from DFL.core.parameters import HydroParameters
from DFL.core.models import BoundedLogWeightPredictor
from DFL.core.layers import TaylorRegressionLayer, OptiLayer, SimulationLayer

# scipy/ECOS compatibility patch (mirrors validator.py)
import scipy.sparse
if hasattr(scipy.sparse, 'csc_array'):
    _csc = scipy.sparse.csc_array
    if not hasattr(_csc, 'get_shape'):
        _csc.get_shape = lambda self: self.shape

REFERENCE_CSV = "./MIQP/MIQP_piecewise/MIQP_piecewise_results.csv"
MODEL_DB = "MIQP_piecewise_results_random_samples"  # models trained on random sampling


def build_params(config, device):
    """Construct HydroParameters exactly as the validation scripts do."""
    portfolio = load_portfolio_data()
    preprocess_data = load_preprocessed_data()
    head_init, v_low_init = initialize_head_and_volume(
        preprocess_data['h_to_v_low_fitted'], device
    )
    params = HydroParameters(
        time_horizon=config.time_horizon,
        sampling_rate=config.sampling_rate,
        δ_p=config.δ_p, δ_h=config.δ_h, δ_q=config.δ_q,
        operational_cost=config.operational_cost,
        head_min=portfolio['head_min'], head_max=portfolio['head_max'],
        max_vol_up=portfolio['max_vol_up'], min_vol_low=portfolio['min_vol_low'],
        ramp_up=portfolio['ramp_up'], ramp_down=portfolio['ramp_down'],
        target_head=portfolio['target_head'], target_vol_low=portfolio['target_vol_low'],
        head_init=head_init, v_low_init=v_low_init,
        neg_min_fit=preprocess_data['neg_min_fit'], neg_max_fit=preprocess_data['neg_max_fit'],
        pos_min_fit=preprocess_data['pos_min_fit'], pos_max_fit=preprocess_data['pos_max_fit'],
        neg_min=preprocess_data['neg_min'], neg_max=preprocess_data['neg_max'],
        pos_min=preprocess_data['pos_min'], pos_max=preprocess_data['pos_max'],
        predict_q_poly=preprocess_data['predict_q_poly'],
        h_to_v_low_fitted=preprocess_data['h_to_v_low_fitted'],
        gross_head=portfolio['gross_head'],
        v_low_to_h_fitted=preprocess_data['v_low_to_h_fitted'],
        device=device,
    )
    return params


def load_reference_days(device):
    """Load the 19 clean MIQP-PW reference schedules (warm-start = correct mode)."""
    df = pd.read_csv(REFERENCE_CSV)
    days = {}
    for date_str, g in df.groupby('date'):
        g = g.sort_values('hour')
        days[date_str] = {
            'power': torch.tensor(g['power'].values[:24], dtype=torch.float32, device=device),
            'head':  torch.tensor(g['head'].values[:24],  dtype=torch.float32, device=device),
            'price': torch.tensor(g['price'].values[:24], dtype=torch.float32, device=device),
        }
    return days


def perturb_modes(power, head, price, params, flip_type, rho, rng):
    """
    Return a perturbed copy of `power` with `rho` hours' modes flipped.

    power, head, price: [24] tensors (clean MIQP-PW reference).
    Returns (perturbed_power, n_flipped).
    """
    p = power.clone()
    h = head
    pr = price
    TH = p.shape[0]
    tol = 0.5  # active threshold used throughout the codebase

    active_idx = [t for t in range(TH) if abs(p[t].item()) > tol]
    idle_idx = [t for t in range(TH) if abs(p[t].item()) <= tol]

    if flip_type == 'idle2active':
        # Most "temptingly priced" idle hours: extreme prices (far from median)
        # are where committing is most tempting / ambiguous.
        med = torch.median(pr).item()
        cand = sorted(idle_idx, key=lambda t: -abs(pr[t].item() - med))
        chosen = cand[:rho]
        for t in chosen:
            if pr[t].item() >= med:
                # high price -> turbine, power at head-dependent minimum
                p[t] = params.pos_min(h[t])
            else:
                # low price -> pump (negative power), at minimum magnitude
                p[t] = params.neg_min(h[t])
        return p, len(chosen)

    if flip_type == 'active2idle':
        # Smallest-|p| active hours = most marginal commitments.
        cand = sorted(active_idx, key=lambda t: abs(p[t].item()))
        chosen = cand[:rho]
        for t in chosen:
            p[t] = torch.zeros_like(p[t])
        return p, len(chosen)

    if flip_type == 'sign':
        # Lowest-|p| active hours, reverse sign (turbine<->pump).
        cand = sorted(active_idx, key=lambda t: abs(p[t].item()))
        chosen = cand[:rho]
        for t in chosen:
            p[t] = -p[t]
        return p, len(chosen)

    raise ValueError(f"unknown flip_type {flip_type}")


def run_dfl(power_init, head_init_traj, price, params, config, model_path, device):
    """
    Run the solver-free DFL inner loop for one warm-start and return ex-post profit.
    Mirrors validate_single_configuration's core loop.
    """
    power_init = power_init.clone().detach()
    head_init = head_init_traj.clone().detach()
    flow_init = params.predict_q_poly(power_init.unsqueeze(0), head_init.unsqueeze(0)).squeeze(0)

    # Load this date's pretrained penalty predictor
    net = BoundedLogWeightPredictor(
        input_size=4, hidden_size=config.hidden_size, num_layers=config.num_layers,
        dropout=config.dropout, time_horizon=params.time_horizon,
        archetype=config.architecture,
        init_w_p=config.init_w_p, init_w_q=config.init_w_q, init_w_h=config.init_w_h,
        w_p_min=config.w_p_min, w_p_max=config.w_p_max,
        w_q_min=config.w_q_min, w_q_max=config.w_q_max,
        w_h_min=config.w_h_min, w_h_max=config.w_h_max,
    ).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    x = torch.stack([price, power_init, flow_init, head_init], dim=1)
    with torch.no_grad():
        log_w_p, log_w_q, log_w_h = net(x)
        w_p, w_q, w_h = torch.exp(log_w_p), torch.exp(log_w_q), torch.exp(log_w_h)

    regression_layer = TaylorRegressionLayer(params)
    optimizer_layer = OptiLayer(params)

    p_cur = power_init.clone().detach()
    h_cur = head_init.clone().detach()
    q_cur = flow_init.clone().detach()

    p_opt = q_opt = h_opt = None
    for it in range(config.max_iterations):
        gf = config.penalty_growth_rate ** it
        c, d, e, a, b = regression_layer.run_regression(p_cur, h_cur, q_cur)
        optimizer_layer.initialize_layer(p_cur.cpu(), h_cur.cpu(), q_cur.cpu())
        p_opt, q_opt, h_opt, v_opt, exp_profit, _ = optimizer_layer.forward(
            price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
            p_cur.cpu(), h_cur.cpu(), q_cur.cpu(),
            (w_p * gf).cpu(), (w_h * gf).cpu(), (w_q * gf).cpu(),
        )
        if it < config.max_iterations - 1:
            p_cur = p_opt.clone().detach().to(p_cur.device)
            h_cur = h_opt.clone().detach().to(h_cur.device)
            q_cur = q_opt.clone().detach().to(q_cur.device)

    sim = SimulationLayer(params)
    p_sim, q_sim, h_sim, v_low_sim = sim.simulate_operation(
        p_opt.to(device), q_opt.to(device), h_opt.to(device)
    )
    ex_post, SI_pen, vol_pen, op_cost = sim.calc_profit(
        p_sim, p_opt.to(device), v_low_sim, price.to(device)
    )
    return dict(ex_post=ex_post.item(), SI=SI_pen.item(), vol=vol_pen.item())


def main():
    ap = argparse.ArgumentParser(description="Mode-perturbation robustness experiment")
    ap.add_argument('--flip-type', required=True,
                    choices=['idle2active', 'active2idle', 'sign'])
    ap.add_argument('--rhos', default='1,2,3,4,5',
                    help='comma-separated hour counts to flip')
    ap.add_argument('--seeds', type=int, default=3, help='random seeds per rho')
    ap.add_argument('--out', default=None, help='output CSV path')
    args = ap.parse_args()

    rhos = [int(x) for x in args.rhos.split(',')]
    device = setup_device()
    config = PWConfig()
    params = build_params(config, device)
    days = load_reference_days(device)

    out_path = args.out or f"./DFL/outputs/mode_perturbation_{args.flip_type}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # --- Baseline (rho=0): clean MIQP-PW warm-start, correct modes ---
    print(f"\n=== Baseline (rho=0), correct warm-start modes ===")
    base_profits = []
    for date_str, d in days.items():
        model_path = (Path(config.output_base_dir) / MODEL_DB /
                      f"{config.architecture}_{config.num_layers}layer_{config.max_iterations}iter" /
                      date_str / "best_model.pt")
        if not model_path.exists():
            print(f"  [skip] no model for {date_str}")
            continue
        with suppress_stdout_fd():
            r = run_dfl(d['power'], d['head'], d['price'], params, config, model_path, device)
        base_profits.append(r['ex_post'])
        rows.append(dict(flip_type=args.flip_type, rho=0, seed=-1, date=date_str,
                         n_flipped=0, **r))
    print(f"  baseline mean ex-post profit = {np.mean(base_profits):.1f} EUR "
          f"(n={len(base_profits)})")

    # --- Perturbation sweep ---
    for rho in rhos:
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 * rho + seed)
            profits = []
            for date_str, d in days.items():
                model_path = (Path(config.output_base_dir) / MODEL_DB /
                              f"{config.architecture}_{config.num_layers}layer_{config.max_iterations}iter" /
                              date_str / "best_model.pt")
                if not model_path.exists():
                    continue
                p_pert, n_flip = perturb_modes(d['power'], d['head'], d['price'],
                                               params, args.flip_type, rho, rng)
                with suppress_stdout_fd():
                    r = run_dfl(p_pert, d['head'], d['price'], params, config, model_path, device)
                profits.append(r['ex_post'])
                rows.append(dict(flip_type=args.flip_type, rho=rho, seed=seed,
                                 date=date_str, n_flipped=n_flip, **r))
            print(f"  rho={rho} seed={seed}: mean ex-post profit = "
                  f"{np.mean(profits):.1f} EUR (n={len(profits)})")

    # Write CSV
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    # Summary table by rho
    dfres = pd.DataFrame(rows)
    print("\n=== Summary: mean ex-post profit by rho ===")
    summ = dfres.groupby('rho')['ex_post'].agg(['mean', 'std', 'count'])
    print(summ.to_string())


if __name__ == "__main__":
    main()
