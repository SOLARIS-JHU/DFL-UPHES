#!/usr/bin/env python3
"""
Penalty Sensitivity Sweep (Reviewer 3, Round 2, Issue 2)
========================================================

For each penalty cell:
  * retrain the headline DFL-PW (random-samples / LSTM-3layer / 7-iter),
  * validate it (solver-free) -> DFL-PW ex-post profit + SI/Vol penalties,
  * re-score the saved MIQP-PW schedules under the SAME penalty,
  * write one summary row.

Both sides are scored under identical penalties per cell, so the relative ranking
and margin are the honest comparison. MIQP is re-scored only (no SI/volume terms in
its objective); DFL is retrained (the penalties are its training loss).

Run baseline FIRST and verify it reproduces tab:main_results before the rest.

Usage:
    python DFL/scripts/run_penalty_sensitivity.py --cells baseline
    python DFL/scripts/run_penalty_sensitivity.py --cells all --n-jobs 8
"""
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

PENALTY_CELLS = {
    "baseline":     dict(si_shortage_mult=-2.0, si_surplus_mult=-0.5, vol_water_value_mult=1.0),
    "si_symmetric": dict(si_shortage_mult=-1.0, si_surplus_mult=-1.0, vol_water_value_mult=1.0),
    "si_mild":      dict(si_shortage_mult=-1.5, si_surplus_mult=-0.75, vol_water_value_mult=1.0),
    "vol_low":      dict(si_shortage_mult=-2.0, si_surplus_mult=-0.5, vol_water_value_mult=0.8),
    "vol_high":     dict(si_shortage_mult=-2.0, si_surplus_mult=-0.5, vol_water_value_mult=1.2),
}

import numpy as np
import torch
import pandas as pd
from pathlib import Path

from DFL.config.pw_config import PWConfig
from DFL.utils.helpers import (
    setup_device, load_portfolio_data,
    load_preprocessed_data, initialize_head_and_volume,
)
from DFL.core.parameters import HydroParameters
from DFL.scripts.rescore_miqp_penalties import rescore_miqp_file

OUT_ROOT = Path("./DFL/outputs/penalty_sensitivity")
MIQP_PW_RESULTS = "MIQP/MIQP_piecewise/MIQP_piecewise_results.csv"


def build_context(cell, device):
    """Return (config, params) wired for a penalty cell.

    Config is restricted to the headline DFL-PW: random-samples training set,
    LSTM / 3 layers / 7 iterations. Output + results dirs are redirected under
    OUT_ROOT/<cell> so the main trained_models/ tree is untouched.
    """
    portfolio = load_portfolio_data()
    preprocess_data = load_preprocessed_data()
    head_init, v_low_init = initialize_head_and_volume(
        preprocess_data['h_to_v_low_fitted'], device)

    config = PWConfig()
    config.architecture = 'LSTM'
    config.num_layers = 3
    config.max_iterations = 7
    config.use_neural_network = True

    params = HydroParameters(
        time_horizon=config.time_horizon,
        sampling_rate=config.sampling_rate,
        δ_p=config.δ_p, δ_h=config.δ_h, δ_q=config.δ_q,
        operational_cost=config.operational_cost,
        si_shortage_mult=cell["si_shortage_mult"],
        si_surplus_mult=cell["si_surplus_mult"],
        vol_water_value_mult=cell["vol_water_value_mult"],
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
    return config, params


# The RS database name MUST equal Path(get_data_file_pattern(random_samples=True)).stem,
# because comprehensive_validation derives db_name that way and looks for models at
# output_base_dir/db_name/config_name/<date>/best_model.pt. train_single_model(...,
# db_name=RS_DB) saves to exactly that path. pretraining_single_noise_level uses the
# WRONG dir name ("random_samples"), so we drive train_single_model directly.
from joblib import Parallel, delayed
from DFL.data.loaders import load_data_for_pretraining
from DFL.training.trainer import train_single_model

RS_DB = "MIQP_piecewise_results_random_samples"


def train_dfl_for_cell(cell_name, config, params, device, n_jobs):
    """Retrain the headline DFL-PW (random samples only) under the cell penalties.

    Saves models to the exact path comprehensive_validation expects.
    """
    config.output_base_dir = str(OUT_ROOT / cell_name / "trained_models")
    config.results_base_dir = str(OUT_ROOT / cell_name / "validation_results")
    Path(config.output_base_dir).mkdir(parents=True, exist_ok=True)

    rs_file = config.get_data_file_pattern(random_samples=True)
    assert Path(rs_file).stem == RS_DB, f"RS db name mismatch: {Path(rs_file).stem}"
    historical = load_data_for_pretraining(rs_file, RS_DB, config, device)
    if not historical:
        raise RuntimeError(f"No RS training data loaded from {rs_file}")

    Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(train_single_model)(
            config, config.architecture, config.num_layers, config.max_iterations,
            date_str, date_data, params, device, RS_DB)
        for date_str, date_data in historical.items())


def validate_dfl_for_cell(config, params, device, price_file):
    from DFL.validation.validator import comprehensive_validation
    comprehensive_validation(config=config, params=params, device=device,
                             new_price_file=price_file)


def aggregate_dfl_profit(config):
    """Mean/std ex-post profit + mean SI/Vol penalties from the cell's RS LSTM run."""
    src = Path(config.results_base_dir) / RS_DB \
        / config.get_model_config_name() / "scheduling_benchmarks.csv"
    # The CSV HAS a header row (validator writes New_Date, ...). Read by name.
    df = pd.read_csv(src)
    ex_post = df["Ex_post_Profit"].astype(float)
    si = df["SI_Penalty"].astype(float)
    vol = df.iloc[:, 6].astype(float)   # Volume_Penalty column (7th)
    t = df.iloc[:, 8].astype(float)     # Processing_Time_Seconds column (9th)
    return dict(dfl_profit_mean=ex_post.mean(), dfl_profit_std=ex_post.std(),
                dfl_si=si.mean(), dfl_vol=vol.mean(), dfl_time=t.mean(),
                dfl_n=len(ex_post))


def run_cell(cell_name, device, price_file, n_jobs):
    cell = PENALTY_CELLS[cell_name]
    config, params = build_context(cell, device)
    train_dfl_for_cell(cell_name, config, params, device, n_jobs)
    validate_dfl_for_cell(config, params, device, price_file)
    dfl = aggregate_dfl_profit(config)

    miqp_df = rescore_miqp_file(MIQP_PW_RESULTS, params, cell)
    row = dict(cell=cell_name, **cell, **dfl,
               miqp_profit_mean=miqp_df["ex_post_profit"].mean(),
               miqp_profit_std=miqp_df["ex_post_profit"].std(),
               miqp_si=miqp_df["SI_penalty"].mean(),
               miqp_vol=miqp_df["volume_penalty"].mean())
    row["gap_pct"] = 100.0 * (row["dfl_profit_mean"] - row["miqp_profit_mean"]) \
        / row["miqp_profit_mean"]
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="baseline",
                    help="'all', or comma-separated cell names")
    ap.add_argument("--price-file", default="./Data/price_data_2024.csv")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    np.random.seed(42); torch.manual_seed(42)
    device = setup_device()

    names = list(PENALTY_CELLS) if args.cells == "all" else args.cells.split(",")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        print(f"\n{'='*70}\nPenalty cell: {name}\n{'='*70}")
        rows.append(run_cell(name, device, args.price_file, args.n_jobs))

    summary = OUT_ROOT / "summary.csv"
    df = pd.DataFrame(rows)
    if summary.exists():
        prev = pd.read_csv(summary)
        df = pd.concat([prev[~prev["cell"].isin(df["cell"])], df], ignore_index=True)
    df.to_csv(summary, index=False)
    print(f"\nWrote {summary}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
