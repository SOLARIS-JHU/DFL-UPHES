#!/usr/bin/env python3
"""
IPOPT NLP Comparison Script

Solves the fixed-mode continuous NLP using IPOPT with true nonlinear constraints
(no Taylor approximation). Uses MIQP solutions as warm-starts to fix integer
modes and initialize continuous variables.

This benchmarks DFL's linearization quality by comparing against the exact NLP
solution for the same mode schedule.

Usage:
    python DFL/scripts/run_ipopt_comparison.py
    python DFL/scripts/run_ipopt_comparison.py --miqp-variant PW
    python DFL/scripts/run_ipopt_comparison.py --miqp-variant GL
    python DFL/scripts/run_ipopt_comparison.py --miqp-variant both --tee
    python DFL/scripts/run_ipopt_comparison.py --price-file ./custom_prices.csv
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path

# Add repository root to Python path to enable DFL imports
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

from DFL.config.ipopt_config import IPOPTConfigPW, IPOPTConfigGL
from DFL.utils.helpers import (
    setup_device, load_portfolio_data,
    load_preprocessed_data, initialize_head_and_volume
)
from DFL.core.parameters import HydroParameters
from DFL.core.ipopt_solver import check_ipopt_available
from DFL.data.loaders import load_data_for_validation, load_new_price_data
from DFL.validation.ipopt_validator import validate_ipopt_scenarios


def run_ipopt_for_config(config, params, preprocess_data, device, price_file, tee=False):
    """
    Run IPOPT validation for a single configuration (PW or GL).

    Args:
        config: IPOPTConfigPW or IPOPTConfigGL instance
        params: HydroParameters instance
        preprocess_data: Preprocessed data dictionary
        device: PyTorch device
        price_file: Path to validation price data
        tee: Whether to show IPOPT solver output
    """
    # Load new price data
    new_price_data = load_new_price_data(price_file, device)
    if not new_price_data:
        print("Error: Could not load new price data")
        return

    # Only run on random samples dataset
    databases = ['random_samples']

    total = len(databases)

    for db_idx, db_source in enumerate(databases):
        if db_source == 'random_samples':
            file_path = config.get_data_file_pattern(random_samples=True)
            db_name = Path(file_path).stem
        else:
            file_path = config.get_data_file_pattern(noise_level=db_source)
            db_name = Path(file_path).stem

        print(f"\n{'=' * 80}")
        print(f"[{db_idx + 1}/{total}] IPOPT-NLP: {db_name}")
        print(f"{'=' * 80}")

        # Load historical data
        historical_data = load_data_for_validation(file_path, db_name, config, device)
        if not historical_data:
            print(f"Warning: Could not load historical data for {db_name}")
            continue

        # Run IPOPT validation
        validate_ipopt_scenarios(
            config=config,
            params=params,
            preprocess_data=preprocess_data,
            device=device,
            new_price_data=new_price_data,
            historical_data=historical_data,
            db_name=db_name,
            tee=tee
        )


def main():
    """Main entry point for IPOPT comparison."""
    parser = argparse.ArgumentParser(
        description='Run IPOPT NLP comparison against DFL methods'
    )
    parser.add_argument('--price-file', type=str, default='./Data/price_data_2024.csv',
                        help='Path to price data for validation')
    parser.add_argument('--miqp-variant', type=str, default='both',
                        choices=['PW', 'GL', 'both'],
                        help='Which MIQP warm-starts to use (default: both)')
    parser.add_argument('--tee', action='store_true',
                        help='Show IPOPT solver output')
    args = parser.parse_args()

    print("=" * 80)
    print("IPOPT NLP Comparison for UPHES Scheduling")
    print("=" * 80)

    # 1. Check IPOPT availability
    if not check_ipopt_available():
        print("\nERROR: IPOPT solver is not available.")
        print("Install via: conda install -c conda-forge ipopt")
        print("Or download from: https://github.com/coin-or/Ipopt")
        sys.exit(1)
    print("IPOPT solver: available")

    # 2. Setup device
    device = setup_device()
    print(f"Using device: {device}")

    # 3. Load portfolio data
    print("\nLoading portfolio data...")
    portfolio = load_portfolio_data()
    if portfolio is None:
        print("Error: Could not load portfolio data")
        return

    # 4. Load preprocessed data
    print("Loading preprocessed data...")
    preprocess_data = load_preprocessed_data()
    if preprocess_data is None:
        print("Error: Could not load preprocessed data")
        return

    # 5. Initialize head and volume
    head_init, v_low_init = initialize_head_and_volume(
        preprocess_data['h_to_v_low_fitted'], device
    )

    # 6. Initialize HydroParameters (shared across both configs)
    params = HydroParameters(
        time_horizon=24,
        sampling_rate=50,
        δ_p=0.5,
        δ_h=1.0,
        δ_q=0.5,
        operational_cost=0.4,
        head_min=portfolio['head_min'],
        head_max=portfolio['head_max'],
        max_vol_up=portfolio['max_vol_up'],
        min_vol_low=portfolio['min_vol_low'],
        ramp_up=portfolio['ramp_up'],
        ramp_down=portfolio['ramp_down'],
        target_head=portfolio['target_head'],
        target_vol_low=portfolio['target_vol_low'],
        head_init=head_init,
        v_low_init=v_low_init,
        neg_min_fit=preprocess_data['neg_min_fit'],
        neg_max_fit=preprocess_data['neg_max_fit'],
        pos_min_fit=preprocess_data['pos_min_fit'],
        pos_max_fit=preprocess_data['pos_max_fit'],
        neg_min=preprocess_data['neg_min'],
        neg_max=preprocess_data['neg_max'],
        pos_min=preprocess_data['pos_min'],
        pos_max=preprocess_data['pos_max'],
        predict_q_poly=preprocess_data['predict_q_poly'],
        h_to_v_low_fitted=preprocess_data['h_to_v_low_fitted'],
        gross_head=portfolio['gross_head'],
        v_low_to_h_fitted=preprocess_data['v_low_to_h_fitted'],
        device=device
    )

    # 7. Run IPOPT for selected variants
    if args.miqp_variant in ('PW', 'both'):
        print("\n" + "=" * 80)
        print("Running IPOPT-NLP with Piecewise (PW) MIQP warm-starts")
        print("=" * 80)
        config_pw = IPOPTConfigPW()
        run_ipopt_for_config(config_pw, params, preprocess_data, device,
                             args.price_file, tee=args.tee)

    if args.miqp_variant in ('GL', 'both'):
        print("\n" + "=" * 80)
        print("Running IPOPT-NLP with Global Linear (GL) MIQP warm-starts")
        print("=" * 80)
        config_gl = IPOPTConfigGL()
        run_ipopt_for_config(config_gl, params, preprocess_data, device,
                             args.price_file, tee=args.tee)

    print("\n" + "=" * 80)
    print("IPOPT NLP comparison completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
