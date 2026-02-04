#!/usr/bin/env python3
"""
Validation Script for Piecewise No-Recursion (PW-no-Rec) Variant

This script validates the PW-no-Rec variant (1 max iteration with neural network)
using the refactored DFL framework.

It validates trained models for:
- Noise levels: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%
- Random samples dataset
- Architecture: LSTM with 3 layers
- Max iterations: 1 (no recursive refinement)
- On new price scenarios from 2024
"""

import sys
import os
import argparse
import numpy as np
import torch

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

from DFL.config.pw_norec_config import PWNoRecConfig
from DFL.utils.helpers import (
    setup_device, load_portfolio_data,
    load_preprocessed_data, initialize_head_and_volume
)
from DFL.core.parameters import HydroParameters
from DFL.validation.validator import comprehensive_validation


def main():
    """Main entry point for PW-no-Rec validation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Validate DFL models for PW-no-Rec variant')
    parser.add_argument('--price-file', type=str, default='./Data/price_data_2024.csv',
                        help='Path to price data for validation')
    args = parser.parse_args()

    print("=" * 80)
    print("DFL Validation - Piecewise No-Recursion (PW-no-Rec) Variant")
    print("=" * 80)

    # 1. Setup device
    device = setup_device()
    print(f"Using device: {device}")

    # 2. Load portfolio data
    print("\nLoading portfolio data...")
    portfolio = load_portfolio_data()
    if portfolio is None:
        print("Error: Could not load portfolio data")
        return

    # 3. Load preprocessed data
    print("Loading preprocessed data...")
    preprocess_data = load_preprocessed_data()
    if preprocess_data is None:
        print("Error: Could not load preprocessed data")
        return

    # 4. Initialize head and volume
    head_init, v_low_init = initialize_head_and_volume(
        preprocess_data['h_to_v_low_fitted'], device
    )

    # 5. Create configuration (PW-no-Rec)
    config = PWNoRecConfig()
    print(f"\nConfiguration: PW-no-Rec variant")
    print(f"  Data pattern: {config.data_file_pattern}")
    print(f"  Architecture: {config.architecture}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Model directory: {config.output_base_dir}")
    print(f"  Results directory: {config.results_base_dir}")

    # 6. Initialize HydroParameters
    params = HydroParameters(
        time_horizon=config.time_horizon,
        sampling_rate=config.sampling_rate,
        δ_p=config.δ_p,
        δ_h=config.δ_h,
        δ_q=config.δ_q,
        operational_cost=config.operational_cost,
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

    # 7. Run comprehensive validation
    print("\nStarting comprehensive validation...")
    print("This will validate all trained models on new price scenarios")

    comprehensive_validation(
        config=config,
        params=params,
        device=device,
        new_price_file=args.price_file
    )

    print("\n" + "=" * 80)
    print("Validation completed!")
    print(f"Results saved to: {config.results_base_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
