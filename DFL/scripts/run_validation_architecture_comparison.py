#!/usr/bin/env python3
"""
Architecture Comparison Validation Script

Validates trained DFL weight predictors across multiple architectures
(LSTM, BILSTM, CNN, TRANSFORMER) for either the GL or PW variant.

Usage:
    python DFL/scripts/run_validation_architecture_comparison.py --variant GL
    python DFL/scripts/run_validation_architecture_comparison.py --variant PW
    python DFL/scripts/run_validation_architecture_comparison.py --variant GL --architectures LSTM,BILSTM
    python DFL/scripts/run_validation_architecture_comparison.py --variant PW --price-file ./custom.csv
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

from DFL.config.gl_config import GLConfig
from DFL.config.pw_config import PWConfig
from DFL.utils.helpers import (
    setup_device, load_portfolio_data,
    load_preprocessed_data, initialize_head_and_volume
)
from DFL.core.parameters import HydroParameters
from DFL.validation.validator import comprehensive_validation


def main():
    """Main entry point for architecture comparison validation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Validate DFL models across architectures')
    parser.add_argument('--variant', type=str, default='PW', choices=['GL', 'PW'],
                        help='MIQP variant: GL (global linear) or PW (piecewise)')
    parser.add_argument('--price-file', type=str, default='./Data/price_data_2024.csv',
                        help='Path to price data for validation')
    parser.add_argument('--architectures', type=str, default='LSTM,BILSTM,CNN,TRANSFORMER',
                        help='Comma-separated list of architectures to validate')
    args = parser.parse_args()

    architectures = [a.strip().upper() for a in args.architectures.split(',')]
    variant = args.variant.upper()

    print("=" * 80)
    print(f"DFL Architecture Comparison Validation - {variant} Variant")
    print("=" * 80)
    print(f"Variant: {variant}")
    print(f"Architectures: {architectures}\n")

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

    # 5. Initialize HydroParameters (shared across architectures)
    config_base = GLConfig() if variant == 'GL' else PWConfig()
    params = HydroParameters(
        time_horizon=config_base.time_horizon,
        sampling_rate=config_base.sampling_rate,
        δ_p=config_base.δ_p,
        δ_h=config_base.δ_h,
        δ_q=config_base.δ_q,
        operational_cost=config_base.operational_cost,
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

    # 6. Run validation for each architecture
    for arch in architectures:
        print(f"\n{'=' * 80}")
        print(f"Validating architecture: {arch}")
        print(f"{'=' * 80}")

        config = GLConfig() if variant == 'GL' else PWConfig()
        config.architecture = arch
        config.num_layers = 3
        config.max_iterations = 7

        print(f"  Config name: {config.get_model_config_name()}")
        print(f"  Model directory: {config.output_base_dir}")
        print(f"  Results directory: {config.results_base_dir}")

        comprehensive_validation(
            config=config,
            params=params,
            device=device,
            new_price_file=args.price_file
        )

    print(f"\n{'=' * 80}")
    print(f"Architecture comparison validation completed! ({variant} variant)")
    print(f"Architectures validated: {architectures}")
    print(f"Results saved to: {config_base.results_base_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
