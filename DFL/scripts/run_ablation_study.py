#!/usr/bin/env python3
"""
Full Ablation Study Script (No Neural Network)

This script reproduces the exact ablation study from DFL_no-NN/
using the refactored DFL framework.

It runs validation WITHOUT neural network using fixed weights for:
- Noise levels: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%
- Random samples dataset
- Fixed weights: w_p=0.1, w_q=0.01, w_h=0.05
- Max iterations: [1, 5]
"""

import sys
sys.path.append('..')

from DFL.config.ablation_config import AblationConfig
from DFL.utils.helpers import (
    setup_device, load_portfolio_data,
    load_preprocessed_data, initialize_head_and_volume
)
from DFL.core.parameters import HydroParameters
from DFL.validation.validator import ablation_validation


def main():
    """Main entry point for ablation study."""
    print("=" * 80)
    print("DFL Ablation Study - No Neural Network (Baseline)")
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

    # 5. Create configuration (Ablation)
    config = AblationConfig()
    print(f"\nConfiguration: Ablation Study (No NN)")
    print(f"  Data pattern: {config.data_file_pattern}")
    print(f"  Neural network: {config.use_neural_network}")
    print(f"  Fixed weights: w_p={config.fixed_w_p}, w_q={config.fixed_w_q}, w_h={config.fixed_w_h}")

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

    # 7. Run ablation validation
    print("\nStarting ablation validation...")
    print("This will validate using FIXED weights (no learning)")

    ablation_validation(
        config=config,
        params=params,
        device=device,
        new_price_file="../Data/price_data_2024.csv"
    )

    print("\n" + "=" * 80)
    print("Ablation study completed!")
    print("Results saved to: ./validation_results/ablation_study/")
    print("=" * 80)


if __name__ == "__main__":
    main()
