#!/usr/bin/env python3
"""
Generate noisy training datasets from MIQP results.

This script generates multiple noisy variants of the original MIQP optimization results:
1. Relative noise datasets: 0%, 10%, 20%, ..., 80% noise
2. Random samples dataset: Random power within feasible ranges preserving modes

Usage:
    python generate_noisy_data.py --variant GL  # For Global Linear variant
    python generate_noisy_data.py --variant PW  # For Piecewise variant
"""

import sys
import argparse
sys.path.append('..')

# Import configurations
from DFL.config.gl_config import GLConfig
from DFL.config.pw_config import PWConfig

# Import utilities
from DFL.utils.helpers import setup_device, load_portfolio_data, load_preprocessed_data, initialize_head_and_volume

# Import core components
from DFL.core.parameters import HydroParameters

# Import noise generation
from DFL.data.noise import generate_noisy_dataset, generate_random_samples_dataset


def main():
    """Main entry point for noise generation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate noisy training datasets')
    parser.add_argument('--variant', type=str, choices=['GL', 'PW'], default='PW',
                        help='Variant to use (GL=Global Linear, PW=Piecewise)')
    parser.add_argument('--noise-levels', type=str, default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8',
                        help='Comma-separated noise levels (e.g., "0.1,0.2,0.3")')
    parser.add_argument('--random-samples', action='store_true',
                        help='Also generate random samples dataset')
    args = parser.parse_args()

    print("="*80)
    print(f"DFL Noise Generation - {args.variant} Variant")
    print("="*80)

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

    # 5. Create configuration based on variant
    if args.variant == 'GL':
        config = GLConfig()
    else:  # PW
        config = PWConfig()

    print(f"\nConfiguration: {config}")
    print(f"Original MIQP file: {config.get_miqp_file_path()}")

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

    # 7. Parse noise levels
    noise_levels = [float(x) for x in args.noise_levels.split(',')]
    print(f"\nNoise levels to generate: {[f'{int(x*100)}%' for x in noise_levels]}")

    # 8. Generate noisy datasets
    print("\n" + "="*80)
    print("Generating noisy datasets...")
    print("="*80)

    generate_noisy_dataset(
        config=config,
        params=params,
        device=device,
        noise_levels=noise_levels
    )

    # 9. Generate random samples dataset if requested
    if args.random_samples:
        print("\n" + "="*80)
        print("Generating random samples dataset...")
        print("="*80)

        generate_random_samples_dataset(
            config=config,
            params=params,
            device=device
        )

    print("\n" + "="*80)
    print("Noise generation completed!")
    print("="*80)
    print("\nGenerated files:")
    for noise_level in noise_levels:
        filename = config.get_data_file_pattern(noise_level=noise_level)
        print(f"  - {filename}")

    if args.random_samples:
        filename = config.get_data_file_pattern(random_samples=True)
        print(f"  - {filename}")

    print("\nYou can now use these files for pretraining:")
    print("  python example_pretraining_gl.py")
    print("  python example_pretraining_pw.py")


if __name__ == "__main__":
    main()
