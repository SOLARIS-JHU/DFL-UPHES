#!/usr/bin/env python3
"""
Full Pretraining Script for Piecewise (PW) Variant

This script reproduces the PW pretraining process using the refactored
DFL framework and data under DFL/outputs/noisy_data.

It trains models for:
- Noise levels: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%
- Random samples dataset
- Architecture: LSTM with 3 layers
- Max iterations: [1, 5]
- Parallel processing with 20 jobs
"""

import sys
import os
import argparse
import numpy as np
import torch
import itertools
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed

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

# Import refactored components
from DFL.config.pw_config import PWConfig
from DFL.utils.helpers import (
    setup_device, load_portfolio_data,
    load_preprocessed_data, initialize_head_and_volume
)
from DFL.core.parameters import HydroParameters
from DFL.data.loaders import load_data_for_pretraining
from DFL.training.trainer import train_single_model


def main():
    """Main entry point for PW pretraining."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train DFL models for PW variant')
    parser.add_argument('--n-jobs', type=int, default=20,
                        help='Number of parallel jobs for training')
    args = parser.parse_args()

    print("=" * 80)
    print("DFL Full Pretraining - Piecewise (PW) Variant")
    print("=" * 80)

    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Started at: {start_timestamp}\n")

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

    # 5. Create base configuration (PW)
    config = PWConfig()
    print(f"\nBase Configuration: PW variant")
    print(f"  Data pattern: {config.data_file_pattern}")
    print(f"  Architecture: {config.architecture}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Output directory: {config.output_base_dir}")

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

    # 7. Define grid search parameters
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Excludes 0% noise
    architectures = ['LSTM']
    num_layers_list = [3]
    max_iterations_list = [7]  # Optimal iteration count selected from validation results

    # 8. Prepare all training jobs
    all_jobs = []

    print("\nPreparing training jobs...")

    # Process noise level databases
    for noise_level in noise_levels:
        # Load from generated noisy data directory
        noisy_data_path = config.get_data_file_pattern(noise_level=noise_level)
        # Use the source data name for organizing output models (for consistency with validation)
        source_data_path = config.get_data_file_pattern(noise_level=noise_level)
        db_name = os.path.splitext(os.path.basename(source_data_path))[0]

        # Load historical data from noisy files
        historical_data = load_data_for_pretraining(
            noisy_data_path, db_name, config, device
        )

        if not historical_data:
            print(f"Warning: Could not load data for noise level {noise_level}")
            continue

        # Create output directory
        root_dir = Path(config.output_base_dir) / db_name
        root_dir.mkdir(exist_ok=True, parents=True)

        print(f"  Noise {int(noise_level*100)}%: {len(historical_data)} dates loaded")

        for architecture, num_layers, max_iterations in itertools.product(
                architectures, num_layers_list, max_iterations_list):

            for date_str, date_data in historical_data.items():
                all_jobs.append((
                    config, architecture, num_layers, max_iterations,
                    date_str, date_data, params, device, db_name
                ))

    # Process random samples database
    noisy_random_file = config.get_data_file_pattern(random_samples=True)
    source_random_file = config.get_data_file_pattern(random_samples=True)
    db_name = os.path.splitext(os.path.basename(source_random_file))[0]

    random_samples_data = load_data_for_pretraining(
        noisy_random_file, db_name, config, device
    )

    if random_samples_data:
        root_dir = Path(config.output_base_dir) / db_name
        root_dir.mkdir(exist_ok=True, parents=True)

        print(f"  Random samples: {len(random_samples_data)} dates loaded")

        for architecture, num_layers, max_iterations in itertools.product(
                architectures, num_layers_list, max_iterations_list):

            for date_str, date_data in random_samples_data.items():
                all_jobs.append((
                    config, architecture, num_layers, max_iterations,
                    date_str, date_data, params, device, db_name
                ))

    print(f"\nTotal training jobs: {len(all_jobs)}")
    print(f"Configurations: {len(noise_levels) + 1} databases × {len(architectures)} archs × {len(num_layers_list)} layers × {len(max_iterations_list)} iters")

    # 9. Run in parallel
    print(f"\nStarting parallel training with {args.n_jobs} workers...")
    results = Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(train_single_model)(*job) for job in all_jobs
    )

    # 10. Report summary
    successful = sum(1 for r in results if r and r.get('success', False))
    print(f"\n{'=' * 80}")
    print(f"Pretraining completed!")
    print(f"  Successful: {successful}/{len(all_jobs)}")
    print(f"  Models saved to: {config.output_base_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
