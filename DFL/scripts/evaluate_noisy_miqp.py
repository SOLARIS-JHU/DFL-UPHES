#!/usr/bin/env python3
"""
Evaluate noisy MIQP schedules with the simulator layer.

This script:
1. Loads MIQP schedules optimized with noisy price data
2. Evaluates them using the SimulationLayer with clean validation prices
3. Computes ex-post profit, SI penalty, volume penalty, and operating cost
4. Saves results for visualization and comparison with DFL methods

The MIQP methods are optimized with noisy prices but evaluated with clean prices,
just like the DFL methods, to ensure fair comparison.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import argparse

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from DFL.core.parameters import HydroParameters
from DFL.core.layers import SimulationLayer
from DFL.utils.helpers import (
    setup_device, load_portfolio_data,
    load_preprocessed_data, initialize_head_and_volume
)


def standardize_date_format(date_str):
    """Convert various date formats to YYYY-MM-DD format."""
    date_str = str(date_str).strip()

    # Handle YYYY/M/D or YYYY/MM/DD format
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            year = parts[0].zfill(4)
            month = parts[1].zfill(2)
            day = parts[2].zfill(2)
            return f"{year}-{month}-{day}"

    # Handle YYYY-MM-DD format (already standardized)
    if '-' in date_str:
        parts = date_str.split('-')
        if len(parts) == 3:
            year = parts[0].zfill(4)
            month = parts[1].zfill(2)
            day = parts[2].zfill(2)
            return f"{year}-{month}-{day}"

    return date_str


def load_miqp_schedule(file_path):
    """Load MIQP schedule from noisy data file."""
    df = pd.read_csv(file_path)

    # Standardize date format
    df['date'] = df['date'].apply(standardize_date_format)

    # Group by date to get all schedules
    schedules = []
    for date, group in df.groupby('date'):
        if len(group) != 24:
            continue  # Skip incomplete days

        schedule = {
            'date': date,
            'power': torch.tensor(group['power'].values, dtype=torch.float32),
            'head': torch.tensor(group['head'].values, dtype=torch.float32),
            'volume': torch.tensor(group['volume'].values, dtype=torch.float32),
            'flow': torch.tensor(group['flow'].values, dtype=torch.float32),
            'noisy_price': torch.tensor(group['price'].values, dtype=torch.float32)
        }
        schedules.append(schedule)

    return schedules


def load_validation_prices(price_file):
    """Load clean validation prices."""
    df = pd.read_csv(price_file)

    prices_dict = {}
    for _, row in df.iterrows():
        date = row['date']
        prices = np.array([float(p) for p in row['prices_hourly'].split(',')])
        prices_dict[date] = torch.tensor(prices, dtype=torch.float32)

    return prices_dict


def evaluate_schedules(schedules, validation_prices, params, sim_layer):
    """Evaluate MIQP schedules with simulator using clean validation prices."""
    results = []

    for schedule in schedules:
        date = schedule['date']

        # Get clean validation price for this date
        if date not in validation_prices:
            continue

        clean_price = validation_prices[date]

        # Extract schedule
        p_opt = schedule['power']
        q_opt = schedule['flow']
        h_opt = schedule['head']

        # Simulate with true physical dynamics
        p_sim, q_sim, h_sim, v_sim = sim_layer.simulate_operation(p_opt, q_opt, h_opt)

        # Calculate ex-post profit with clean validation prices
        ex_post_profit, si_penalty, vol_penalty, op_cost = sim_layer.calc_profit(
            p_sim, p_opt, v_sim, clean_price
        )

        # Calculate expected profit (what MIQP optimized for with noisy prices)
        noisy_price = schedule['noisy_price']
        expected_revenue = torch.sum(noisy_price * p_opt)
        expected_profit = expected_revenue - op_cost

        results.append({
            'Date': date,
            'Ex_post_Profit': ex_post_profit.item(),
            'Expected_Profit': expected_profit.item(),
            'SI_Penalty': si_penalty.item(),
            'Volume_Penalty': vol_penalty.item(),
            'Operating_Cost': op_cost.item()
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate noisy MIQP schedules with simulator')
    parser.add_argument('--price-file', type=str,
                       default='Data/price_data_2024.csv',
                       help='Path to validation price data')
    parser.add_argument('--output-dir', type=str,
                       default='DFL/outputs/validation_results/noisy_miqp',
                       help='Output directory for results')
    args = parser.parse_args()

    print("=" * 80)
    print("Evaluating Noisy MIQP Schedules with Simulator Layer")
    print("=" * 80)

    # Setup device and load data
    device = setup_device()

    # Load portfolio and preprocessed data
    print("\nLoading system data...")
    portfolio = load_portfolio_data()
    if portfolio is None:
        print("Error: Could not load portfolio data")
        return

    preprocess_data = load_preprocessed_data()
    if preprocess_data is None:
        print("Error: Could not load preprocessed data")
        return

    head_init, v_low_init = initialize_head_and_volume(
        preprocess_data['h_to_v_low_fitted'], device
    )

    # Initialize parameters
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

    sim_layer = SimulationLayer(params)

    # Load validation prices
    print(f"\nLoading validation prices from {args.price_file}...")
    price_file = repo_root / args.price_file
    validation_prices = load_validation_prices(price_file)
    print(f"  Loaded {len(validation_prices)} validation price scenarios")

    # Define noisy data directory and output directory
    noisy_data_dir = repo_root / 'DFL' / 'outputs' / 'noisy_data'
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process both MIQP-Linear and MIQP-Piecewise
    variants = ['MIQP_linear', 'MIQP_piecewise']

    for variant in variants:
        print(f"\n{'=' * 80}")
        print(f"Processing {variant} variant")
        print(f"{'=' * 80}")

        # Find all noisy data files for this variant
        pattern = f"{variant}_results_*.csv"
        noisy_files = sorted(noisy_data_dir.glob(pattern))

        if not noisy_files:
            print(f"  ⚠ Warning: No noisy data files found for {variant}")
            continue

        print(f"  Found {len(noisy_files)} noisy data files")

        for noisy_file in noisy_files:
            print(f"\n  Processing: {noisy_file.name}")

            # Load schedules
            schedules = load_miqp_schedule(noisy_file)
            print(f"    Loaded {len(schedules)} schedules")

            # Evaluate with simulator
            results_df = evaluate_schedules(schedules, validation_prices, params, sim_layer)
            print(f"    Evaluated {len(results_df)} schedules")

            # Compute statistics
            if len(results_df) > 0:
                mean_profit = results_df['Ex_post_Profit'].mean()
                mean_si = results_df['SI_Penalty'].mean()
                mean_vol = results_df['Volume_Penalty'].mean()
                print(f"    Mean ex-post profit: €{mean_profit:.2f}")
                print(f"    Mean SI penalty: €{mean_si:.2f}")
                print(f"    Mean volume penalty: €{mean_vol:.2f}")

            # Save results
            output_file = output_dir / noisy_file.name.replace('_results_', '_evaluated_')
            results_df.to_csv(output_file, index=False)
            print(f"    Saved: {output_file}")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
