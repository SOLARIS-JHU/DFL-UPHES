"""
IPOPT NLP validation for hydropower scheduling.

Validates IPOPT NLP solutions against the same ex-post simulation used
by all DFL variants, enabling direct comparison.
"""

import torch
import numpy as np
import csv
import time
import traceback
from pathlib import Path
from datetime import datetime

from ..data.loaders import load_data_for_validation, load_new_price_data
from ..core.layers import SimulationLayer
from ..core.ipopt_solver import IPOPTHydroSolver
from .validator import find_closest_date


def validate_ipopt_scenarios(config, params, preprocess_data, device,
                             new_price_data, historical_data, db_name,
                             tee=False):
    """
    Validate IPOPT NLP solutions on new price scenarios.

    For each validation date:
    1. Find closest historical date (reuse find_closest_date)
    2. Extract modes from MIQP solution
    3. Build and solve NLP with IPOPTHydroSolver
    4. Pass solution through SimulationLayer for ex-post evaluation
    5. Write scheduling_benchmarks.csv

    Args:
        config: IPOPT config instance (IPOPTConfigPW or IPOPTConfigGL)
        params: HydroParameters instance
        preprocess_data: Dictionary from load_preprocessed_data()
        device: PyTorch device
        new_price_data: Dictionary of new price scenarios by date
        historical_data: Dictionary of historical operational data
        db_name: Database name for output organization
        tee: Whether to print IPOPT solver output

    Returns:
        list: Validation results for each date
    """
    config_name = "IPOPT_0layer_0iter"

    # Create output directory
    config_dir = Path(config.results_base_dir) / db_name / config_name
    config_dir.mkdir(exist_ok=True, parents=True)

    # Create benchmark CSV file
    benchmark_file = config_dir / "scheduling_benchmarks.csv"
    with open(benchmark_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
            'Expected_Profit', 'Ex_post_Profit', 'SI_Penalty',
            'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
            'Timestamp'
        ])

    # Initialize IPOPT solver
    solver = IPOPTHydroSolver(
        params=params,
        preprocess_data=preprocess_data,
        ipopt_max_iter=config.ipopt_max_iter,
        ipopt_tol=config.ipopt_tol,
        ipopt_time_limit=config.ipopt_time_limit
    )

    results = []

    for date_idx, (new_date, new_price) in enumerate(new_price_data.items()):
        print(f"\n[{date_idx + 1}/{len(new_price_data)}] Processing date: {new_date}")

        try:
            start_time = time.time()

            # 1. Find closest historical date
            closest_date, distance = find_closest_date(new_price, historical_data)
            print(f"Closest historical date: {closest_date} (distance: {distance:.2f})")

            # 2. Get MIQP solution data from closest date
            closest_data = historical_data[closest_date]
            power_hist = closest_data['power'][:24].numpy()
            head_hist = closest_data['head'][:24].numpy()
            flow_hist = closest_data['flow'][:24].numpy()

            # 3. Extract modes from MIQP solution
            modes = power_hist  # IPOPTHydroSolver._parse_modes handles numeric

            # 4. Solve NLP with IPOPT
            prices_np = new_price.numpy()
            nlp_result = solver.solve(
                prices=prices_np,
                modes=modes,
                warm_start_power=power_hist,
                warm_start_head=head_hist,
                warm_start_flow=flow_hist,
                tee=tee
            )

            print(f"IPOPT status: {nlp_result['status']}, objective: {nlp_result['objective']:.2f}")

            # 5. Convert IPOPT solution to tensors for simulation
            p_opt = torch.tensor(nlp_result['power'], dtype=torch.float32, device=device)
            q_opt = torch.tensor(nlp_result['flow'], dtype=torch.float32, device=device)
            h_opt = torch.tensor(nlp_result['head'], dtype=torch.float32, device=device)

            expected_profit = nlp_result['objective']

            # 6. Run simulation for ex-post evaluation
            simulator = SimulationLayer(params)
            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(
                p_opt, q_opt, h_opt
            )

            # 7. Calculate ex-post profit
            ex_post_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
                p_sim, p_opt, v_low_sim, new_price.to(device)
            )

            processing_time = time.time() - start_time

            # 8. Save results
            result = {
                'new_date': new_date,
                'closest_date': closest_date,
                'distance': distance,
                'expected_profit': expected_profit,
                'ex_post_profit': ex_post_profit.item(),
                'SI_penalty': SI_penalty.item(),
                'volume_penalty': volume_penalty.item(),
                'operating_cost': operating_cost.item(),
                'processing_time': processing_time,
                'ipopt_status': nlp_result['status'],
                'p_opt': nlp_result['power'],
                'q_opt': nlp_result['flow'],
                'h_opt': nlp_result['head'],
                'v_low': nlp_result['v_low']
            }

            # Append to benchmark CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(benchmark_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    new_date.replace('/', '-'), closest_date, f"{distance:.2f}",
                    f"{expected_profit:.2f}", f"{ex_post_profit.item():.2f}",
                    f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                    f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                    timestamp
                ])

            results.append(result)

            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Expected profit: {expected_profit:.2f}")
            print(f"  Ex-post profit: {ex_post_profit.item():.2f}")

        except Exception as e:
            print(f"Error processing date {new_date}: {e}")
            print(traceback.format_exc())

            # Log the error
            with open(config_dir / "error_log.txt", 'a') as f:
                f.write(f"\n[{datetime.now()}] Error processing {new_date}:\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-" * 50 + "\n")

    print(f"\nIPOPT validation completed for {db_name}")
    print(f"Results saved to: {config_dir}")

    return results
