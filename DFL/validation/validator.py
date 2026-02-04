"""
Validation functions for DFL models.

This module handles validation of trained models on new price scenarios.
"""

import torch
import numpy as np
import csv
import json
import time
import traceback
import itertools
import scipy.sparse
from pathlib import Path
from datetime import datetime

from ..data.loaders import load_data_for_validation, load_new_price_data
from ..core.models import BoundedLogWeightPredictor
from ..core.layers import TaylorRegressionLayer, OptiLayer, SimulationLayer
from ..core.pipeline import RecursiveLinearizationPipeline

# Monkey patch for scipy/ECOS compatibility (scipy 1.13+ changed sparse matrix API)
# This patch adds get_shape() method to csc_array objects for backward compatibility with ECOS
if hasattr(scipy.sparse, 'csc_array'):
    _csc_array_cls = scipy.sparse.csc_array

    # Add get_shape method to the class if it doesn't exist
    if not hasattr(_csc_array_cls, 'get_shape'):
        def get_shape_method(self):
            """Return shape for backward compatibility with ECOS."""
            return self.shape

        _csc_array_cls.get_shape = get_shape_method


def find_closest_date(new_price, historical_data):
    """
    Find the date in historical data with the most similar price signal.

    Args:
        new_price: Tensor of shape [24] with hourly prices
        historical_data: Dictionary of historical data

    Returns:
        tuple: (date_str, distance) - closest date and Euclidean distance
    """
    closest_date = None
    min_distance = float('inf')

    for date_str, date_data in historical_data.items():
        historical_price = date_data['price'][:24]

        # Calculate Euclidean distance between price profiles
        distance = torch.norm(new_price - historical_price).item()

        if distance < min_distance:
            min_distance = distance
            closest_date = date_str

    return closest_date, min_distance


def validate_single_configuration(config, params, device, new_price_data, historical_data,
                                   architecture, num_layers, max_iterations, db_name):
    """
    Validate a single model configuration on new price scenarios.

    Args:
        config: DFLConfig instance
        params: HydroParameters instance
        device: PyTorch device
        new_price_data: Dictionary of new price scenarios by date
        historical_data: Dictionary of historical operational data
        architecture: Network architecture ('LSTM', 'RNN', 'FC')
        num_layers: Number of network layers
        max_iterations: Number of recursive linearization iterations
        db_name: Database name for output organization

    Returns:
        list: Validation results for each date
    """
    config_name = f"{architecture}_{num_layers}layer_{max_iterations}iter"

    # Create output directory (centralized in outputs folder)
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

    # Initialize layers
    regression_layer = TaylorRegressionLayer(params)
    optimizer_layer = OptiLayer(params)

    results = []

    # Process each new date
    for date_idx, (new_date, new_price) in enumerate(new_price_data.items()):
        print(f"\n[{date_idx+1}/{len(new_price_data)}] Processing date: {new_date}")

        # Create directory for this date
        safe_date = new_date.replace('/', '-')
        date_dir = config_dir / safe_date
        date_dir.mkdir(exist_ok=True, parents=True)

        try:
            start_time = time.time()

            # 1. Find the closest historical date
            closest_date, distance = find_closest_date(new_price, historical_data)
            print(f"Closest historical date: {closest_date} (distance: {distance:.2f})")

            # 2. Get initial conditions from closest date
            closest_data = historical_data[closest_date]
            power_init = closest_data['power'][:24].clone()
            head_init = closest_data['head'][:24].clone()
            flow_init = params.predict_q_poly(power_init.unsqueeze(0), head_init.unsqueeze(0)).squeeze(0)

            # 3. Determine weights based on configuration
            if config.use_neural_network:
                # Neural network variant: load model and predict weights
                model_path = Path(config.output_base_dir) / db_name / config_name / closest_date / "best_model.pt"

                if not model_path.exists():
                    print(f"Warning: No best model found at {model_path}. Skipping this date.")
                    continue

                # Initialize weight network
                weight_network = BoundedLogWeightPredictor(
                    input_size=4,
                    hidden_size=config.hidden_size,
                    num_layers=num_layers,
                    dropout=config.dropout,
                    time_horizon=params.time_horizon,
                    archetype=architecture,
                    init_w_p=config.init_w_p,
                    init_w_q=config.init_w_q,
                    init_w_h=config.init_w_h,
                    w_p_min=config.w_p_min,
                    w_p_max=config.w_p_max,
                    w_q_min=config.w_q_min,
                    w_q_max=config.w_q_max,
                    w_h_min=config.w_h_min,
                    w_h_max=config.w_h_max
                ).to(device)

                # Load the pretrained weights
                weight_network.load_state_dict(torch.load(model_path, map_location=device))
                weight_network.eval()

                # Predict weights using the network
                x = torch.stack([new_price, power_init, flow_init, head_init], dim=1)

                with torch.no_grad():
                    log_w_p, log_w_q, log_w_h = weight_network(x)
                    w_p = torch.exp(log_w_p)
                    w_q = torch.exp(log_w_q)
                    w_h = torch.exp(log_w_h)
            else:
                # Ablation variant: use fixed weights (no neural network)
                print(f"Using fixed weights: w_p={config.fixed_w_p}, w_q={config.fixed_w_q}, w_h={config.fixed_w_h}")
                w_p = torch.ones(24, device=device) * config.fixed_w_p
                w_q = torch.ones(24, device=device) * config.fixed_w_q
                w_h = torch.ones(24, device=device) * config.fixed_w_h

            # 7. Run recursive linearization
            p_current = power_init.clone().detach()
            h_current = head_init.clone().detach()
            flow_current = flow_init.clone().detach()

            iter_results = []

            for iteration in range(max_iterations):
                # Apply growth to weights
                growth_factor = config.penalty_growth_rate ** iteration
                w_p_iter = w_p * growth_factor
                w_q_iter = w_q * growth_factor
                w_h_iter = w_h * growth_factor

                # Compute linearization coefficients
                c, d, e, a, b = regression_layer.run_regression(p_current, h_current, flow_current)

                # Initialize OptiLayer
                optimizer_layer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())

                # Run optimization
                p_opt, q_opt, h_opt, v_opt, expected_profit, optimized_objective = optimizer_layer.forward(
                    new_price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                    p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                    w_p_iter.cpu(), w_h_iter.cpu(), w_q_iter.cpu()
                )

                # Store iteration results
                iter_results.append({
                    'iteration': iteration,
                    'p_opt': p_opt.detach().cpu().numpy(),
                    'q_opt': q_opt.detach().cpu().numpy(),
                    'h_opt': h_opt.detach().cpu().numpy(),
                    'expected_profit': expected_profit.item()
                })

                # Update for next iteration
                if iteration < max_iterations - 1:
                    p_current = p_opt.clone().detach().to(device=power_init.device)
                    h_current = h_opt.clone().detach().to(device=head_init.device)
                    flow_current = q_opt.clone().detach().to(device=flow_init.device)

            # 8. Run simulation
            simulator = SimulationLayer(params)
            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(
                p_opt.to(device), q_opt.to(device), h_opt.to(device)
            )

            # 9. Calculate ex-post profit
            ex_post_profit, SI_penalty, volume_penalty, operating_cost = simulator.calc_profit(
                p_sim, p_opt.to(device), v_low_sim, new_price.to(device)
            )

            processing_time = time.time() - start_time

            # 10. Save results
            result = {
                'p_opt': p_opt.detach().cpu().numpy(),
                'q_opt': q_opt.detach().cpu().numpy(),
                'h_opt': h_opt.detach().cpu().numpy(),
                'v_opt': v_opt.detach().cpu().numpy(),
                'p_sim': p_sim.detach().cpu().numpy(),
                'q_sim': q_sim.detach().cpu().numpy(),
                'h_sim': h_sim.detach().cpu().numpy(),
                'v_low_sim': v_low_sim.detach().cpu().numpy(),
                'new_price': new_price.detach().cpu().numpy(),
                'closest_price': closest_data['price'][:24].detach().cpu().numpy(),
                'closest_power': closest_data['power'][:24].detach().cpu().numpy(),
                'new_date': new_date,
                'closest_date': closest_date,
                'distance': distance,
                'processing_time': processing_time,
                'expected_profit': expected_profit.item(),
                'ex_post_profit': ex_post_profit.item(),
                'SI_penalty': SI_penalty.item(),
                'volume_penalty': volume_penalty.item(),
                'operating_cost': operating_cost.item(),
                'iter_results': iter_results,
                'database': db_name,
                'architecture': architecture,
                'num_layers': num_layers,
                'max_iterations': max_iterations
            }

            # Save as numpy array
            np.save(date_dir / "results.npy", result)

            # Append to benchmark CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(benchmark_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    safe_date, closest_date, f"{distance:.2f}",
                    f"{expected_profit.item():.2f}", f"{ex_post_profit.item():.2f}",
                    f"{SI_penalty.item():.2f}", f"{volume_penalty.item():.2f}",
                    f"{operating_cost.item():.2f}", f"{processing_time:.2f}",
                    timestamp
                ])

            results.append(result)

            print(f"✓ Validation completed:")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Expected profit: {expected_profit.item():.2f}")
            print(f"  Ex-post profit: {ex_post_profit.item():.2f}")

        except Exception as e:
            print(f"✗ Error processing date {new_date}: {e}")
            print(traceback.format_exc())

            # Log the error
            with open(config_dir / "error_log.txt", 'a') as f:
                f.write(f"\n[{datetime.now()}] Error processing {new_date}:\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-"*50 + "\n")

    print(f"\nValidation completed for {config_name}")
    print(f"Results saved to: {config_dir}")

    return results


def comprehensive_validation(config, params, device, new_price_file="./Data/price_data_2024.csv"):
    """
    Perform comprehensive validation across all model configurations.

    Args:
        config: DFLConfig instance
        params: HydroParameters instance
        device: PyTorch device
        new_price_file: Path to new price data CSV

    Returns:
        None (results are saved to disk)
    """
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting comprehensive validation at {start_timestamp}...")

    # Load new price data
    new_price_data = load_new_price_data(new_price_file, device)
    if not new_price_data:
        print("Error: Could not load new price data")
        return

    # Define validation parameters (must match pretraining)
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Excludes 0% noise
    architectures = [config.architecture] if config.use_neural_network else ['NoNN']
    num_layers_list = [config.num_layers] if config.use_neural_network else [0]
    max_iterations_list = [config.max_iterations]  # Use iteration count from config

    # Create master results directory (centralized in outputs folder)
    master_dir = Path(config.results_base_dir) / "comprehensive"
    master_dir.mkdir(exist_ok=True, parents=True)

    # Create master benchmark file (append mode to preserve previous results)
    master_benchmark_file = master_dir / "master_validation_benchmarks.csv"
    file_exists = master_benchmark_file.exists()

    header = []
    base_columns = [
        'Database', 'Noise_Level', 'Architecture', 'Num_Layers', 'Max_Iterations',
        'New_Date', 'Closest_Historical_Date', 'Distance_Metric',
        'Expected_Profit', 'Ex_post_Profit', 'SI_Penalty',
        'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds',
        'Timestamp'
    ]
    columns = base_columns + ['Method_Type']
    include_method_type = True

    if file_exists:
        with open(master_benchmark_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, [])
        if header:
            columns = header
            include_method_type = 'Method_Type' in header

    # Only write header if file is new or empty
    if not file_exists or not header:
        with open(master_benchmark_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    # Set method type based on variant
    variant = str(config.variant_name).upper()
    if variant == "GL":
        method_type = "DFL-GL-RS"
    elif variant == "PW":
        method_type = "DFL-PW-RS"
    elif variant == "PW-NO-REC":
        method_type = "DFL-PW-no-Rec"
    elif variant == "ABLATION" or not config.use_neural_network:
        method_type = "DFL-PW-no-NN"
    else:
        # Fallback
        method_type = "DFL-RS"

    # REMOVED: best_configurations.json generation (user will manually select best iteration)

    # Total configurations
    databases = noise_levels + ['random_samples']
    total_configs = len(databases) * len(architectures) * len(num_layers_list) * len(max_iterations_list)
    config_counter = 0

    # Iterate through all configurations
    for db_source in databases:
        if db_source == 'random_samples':
            # Load from generated noisy data directory
            file_path = config.get_data_file_pattern(random_samples=True)
            # Use the source data name for organizing output models
            source_file_path = config.get_data_file_pattern(random_samples=True)
            db_name = Path(source_file_path).stem
            noise_level = None
        else:
            # Load from generated noisy data directory
            file_path = config.get_data_file_pattern(noise_level=db_source)
            # Use the source data name for organizing output models
            source_file_path = config.get_data_file_pattern(noise_level=db_source)
            db_name = Path(source_file_path).stem
            noise_level = db_source

        # Load historical data from noisy files
        historical_data = load_data_for_validation(file_path, db_name, config, device)
        if not historical_data:
            print(f"Warning: Could not load historical data for {db_name}")
            continue

        for arch, num_layers, max_iter in itertools.product(
                architectures, num_layers_list, max_iterations_list):

            config_counter += 1
            print(f"\n{'='*80}")
            print(f"[{config_counter}/{total_configs}] Validating: {db_name}/{arch}_{num_layers}layer_{max_iter}iter")
            print(f"{'='*80}")

            # Run validation
            results = validate_single_configuration(
                config, params, device, new_price_data, historical_data,
                arch, num_layers, max_iter, db_name
            )

            # Update master benchmark file
            for result in results:
                row_data = {
                    'Database': db_name,
                    'Noise_Level': noise_level if noise_level is not None else 'random',
                    'Architecture': arch,
                    'Num_Layers': num_layers,
                    'Max_Iterations': max_iter,
                    'New_Date': result.get('new_date', 'N/A'),
                    'Closest_Historical_Date': result.get('closest_date', 'N/A'),
                    'Distance_Metric': result.get('distance', 'N/A'),
                    'Expected_Profit': f"{result['expected_profit']:.2f}",
                    'Ex_post_Profit': f"{result['ex_post_profit']:.2f}",
                    'SI_Penalty': f"{result['SI_penalty']:.2f}",
                    'Volume_Penalty': f"{result['volume_penalty']:.2f}",
                    'Operating_Cost': f"{result['operating_cost']:.2f}",
                    'Processing_Time_Seconds': result.get('processing_time', 'N/A'),
                    'Timestamp': start_timestamp
                }
                if include_method_type:
                    row_data['Method_Type'] = method_type

                with open(master_benchmark_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([row_data.get(col, '') for col in columns])

    # REMOVED: best_configurations.json generation (user will manually select best iteration)

    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nComprehensive validation completed!")
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Master benchmark saved to: {master_benchmark_file}")
