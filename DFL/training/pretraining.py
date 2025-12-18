"""
Pretraining orchestration with grid search.

This module handles the high-level orchestration of pretraining across
multiple configurations, noise levels, and dates.
"""

import itertools
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed

from ..data.loaders import load_data_for_pretraining
from .trainer import train_single_model


def pretraining_with_grid_search(config, params, device, n_jobs=20):
    """
    Perform pretraining with parallel execution and grid search.

    Args:
        config: DFLConfig instance
        params: HydroParameters instance
        device: PyTorch device
        n_jobs: Number of parallel jobs

    Returns:
        None (models are saved to disk)
    """
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define noise levels for grid search
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Grid search parameters
    architectures = [config.architecture] if config.use_neural_network else ['NoNN']
    num_layers_list = [config.num_layers] if config.use_neural_network else [0]
    max_iterations_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Prepare all training jobs
    all_jobs = []

    # Process noise level databases
    for noise_level in noise_levels:
        file_path = config.get_data_file_pattern(noise_level=noise_level)

        # Load data
        historical_data = load_data_for_pretraining(
            file_path, f"noise_{int(noise_level*100):02d}pct", config, device
        )

        if not historical_data:
            print(f"Warning: Could not load data for noise level {noise_level}")
            continue

        # Create output directory
        source_name = config.get_data_file_pattern(noise_level=noise_level).replace('.csv', '')
        root_dir = Path(config.output_base_dir) / source_name
        root_dir.mkdir(exist_ok=True, parents=True)

        for architecture, num_layers, max_iterations in itertools.product(
                architectures, num_layers_list, max_iterations_list):

            # Update config for this iteration
            config.architecture = architecture if architecture != 'NoNN' else config.architecture
            config.num_layers = num_layers
            config.max_iterations = max_iterations

            for date_str, date_data in historical_data.items():
                all_jobs.append((
                    config, architecture, num_layers, max_iterations,
                    date_str, date_data, params, device
                ))

    # Process random samples database
    random_samples_file = config.get_data_file_pattern(random_samples=True)

    random_samples_data = load_data_for_pretraining(
        random_samples_file, "random_samples", config, device
    )

    if random_samples_data:
        source_name = random_samples_file.replace('.csv', '')
        root_dir = Path(config.output_base_dir) / source_name
        root_dir.mkdir(exist_ok=True, parents=True)

        for architecture, num_layers, max_iterations in itertools.product(
                architectures, num_layers_list, max_iterations_list):

            # Update config for this iteration
            config.architecture = architecture if architecture != 'NoNN' else config.architecture
            config.num_layers = num_layers
            config.max_iterations = max_iterations

            for date_str, date_data in random_samples_data.items():
                all_jobs.append((
                    config, architecture, num_layers, max_iterations,
                    date_str, date_data, params, device
                ))

    print(f"Training {len(all_jobs)} models in parallel with {n_jobs} workers...")

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(train_single_model)(*job) for job in all_jobs
    )

    # Count successes and failures
    successes = sum(1 for r in results if r.get('success', False))
    failures = len(results) - successes

    print(f"\nPretraining completed!")
    print(f"Successful: {successes}/{len(results)}")
    print(f"Failed: {failures}/{len(results)}")
    print(f"Start timestamp: {start_timestamp}")


def pretraining_single_noise_level(config, params, device, noise_level=None, random_samples=False):
    """
    Perform pretraining for a single noise level or random samples.

    Args:
        config: DFLConfig instance
        params: HydroParameters instance
        device: PyTorch device
        noise_level: Float between 0 and 1, or None
        random_samples: Boolean, whether to use random samples

    Returns:
        None (models are saved to disk)
    """
    # Get file path
    if random_samples:
        file_path = config.get_data_file_pattern(random_samples=True)
        source_name = "random_samples"
    else:
        file_path = config.get_data_file_pattern(noise_level=noise_level)
        source_name = f"noise_{int(noise_level*100):02d}pct" if noise_level is not None else "base"

    # Load data
    historical_data = load_data_for_pretraining(file_path, source_name, config, device)

    if not historical_data:
        print(f"Error: Could not load data from {file_path}")
        return

    print(f"Loaded {len(historical_data)} days of data")

    # Create output directory
    root_dir = Path(config.output_base_dir) / source_name
    root_dir.mkdir(exist_ok=True, parents=True)

    # Train for each date
    results = []
    for date_str, date_data in historical_data.items():
        print(f"\nTraining for date: {date_str}")

        result = train_single_model(
            config=config,
            architecture=config.architecture,
            num_layers=config.num_layers,
            max_iterations=config.max_iterations,
            date_str=date_str,
            date_data=date_data,
            params=params,
            device=device
        )

        results.append(result)

        if result['success']:
            print(f"  ✓ Training completed in {result['training_time']:.2f}s")
        else:
            print(f"  ✗ Training failed: {result.get('error', 'Unknown error')}")

    # Summary
    successes = sum(1 for r in results if r['success'])
    print(f"\n{'='*60}")
    print(f"Pretraining summary for {source_name}:")
    print(f"  Successful: {successes}/{len(results)}")
    print(f"  Failed: {len(results) - successes}/{len(results)}")
    print(f"{'='*60}")
