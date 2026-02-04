#!/usr/bin/env python3
"""
Aggregate validation results from all DFL approaches and noisy MIQP baselines:
- DFL-GL-RS: GL-based (GL with 7 iterations, LSTM)
- DFL-PW-RS: PW-based (PW with 7 iterations, LSTM)
- DFL-PW-no-Rec: PW with 1 iteration (LSTM, no recursive refinement)
- DFL-PW-no-NN: PW with 7 iterations (fixed weights, no neural network)
- MIQP-GL-noised: MIQP global linear optimized with noisy data, evaluated with simulator
- MIQP-PW-noised: MIQP piecewise optimized with noisy data, evaluated with simulator
into a comprehensive master validation benchmarks file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')


def get_repo_root():
    """Find repo root by looking for DFL directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / 'DFL').exists():
            return current
        current = current.parent
    return Path.cwd()


def extract_architecture_and_layers(config_dir):
    """Extract architecture name and layer count from directory name."""
    # Format: LSTM_3layer_5iter or similar
    match = re.match(r'([A-Za-z0-9]+)_(\d+)layer(?:_(\d+)iter)?', config_dir)
    if match:
        arch = match.group(1)  # e.g., LSTM
        layers = int(match.group(2))  # e.g., 3
        iters = int(match.group(3)) if match.group(3) else 1  # e.g., 5
        return arch, layers, iters
    return None, None, None


def extract_noise_level(data_dir):
    """Extract noise level from directory name."""
    # Format: MIQP_linear_results_relative_noise_10pct or MIQP_piecewise_results_random_samples
    if 'random_samples' in data_dir:
        return 'random'

    match = re.search(r'noise_(\d+)pct', data_dir)
    if match:
        return float(match.group(1)) / 100

    return None


def aggregate_method_results(validation_base, method_name, base_arch_type,
                             data_prefixes=None, arch_filter=None, exclude_arches=None,
                             iter_filter=None):
    """
    Aggregate results for a specific method.

    Args:
        validation_base: Base directory for validation results
        method_name: Name for this method type (e.g., 'DFL-GL-RS')
        base_arch_type: Base architecture type (e.g., 'GL-based')
        data_prefixes: List of data source prefixes to include (e.g., ['MIQP_linear_results'])
        arch_filter: Set of architectures to include (e.g., {'LSTM'})
        exclude_arches: Set of architectures to exclude (e.g., {'NoNN'})
        iter_filter: Filter for max iterations (e.g., 7 or [1, 7])
    """
    all_records = []

    if not validation_base.exists():
        print(f"Validation base not found: {validation_base}")
        return []

    # Iterate through data type directories (noise levels)
    for data_dir in sorted(validation_base.iterdir()):
        if not data_dir.is_dir():
            continue
        if data_dir.name in {'comprehensive'}:
            continue
        if data_prefixes and not any(data_dir.name.startswith(prefix) for prefix in data_prefixes):
            continue

        noise_level = extract_noise_level(data_dir.name)
        if noise_level is None:
            print(f"Warning: Could not extract noise level from {data_dir.name}")
            continue

        # Iterate through architecture/configuration directories
        for config_dir in sorted(data_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            arch, num_layers, max_iters = extract_architecture_and_layers(config_dir.name)
            if arch is None:
                print(f"Warning: Could not parse config {config_dir.name}")
                continue
            if arch_filter and arch not in arch_filter:
                continue
            if exclude_arches and arch in exclude_arches:
                continue
            if iter_filter is not None:
                if isinstance(iter_filter, (list, tuple)):
                    if max_iters not in iter_filter:
                        continue
                else:
                    if max_iters != iter_filter:
                        continue

            # Load the scheduling_benchmarks.csv
            bench_file = config_dir / 'scheduling_benchmarks.csv'
            if not bench_file.exists():
                print(f"Warning: Benchmark file not found: {bench_file}")
                continue

            try:
                df = pd.read_csv(bench_file)

                # Add metadata columns
                df['Database'] = data_dir.name  # Add database name (e.g., MIQP_piecewise_results_relative_noise_10pct)
                df['Architecture'] = arch
                df['Num_Layers'] = num_layers
                df['Max_Iterations'] = max_iters
                # Noise_Level should be the decimal format (0.1, 0.2) for compatibility with print_tables.py
                # Noise_Level_Numeric is the same value but explicitly numeric
                if noise_level == 'random':
                    df['Noise_Level'] = 'random'
                    df['Noise_Level_Numeric'] = np.nan
                else:
                    df['Noise_Level'] = noise_level  # Keep as decimal (0.1, 0.2, etc.)
                    df['Noise_Level_Numeric'] = noise_level
                df['Method_Type'] = method_name
                df['Base_Architecture'] = base_arch_type

                all_records.append(df)

            except Exception as e:
                print(f"Error reading {bench_file}: {e}")
                continue

    return all_records


def aggregate_noisy_miqp(validation_base, variant_name, method_name):
    """Aggregate noisy MIQP evaluation results.

    Args:
        validation_base: Base directory for validation results
        variant_name: 'linear' or 'piecewise'
        method_name: 'MIQP-GL-noised' or 'MIQP-PW-noised'
    """
    all_records = []
    noisy_miqp_dir = validation_base / 'noisy_miqp'

    if not noisy_miqp_dir.exists():
        print(f"Noisy MIQP directory not found: {noisy_miqp_dir}")
        return []

    # Find all evaluated files for this variant
    pattern = f"MIQP_{variant_name}_evaluated_*.csv"
    eval_files = sorted(noisy_miqp_dir.glob(pattern))

    if not eval_files:
        print(f"No evaluated files found matching: {pattern}")
        return []

    print(f"Found {len(eval_files)} evaluated files")

    for eval_file in eval_files:
        # Extract noise level from filename
        # Format: MIQP_linear_evaluated_relative_noise_10pct.csv or MIQP_linear_evaluated_random_samples.csv
        if 'random_samples' in eval_file.name:
            noise_level = 'random'
            noise_numeric = np.nan
        else:
            match = re.search(r'noise_(\d+)pct', eval_file.name)
            if match:
                noise_numeric = float(match.group(1)) / 100
                noise_level = noise_numeric
            else:
                print(f"Warning: Could not extract noise level from {eval_file.name}")
                continue

        try:
            df = pd.read_csv(eval_file)

            # Add metadata columns to match DFL format
            df['Method_Type'] = method_name
            df['Base_Architecture'] = method_name  # e.g., 'MIQP-GL-noised'
            df['Architecture'] = 'MIQP'
            df['Num_Layers'] = 0  # N/A for MIQP
            df['Max_Iterations'] = 0  # N/A for MIQP
            df['Noise_Level'] = noise_level
            df['Noise_Level_Numeric'] = noise_numeric
            df['Database'] = f"MIQP_{variant_name}_results_{eval_file.name.split('_evaluated_')[1].replace('.csv', '')}"

            all_records.append(df)

        except Exception as e:
            print(f"Error reading {eval_file}: {e}")
            continue

    return all_records


def main():
    """Main execution."""
    print("="*80)
    print("DFL & Noisy MIQP Validation Results Aggregation")
    print("="*80)

    repo_root = get_repo_root()
    validation_base = repo_root / 'DFL' / 'outputs' / 'validation_results'
    output_file = validation_base / 'comprehensive' / 'master_validation_benchmarks.csv'

    print(f"\nRepository root: {repo_root}")
    print(f"Validation base: {validation_base}")
    print(f"Output file: {output_file}")

    # Aggregate from all methods
    all_dfs = []

    # Process GL-based (7 iterations)
    print("\n" + "-"*80)
    print("Processing GL-based validation results...")
    print("-"*80)
    gl_records = aggregate_method_results(
        validation_base,
        'DFL-GL-RS',
        'GL-based',
        data_prefixes=['MIQP_linear_results'],
        iter_filter=7
    )
    if gl_records:
        gl_df = pd.concat(gl_records, ignore_index=True)
        print(f"✓ Aggregated {len(gl_df)} GL-based records")
        all_dfs.append(gl_df)
    else:
        print("✗ No GL-based records found")

    # Process PW-based (7 iterations)
    print("\n" + "-"*80)
    print("Processing PW-based validation results...")
    print("-"*80)
    pw_records = aggregate_method_results(
        validation_base,
        'DFL-PW-RS',
        'PW-based',
        data_prefixes=['MIQP_piecewise_results'],
        exclude_arches={'NoNN'},
        iter_filter=7
    )
    if pw_records:
        pw_df = pd.concat(pw_records, ignore_index=True)
        print(f"✓ Aggregated {len(pw_df)} PW-based records")
        all_dfs.append(pw_df)
    else:
        print("✗ No PW-based records found")

    # Process PW-no-Rec (1 iteration)
    print("\n" + "-"*80)
    print("Processing PW-no-Rec validation results...")
    print("-"*80)
    pwnorec_records = aggregate_method_results(
        validation_base,
        'DFL-PW-no-Rec',
        'PW-no-Rec',
        data_prefixes=['MIQP_piecewise_results'],
        exclude_arches={'NoNN'},
        iter_filter=1
    )
    if pwnorec_records:
        pwnorec_df = pd.concat(pwnorec_records, ignore_index=True)
        print(f"✓ Aggregated {len(pwnorec_df)} PW-no-Rec records")
        all_dfs.append(pwnorec_df)
    else:
        print("✗ No PW-no-Rec records found")

    # Process PW-no-NN (Fixed weights, no NN, 7 iterations)
    print("\n" + "-"*80)
    print("Processing PW-no-NN validation results...")
    print("-"*80)
    pwnonn_records = aggregate_method_results(
        validation_base,
        'DFL-PW-no-NN',
        'PW-no-NN',
        data_prefixes=['MIQP_piecewise_results'],
        arch_filter={'NoNN'},
        iter_filter=7
    )
    if pwnonn_records:
        pwnonn_df = pd.concat(pwnonn_records, ignore_index=True)
        print(f"✓ Aggregated {len(pwnonn_df)} PW-no-NN records")
        all_dfs.append(pwnonn_df)
    else:
        print("✗ No PW-no-NN records found")

    # Process noisy MIQP-GL evaluations
    print("\n" + "-"*80)
    print("Processing noisy MIQP-GL evaluations...")
    print("-"*80)
    miqp_gl_noised_records = aggregate_noisy_miqp(validation_base, 'linear', 'MIQP-GL-noised')
    if miqp_gl_noised_records:
        miqp_gl_noised_df = pd.concat(miqp_gl_noised_records, ignore_index=True)
        print(f"✓ Aggregated {len(miqp_gl_noised_df)} MIQP-GL-noised records")
        all_dfs.append(miqp_gl_noised_df)
    else:
        print("✗ No MIQP-GL-noised records found")

    # Process noisy MIQP-PW evaluations
    print("\n" + "-"*80)
    print("Processing noisy MIQP-PW evaluations...")
    print("-"*80)
    miqp_pw_noised_records = aggregate_noisy_miqp(validation_base, 'piecewise', 'MIQP-PW-noised')
    if miqp_pw_noised_records:
        miqp_pw_noised_df = pd.concat(miqp_pw_noised_records, ignore_index=True)
        print(f"✓ Aggregated {len(miqp_pw_noised_df)} MIQP-PW-noised records")
        all_dfs.append(miqp_pw_noised_df)
    else:
        print("✗ No MIQP-PW-noised records found")

    # Combine all
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        print("\n" + "-"*80)
        print(f"Combined master dataframe: {len(master_df)} total records")
        print("-"*80)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save
        master_df.to_csv(output_file, index=False)
        print(f"✓ Saved master validation benchmarks to: {output_file}")

        print(f"\nMaster dataframe summary:")
        print(f"  Total records: {len(master_df)}")
        print(f"  Columns: {len(master_df.columns)}")
        print(f"  Methods: {master_df['Method_Type'].unique()}")
        print(f"  Architectures: {master_df['Architecture'].unique()}")

        # Sort noise levels properly (numbers first, then 'random')
        noise_vals = master_df['Noise_Level'].unique()
        numeric_noise = sorted([x for x in noise_vals if isinstance(x, (int, float)) or (isinstance(x, str) and x != 'random')])
        random_noise = ['random'] if any(isinstance(x, str) and x == 'random' for x in noise_vals) else []
        print(f"  Noise levels: {numeric_noise + random_noise}")

    else:
        print("\n✗ ERROR: No records found from any method!")

    print("\n" + "="*80)
    print("Aggregation completed!")
    print("="*80)


if __name__ == "__main__":
    main()
