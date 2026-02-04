"""
Noise generation for creating robust training datasets.

This module generates noisy variants of MIQP optimization results:
1. Relative noise: Adds noise proportional to capacity range
2. Random sampling: Generates random schedules preserving operational modes
"""

import torch
import numpy as np
import pandas as pd
import os
from copy import deepcopy

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class BaselineSimulator:
    """Baseline simulator without noise for generating reference results."""

    def __init__(self, params):
        """
        Initialize BaselineSimulator.

        Args:
            params: HydroParameters instance
        """
        self.params = params

    def simulate_daily_operation(self, p_schedule, initial_head, track_changes=False):
        """
        Simulate daily operation without noise.

        Args:
            p_schedule: Power schedule tensor [time_horizon]
            initial_head: Initial head value
            track_changes: Whether to track mode changes and clamping

        Returns:
            tuple: (p_sim, q_sim, h_sim, v_low_sim) or with tracking info if enabled
        """
        TH = len(p_schedule)
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        # Tracking variables
        mode_changes = []
        power_clamps = []

        # Start with initial volume corresponding to initial head
        v_current = self.params.h_to_v_low_fitted(initial_head)

        for i in range(TH):
            p_current = p_schedule[i]
            original_power = p_current.item()

            # For hour 0, use initial head; for other hours, calculate from volume
            if i == 0:
                h_current = initial_head
            else:
                h_current = self.params.v_low_to_h_fitted(v_current)

            q_candidate = torch.zeros_like(p_current)
            p_clamped = p_current
            power_was_clamped = False
            original_mode = None

            # Determine original mode
            if original_power > 0.5:
                original_mode = "Turbine"
            elif original_power < -0.5:
                original_mode = "Pump"
            else:
                original_mode = "Idle"

            if p_current > 0.5:  # Turbine mode
                p_min_turb = self.params.pos_min(h_current)
                p_max_turb = self.params.pos_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                if abs(p_clamped.item() - p_current.item()) > 1e-6:
                    power_was_clamped = True
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
            elif p_current < -0.5:  # Pump mode
                p_min_pump = self.params.neg_min(h_current)
                p_max_pump = self.params.neg_max(h_current)
                p_clamped = torch.clamp(p_current, min=p_min_pump, max=p_max_pump)
                if abs(p_clamped.item() - p_current.item()) > 1e-6:
                    power_was_clamped = True
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)

            v_next = v_current + q_candidate * 3600
            out_of_bounds = (v_next > self.params.max_vol_up) | (v_next < self.params.min_vol_low)

            if out_of_bounds:
                p_final = torch.zeros_like(p_current)
                q_final = torch.zeros_like(q_candidate)
                v_next = v_current
                h_final = h_current
                final_mode = "Idle"
                if track_changes and original_mode != "Idle":
                    mode_changes.append({
                        'hour': i,
                        'original_mode': original_mode,
                        'final_mode': final_mode,
                        'reason': 'Volume out of bounds',
                        'original_power': original_power,
                        'final_power': 0.0
                    })
            else:
                p_final = p_clamped if abs(p_current) > 0.5 else torch.zeros_like(p_current)
                q_final = q_candidate
                h_final = self.params.v_low_to_h_fitted(v_next)

                # Determine final mode
                final_power = p_final.item()
                if final_power > 0.5:
                    final_mode = "Turbine"
                elif final_power < -0.5:
                    final_mode = "Pump"
                else:
                    final_mode = "Idle"

            # Track power clamping
            if track_changes and power_was_clamped:
                power_clamps.append({
                    'hour': i,
                    'original_power': original_power,
                    'clamped_power': p_clamped.item(),
                    'mode': original_mode,
                    'head': h_current.item()
                })

            # Track mode changes
            if track_changes and not out_of_bounds and original_mode != final_mode:
                mode_changes.append({
                    'hour': i,
                    'original_mode': original_mode,
                    'final_mode': final_mode,
                    'reason': 'Mode transition',
                    'original_power': original_power,
                    'final_power': p_final.item()
                })

            p_list.append(p_final)
            q_list.append(q_final)
            h_list.append(h_final)
            v_list.append(v_next)
            v_current = v_next

        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list)
        v_low_sim = torch.stack(v_list)

        if track_changes:
            return p_sim, q_sim, h_sim, v_low_sim, mode_changes, power_clamps
        else:
            return p_sim, q_sim, h_sim, v_low_sim


class NoiseSimulator:
    """Simulator with relative noise addition and retry mechanism."""

    def __init__(self, params, relative_noise_level=0.15):
        """
        Initialize NoiseSimulator.

        Args:
            params: HydroParameters instance
            relative_noise_level: Noise level as fraction of capacity range (e.g., 0.15 = 15%)
        """
        self.params = params
        self.relative_noise_level = relative_noise_level

    def get_capacity_range(self, head):
        """Get capacity range for turbine and pump modes at given head."""
        pos_min_val = self.params.pos_min(head).item()
        pos_max_val = self.params.pos_max(head).item()
        neg_min_val = self.params.neg_min(head).item()
        neg_max_val = self.params.neg_max(head).item()

        return pos_min_val, pos_max_val, neg_min_val, neg_max_val

    def generate_noisy_power(self, baseline_power, head, debug=False):
        """
        Generate noisy power in overlapping region between noise region and feasible region.

        Args:
            baseline_power: Original power value
            head: Current head level
            debug: Print debug information

        Returns:
            tuple: (noisy_power, relative_error, debug_info) or (None, None, debug_info) if invalid
        """
        pos_min_val, pos_max_val, neg_min_val, neg_max_val = self.get_capacity_range(head)

        debug_info = {
            'head': head.item(),
            'baseline_power': baseline_power,
            'pos_range': (pos_min_val, pos_max_val),
            'neg_range': (neg_min_val, neg_max_val)
        }

        # Determine mode and capacity range
        if baseline_power > 0.5:  # Turbine mode
            capacity_range = pos_max_val - pos_min_val
            feasible_min, feasible_max = pos_min_val, pos_max_val
            mode = "turbine"
        elif baseline_power < -0.5:  # Pump mode
            capacity_range = neg_max_val - neg_min_val
            feasible_min, feasible_max = neg_min_val, neg_max_val
            mode = "pump"
        else:  # Idle mode
            debug_info['mode'] = "idle"
            return 0.0, 0.0, debug_info

        # Calculate noise range based on capacity
        noise_magnitude = self.relative_noise_level * capacity_range
        noise_min = baseline_power - noise_magnitude
        noise_max = baseline_power + noise_magnitude

        # Find overlapping region with feasible region
        overlap_min = max(noise_min, feasible_min)
        overlap_max = min(noise_max, feasible_max)

        debug_info.update({
            'mode': mode,
            'capacity_range': capacity_range,
            'noise_magnitude': noise_magnitude,
            'noise_range': (noise_min, noise_max),
            'feasible_range': (feasible_min, feasible_max),
            'overlap_range': (overlap_min, overlap_max)
        })

        # Check if there's a valid overlap
        if overlap_min >= overlap_max:
            debug_info['error'] = f"No overlap: noise range [{noise_min:.2f}, {noise_max:.2f}] vs feasible [{feasible_min:.2f}, {feasible_max:.2f}]"
            return None, None, debug_info

        # Generate random power in overlapping region
        noisy_power = np.random.uniform(overlap_min, overlap_max)
        relative_error = abs(noisy_power - baseline_power) / capacity_range

        debug_info.update({
            'noisy_power': noisy_power,
            'relative_error': relative_error
        })

        return noisy_power, relative_error, debug_info

    def simulate_daily_operation_with_noise(self, baseline_schedule, initial_head, max_retries=100, debug=False):
        """
        Simulate daily operation with noise, retrying if constraints are violated.

        Args:
            baseline_schedule: Baseline power schedule
            initial_head: Initial head value
            max_retries: Maximum number of retry attempts
            debug: Print debug information

        Returns:
            dict: Results with noisy schedule and metadata, or None if failed
        """
        for attempt in range(max_retries):
            # Initialize
            TH = len(baseline_schedule)
            p_list = []
            q_list = []
            h_list = []
            v_list = []
            relative_errors = []

            v_current = self.params.h_to_v_low_fitted(initial_head)
            success = True

            for i in range(TH):
                baseline_power = baseline_schedule[i].item()

                if i == 0:
                    h_current = initial_head
                else:
                    h_current = self.params.v_low_to_h_fitted(v_current)

                # Generate noisy power
                noisy_power, rel_error, _ = self.generate_noisy_power(baseline_power, h_current, debug=debug)

                if noisy_power is None:
                    success = False
                    break

                relative_errors.append(rel_error)
                p_noisy = torch.tensor(noisy_power, dtype=torch.float32)

                # Calculate flow
                if abs(noisy_power) > 0.5:
                    q_noisy = self.params.predict_q_poly(p_noisy.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
                else:
                    q_noisy = torch.zeros_like(p_noisy)

                # Update volume
                v_next = v_current + q_noisy * 3600

                # Check volume constraints
                if v_next > self.params.max_vol_up or v_next < self.params.min_vol_low:
                    success = False
                    break

                h_next = self.params.v_low_to_h_fitted(v_next)

                p_list.append(p_noisy)
                q_list.append(q_noisy)
                h_list.append(h_next)
                v_list.append(v_next)
                v_current = v_next

            if success:
                return {
                    'p_sim': torch.stack(p_list),
                    'q_sim': torch.stack(q_list),
                    'h_sim': torch.stack(h_list),
                    'v_low_sim': torch.stack(v_list),
                    'relative_errors': relative_errors,
                    'attempts': attempt + 1
                }

        # Failed after max retries
        if debug:
            print(f"Failed to generate valid noisy schedule after {max_retries} attempts")
        return None


class RandomModeSampler:
    """Generates random power schedules while preserving operational modes."""

    def __init__(self, params):
        """
        Initialize RandomModeSampler.

        Args:
            params: HydroParameters instance
        """
        self.params = params

    def sample_random_power(self, mode, head):
        """
        Sample random power within feasible range for given mode and head.

        Args:
            mode: Operational mode ('Turbine', 'Pump', 'Idle')
            head: Current head level

        Returns:
            float: Random power value
        """
        if mode == "Idle":
            return 0.0
        elif mode == "Turbine":
            p_min = self.params.pos_min(head).item()
            p_max = self.params.pos_max(head).item()
            return np.random.uniform(p_min, p_max)
        elif mode == "Pump":
            p_min = self.params.neg_min(head).item()
            p_max = self.params.neg_max(head).item()
            return np.random.uniform(p_min, p_max)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate_random_schedule(self, original_modes, initial_head, max_retries=100):
        """
        Generate random power schedule preserving modes from original.

        Args:
            original_modes: List of modes from original schedule
            initial_head: Initial head value
            max_retries: Maximum retry attempts

        Returns:
            dict: Results with random schedule, or None if failed
        """
        for attempt in range(max_retries):
            TH = len(original_modes)
            p_list = []
            q_list = []
            h_list = []
            v_list = []

            v_current = self.params.h_to_v_low_fitted(initial_head)
            success = True

            for i in range(TH):
                mode = original_modes[i]

                if i == 0:
                    h_current = initial_head
                else:
                    h_current = self.params.v_low_to_h_fitted(v_current)

                # Sample random power for this mode
                random_power = self.sample_random_power(mode, h_current)
                p_random = torch.tensor(random_power, dtype=torch.float32)

                # Calculate flow
                if abs(random_power) > 0.5:
                    q_random = self.params.predict_q_poly(p_random.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
                else:
                    q_random = torch.zeros_like(p_random)

                # Update volume
                v_next = v_current + q_random * 3600

                # Check constraints
                if v_next > self.params.max_vol_up or v_next < self.params.min_vol_low:
                    success = False
                    break

                h_next = self.params.v_low_to_h_fitted(v_next)

                p_list.append(p_random)
                q_list.append(q_random)
                h_list.append(h_next)
                v_list.append(v_next)
                v_current = v_next

            if success:
                return {
                    'p_sim': torch.stack(p_list),
                    'q_sim': torch.stack(q_list),
                    'h_sim': torch.stack(h_list),
                    'v_low_sim': torch.stack(v_list),
                    'attempts': attempt + 1
                }

        return None


def generate_noisy_dataset(config, params, device, noise_levels=None):
    """
    Generate noisy datasets for all specified noise levels.

    Args:
        config: DFLConfig instance
        params: HydroParameters instance
        device: PyTorch device
        noise_levels: List of noise levels (e.g., [0.1, 0.2, 0.3]) or None for default

    Returns:
        None (saves CSV files)
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Load original MIQP results
    original_file = config.get_miqp_file_path()
    df_original = pd.read_csv(original_file)

    print(f"Loaded {len(df_original)} rows from {original_file}")

    # Process each noise level
    for noise_level in noise_levels:
        print(f"\nGenerating dataset with {int(noise_level*100)}% noise...")

        # Set random seed for each noise level to ensure reproducibility
        np.random.seed(42 + int(noise_level * 1000))  # Different seed for each noise level

        # Handle 0% noise as direct copy (no simulation needed)
        if noise_level == 0.0:
            df_output = df_original.copy()
        else:
            baseline_sim = BaselineSimulator(params)
            noise_sim = NoiseSimulator(params, relative_noise_level=noise_level)
            results = []

            # Process each date
            dates = df_original['date'].unique()
            for date in dates:
                date_data = df_original[df_original['date'] == date].sort_values('hour')

                if len(date_data) != 24:
                    continue

                # Get original power schedule
                original_power = torch.tensor(date_data['power'].values, dtype=torch.float32, device=device)
                initial_head = torch.tensor(params.head_init, dtype=torch.float32, device=device)

                # STEP 1: Run baseline simulation on original data
                p_baseline, q_baseline, h_baseline, v_baseline = baseline_sim.simulate_daily_operation(
                    original_power, initial_head
                )

                # STEP 2: Apply relative noise to baseline results and re-simulate
                result = noise_sim.simulate_daily_operation_with_noise(
                    p_baseline, initial_head, max_retries=50
                )

                if result is not None:
                    # Create dataframe for this date
                    for hour in range(24):
                        row_dict = {
                            'date': date,
                            'hour': hour,
                            'power': result['p_sim'][hour].item(),
                            'head': result['h_sim'][hour].item(),
                            'volume': result['v_low_sim'][hour].item(),
                            'flow': result['q_sim'][hour].item(),
                        }
                        # Add price if available
                        if 'price' in date_data.columns:
                            row_dict['price'] = date_data.iloc[hour]['price']
                        results.append(row_dict)

            df_output = pd.DataFrame(results)
            # Reorder columns to match original format
            cols = ['date', 'hour', 'power', 'head', 'volume', 'flow']
            if 'price' in df_output.columns:
                cols.append('price')
            df_output = df_output[cols]

        # Save to CSV using config-based path
        output_file = config.get_results_file(noise_level=noise_level)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_output.to_csv(output_file, index=False)
        print(f"Saved {len(df_output)} rows to {output_file}")


def generate_random_samples_dataset(config, params, device):
    """
    Generate random samples dataset preserving operational modes.

    Args:
        config: DFLConfig instance
        params: HydroParameters instance
        device: PyTorch device

    Returns:
        None (saves CSV file)
    """
    print("\nGenerating random samples dataset...")

    # Set random seed for random samples generation
    np.random.seed(42)

    # Load original MIQP results
    original_file = config.get_miqp_file_path()
    df_original = pd.read_csv(original_file)

    random_sampler = RandomModeSampler(params)
    results = []

    # Process each date
    dates = df_original['date'].unique()
    for date in dates:
        date_data = df_original[df_original['date'] == date].sort_values('hour')

        if len(date_data) != 24:
            continue

        # Extract original modes
        original_modes = []
        for _, row in date_data.iterrows():
            power = row['power']
            if power > 0.5:
                original_modes.append("Turbine")
            elif power < -0.5:
                original_modes.append("Pump")
            else:
                original_modes.append("Idle")

        initial_head = torch.tensor(params.head_init, dtype=torch.float32, device=device)

        # Generate random schedule
        result = random_sampler.generate_random_schedule(original_modes, initial_head, max_retries=100)

        if result is not None:
            for hour in range(24):
                row_dict = {
                    'date': date,
                    'hour': hour,
                    'power': result['p_sim'][hour].item(),
                    'head': result['h_sim'][hour].item(),
                    'volume': result['v_low_sim'][hour].item(),
                    'flow': result['q_sim'][hour].item(),
                }
                # Add price if available
                if 'price' in date_data.columns:
                    row_dict['price'] = date_data.iloc[hour]['price']
                results.append(row_dict)

    # Save to CSV using config-based path
    df_output = pd.DataFrame(results)
    # Reorder columns to match original format
    cols = ['date', 'hour', 'power', 'head', 'volume', 'flow']
    if 'price' in df_output.columns:
        cols.append('price')
    df_output = df_output[cols]

    output_file = config.get_results_file(random_samples=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_output.to_csv(output_file, index=False)
    print(f"Saved {len(df_output)} rows to {output_file}")
