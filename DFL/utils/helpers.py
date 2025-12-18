"""
Utility helper functions for DFL.

This module contains common utility functions used across the DFL framework.
"""

import torch
import sys


def hourly_to_quarterly(tensor_data):
    """
    Convert hourly data to quarterly (15-minute) data by repeating each value 4 times.

    Args:
        tensor_data: Tensor with hourly data

    Returns:
        Tensor with quarterly data (4x longer)
    """
    return tensor_data.repeat_interleave(4)


def setup_device():
    """
    Setup and return the PyTorch device.

    Returns:
        torch.device: CPU or CUDA device
    """
    # Use CPU for now (can be changed to support CUDA if needed)
    device = torch.device("cpu")
    return device


def load_portfolio_data(library_path="../Library"):
    """
    Load portfolio data from the Library module.

    Args:
        library_path: Path to the Library directory

    Returns:
        dict: Dictionary containing portfolio parameters
    """
    sys.path.append(library_path)

    try:
        from V_H_relations import load_portfolio_data as load_portfolio
        load_portfolio()

        # Import all the parameters
        from V_H_relations import (
            r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R,
            height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low,
            max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up,
            target_vol_low, target_head, gross_head, get_v_low
        )

        return {
            'r': r, 'm': m, 'head_max': head_max, 'head_min': head_min,
            'h_dead_up': h_dead_up, 'h_normal_up': h_normal_up, 'height_up': height_up,
            'R': R, 'height_low': height_low, 'n': n, 'h_dead_low': h_dead_low,
            'h_normal_low': h_normal_low, 'max_vol_up': max_vol_up,
            'max_vol_low': max_vol_low, 'max_vol': max_vol,
            'ramp_down': ramp_down, 'ramp_up': ramp_up, 'min_vol_low': min_vol_low,
            'target_vol_up': target_vol_up, 'target_vol_low': target_vol_low,
            'target_head': target_head, 'gross_head': gross_head, 'get_v_low': get_v_low
        }

    except ImportError as e:
        print(f"Warning: Could not load portfolio data: {e}")
        return None


def load_preprocessed_data(preprocess_file="../preprocess.pkl"):
    """
    Load preprocessed functions and data from pickle file.

    Args:
        preprocess_file: Path to the preprocess.pkl file

    Returns:
        dict: Dictionary containing all preprocessed data and functions
    """
    import dill as pickle

    try:
        with open(preprocess_file, 'rb') as f:
            data = pickle.load(f)

        # Data is a tuple, unpack it
        (v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin,
         coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin,
         predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit,
         neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs,
         DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly,
         neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model,
         get_UPC_bound, LR_UPC_bound) = data

        return {
            'v_low_h_coeffs': v_low_h_coeffs,
            'h_v_coeffs': h_v_coeffs,
            'v_low_to_h_fitted': v_low_to_h_fitted,
            'v_low_h_poly': v_low_h_poly,
            'h_vlow_coeff_lin': h_vlow_coeff_lin,
            'coefs_tur_lin': coefs_tur_lin,
            'intercept_tur_lin': intercept_tur_lin,
            'coefs_pump_lin': coefs_pump_lin,
            'intercept_pump_lin': intercept_pump_lin,
            'predict_q_linear_tur': predict_q_linear_tur,
            'predict_q_linear_pump': predict_q_linear_pump,
            'h_to_v_low_lin': h_to_v_low_lin,
            'h_fit': h_fit,
            'neg_min_fit': neg_min_fit,
            'neg_max_fit': neg_max_fit,
            'pos_min_fit': pos_min_fit,
            'pos_max_fit': pos_max_fit,
            'h_v_poly': h_v_poly,
            'h_v_coeffs': h_v_coeffs,
            'DA_price_hour': DA_price_hour,
            'DA_price_quarter': DA_price_quarter,
            'h_to_v_low_fitted': h_to_v_low_fitted,
            'predict_q_poly': predict_q_poly,
            'neg_min': neg_min,
            'neg_max': neg_max,
            'pos_min': pos_min,
            'pos_max': pos_max,
            'prepare_and_fit_model': prepare_and_fit_model,
            'get_UPC_bound': get_UPC_bound,
            'LR_UPC_bound': LR_UPC_bound
        }

    except Exception as e:
        print(f"Warning: Could not load preprocessed data: {e}")
        return None


def initialize_head_and_volume(h_to_v_low_fitted, device=None):
    """
    Initialize head and volume values.

    Args:
        h_to_v_low_fitted: Fitted function to convert head to lower volume
        device: PyTorch device

    Returns:
        tuple: (head_init, v_low_init) as tensors
    """
    if device is None:
        device = torch.device("cpu")

    head_init = torch.tensor(77.0, device=device, dtype=torch.float32)
    v_low_init = torch.tensor(h_to_v_low_fitted(head_init), device=device, dtype=torch.float32)

    print(f"Initial head: {head_init.item()}, Initial v_low: {v_low_init.item()}")

    return head_init, v_low_init
