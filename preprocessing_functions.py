"""
Preprocessing Functions Module

This module loads the existing preprocess.pkl file and provides clean, 
importable functions without namespace issues.
"""

import torch
import numpy as np
import pandas as pd
import dill as pickle
from pathlib import Path
import sys

# Set device
device = torch.device("cpu")

# Global variables to store loaded data
_data_loaded = False
_preprocessing_data = {}

def load_preprocessing_data():
    """Load data from the existing preprocess.pkl file"""
    global _data_loaded, _preprocessing_data
    
    if _data_loaded:
        return _preprocessing_data
    
    try:
        with open('preprocess.pkl', 'rb') as f:
            # Your original pickle contains a tuple with all the data
            loaded_tuple = pickle.load(f)
            
        # Unpack the tuple based on your original save order
        (v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted_orig, v_low_h_poly, 
         h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, 
         intercept_pump_lin, predict_q_linear_tur_orig, predict_q_linear_pump_orig, 
         h_to_v_low_lin_orig, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, 
         pos_max_fit, h_v_poly, h_v_coeffs_new, DA_price_hour, DA_price_quarter, 
         h_to_v_low_fitted_orig_new, predict_q_poly_orig, neg_min_orig, neg_max_orig, 
         pos_min_orig, pos_max_orig, prepare_and_fit_model_orig, get_UPC_bound_orig, 
         LR_UPC_bound_orig) = loaded_tuple
        
        # Store the data we need
        _preprocessing_data = {
            'v_low_h_coeffs': v_low_h_coeffs,
            'h_v_coeffs': h_v_coeffs,
            'h_vlow_coeff_lin': h_vlow_coeff_lin,
            'coefs_tur_lin': coefs_tur_lin,
            'intercept_tur_lin': intercept_tur_lin,
            'coefs_pump_lin': coefs_pump_lin,
            'intercept_pump_lin': intercept_pump_lin,
            'h_fit': h_fit,
            'neg_min_fit': neg_min_fit,
            'neg_max_fit': neg_max_fit,
            'pos_min_fit': pos_min_fit,
            'pos_max_fit': pos_max_fit,
            'h_v_poly': h_v_poly,
            'h_v_coeffs_new': h_v_coeffs_new,
            'DA_price_hour': DA_price_hour,
            'DA_price_quarter': DA_price_quarter,
        }
        
        # Try to extract polynomial coefficients if they exist as torch tensors
        try:
            # Look for polynomial coefficients in the original functions
            # We'll extract them by calling the functions with dummy values and reverse engineering
            # Or if they're stored as global variables, we can access them
            
            # For now, we'll set these to None and handle them in the functions
            _preprocessing_data['poly_neg_min_fit'] = None
            _preprocessing_data['poly_neg_max_fit'] = None
            _preprocessing_data['poly_pos_min_fit'] = None
            _preprocessing_data['poly_pos_max_fit'] = None
            _preprocessing_data['coefs_tur'] = None
            _preprocessing_data['intercept_tur'] = None
            _preprocessing_data['coefs_pump'] = None
            _preprocessing_data['intercept_pump'] = None
            
        except Exception as e:
            print(f"Warning: Could not extract polynomial coefficients: {e}")
        
        _data_loaded = True
        return _preprocessing_data
        
    except Exception as e:
        print(f"Error loading preprocessing data: {e}")
        print("Make sure preprocess.pkl exists in the current directory")
        return {}

# Load data when module is imported
_preprocessing_data = load_preprocessing_data()

def get_coefficients():
    """Get all coefficients as a dictionary"""
    return _preprocessing_data

def h_to_v_low_fitted(head=None):
    """
    Convert head to v_low using fitted polynomial
    
    Args:
        head: head value (can be tensor or scalar)
    
    Returns:
        torch.Tensor: corresponding v_low value
    """
    if head is None:
        return None
        
    h_v_coeffs = _preprocessing_data.get('h_v_coeffs_new', _preprocessing_data.get('h_v_coeffs'))
    if h_v_coeffs is None:
        raise ValueError("h_v_coeffs not found in preprocessing data")
    
    # Convert input to tensor if it's not already
    if not isinstance(head, torch.Tensor):
        head = torch.tensor(head, dtype=torch.float32, device=device)
    
    # Ensure coefficients are on the correct device
    if hasattr(h_v_coeffs, 'to'):
        h_v_coeffs = h_v_coeffs.to(device)
    
    # Evaluate the polynomial using Horner's method
    result = h_v_coeffs[0]
    for i in range(1, len(h_v_coeffs)):
        result = result * head + h_v_coeffs[i]
    
    return result

def v_low_to_h_fitted(v_low=None):
    """
    Convert v_low to head using fitted polynomial
    
    Args:
        v_low: v_low value (can be tensor or scalar)
    
    Returns:
        torch.Tensor: corresponding head value
    """
    if v_low is None:
        return None
        
    v_low_h_coeffs = _preprocessing_data.get('v_low_h_coeffs')
    if v_low_h_coeffs is None:
        raise ValueError("v_low_h_coeffs not found in preprocessing data")
    
    # Convert input to tensor if it's not already
    if not isinstance(v_low, torch.Tensor):
        v_low = torch.tensor(v_low, dtype=torch.float32, device=device)
    elif v_low.device != device:
        v_low = v_low.to(device)
    
    # Ensure coefficients are on the correct device
    if hasattr(v_low_h_coeffs, 'to'):
        v_low_h_coeffs = v_low_h_coeffs.to(device)
    
    # Evaluate the polynomial using Horner's method
    result = v_low_h_coeffs[0]
    for i in range(1, len(v_low_h_coeffs)):
        result = result * v_low + v_low_h_coeffs[i]
    
    return result

def h_to_v_low_lin(head):
    """
    Linear conversion from head to v_low
    
    Args:
        head: head value
    
    Returns:
        v_low value using linear relationship
    """
    h_v_poly = _preprocessing_data.get('h_v_poly')
    if h_v_poly is None:
        # Fallback to linear coefficients
        h_vlow_coeff_lin = _preprocessing_data.get('h_vlow_coeff_lin')
        if h_vlow_coeff_lin is not None:
            return np.polyval(h_vlow_coeff_lin, head)
        else:
            raise ValueError("Neither h_v_poly nor h_vlow_coeff_lin found in preprocessing data")
    
    return h_v_poly(head)

def neg_min(h, coefficients=None):
    """p >= neg_min(h), in pump mode"""
    if coefficients is None:
        coefficients = _preprocessing_data.get('poly_neg_min_fit')
        if coefficients is None:
            # Fallback to linear fit
            neg_min_fit = _preprocessing_data.get('neg_min_fit')
            if neg_min_fit is not None:
                return torch.tensor(np.polyval(neg_min_fit, h.cpu().numpy() if isinstance(h, torch.Tensor) else h), 
                                  dtype=torch.float32, device=device)
            else:
                raise ValueError("No neg_min coefficients found")
    
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def neg_max(h, coefficients=None):
    """p <= neg_max(h), in pump mode"""
    if coefficients is None:
        coefficients = _preprocessing_data.get('poly_neg_max_fit')
        if coefficients is None:
            # Fallback to linear fit
            neg_max_fit = _preprocessing_data.get('neg_max_fit')
            if neg_max_fit is not None:
                return torch.tensor(np.polyval(neg_max_fit, h.cpu().numpy() if isinstance(h, torch.Tensor) else h), 
                                  dtype=torch.float32, device=device)
            else:
                raise ValueError("No neg_max coefficients found")
    
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def pos_min(h, coefficients=None):
    """p >= pos_min(h), in turbine mode"""
    if coefficients is None:
        coefficients = _preprocessing_data.get('poly_pos_min_fit')
        if coefficients is None:
            # Fallback to linear fit
            pos_min_fit = _preprocessing_data.get('pos_min_fit')
            if pos_min_fit is not None:
                return torch.tensor(np.polyval(pos_min_fit, h.cpu().numpy() if isinstance(h, torch.Tensor) else h), 
                                  dtype=torch.float32, device=device)
            else:
                raise ValueError("No pos_min coefficients found")
    
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def pos_max(h, coefficients=None):
    """p <= pos_max(h), in turbine mode"""
    if coefficients is None:
        coefficients = _preprocessing_data.get('poly_pos_max_fit')
        if coefficients is None:
            # Fallback to linear fit
            pos_max_fit = _preprocessing_data.get('pos_max_fit')
            if pos_max_fit is not None:
                return torch.tensor(np.polyval(pos_max_fit, h.cpu().numpy() if isinstance(h, torch.Tensor) else h), 
                                  dtype=torch.float32, device=device)
            else:
                raise ValueError("No pos_max coefficients found")
    
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def predict_q_linear_tur(p, h, coefs=None, intercept=None):
    """
    Linear prediction of flow rate for turbine mode.
    """
    if coefs is None:
        coefs = _preprocessing_data.get('coefs_tur_lin')
    if intercept is None:
        intercept = _preprocessing_data.get('intercept_tur_lin')
    
    if coefs is None or intercept is None:
        raise ValueError("Turbine coefficients not found in preprocessing data")
    
    # Convert to tensors if needed
    if not isinstance(coefs, torch.Tensor):
        coefs = torch.tensor(coefs, dtype=torch.float32, device=device)
    if not isinstance(intercept, torch.Tensor):
        intercept = torch.tensor(intercept, dtype=torch.float32, device=device)
    
    # Create feature matrix [p, h]
    features = torch.stack([p, h], dim=-1)
    
    # Compute linear prediction q = c_p*p + c_h*h + intercept
    q = torch.einsum('...d,d->...', features, coefs) + intercept
    
    # Zero out predictions for non-turbine mode (p ≤ 0)
    mask_tur = (p > 0)
    return torch.where(mask_tur, q, torch.zeros_like(q).to(device))

def predict_q_linear_pump(p, h, coefs=None, intercept=None):
    """
    Linear prediction of flow rate for pump mode.
    """
    if coefs is None:
        coefs = _preprocessing_data.get('coefs_pump_lin')
    if intercept is None:
        intercept = _preprocessing_data.get('intercept_pump_lin')
    
    if coefs is None or intercept is None:
        raise ValueError("Pump coefficients not found in preprocessing data")
    
    # Convert to tensors if needed
    if not isinstance(coefs, torch.Tensor):
        coefs = torch.tensor(coefs, dtype=torch.float32, device=device)
    if not isinstance(intercept, torch.Tensor):
        intercept = torch.tensor(intercept, dtype=torch.float32, device=device)
    
    # Create feature matrix [p, h]
    features = torch.stack([p, h], dim=-1)
    
    # Compute linear prediction q = c_p*p + c_h*h + intercept
    q = torch.einsum('...d,d->...', features, coefs) + intercept
    
    # Zero out predictions for non-pump mode (p ≥ 0)
    mask_pump = (p < 0)
    return torch.where(mask_pump, q, torch.zeros_like(q).to(device))

def predict_q_poly(p, h, coefs_tur=None, intercept_tur=None,
                  coefs_pump=None, intercept_pump=None, 
                  poly_degree=5):
    """
    Vectorized prediction of q values from p and h using polynomial features
    """
    # Use default coefficients if not provided
    if coefs_tur is None:
        coefs_tur = _preprocessing_data.get('coefs_tur')
    if intercept_tur is None:
        intercept_tur = _preprocessing_data.get('intercept_tur')
    if coefs_pump is None:
        coefs_pump = _preprocessing_data.get('coefs_pump')
    if intercept_pump is None:
        intercept_pump = _preprocessing_data.get('intercept_pump')
    
    # Check if coefficients are available
    if any(x is None for x in [coefs_tur, intercept_tur, coefs_pump, intercept_pump]):
        raise ValueError("Polynomial coefficients not found. You may need to run the polynomial fitting section in preprocessing.py")
    
    # Ensure inputs are torch tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32, device=device)
    if not isinstance(h, torch.Tensor):
        h = torch.tensor(h, dtype=torch.float32, device=device)
        
    # Generate power matrix
    max_power = poly_degree + 1
    
    # Precompute all possible power combinations (p^a * h^b, a + b <= poly_degree)
    powers = torch.arange(max_power, device=p.device)
    p_pows = torch.pow(p.unsqueeze(-1), powers)  # [..., max_power]
    h_pows = torch.pow(h.unsqueeze(-1), powers)  # [..., max_power]
    
    # Generate polynomial feature matrix (avoid loops)
    terms = []
    for total_degree in range(1, poly_degree + 1):
        for a in range(total_degree, -1, -1):
            b = total_degree - a
            if b <= total_degree and a + b <= poly_degree:
                terms.append(p_pows[..., a] * h_pows[..., b])
    
    features = torch.stack(terms, dim=-1)  # [..., num_features]
    
    # Combine turbine and pump coefficient calculations
    q_tur = torch.einsum('...f,f->...', features, coefs_tur) + intercept_tur
    q_pump = torch.einsum('...f,f->...', features, coefs_pump) + intercept_pump
    
    # Vectorized conditional selection
    mask_tur = (p > 0)
    mask_pump = (p < 0)
    q = torch.where(mask_tur, q_tur, torch.where(mask_pump, q_pump, torch.zeros_like(p).to(p.device)))
        
    return q

def get_DA_price_hour():
    """Get day-ahead price hourly data"""
    return _preprocessing_data.get('DA_price_hour')

def get_DA_price_quarter():
    """Get day-ahead price quarterly data"""
    return _preprocessing_data.get('DA_price_quarter')

def read_da_price(date, file_path="./Data/Day-ahead Prices_202301010000-202401010000-2.csv"):
    """
    Read day-ahead prices for a specific date
    
    Args:
        date: Date string in format "MM.DD.YYYY"
        file_path: Path to the CSV file
    
    Returns:
        torch.Tensor: Price data for the specified date
    """
    # Load the data
    data = pd.read_csv(file_path)
    
    # Convert the date string to datetime format for easier filtering
    data['Date'] = pd.to_datetime(data['MTU (CET/CEST)'].str[:10], format='%d.%m.%Y')
    
    # Filter the data for the given date
    filtered_data = data[data['Date'] == date]['Day-ahead Price [EUR/MWh]']
    
    # Convert the series to a torch tensor
    tensor_data = torch.tensor(filtered_data.values, dtype=torch.float).to(device)
    
    return tensor_data

def hourly_to_quarterly(tensor_data):
    """Convert hourly data to quarterly by repeating each element 4 times"""
    quarterly_data = tensor_data.repeat_interleave(4)
    return quarterly_data

# Utility function to check what data is available
def list_available_data():
    """Print all available data keys"""
    print("Available preprocessing data:")
    for key in _preprocessing_data.keys():
        value = _preprocessing_data[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: torch.Tensor with shape {value.shape}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: numpy.ndarray with shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

if __name__ == "__main__":
    # Test the functions when run directly
    print("Testing preprocessing functions...")
    
    try:
        # Test basic functions
        test_head = torch.tensor(80.0, device=device)
        v_low_result = h_to_v_low_fitted(test_head)
        print(f"h_to_v_low_fitted(80): {v_low_result}")
        
        head_result = v_low_to_h_fitted(v_low_result)
        print(f"v_low_to_h_fitted({v_low_result}): {head_result}")
        
        # List available data
        list_available_data()
        
    except Exception as e:
        print(f"Error testing functions: {e}")
        import traceback
        traceback.print_exc()
