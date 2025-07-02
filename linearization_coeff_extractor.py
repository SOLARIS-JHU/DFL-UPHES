# %% Import libraries
import torch
import dill as pickle
import pandas as pd
import sys
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# torch.autograd.set_detect_anomaly(True)


device = torch.device("cpu")

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

head_init = 77.0 # Initial head value
v_low_init = h_to_v_low_fitted(head_init) # Initial lower reservoir volume
# convert v_low_init to float
v_low_init = float(v_low_init)
# convert h_vlow_coeff_lin to python array
h_vlow_coeff_lin = h_vlow_coeff_lin.detach().numpy()
# Import from pipeline file
# from pipeline_learnable_weight_ver import HydroParameters, RegressionLayer, TaylorRegressionLayer

# %%
class HydroParameters:
    def __init__(
        self,
        time_horizon=24, # number of time periods
        sampling_rate=50, # number of samples for regression
        δ_p=1,
        δ_h=2,
        δ_q=1,
        operational_cost=3.8,
        rho=1000,
        g=9.81,
        mu=0.9,
        head_min=head_min,
        head_max=head_max,
        max_vol_up=max_vol_up,
        min_vol_low=min_vol_low,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        target_head=target_head,
        target_vol_low=target_vol_low,
        head_init=head_init,
        v_low_init=v_low_init,
        neg_min_fit=neg_min_fit, 
        neg_max_fit=neg_max_fit,   
        pos_min_fit=pos_min_fit,     
        pos_max_fit=pos_max_fit,
        neg_min=neg_min,
        neg_max=neg_max,
        pos_min=pos_min,
        pos_max=pos_max,
        predict_q_poly=predict_q_poly,
        h_to_v_low_fitted=h_to_v_low_fitted,
        gross_head=gross_head
    ):
        self.time_horizon = time_horizon
        self.sampling_rate = sampling_rate
        self.operational_cost = operational_cost
        
        self.δ_p = torch.tensor(δ_p, dtype=torch.float32)
        self.δ_h = torch.tensor(δ_h, dtype=torch.float32)
        self.δ_q = torch.tensor(δ_q, dtype=torch.float32)
        self.rho = torch.tensor(rho, dtype=torch.float32)
        self.g = torch.tensor(g, dtype=torch.float32)
        self.mu = torch.tensor(mu, dtype=torch.float32)

        self.head_min = torch.tensor(head_min, dtype=torch.float32)
        self.head_max = torch.tensor(head_max, dtype=torch.float32)
        self.max_vol_up = torch.tensor(max_vol_up, dtype=torch.float32)
        self.min_vol_low = torch.tensor(min_vol_low, dtype=torch.float32)
        self.ramp_up = torch.tensor(ramp_up, dtype=torch.float32)
        self.ramp_down = torch.tensor(ramp_down, dtype=torch.float32)

        self.target_head = torch.tensor(target_head, dtype=torch.float32)
        self.target_vol_low = torch.tensor(target_vol_low, dtype=torch.float32)
        self.head_init = torch.tensor(head_init, dtype=torch.float32)
        self.v_low_init = torch.tensor(v_low_init, dtype=torch.float32)

        self.neg_min_fit = torch.tensor(neg_min_fit, dtype=torch.float32)
        self.neg_max_fit = torch.tensor(neg_max_fit, dtype=torch.float32)
        self.pos_min_fit = torch.tensor(pos_min_fit, dtype=torch.float32)
        self.pos_max_fit = torch.tensor(pos_max_fit, dtype=torch.float32)

        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.predict_q_poly = predict_q_poly
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.gross_head = gross_head

class RegressionLayer: # with hard constraints
    def __init__(self, params: HydroParameters):
        self.params = params

    def least_squares_UPC_torch(self, p_samples, h_samples, q_values):
        """
        Perform least squares for q = c*p + d*h + e with gradient tracking.
        Ensures proper shape handling for matrix operations.
        """
        # Ensure inputs are tensors with gradients and proper shapes
        p_samples = p_samples.detach().clone().requires_grad_(True)
        h_samples = h_samples.detach().clone().requires_grad_(True)
        q_values = q_values.detach().clone().requires_grad_(True)
        
        # Reshape inputs to ensure proper dimensions
        p_samples = p_samples.view(-1)  # Flatten to 1D
        h_samples = h_samples.view(-1)  # Flatten to 1D
        q_values = q_values.view(-1)    # Flatten to 1D
        
        # Create design matrix with gradient tracking
        ones = torch.ones_like(p_samples, requires_grad=True)
        X = torch.stack([p_samples, h_samples, ones], dim=1)  # Shape: [n_samples, 3]
        y = q_values.view(-1, 1)  # Shape: [n_samples, 1]
        
        # Compute least squares solution with gradient tracking
        XTX = torch.matmul(X.t(), X)  # Shape: [3, 3]
        XTy = torch.matmul(X.t(), y)  # Shape: [3, 1]
        
        # Add small regularization for numerical stability
        epsilon = 1e-6
        reg_matrix = epsilon * torch.eye(3, device=XTX.device)
        XTX_reg = XTX + reg_matrix
        
        beta = torch.matmul(torch.inverse(XTX_reg), XTy)
        return beta.squeeze()

    def least_squares_v_low_torch(self, h_samples, v_low_samples):
        """
        Perform least squares for v_low = a*h + b with gradient tracking.
        Ensures proper shape handling for matrix operations.
        """
        # Ensure inputs are tensors with gradients and proper shapes
        h_samples = h_samples.detach().clone().requires_grad_(True)
        v_low_samples = v_low_samples.detach().clone().requires_grad_(True)
        
        # Reshape inputs to ensure proper dimensions
        h_samples = h_samples.view(-1)      # Flatten to 1D
        v_low_samples = v_low_samples.view(-1)  # Flatten to 1D
        
        # Create design matrix with gradient tracking
        ones = torch.ones_like(h_samples, requires_grad=True)
        X = torch.stack([h_samples, ones], dim=1)  # Shape: [n_samples, 2]
        y = v_low_samples.view(-1, 1)  # Shape: [n_samples, 1]
        
        # Compute least squares solution with gradient tracking
        XTX = torch.matmul(X.t(), X)  # Shape: [2, 2]
        XTy = torch.matmul(X.t(), y)  # Shape: [2, 1]
        
        # Add small regularization for numerical stability
        epsilon = 1e-6
        reg_matrix = epsilon * torch.eye(2, device=XTX.device)
        XTX_reg = XTX + reg_matrix
        
        beta = torch.matmul(torch.inverse(XTX_reg), XTy)
        return beta.squeeze()

    def run_regression(self, power, head):
        """
        Run regression with gradient tracking enabled.
        Handles batch operations properly.
        """
        TH = self.params.time_horizon
        device = power.device
        c_list, d_list, e_list = [], [], []
        a_list, b_list = [], []

        for t in range(TH):
            # Sample points around current operating point
            p_center = power[t].detach()
            h_center = head[t].detach()
            
            # Create sample points with gradient tracking
            p_samples = torch.linspace(
                float(p_center - self.params.δ_p),
                float(p_center + self.params.δ_p),
                self.params.sampling_rate,
                device=device,
                requires_grad=True
            )
            
            h_lo = torch.max(self.params.head_min, h_center - self.params.δ_h)
            h_hi = torch.min(self.params.head_max, h_center + self.params.δ_h)
            h_samples = torch.linspace(
                float(h_lo), 
                float(h_hi), 
                self.params.sampling_rate,
                device=device,
                requires_grad=True
            )
            
            # Idle mode check - if power is approximately zero, don't try to sample around it
            if abs(p_center) < 1e-6:  # Idle mode
                c_list.append(torch.zeros(1, device=device, requires_grad=True))
                d_list.append(torch.zeros(1, device=device, requires_grad=True))
                e_list.append(torch.zeros(1, device=device, requires_grad=True))
            else:
                # Create meshgrid
                p_mesh, h_mesh = torch.meshgrid(p_samples, h_samples, indexing="ij")
                p_flat = p_mesh.flatten()
                h_flat = h_mesh.flatten()

                # Filter valid regions
                mask_turbine = (
                    (p_flat >= self.params.pos_min_fit[0]*h_flat + self.params.pos_min_fit[1]) &
                    (p_flat <= self.params.pos_max_fit[0]*h_flat + self.params.pos_max_fit[1])
                )
                mask_pump = (
                    (p_flat >= self.params.neg_min_fit[0]*h_flat + self.params.neg_min_fit[1]) &
                    (p_flat <= self.params.neg_max_fit[0]*h_flat + self.params.neg_max_fit[1])
                )
                mask = mask_turbine | mask_pump
                
                if not mask.any():
                    # Handle case with no valid points - use zeros with proper size
                    c_list.append(torch.zeros(1, device=device, requires_grad=True))
                    d_list.append(torch.zeros(1, device=device, requires_grad=True))
                    e_list.append(torch.zeros(1, device=device, requires_grad=True))
                    print(f"Warning: No valid points found for timestep {t}")
                else:
                    p_valid = p_flat[mask]
                    h_valid = h_flat[mask]
                    
                    # Vectorized q_values calculation
                    q_values = self.params.predict_q_poly(p_valid, h_valid)
                    
                    try:
                        beta = self.least_squares_UPC_torch(p_valid, h_valid, q_values)
                        c_list.append(beta[0].view(1))  # Ensure consistent shape
                        d_list.append(beta[1].view(1))
                        e_list.append(beta[2].view(1))
                    except Exception as e:
                        print(f"Error in least squares for timestep {t}: {e}")
                        # Use zeros for this timestep
                        c_list.append(torch.zeros(1, device=device, requires_grad=True))
                        d_list.append(torch.zeros(1, device=device, requires_grad=True))
                        e_list.append(torch.zeros(1, device=device, requires_grad=True))

            # Regression for v_low - always calculate this regardless of operating mode
            h_samples_2 = torch.linspace(
                float(h_lo), 
                float(h_hi), 
                self.params.sampling_rate,
                device=device,
                requires_grad=True
            )
            
            # Calculate v_low values
            v_low_values = torch.tensor([
                self.params.h_to_v_low_fitted(h.item()) 
                for h in h_samples_2
            ], dtype=torch.float32, device=device, requires_grad=True)
            
            try:
                beta_v = self.least_squares_v_low_torch(h_samples_2, v_low_values)
                a_list.append(beta_v[0].view(1))  # Ensure consistent shape
                b_list.append(beta_v[1].view(1))
            except Exception as e:
                print(f"Error in v_low regression for timestep {t}: {e}")
                # Use zeros for this timestep
                a_list.append(torch.zeros(1, device=device, requires_grad=True))
                b_list.append(torch.zeros(1, device=device, requires_grad=True))

        # Stack results with gradient tracking
        c_tensor = torch.stack(c_list)
        d_tensor = torch.stack(d_list)
        e_tensor = torch.stack(e_list)
        a_tensor = torch.stack(a_list)
        b_tensor = torch.stack(b_list)

        return c_tensor, d_tensor, e_tensor, a_tensor, b_tensor
    
class TaylorRegressionLayer:
    def __init__(self, params: HydroParameters):
        self.params = params
        
    def compute_UPC_derivatives(self, p, h):
        """
        Compute partial derivatives of q = predict_q_poly(p, h) at point (p, h)
        using numerical differentiation since predict_q_poly may not be directly differentiable.
        
        Args:
            p (float): Power value to evaluate derivative at
            h (float): Head value to evaluate derivative at
            
        Returns:
            tuple: (dq/dp, dq/dh) evaluated at (p, h)
        """
        eps = 1e-6  # Small value for numerical differentiation
        
        # Compute dq/dp using central difference
        q_p_plus = self.params.predict_q_poly(torch.tensor(p + eps), torch.tensor(h))
        q_p_minus = self.params.predict_q_poly(torch.tensor(p - eps), torch.tensor(h))
        dq_dp = (q_p_plus - q_p_minus) / (2 * eps)
        
        # Compute dq/dh using central difference
        q_h_plus = self.params.predict_q_poly(torch.tensor(p), torch.tensor(h + eps))
        q_h_minus = self.params.predict_q_poly(torch.tensor(p), torch.tensor(h - eps))
        dq_dh = (q_h_plus - q_h_minus) / (2 * eps)
        
        # Convert to tensors with gradient tracking
        dq_dp = torch.tensor(float(dq_dp), requires_grad=True)
        dq_dh = torch.tensor(float(dq_dh), requires_grad=True)
        
        return dq_dp, dq_dh

    def compute_volume_derivatives(self, h):
        """
        Compute derivative of v_low = h_to_v_low_fitted(h) at point h
        using numerical differentiation.
        
        Args:
            h (float): Head value to evaluate derivative at
            
        Returns:
            torch.Tensor: dv/dh evaluated at h
        """
        eps = 1e-6  # Small value for numerical differentiation
        
        # Compute dv/dh using central difference
        v_plus = self.params.h_to_v_low_fitted(h + eps)
        v_minus = self.params.h_to_v_low_fitted(h - eps)
        dv_dh = (v_plus - v_minus) / (2 * eps)
        
        # Convert to tensor with gradient tracking
        return torch.tensor(float(dv_dh), requires_grad=True)

    def run_regression(self, power, head):
        """
        Compute local linear approximations using numerical derivatives
        around each operating point.
        
        Args:
            power (torch.Tensor): Power schedule [time_horizon]
            head (torch.Tensor): Head schedule [time_horizon]
            
        Returns:
            tuple: Coefficients (c, d, e, a, b) for linear approximations
                  q ≈ c*p + d*h + e
                  v_low ≈ a*h + b
        """
        TH = self.params.time_horizon
        device = power.device
        
        c_list = []
        d_list = []
        e_list = []
        a_list = []
        b_list = []
        
        for t in range(TH):
            p_t = float(power[t])  # Convert to Python float
            h_t = float(head[t])   # Convert to Python float
            
            if abs(p_t) < 1e-6:  # Idle mode
                c_list.append(torch.zeros(1, device=device, requires_grad=True))
                d_list.append(torch.zeros(1, device=device, requires_grad=True))
                e_list.append(torch.zeros(1, device=device, requires_grad=True))
            else:
                try:
                    # Compute UPC derivatives
                    dq_dp, dq_dh = self.compute_UPC_derivatives(p_t, h_t)
                    
                    # Get q value at operating point
                    q_t = float(self.params.predict_q_poly(
                        torch.tensor(p_t), 
                        torch.tensor(h_t)
                    ))
                    
                    # Compute coefficients ensuring gradient tracking
                    c_list.append(dq_dp.to(device))
                    d_list.append(dq_dh.to(device))
                    e_list.append(torch.tensor(
                        q_t - float(dq_dp)*p_t - float(dq_dh)*h_t,
                        device=device,
                        requires_grad=True
                    ))
                except Exception as e:
                    print(f"Error in UPC derivatives at t={t}: {e}")
                    raise
            
            try:
                # Compute volume derivatives
                dv_dh = self.compute_volume_derivatives(h_t)
                v_t = float(self.params.h_to_v_low_fitted(h_t))
                
                # Compute coefficients ensuring gradient tracking
                a_list.append(dv_dh.to(device))
                b_list.append(torch.tensor(
                    v_t - float(dv_dh)*h_t,
                    device=device,
                    requires_grad=True
                ))
            except Exception as e:
                print(f"Error in volume derivatives at t={t}: {e}")
                raise
        
        # Stack results maintaining gradient tracking
        c = torch.stack(c_list)
        d = torch.stack(d_list)
        e = torch.stack(e_list)
        a = torch.stack(a_list)
        b = torch.stack(b_list)
        
        return c, d, e, a, b

# %% Database reading functions
def read_database(file_path="./Data/database_no_piecewise.csv"):
    """
    Read the hydro pump-storage optimization database.
    
    Parameters:
        file_path (str): Path to the CSV database file
        
    Returns:
        pd.DataFrame: The loaded database with additional calculated columns
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['Time', 'Power', 'Head', 'Flow', 'Price', 'Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Add 'Mode' column if not present
        if 'Mode' not in df.columns:
            # Determine mode based on power and flow values
            conditions = [
                (abs(df['Power']) < 0.01),  # Idle mode (power close to zero)
                (df['Power'] > 0),          # Turbine mode (positive power)
                (df['Power'] < 0)           # Pump mode (negative power)
            ]
            choices = ['Idle', 'Turbine', 'Pump']
            df['Mode'] = np.select(conditions, choices, default='Unknown')
        
        # Convert Date to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Calculate additional metrics
        df['Revenue'] = df['Power'] * df['Price']  # Revenue calculation
        
        print(f"Successfully loaded database with {len(df)} entries.")
        print(f"Number of unique dates: {df['Date'].nunique()}")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading database: {e}")
        return None

def visualize_date(df, date):
    """
    Visualize results for a specific date.
    
    Parameters:
        df (pd.DataFrame): The database dataframe
        date (str): Date to visualize in 'YYYY-MM-DD' format
    """
    if df is None:
        return
    
    # Convert string date to datetime if needed
    target_date = pd.to_datetime(date)
    day_data = df[df['Date'] == target_date]
    
    if len(day_data) == 0:
        print(f"No data found for date: {date}")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Power and Price
    ax1 = axes[0]
    ax1.set_title(f"Power and Price for {date}")
    ax1.plot(day_data['Time'], day_data['Power'], 'b-', label='Power (MW)')
    ax1.set_ylabel('Power (MW)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Price on secondary y-axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(day_data['Time'], day_data['Price'], 'r--', label='Price ($/MWh)')
    ax1_twin.set_ylabel('Price ($/MWh)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Highlight operation modes with background colors
    for mode, color, alpha in zip(['Turbine', 'Pump', 'Idle'], 
                                 ['green', 'red', 'gray'], 
                                 [0.2, 0.2, 0.1]):
        for t in day_data[day_data['Mode'] == mode]['Time']:
            ax1.axvspan(t-0.5, t+0.5, alpha=alpha, color=color)
    
    # Add legend for modes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Turbine Mode'),
        Patch(facecolor='red', alpha=0.2, label='Pump Mode'),
        Patch(facecolor='gray', alpha=0.1, label='Idle Mode'),
        plt.Line2D([0], [0], color='b', label='Power'),
        plt.Line2D([0], [0], color='r', linestyle='--', label='Price')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Head
    ax2 = axes[1]
    ax2.plot(day_data['Time'], day_data['Head'], 'g-')
    ax2.set_ylabel('Head (m)')
    ax2.set_title('Reservoir Head')
    
    # Plot 3: Flow
    ax3 = axes[2]
    ax3.plot(day_data['Time'], day_data['Flow'], 'm-')
    ax3.set_ylabel('Flow (m³/s)')
    ax3.set_title('Water Flow')
    ax3.set_xlabel('Time (hours)')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics for {date}:")
    print(f"Total Energy Generated: {day_data[day_data['Power'] > 0]['Power'].sum():.2f} MWh")
    print(f"Total Energy Consumed: {abs(day_data[day_data['Power'] < 0]['Power'].sum()):.2f} MWh")
    print(f"Net Energy: {day_data['Power'].sum():.2f} MWh")
    print(f"Total Revenue: ${day_data['Revenue'].sum():.2f}")
    
    # Mode statistics
    mode_counts = day_data['Mode'].value_counts()
    print("\nOperation Mode Distribution:")
    for mode, count in mode_counts.items():
        print(f"{mode}: {count} hours ({count/24*100:.1f}%)")

def analyze_database(df):
    """
    Perform comprehensive analysis on the entire database.
    
    Parameters:
        df (pd.DataFrame): The database dataframe
        
    Returns:
        dict: Analysis results
    """
    if df is None:
        return {}
    
    results = {}
    
    # Overall statistics
    results['total_days'] = df['Date'].nunique()
    results['total_revenue'] = df['Revenue'].sum()
    results['total_energy_generated'] = df[df['Power'] > 0]['Power'].sum()
    results['total_energy_consumed'] = abs(df[df['Power'] < 0]['Power'].sum())
    results['net_energy'] = df['Power'].sum()
    
    # Mode distribution
    mode_dist = df['Mode'].value_counts(normalize=True) * 100
    results['mode_distribution'] = mode_dist.to_dict()
    
    # Average statistics by day
    daily_stats = df.groupby('Date').agg({
        'Power': ['mean', 'min', 'max', 'sum'],
        'Revenue': 'sum',
        'Head': ['mean', 'min', 'max']
    })
    
    results['avg_daily_revenue'] = daily_stats['Revenue']['sum'].mean()
    results['avg_daily_net_energy'] = daily_stats['Power']['sum'].mean()
    
    # Print summary
    print("\n=== Database Analysis Summary ===")
    print(f"Total Days: {results['total_days']}")
    print(f"Total Revenue: ${results['total_revenue']:.2f}")
    print(f"Total Energy Generated: {results['total_energy_generated']:.2f} MWh")
    print(f"Total Energy Consumed: {results['total_energy_consumed']:.2f} MWh")
    print(f"Net Energy: {results['net_energy']:.2f} MWh")
    print(f"Average Daily Revenue: ${results['avg_daily_revenue']:.2f}")
    
    print("\nOperation Mode Distribution:")
    for mode, percentage in results['mode_distribution'].items():
        print(f"{mode}: {percentage:.1f}%")
    
    return results

if __name__ == "__main__":
    # Read the database
    df = read_database()
    
    if df is not None:
        # Display the first few rows
        print("\nDatabase Preview:")
        print(df.head())
        
        # Get the first date in the dataset
        first_date = df['Date'].min().strftime('%Y-%m-%d')
        
        # Visualize the first date
        print(f"\nVisualizing results for {first_date}:")
        visualize_date(df, first_date)
        
        # Analyze entire database
        analysis_results = analyze_database(df)

# %% Linearization Coefficient Extraction
def extract_linearization_coefficients(df, regression_method='sampling', save_path="./Data/database_with_coefficients.csv"):
    """
    Extract linearization coefficients for all records in the database.
    
    Parameters:
        df (pd.DataFrame): The database dataframe
        regression_method (str): 'sampling' or 'taylor' for different regression approaches
        save_path (str): Path to save the enhanced CSV file
        
    Returns:
        pd.DataFrame: Database with added linearization coefficients
    """
    print(f"Extracting linearization coefficients using {regression_method} method...")
    
    # Create hydroparameters 
    params = HydroParameters()
    
    # Initialize regression layer based on method
    if regression_method.lower() == 'sampling':
        reg_layer = RegressionLayer(params)
    elif regression_method.lower() == 'taylor':
        reg_layer = TaylorRegressionLayer(params)
    else:
        raise ValueError(f"Unknown regression method: {regression_method}. Use 'sampling' or 'taylor'.")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_with_coeff = df.copy()
    
    # Initialize coefficient columns with NaN
    coeff_columns = ['c', 'd', 'e', 'a', 'b']
    for col in coeff_columns:
        df_with_coeff[col] = np.nan
    
    # Process each date separately
    unique_dates = df_with_coeff['Date'].unique()
    
    # Track success and failure statistics
    success_count = 0
    failure_count = 0
    
    # Process date by date
    for date_idx, date in enumerate(tqdm(unique_dates, desc="Processing dates")):
        try:
            date_df = df_with_coeff[df_with_coeff['Date'] == date].copy()
            
            # Skip if not exactly 24 hours of data
            if len(date_df) != 24:
                print(f"Warning: Date {date} has {len(date_df)} records instead of 24, skipping.")
                continue
            
            # Get indices for this date (to update the original dataframe later)
            date_indices = date_df.index.tolist()
            
            # Get power and head values for this date
            power_values = torch.tensor(date_df['Power'].values, dtype=torch.float32)
            head_values = torch.tensor(date_df['Head'].values, dtype=torch.float32)
            
            # Process all 24 hours at once with error handling
            try:
                # Run regression for the entire day
                c, d, e, a, b = reg_layer.run_regression(power_values, head_values)
                
                # Verify dimensions
                if (c.shape[0] != 24 or d.shape[0] != 24 or e.shape[0] != 24 or 
                    a.shape[0] != 24 or b.shape[0] != 24):
                    raise ValueError(f"Unexpected output dimensions for date {date}")
                
                # Update dataframe with coefficients
                for t in range(24):
                    idx = date_indices[t]
                    df_with_coeff.loc[idx, 'c'] = c[t].item()
                    df_with_coeff.loc[idx, 'd'] = d[t].item()
                    df_with_coeff.loc[idx, 'e'] = e[t].item()
                    df_with_coeff.loc[idx, 'a'] = a[t].item()
                    df_with_coeff.loc[idx, 'b'] = b[t].item()
                
                success_count += 24  # 24 hours successfully processed
                
                # Save progress every 10 dates
                if (date_idx + 1) % 10 == 0:
                    temp_save_path = save_path.replace('.csv', f'_temp_{date_idx+1}.csv')
                    df_with_coeff.to_csv(temp_save_path, index=False)
                    print(f"Progress saved to {temp_save_path}")
                
            except Exception as e:
                print(f"Error processing entire day for date {date}: {str(e)}")
                print("Trying hour-by-hour approach...")
                
                # If batch processing fails, try hour by hour
                for t in range(24):
                    try:
                        # Extract single timestep data
                        p_t = power_values[t:t+1]  # Shape: [1]
                        h_t = head_values[t:t+1]   # Shape: [1]
                        
                        # Skip processing for idle mode
                        if abs(p_t.item()) < 1e-6:
                            # For idle mode, set q-related coefficients to 0
                            df_with_coeff.loc[date_indices[t], 'c'] = 0.0
                            df_with_coeff.loc[date_indices[t], 'd'] = 0.0
                            df_with_coeff.loc[date_indices[t], 'e'] = 0.0
                            
                            # Still need to compute volume-head relationship
                            temp_params = HydroParameters()
                            temp_params.time_horizon = 1
                            temp_reg = RegressionLayer(temp_params)
                            
                            # Perform only the volume-head regression
                            h_lo = max(float(temp_params.head_min), float(h_t.item() - temp_params.δ_h))
                            h_hi = min(float(temp_params.head_max), float(h_t.item() + temp_params.δ_h))
                            h_samples = torch.linspace(h_lo, h_hi, temp_params.sampling_rate, requires_grad=True)
                            
                            v_low_values = torch.tensor([
                                temp_params.h_to_v_low_fitted(h.item()) 
                                for h in h_samples
                            ], dtype=torch.float32, requires_grad=True)
                            
                            beta_v = temp_reg.least_squares_v_low_torch(h_samples, v_low_values)
                            
                            df_with_coeff.loc[date_indices[t], 'a'] = beta_v[0].item()
                            df_with_coeff.loc[date_indices[t], 'b'] = beta_v[1].item()
                            
                            success_count += 1
                            continue
                        
                        # For non-idle mode, use the regression layer
                        temp_params = HydroParameters()
                        temp_params.time_horizon = 1
                        
                        if regression_method.lower() == 'sampling':
                            temp_reg = RegressionLayer(temp_params)
                        else:
                            temp_reg = TaylorRegressionLayer(temp_params)
                        
                        # Perform regression for this hour
                        c_t, d_t, e_t, a_t, b_t = temp_reg.run_regression(p_t, h_t)
                        
                        # Update coefficients
                        df_with_coeff.loc[date_indices[t], 'c'] = c_t[0].item()
                        df_with_coeff.loc[date_indices[t], 'd'] = d_t[0].item()
                        df_with_coeff.loc[date_indices[t], 'e'] = e_t[0].item()
                        df_with_coeff.loc[date_indices[t], 'a'] = a_t[0].item()
                        df_with_coeff.loc[date_indices[t], 'b'] = b_t[0].item()
                        
                        success_count += 1
                        
                    except Exception as hour_e:
                        print(f"  Error at hour {t}: {str(hour_e)}")
                        failure_count += 1
        
        except Exception as date_e:
            print(f"Error with date {date}: {str(date_e)}")
            failure_count += 24  # Count all hours as failures
    
    # Print statistics
    print(f"\nCoefficient extraction complete:")
    print(f"Successful extractions: {success_count}")
    print(f"Failed extractions: {failure_count}")
    print(f"Success rate: {success_count / (success_count + failure_count) * 100:.2f}%")
    
    # Check for NaN values
    nan_counts = df_with_coeff[coeff_columns].isna().sum()
    print(f"\nMissing coefficient counts:")
    for col in coeff_columns:
        print(f"  {col}: {nan_counts[col]} missing values")
    
    # Save the final dataframe
    df_with_coeff.to_csv(save_path, index=False)
    print(f"Enhanced database saved to {save_path}")
    
    return df_with_coeff

def visualize_coefficients(df, date):
    """
    Visualize linearization coefficients for a specific date.
    
    Parameters:
        df (pd.DataFrame): The database dataframe with coefficients
        date (str): Date to visualize in 'YYYY-MM-DD' format
    """
    if 'c' not in df.columns:
        print("No linearization coefficients found in the dataframe.")
        return
    
    # Convert string date to datetime if needed
    target_date = pd.to_datetime(date)
    day_data = df[df['Date'] == target_date]
    
    if len(day_data) == 0:
        print(f"No data found for date: {date}")
        return
    
    # Create figure for UPC coefficients (c, d, e)
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig1.suptitle(f"UPC Linearization Coefficients for {date}", fontsize=16)
    
    # Plot c coefficient
    axes1[0].plot(day_data['Time'], day_data['c'], 'b-o')
    axes1[0].set_ylabel('c coefficient')
    axes1[0].set_title('Flow-Power Coefficient')
    axes1[0].grid(True)
    
    # Plot d coefficient
    axes1[1].plot(day_data['Time'], day_data['d'], 'g-o')
    axes1[1].set_ylabel('d coefficient')
    axes1[1].set_title('Flow-Head Coefficient')
    axes1[1].grid(True)
    
    # Plot e coefficient
    axes1[2].plot(day_data['Time'], day_data['e'], 'r-o')
    axes1[2].set_ylabel('e coefficient')
    axes1[2].set_title('Flow Intercept')
    axes1[2].set_xlabel('Time (hours)')
    axes1[2].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Create figure for volume-head coefficients (a, b)
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle(f"Volume-Head Linearization Coefficients for {date}", fontsize=16)
    
    # Plot a coefficient
    axes2[0].plot(day_data['Time'], day_data['a'], 'm-o')
    axes2[0].set_ylabel('a coefficient')
    axes2[0].set_title('Volume-Head Slope')
    axes2[0].grid(True)
    
    # Plot b coefficient
    axes2[1].plot(day_data['Time'], day_data['b'], 'c-o')
    axes2[1].set_ylabel('b coefficient')
    axes2[1].set_title('Volume Intercept')
    axes2[1].set_xlabel('Time (hours)')
    axes2[1].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# %% Main script
if __name__ == "__main__":
    # Read the database
    print("Reading database...")
    df = read_database()
      
    if df is not None:
        # Extract linearization coefficients
        # Choose between 'sampling' and 'taylor' methods
        regression_method = 'sampling'  # or 'taylor'
        df_with_coeff = extract_linearization_coefficients(
            df,
            regression_method=regression_method,
            save_path="./Data/database_no_piecewise_with_coeff.csv"
        )
        
        # Display the first few rows with coefficients
        print("\nDatabase with coefficients preview:")
        print(df_with_coeff.head())
        
        # Get the first date in the dataset for visualization
        first_date = df_with_coeff['Date'].min().strftime('%Y-%m-%d')
        
        # Visualize coefficients for the first date
        print(f"\nVisualizing coefficients for {first_date}:")
        visualize_coefficients(df_with_coeff, first_date)
        
        print("\nProcess completed successfully!")


# %%
import torch
import matplotlib.pyplot as plt

def test_regression_layer():
    # Initialize test data with power, head, and flow values
    power = torch.tensor([0.0, 0.0, -7.32, -7.63, -7.95, -8.26, -8.19, 4.27, 4.11, 4.43, 4.23, 4.01, 3.78, 3.55, 
                        3.37, 3.3, 3.23, 4.17, 4.8, 4.55, 3.91, 3.66, 2.64, 2.57], dtype=torch.float32, requires_grad=True)
    head = torch.tensor([76.96, 79.39, 81.82, 84.25, 86.67, 89.12, 91.47, 90.13, 88.82, 87.34, 85.89, 84.48, 83.13, 81.85, 
                        80.6, 79.35, 78.09, 76.37, 74.23, 72.18, 70.49, 68.9, 67.77, 66.67], dtype=torch.float32, requires_grad=True)
    flow = torch.tensor([0.0, 0.0, -10.12, -10.11, -10.11, -10.18, -9.79, 5.55, 5.48, 6.16, 6.06, 5.87, 5.61, 5.36, 
                        5.19, 5.21, 5.24, 7.15, 8.95, 8.53, 7.04, 6.64, 4.68, 4.61], dtype=torch.float32, requires_grad=True)
    
    idle_indices = [0, 1]
    
    # Initialize parameters and the regression layer
    params = HydroParameters()
    regression = RegressionLayer(params)
    
    # Run regression
    print("Running regression...")
    c, d, e, a, b = regression.run_regression(power, head)
    
    # Print shapes of output tensors
    print(f"Output tensor shapes: c: {c.shape}, d: {d.shape}, e: {e.shape}, a: {a.shape}, b: {b.shape}")
    
    # Check for NaN or Inf values
    for name, tensor in zip(['c', 'd', 'e', 'a', 'b'], [c, d, e, a, b]):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: {name} contains NaN or Inf values!")
    
    # Display coefficients for idle indices
    print("\nCoefficients at idle indices:")
    for idx in idle_indices:
        print(f"At index {idx} (idle mode):")
        print(f"  c = {c[idx].item():.6f}, d = {d[idx].item():.6f}, e = {e[idx].item():.6f}")
        print(f"  a = {a[idx].item():.6f}, b = {b[idx].item():.6f}")
    
    # Sample non-idle indices for comparison
    non_idle_indices = [0, 7]  # One from pump mode, one from turbine mode
    print("\nCoefficients at non-idle indices for comparison:")
    for idx in non_idle_indices:
        mode = "Pump" if power[idx] < 0 else "Turbine"
        print(f"At index {idx} ({mode} mode):")
        print(f"  c = {c[idx].item():.6f}, d = {d[idx].item():.6f}, e = {e[idx].item():.6f}")
        print(f"  a = {a[idx].item():.6f}, b = {b[idx].item():.6f}")
    
    # Plot the coefficients
    plt.figure(figsize=(15, 10))
    
    # Mark idle indices with vertical lines
    for idx in idle_indices:
        plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
    
    plt.subplot(3, 1, 1)
    plt.title('Flow-Power Coefficient (c)')
    plt.plot(c.detach().numpy(), 'b-o')
    for idx in idle_indices:
        plt.scatter(idx, c[idx].item(), color='r', s=100, zorder=5)
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.title('Flow-Head Coefficient (d)')
    plt.plot(d.detach().numpy(), 'g-o')
    for idx in idle_indices:
        plt.scatter(idx, d[idx].item(), color='r', s=100, zorder=5)
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.title('Flow Intercept (e)')
    plt.plot(e.detach().numpy(), 'r-o')
    for idx in idle_indices:
        plt.scatter(idx, e[idx].item(), color='r', s=100, zorder=5)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot volume-head coefficients
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.title('Volume-Head Slope (a)')
    plt.plot(a.detach().numpy(), 'm-o')
    for idx in idle_indices:
        plt.scatter(idx, a[idx].item(), color='r', s=100, zorder=5)
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title('Volume Intercept (b)')
    plt.plot(b.detach().numpy(), 'c-o')
    for idx in idle_indices:
        plt.scatter(idx, b[idx].item(), color='r', s=100, zorder=5)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return c, d, e, a, b

# Run the test
c, d, e, a, b = test_regression_layer()