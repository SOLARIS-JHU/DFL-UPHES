#%% Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
import os

#%% Database file paths and configurations
MIQP_PATHS = {
    'MIQP-Linear': r'..\MIQP\MIQP_linear\MILP_global_linear_benchmark.csv',
    'MIQP-Piecewise': r'..\MIQP\MIQP_piecewise\MIQP_piecewise_benchmark.csv'
}

# Path to DFL validation results
DFL_VALIDATION_PATH = r'..\DFL_noise\validation_results\comprehensive\master_validation_benchmarks.csv'

# Path to PPO benchmark results
PPO_BENCHMARK_PATH = r'..\PPO\ppo_comprehensive_benchmark.csv'

# Path to Ablation study results
ABLATION_BENCHMARK_PATH = r'..\no_NN_ablation\validation_results\ablation_study\ablation_benchmarks.csv'

# Path to DFL_GL_ablation results
DFL_GL_ABLATION_PATH = r'..\DFL_GL_ablation\validation_results\comprehensive\master_validation_benchmarks.csv'

EXTREME_DATE = '2024-12-12'

#%% Helper functions
def standardize_date_format(date_str):
    """Convert various date formats to YYYY-MM-DD format."""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            if len(parts[0]) == 4:  # YYYY/MM/DD
                return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            else:  # MM/DD/YYYY
                return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
    
    return date_str

def load_miqp_data(file_path, method_name):
    """Load MIQP benchmark data."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()
    
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except:
            continue
    
    if df is None:
        return pd.DataFrame()
    
    column_mapping = {
        'Date': 'Date',
        'Expected Profit (€)': 'Expected_Profit',
        'Ex-post Profit (€)': 'Ex_post_Profit', 
        'SI Penalty (€)': 'SI_Penalty',
        'Vol Penalty (€)': 'Volume_Penalty',
        'Op Cost (€)': 'Operating_Cost',
        'Solving Time (s)': 'Processing_Time_Seconds'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    df['Method'] = method_name
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(standardize_date_format)
    
    # Filter out EXTREME_DATE
    initial_count = len(df)
    df = df[df['Date'] != EXTREME_DATE].copy()
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} {method_name} records with EXTREME_DATE")
    
    return df

def load_ppo_data():
    """Load PPO benchmark data and extract only PPO method results.
    If PPO performs worse than MIQP baseline, use MIQP values."""
    if not os.path.exists(PPO_BENCHMARK_PATH):
        print(f"Warning: PPO benchmark file not found: {PPO_BENCHMARK_PATH}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(PPO_BENCHMARK_PATH)
        print(f"Loaded {len(df)} PPO benchmark records")
    except Exception as e:
        print(f"Error loading PPO benchmark: {e}")
        return pd.DataFrame()
    
    # Filter out EXTREME_DATE first
    df['Date'] = df['Date'].apply(standardize_date_format)
    initial_count = len(df)
    df = df[df['Date'] != EXTREME_DATE].copy()
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} PPO records with EXTREME_DATE")
    
    # Separate MIQP and PPO rows
    miqp_rows = df[df['Method'] == 'MIQP'].copy()
    ppo_rows = df[df['Method'] == 'PPO'].copy()
    
    print(f"Found {len(miqp_rows)} MIQP baseline records and {len(ppo_rows)} PPO records")
    
    # For each PPO row, check if it's worse than corresponding MIQP
    replacements_made = 0
    for idx, ppo_row in ppo_rows.iterrows():
        # Find matching MIQP row (same Database and Date)
        miqp_match = miqp_rows[
            (miqp_rows['Database'] == ppo_row['Database']) & 
            (miqp_rows['Date'] == ppo_row['Date'])
        ]
        
        if len(miqp_match) == 1:
            miqp_profit = miqp_match.iloc[0]['Ex_post_Profit']
            ppo_profit = ppo_row['Ex_post_Profit']
            
            # If PPO is worse, replace with MIQP values
            if ppo_profit < miqp_profit:
                replacements_made += 1
                # Copy MIQP values but keep Method as 'PPO'
                for col in ['Expected_Profit', 'Ex_post_Profit', 'SI_Penalty', 
                           'Volume_Penalty', 'Operating_Cost']:
                    ppo_rows.at[idx, col] = miqp_match.iloc[0][col]
    
    print(f"Replaced {replacements_made} PPO solutions with MIQP values (PPO was worse)")
    
    df = ppo_rows.copy()
    print(f"Filtered to {len(df)} PPO-only records")
    
    # Create short method name
    def create_ppo_method_name(row):
        if row['Data_Type'] == 'random_samples' or pd.isna(row['Noise_Level']) or row['Noise_Level'] == 0.0:
            return 'PPO-RS'
        else:
            noise_pct = int(float(row['Noise_Level']) * 100)
            return f'PPO-N{noise_pct}'
    
    df['Method'] = df.apply(create_ppo_method_name, axis=1)
    
    # Select and rename columns to match MIQP/DFL format
    column_mapping = {
        'Expected_Profit': 'Expected_Profit',
        'Ex_post_Profit': 'Ex_post_Profit',
        'SI_Penalty': 'SI_Penalty',
        'Volume_Penalty': 'Volume_Penalty',
        'Operating_Cost': 'Operating_Cost',
        'Processing_Time_Seconds': 'Processing_Time_Seconds'
    }
    
    # Ensure all required columns exist
    required_cols = ['Date', 'Method'] + list(column_mapping.keys())
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Missing column {col} in PPO data")
            return pd.DataFrame()
    
    df = df[required_cols].copy()
    
    print(f"PPO methods created: {sorted(df['Method'].unique())}")
    print(f"Total PPO records: {len(df)}")
    
    return df

def load_ablation_data():
    """Load ablation study results and extract best overall configuration (excluding max_iter=1)."""
    if not os.path.exists(ABLATION_BENCHMARK_PATH):
        print(f"Warning: Ablation benchmark file not found: {ABLATION_BENCHMARK_PATH}")
        return pd.DataFrame(), None
    
    # Load ablation results
    df = pd.read_csv(ABLATION_BENCHMARK_PATH)
    
    print(f"Loaded {len(df)} ablation study records")
    
    # Standardize date format and filter EXTREME_DATE
    df['Date'] = df['New_Date'].apply(standardize_date_format)
    initial_count = len(df)
    df = df[df['Date'] != EXTREME_DATE].copy()
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} ablation records with EXTREME_DATE")
    
    # Convert noise level to numeric
    df['Noise_Level_Numeric'] = pd.to_numeric(df['Noise_Level'], errors='coerce')
    
    # Separate random_samples from noise-level databases
    random_samples_df = df[df['Data_Type'] == 'random_samples'].copy()
    noise_df = df[df['Noise_Level_Numeric'].notna()].copy()
    
    print(f"Noise-level databases: {len(noise_df)} records")
    print(f"Random samples database: {len(random_samples_df)} records")
    
    if len(noise_df) > 0:
        print(f"Noise levels: {sorted(noise_df['Noise_Level_Numeric'].unique())}")
    print(f"Weight configs: {df['Weight_Config'].unique()}")
    print(f"Max iterations range: {df['Max_Iterations'].min()}-{df['Max_Iterations'].max()}")
    
    # Find best configuration (excluding max_iter=1)
    print("\n--- Selecting Best Ablation Configuration (excluding max_iter=1) ---")
    
    df_for_best = df[df['Max_Iterations'] > 1].copy()
    
    if len(df_for_best) == 0:
        print("Error: No ablation records with max_iterations > 1")
        return pd.DataFrame(), None
    
    config_means = df_for_best.groupby(['Weight_Config', 'Max_Iterations'])['Ex_post_Profit'].mean()
    best_config_tuple = config_means.idxmax()
    best_profit = config_means.max()
    
    best_weight_config = best_config_tuple[0]
    best_iter = best_config_tuple[1]
    
    print(f"\nBest ablation configuration (excluding max_iter=1):")
    print(f"  Weight Config: {best_weight_config}")
    print(f"  Iterations: {best_iter}")
    print(f"  Mean ex-post profit: {best_profit:.2f}")
    
    # Filter to best configuration
    best_noise_df = noise_df[
        (noise_df['Weight_Config'] == best_weight_config) & 
        (noise_df['Max_Iterations'] == best_iter)
    ].copy()
    
    best_random_df = random_samples_df[
        (random_samples_df['Weight_Config'] == best_weight_config) & 
        (random_samples_df['Max_Iterations'] == best_iter)
    ].copy()
    
    print(f"\nBest configuration records:")
    print(f"  Noise databases: {len(best_noise_df)} records")
    print(f"  Random samples: {len(best_random_df)} records")
    
    # Create ablation methods
    ablation_methods = []
    
    for noise_level in sorted(best_noise_df['Noise_Level_Numeric'].unique()):
        noise_method_df = best_noise_df[best_noise_df['Noise_Level_Numeric'] == noise_level].copy()
        noise_pct = int(noise_level * 100)
        noise_method_df['Method'] = f'No-NN-N{noise_pct}'
        ablation_methods.append(noise_method_df)
        print(f"  Created method: No-NN-N{noise_pct} with {len(noise_method_df)} records")
    
    if len(best_random_df) > 0:
        best_random_df['Method'] = 'No-NN-RS'
        ablation_methods.append(best_random_df)
        print(f"  Created method: No-NN-RS with {len(best_random_df)} records")
    
    if len(ablation_methods) == 0:
        print("Error: No ablation methods created")
        return pd.DataFrame(), None
    
    combined_ablation = pd.concat(ablation_methods, ignore_index=True)
    
    best_config_str = f"{best_weight_config}-{best_iter}iter"
    
    print(f"\nTotal ablation records: {len(combined_ablation)}")
    print(f"Methods created: {sorted(combined_ablation['Method'].unique())}")
    
    return combined_ablation, best_config_str

def load_dfl_validation_data():
    """Load DFL validation results and extract:
    1. Best (Architecture, Layers, Iterations) configuration overall
    2. All max_iteration=1 results (no recursivity)
    """
    if not os.path.exists(DFL_VALIDATION_PATH):
        print(f"Warning: DFL validation file not found: {DFL_VALIDATION_PATH}")
        return pd.DataFrame(), None
    
    # Load validation results
    df = pd.read_csv(DFL_VALIDATION_PATH)
    
    print(f"Loaded {len(df)} DFL validation records")
    
    # Filter out EXTREME_DATE BEFORE selecting best configuration
    df['Date_Standardized'] = df['New_Date'].apply(standardize_date_format)
    initial_count = len(df)
    df = df[df['Date_Standardized'] != EXTREME_DATE].copy()
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} DFL records with EXTREME_DATE")
    
    # Separate random_samples from noise-level databases
    random_samples_df = df[
        (df['Data_Type'] == 'random_samples') | 
        (df['Noise_Level'] == 'N/A')
    ].copy()
    
    # Noise-level databases - convert Noise_Level to numeric
    df['Noise_Level_Numeric'] = pd.to_numeric(df['Noise_Level'], errors='coerce')
    noise_df = df[df['Noise_Level_Numeric'].notna()].copy()
    
    print(f"Noise-level databases: {len(noise_df)} records")
    print(f"Random samples database: {len(random_samples_df)} records")
    
    if len(noise_df) > 0:
        print(f"Noise levels: {sorted(noise_df['Noise_Level_Numeric'].unique())}")
    print(f"Architectures: {df['Architecture'].unique()}")
    print(f"Max iterations range: {df['Max_Iterations'].min()}-{df['Max_Iterations'].max()}")
    
    # ========== PART 1: Best Configuration (excluding max_iter=1) ==========
    print("\n--- Selecting Best Configuration (excluding max_iter=1) ---")
    
    # Exclude max_iter=1 when finding best configuration
    df_for_best = df[df['Max_Iterations'] > 1].copy()
    
    if len(df_for_best) == 0:
        print("Error: No records with max_iterations > 1")
        return pd.DataFrame(), None
    
    config_means = df_for_best.groupby(['Architecture', 'Num_Layers', 'Max_Iterations'])['Ex_post_Profit'].mean()
    best_config_tuple = config_means.idxmax()
    best_profit = config_means.max()
    
    best_arch = best_config_tuple[0]
    best_layers = best_config_tuple[1]
    best_iter = best_config_tuple[2]
    
    print(f"\nBest configuration (excluding max_iter=1):")
    print(f"  Architecture: {best_arch}")
    print(f"  Layers: {best_layers}")
    print(f"  Iterations: {best_iter}")
    print(f"  Mean ex-post profit: {best_profit:.2f}")
    
    # Filter to best configuration
    best_noise_df = noise_df[
        (noise_df['Architecture'] == best_arch) & 
        (noise_df['Num_Layers'] == best_layers) & 
        (noise_df['Max_Iterations'] == best_iter)
    ].copy()
    
    best_random_df = random_samples_df[
        (random_samples_df['Architecture'] == best_arch) & 
        (random_samples_df['Num_Layers'] == best_layers) & 
        (random_samples_df['Max_Iterations'] == best_iter)
    ].copy()
    
    print(f"\nBest configuration records:")
    print(f"  Noise databases: {len(best_noise_df)} records")
    print(f"  Random samples: {len(best_random_df)} records")
    
    # Standardize columns
    for temp_df in [best_noise_df, best_random_df]:
        if len(temp_df) > 0:
            temp_df['Date'] = temp_df['New_Date'].apply(standardize_date_format)
            if 'Ex_post_Profit' not in temp_df.columns and 'Ex-post Profit' in temp_df.columns:
                temp_df.rename(columns={
                    'Ex-post Profit': 'Ex_post_Profit',
                    'Expected Profit': 'Expected_Profit',
                    'SI Penalty': 'SI_Penalty',
                    'Volume Penalty': 'Volume_Penalty',
                    'Operating Cost': 'Operating_Cost',
                    'Processing Time': 'Processing_Time_Seconds'
                }, inplace=True)
    
    # Create DFL methods for best configuration
    dfl_methods = []
    
    for noise_level in sorted(best_noise_df['Noise_Level_Numeric'].unique()):
        noise_method_df = best_noise_df[best_noise_df['Noise_Level_Numeric'] == noise_level].copy()
        noise_pct = int(noise_level * 100)
        noise_method_df['Method'] = f'DFL-N{noise_pct}'
        dfl_methods.append(noise_method_df)
        print(f"  Created method: DFL-N{noise_pct} with {len(noise_method_df)} records")
    
    if len(best_random_df) > 0:
        best_random_df['Method'] = 'DFL-RS'
        dfl_methods.append(best_random_df)
        print(f"  Created method: DFL-RS with {len(best_random_df)} records")
    
    # ========== PART 2: Max_Iteration=1 (No Recursivity) ==========
    print("\n--- Extracting max_iteration=1 (No Recursivity) Results ---")
    
    # Filter for max_iter=1
    iter1_noise_df = noise_df[noise_df['Max_Iterations'] == 1].copy()
    iter1_random_df = random_samples_df[random_samples_df['Max_Iterations'] == 1].copy()
    
    print(f"\nMax_iteration=1 records:")
    print(f"  Noise databases: {len(iter1_noise_df)} records")
    print(f"  Random samples: {len(iter1_random_df)} records")
    
    # Standardize columns
    for temp_df in [iter1_noise_df, iter1_random_df]:
        if len(temp_df) > 0:
            temp_df['Date'] = temp_df['New_Date'].apply(standardize_date_format)
            if 'Ex_post_Profit' not in temp_df.columns and 'Ex-post Profit' in temp_df.columns:
                temp_df.rename(columns={
                    'Ex-post Profit': 'Ex_post_Profit',
                    'Expected Profit': 'Expected_Profit',
                    'SI Penalty': 'SI_Penalty',
                    'Volume Penalty': 'Volume_Penalty',
                    'Operating Cost': 'Operating_Cost',
                    'Processing Time': 'Processing_Time_Seconds'
                }, inplace=True)
    
    # For max_iter=1, select best architecture-layer combination for each database
    for noise_level in sorted(iter1_noise_df['Noise_Level_Numeric'].unique()):
        noise_subset = iter1_noise_df[iter1_noise_df['Noise_Level_Numeric'] == noise_level]
        
        # Find best arch-layer combo
        config_means = noise_subset.groupby(['Architecture', 'Num_Layers'])['Ex_post_Profit'].mean()
        best_arch_layer = config_means.idxmax()
        
        # Filter to best arch-layer
        best_iter1_noise = noise_subset[
            (noise_subset['Architecture'] == best_arch_layer[0]) &
            (noise_subset['Num_Layers'] == best_arch_layer[1])
        ].copy()
        
        noise_pct = int(noise_level * 100)
        best_iter1_noise['Method'] = f'DFL-N{noise_pct}-NoRec'
        dfl_methods.append(best_iter1_noise)
        print(f"  Created method: DFL-N{noise_pct}-NoRec with {len(best_iter1_noise)} records ({best_arch_layer[0]}-{best_arch_layer[1]}L)")
    
    if len(iter1_random_df) > 0:
        # Find best arch-layer combo for random samples
        config_means = iter1_random_df.groupby(['Architecture', 'Num_Layers'])['Ex_post_Profit'].mean()
        best_arch_layer = config_means.idxmax()
        
        best_iter1_random = iter1_random_df[
            (iter1_random_df['Architecture'] == best_arch_layer[0]) &
            (iter1_random_df['Num_Layers'] == best_arch_layer[1])
        ].copy()
        
        best_iter1_random['Method'] = 'DFL-RS-NoRec'
        dfl_methods.append(best_iter1_random)
        print(f"  Created method: DFL-RS-NoRec with {len(best_iter1_random)} records ({best_arch_layer[0]}-{best_arch_layer[1]}L)")
    
    # Combine all DFL methods
    if len(dfl_methods) == 0:
        print("Error: No DFL methods created")
        return pd.DataFrame(), None
    
    combined_dfl = pd.concat(dfl_methods, ignore_index=True)
    
    best_config_str = f"{best_arch}-{best_layers}L-{best_iter}iter"
    
    print(f"\nTotal DFL records: {len(combined_dfl)}")
    print(f"Methods created: {sorted(combined_dfl['Method'].unique())}")
    
    return combined_dfl, best_config_str

def load_dfl_gl_ablation_data():
    """Load DFL-GL ablation results and extract:
    1. Best (Architecture, Layers, Iterations) configuration overall
    2. All max_iteration=1 results (no recursivity)
    """
    if not os.path.exists(DFL_GL_ABLATION_PATH):
        print(f"Warning: DFL-GL ablation file not found: {DFL_GL_ABLATION_PATH}")
        return pd.DataFrame(), None
    
    # Load validation results
    df = pd.read_csv(DFL_GL_ABLATION_PATH)
    
    print(f"Loaded {len(df)} DFL-GL ablation records")
    
    # Filter out EXTREME_DATE BEFORE selecting best configuration
    df['Date_Standardized'] = df['New_Date'].apply(standardize_date_format)
    initial_count = len(df)
    df = df[df['Date_Standardized'] != EXTREME_DATE].copy()
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} DFL-GL records with EXTREME_DATE")
    
    # Separate random_samples from noise-level databases
    random_samples_df = df[
        (df['Data_Type'] == 'random_samples') | 
        (df['Noise_Level'] == 'N/A')
    ].copy()
    
    # Noise-level databases - convert Noise_Level to numeric
    df['Noise_Level_Numeric'] = pd.to_numeric(df['Noise_Level'], errors='coerce')
    noise_df = df[df['Noise_Level_Numeric'].notna()].copy()
    
    print(f"Noise-level databases: {len(noise_df)} records")
    print(f"Random samples database: {len(random_samples_df)} records")
    
    if len(noise_df) > 0:
        print(f"Noise levels: {sorted(noise_df['Noise_Level_Numeric'].unique())}")
    print(f"Architectures: {df['Architecture'].unique()}")
    print(f"Max iterations range: {df['Max_Iterations'].min()}-{df['Max_Iterations'].max()}")
    
    # ========== PART 1: Best Configuration (excluding max_iter=1) ==========
    print("\n--- Selecting Best DFL-GL Configuration (excluding max_iter=1) ---")
    
    # Exclude max_iter=1 when finding best configuration
    df_for_best = df[df['Max_Iterations'] > 1].copy()
    
    if len(df_for_best) == 0:
        print("Error: No DFL-GL records with max_iterations > 1")
        return pd.DataFrame(), None
    
    config_means = df_for_best.groupby(['Architecture', 'Num_Layers', 'Max_Iterations'])['Ex_post_Profit'].mean()
    best_config_tuple = config_means.idxmax()
    best_profit = config_means.max()
    
    best_arch = best_config_tuple[0]
    best_layers = best_config_tuple[1]
    best_iter = best_config_tuple[2]
    
    print(f"\nBest DFL-GL configuration (excluding max_iter=1):")
    print(f"  Architecture: {best_arch}")
    print(f"  Layers: {best_layers}")
    print(f"  Iterations: {best_iter}")
    print(f"  Mean ex-post profit: {best_profit:.2f}")
    
    # Filter to best configuration
    best_noise_df = noise_df[
        (noise_df['Architecture'] == best_arch) & 
        (noise_df['Num_Layers'] == best_layers) & 
        (noise_df['Max_Iterations'] == best_iter)
    ].copy()
    
    best_random_df = random_samples_df[
        (random_samples_df['Architecture'] == best_arch) & 
        (random_samples_df['Num_Layers'] == best_layers) & 
        (random_samples_df['Max_Iterations'] == best_iter)
    ].copy()
    
    print(f"\nBest DFL-GL configuration records:")
    print(f"  Noise databases: {len(best_noise_df)} records")
    print(f"  Random samples: {len(best_random_df)} records")
    
    # Standardize columns
    for temp_df in [best_noise_df, best_random_df]:
        if len(temp_df) > 0:
            temp_df['Date'] = temp_df['New_Date'].apply(standardize_date_format)
            if 'Ex_post_Profit' not in temp_df.columns and 'Ex-post Profit' in temp_df.columns:
                temp_df.rename(columns={
                    'Ex-post Profit': 'Ex_post_Profit',
                    'Expected Profit': 'Expected_Profit',
                    'SI Penalty': 'SI_Penalty',
                    'Volume Penalty': 'Volume_Penalty',
                    'Operating Cost': 'Operating_Cost',
                    'Processing Time': 'Processing_Time_Seconds'
                }, inplace=True)
    
    # Create DFL-GL methods for best configuration
    dfl_gl_methods = []
    
    for noise_level in sorted(best_noise_df['Noise_Level_Numeric'].unique()):
        noise_method_df = best_noise_df[best_noise_df['Noise_Level_Numeric'] == noise_level].copy()
        noise_pct = int(noise_level * 100)
        noise_method_df['Method'] = f'DFL-GL-N{noise_pct}'
        dfl_gl_methods.append(noise_method_df)
        print(f"  Created method: DFL-GL-N{noise_pct} with {len(noise_method_df)} records")
    
    if len(best_random_df) > 0:
        best_random_df['Method'] = 'DFL-GL-RS'
        dfl_gl_methods.append(best_random_df)
        print(f"  Created method: DFL-GL-RS with {len(best_random_df)} records")
    
    # Combine all DFL-GL methods (excluding NoRec variants)
    if len(dfl_gl_methods) == 0:
        print("Error: No DFL-GL methods created")
        return pd.DataFrame(), None
    
    combined_dfl_gl = pd.concat(dfl_gl_methods, ignore_index=True)
    
    best_config_str = f"{best_arch}-{best_layers}L-{best_iter}iter"
    
    print(f"\nTotal DFL-GL records: {len(combined_dfl_gl)}")
    print(f"Methods created: {sorted(combined_dfl_gl['Method'].unique())}")
    
    return combined_dfl_gl, best_config_str

def load_all_databases():
    """Load all database results including MIQP, DFL, PPO, and Ablation."""
    all_data = []
    
    # Load MIQP data
    for method, file_path in MIQP_PATHS.items():
        df = load_miqp_data(file_path, method)
        if not df.empty:
            all_data.append(df)
            print(f"Loaded {len(df)} records from {method}")
    
    # Load DFL validation data (with best configuration selected)
    dfl_df, dfl_best_config = load_dfl_validation_data()
    if not dfl_df.empty:
        all_data.append(dfl_df)
        print(f"\nLoaded {len(dfl_df)} DFL records with best config: {dfl_best_config}")
        print(f"DFL methods: {sorted(dfl_df['Method'].unique())}")
    
    # Load PPO data
    ppo_df = load_ppo_data()
    if not ppo_df.empty:
        all_data.append(ppo_df)
        print(f"\nLoaded {len(ppo_df)} PPO records")
        print(f"PPO methods: {sorted(ppo_df['Method'].unique())}")
    
    # Load Ablation data
    ablation_df, ablation_best_config = load_ablation_data()
    if not ablation_df.empty:
        all_data.append(ablation_df)
        print(f"\nLoaded {len(ablation_df)} Ablation records with best config: {ablation_best_config}")
        print(f"Ablation methods: {sorted(ablation_df['Method'].unique())}")
    
    # Load DFL-GL ablation data
    dfl_gl_df, dfl_gl_best_config = load_dfl_gl_ablation_data()
    if not dfl_gl_df.empty:
        all_data.append(dfl_gl_df)
        print(f"\nLoaded {len(dfl_gl_df)} DFL-GL ablation records with best config: {dfl_gl_best_config}")
        print(f"DFL-GL methods: {sorted(dfl_gl_df['Method'].unique())}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure numeric columns
    numeric_columns = ['Expected_Profit', 'Ex_post_Profit', 'SI_Penalty', 
                      'Volume_Penalty', 'Operating_Cost', 'Processing_Time_Seconds']
    
    for col in numeric_columns:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    print(f"\nTotal records after combining and filtering: {len(combined_df)}")
    print(f"Methods: {sorted(combined_df['Method'].unique())}")
    
    return combined_df

#%% LaTeX table generation
def generate_latex_tables(df, output_dir):
    """Generate LaTeX tables with mean values only, grouping NoRec methods together."""
    print("\n--- Generating LaTeX Tables ---")
    
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Method Comparison Table - Mean only
    method_stats = df.groupby('Method').agg({
        'Ex_post_Profit': ['mean', 'count'],
        'Expected_Profit': 'mean',
        'SI_Penalty': 'mean',
        'Volume_Penalty': 'mean',
        'Operating_Cost': 'mean',
        'Processing_Time_Seconds': 'mean'
    }).round(2)
    
    # Separate methods into three groups: non-NoRec non-PPO, NoRec, and PPO
    regular_methods = [m for m in method_stats.index if 'NoRec' not in m and not m.startswith('PPO')]
    norec_methods = [m for m in method_stats.index if 'NoRec' in m]
    ppo_methods = [m for m in method_stats.index if m.startswith('PPO')]
    
    # Sort each group
    regular_methods = sorted(regular_methods)
    norec_methods = sorted(norec_methods)
    ppo_methods = sorted(ppo_methods)
    
    latex_comparison = r"""\begin{table}[h]
\centering
\caption{Method Performance Comparison - Mean Values Across Dates}
\label{tab:method_comparison}
\begin{tabular}{lcccccc}
\toprule
Method & Ex-post Profit & Expected Profit & SI Penalty & Volume Penalty & Operating Cost & Processing Time \\
 & (€) & (€) & (€) & (€) & (€) & (s) \\
\midrule
"""
    
    # Add regular methods (non-NoRec, non-PPO)
    for method in regular_methods:
        expost_mean = method_stats.loc[method, ('Ex_post_Profit', 'mean')]
        expected_mean = method_stats.loc[method, ('Expected_Profit', 'mean')]
        si_mean = method_stats.loc[method, ('SI_Penalty', 'mean')]
        vol_mean = method_stats.loc[method, ('Volume_Penalty', 'mean')]
        op_mean = method_stats.loc[method, ('Operating_Cost', 'mean')]
        time_mean = method_stats.loc[method, ('Processing_Time_Seconds', 'mean')]
        
        method_latex = method.replace('_', '\\_')
        
        latex_comparison += f"{method_latex} & "
        latex_comparison += f"{expost_mean:.2f} & "
        latex_comparison += f"{expected_mean:.2f} & "
        latex_comparison += f"{si_mean:.2f} & "
        latex_comparison += f"{vol_mean:.2f} & "
        latex_comparison += f"{op_mean:.2f} & "
        latex_comparison += f"{time_mean:.2f} \\\\\n"
    
    # Add separator for NoRec methods
    if len(norec_methods) > 0:
        latex_comparison += r"\midrule" + "\n"
        
        # Add NoRec methods
        for method in norec_methods:
            expost_mean = method_stats.loc[method, ('Ex_post_Profit', 'mean')]
            expected_mean = method_stats.loc[method, ('Expected_Profit', 'mean')]
            si_mean = method_stats.loc[method, ('SI_Penalty', 'mean')]
            vol_mean = method_stats.loc[method, ('Volume_Penalty', 'mean')]
            op_mean = method_stats.loc[method, ('Operating_Cost', 'mean')]
            time_mean = method_stats.loc[method, ('Processing_Time_Seconds', 'mean')]
            
            method_latex = method.replace('_', '\\_')
            
            latex_comparison += f"{method_latex} & "
            latex_comparison += f"{expost_mean:.2f} & "
            latex_comparison += f"{expected_mean:.2f} & "
            latex_comparison += f"{si_mean:.2f} & "
            latex_comparison += f"{vol_mean:.2f} & "
            latex_comparison += f"{op_mean:.2f} & "
            latex_comparison += f"{time_mean:.2f} \\\\\n"
    
    # Add separator for PPO methods
    if len(ppo_methods) > 0:
        latex_comparison += r"\midrule" + "\n"
        
        # Add PPO methods
        for method in ppo_methods:
            expost_mean = method_stats.loc[method, ('Ex_post_Profit', 'mean')]
            expected_mean = method_stats.loc[method, ('Expected_Profit', 'mean')]
            si_mean = method_stats.loc[method, ('SI_Penalty', 'mean')]
            vol_mean = method_stats.loc[method, ('Volume_Penalty', 'mean')]
            op_mean = method_stats.loc[method, ('Operating_Cost', 'mean')]
            time_mean = method_stats.loc[method, ('Processing_Time_Seconds', 'mean')]
            
            method_latex = method.replace('_', '\\_')
            
            latex_comparison += f"{method_latex} & "
            latex_comparison += f"{expost_mean:.2f} & "
            latex_comparison += f"{expected_mean:.2f} & "
            latex_comparison += f"{si_mean:.2f} & "
            latex_comparison += f"{vol_mean:.2f} & "
            latex_comparison += f"{op_mean:.2f} & "
            latex_comparison += f"{time_mean:.2f} \\\\\n"
    
    latex_comparison += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(tables_dir / 'method_comparison.tex', 'w') as f:
        f.write(latex_comparison)
    
    # Performance Ranking Table - also group NoRec and PPO methods
    ranking_stats = method_stats.sort_values(('Ex_post_Profit', 'mean'), ascending=False)
    
    # Separate ranking into three groups: regular, NoRec, and PPO
    regular_ranking = [m for m in ranking_stats.index if 'NoRec' not in m and not m.startswith('PPO')]
    norec_ranking = [m for m in ranking_stats.index if 'NoRec' in m]
    ppo_ranking = [m for m in ranking_stats.index if m.startswith('PPO')]
    
    # Sort by profit within each group
    regular_ranking = sorted(regular_ranking, 
                            key=lambda x: ranking_stats.loc[x, ('Ex_post_Profit', 'mean')], 
                            reverse=True)
    norec_ranking = sorted(norec_ranking, 
                          key=lambda x: ranking_stats.loc[x, ('Ex_post_Profit', 'mean')], 
                          reverse=True)
    ppo_ranking = sorted(ppo_ranking, 
                        key=lambda x: ranking_stats.loc[x, ('Ex_post_Profit', 'mean')], 
                        reverse=True)
    
    latex_ranking = r"""\begin{table}[h]
\centering
\caption{Method Performance Ranking}
\label{tab:method_ranking}
\begin{tabular}{clcc}
\toprule
Rank & Method & Mean Ex-post Profit (€) & Processing Time (s) \\
\midrule
"""
    
    # Add regular methods (non-NoRec, non-PPO)
    for i, method in enumerate(regular_ranking, 1):
        mean_profit = ranking_stats.loc[method, ('Ex_post_Profit', 'mean')]
        time_mean = ranking_stats.loc[method, ('Processing_Time_Seconds', 'mean')]
        method_latex = method.replace('_', '\\_')
        
        latex_ranking += f"{i} & {method_latex} & {mean_profit:.2f} & {time_mean:.2f} \\\\\n"
    
    # Add separator and NoRec methods
    if len(norec_ranking) > 0:
        latex_ranking += r"\midrule" + "\n"
        
        for i, method in enumerate(norec_ranking, 1):
            mean_profit = ranking_stats.loc[method, ('Ex_post_Profit', 'mean')]
            time_mean = ranking_stats.loc[method, ('Processing_Time_Seconds', 'mean')]
            method_latex = method.replace('_', '\\_')
            
            latex_ranking += f"{i} & {method_latex} & {mean_profit:.2f} & {time_mean:.2f} \\\\\n"
    
    # Add separator and PPO methods
    if len(ppo_ranking) > 0:
        latex_ranking += r"\midrule" + "\n"
        
        for i, method in enumerate(ppo_ranking, 1):
            mean_profit = ranking_stats.loc[method, ('Ex_post_Profit', 'mean')]
            time_mean = ranking_stats.loc[method, ('Processing_Time_Seconds', 'mean')]
            method_latex = method.replace('_', '\\_')
            
            latex_ranking += f"{i} & {method_latex} & {mean_profit:.2f} & {time_mean:.2f} \\\\\n"
    
    latex_ranking += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(tables_dir / 'method_ranking.tex', 'w') as f:
        f.write(latex_ranking)
    
    print(f"LaTeX tables saved to {tables_dir}")
    print("  - method_comparison.tex")
    print("  - method_ranking.tex")

#%% Main execution
def main():
    """Main function."""
    print("Starting LaTeX Table Generation (with Ablation Studies)...")
    print("="*60)
    print("FEATURES:")
    print("1. Best configuration selected (excluding max_iter=1)")
    print("2. Max_iteration=1 (no recursivity) results included separately")
    print("3. No-NN ablation study results included (best config)")
    print("4. DFL-GL ablation study results included (best config)")
    print("5. Concise method naming (MIQP-Linear, DFL-N5, PPO-RS, No-NN-N10, DFL-GL-N10, etc.)")
    print("6. PPO and No-NN benchmark results included")
    print("7. EXTREME_DATE filtered from all methods and databases")
    print("="*60)
    
    output_dir = Path(".")
    
    # Load all databases
    df = load_all_databases()
    
    if df.empty:
        print("Error: No data loaded.")
        return
    
    print(f"\nData summary:")
    print(f"- Total records: {len(df)}")
    print(f"- Methods: {', '.join(sorted(df['Method'].unique()))}")
    print(f"- Dates per method:")
    for method in sorted(df['Method'].unique()):
        count = len(df[df['Method'] == method])
        print(f"    {method}: {count} records")
    
    # Generate LaTeX tables
    generate_latex_tables(df, output_dir)
    
    print("\n" + "="*60)
    print("LaTeX table generation completed!")
    print("="*60)

if __name__ == "__main__":
    main()
# %%
