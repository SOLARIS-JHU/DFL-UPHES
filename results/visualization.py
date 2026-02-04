#!/usr/bin/env python3
"""
Publication-Quality Plots for UPHES Optimization Methods.
Generate IEEE-style density plots, noise robustness analysis, and trade-off visualizations.
Works from any directory - automatically finds repo root.
Outputs to results/figures/ directory (not repo root).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
import os

#%% Helper to get repo root and handle paths from any directory
def get_repo_root():
    """Find repo root by looking for DFL directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / 'DFL').exists():
            return current
        current = current.parent
    return Path.cwd()  # Fallback to current directory

REPO_ROOT = get_repo_root()
OUTPUT_DIR = REPO_ROOT / 'results' / 'figures'

# IEEE publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.figsize': (3.5, 2.8),
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',
    'legend.fancybox': False
})

LINESTYLES = {
    'DFL-GL': '-', 'DFL': '-', 'MIQP-Linear': '-',
    'MIQP-PW': '-', 'No-NN': '-', 'DFL-NR': '-'
}

# File paths (centralized DFL outputs)
MIQP_PATHS = {
    'MIQP-Linear': REPO_ROOT / 'MIQP' / 'MIQP_linear' / 'MILP_global_linear_benchmark.csv',
    'MIQP-Piecewise': REPO_ROOT / 'MIQP' / 'MIQP_piecewise' / 'MIQP_piecewise_benchmark.csv'
}
DFL_VALIDATION_PATH = REPO_ROOT / 'DFL' / 'outputs' / 'validation_results' / 'comprehensive' / 'master_validation_benchmarks.csv'
COMPREHENSIVE_COMPARISON_PATH = REPO_ROOT / 'results' / 'tables' / 'comprehensive_comparison.csv'

# ============================================================================
# Data Loading Functions
# ============================================================================

def standardize_date_format(date_str):
    """Convert various date formats to YYYY-MM-DD format."""
    if pd.isna(date_str):
        return None
    date_str = str(date_str).strip()
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}" if len(parts[0]) == 4 \
                else f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
    return date_str


def load_csv_with_encoding(file_path):
    """Load CSV with multiple encoding attempts."""
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except:
            continue
    return None


def load_miqp_data(file_path, method_name):
    """Load MIQP benchmark data."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()

    df = load_csv_with_encoding(file_path)
    if df is None:
        return pd.DataFrame()

    column_mapping = {
        'Date': 'Date', 'Expected Profit (€)': 'Expected_Profit',
        'Ex-post Profit (€)': 'Ex_post_Profit', 'SI Penalty (€)': 'SI_Penalty',
        'Vol Penalty (€)': 'Volume_Penalty', 'Op Cost (€)': 'Operating_Cost',
        'Solving Time (s)': 'Processing_Time_Seconds'
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    df['Method'] = method_name

    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(standardize_date_format)

    return df


def find_column(df, candidates):
    """Find first matching column name from candidates."""
    return next((col for col in candidates if col in df.columns), None)


def load_and_prepare_df(file_path, data_name):
    """Load and prepare dataframe with date filtering."""
    if not os.path.exists(file_path):
        print(f"Warning: {data_name} file not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} {data_name} records")

    # Check if date columns exist and have data
    has_date_data = False
    date_col = None

    if 'New_Date' in df.columns:
        non_null_count = df['New_Date'].notna().sum()
        if non_null_count > 0:
            date_col = 'New_Date'
            has_date_data = True

    if not has_date_data and 'Date' in df.columns:
        non_null_count = df['Date'].notna().sum()
        if non_null_count > 0:
            date_col = 'Date'
            has_date_data = True

    # If we have a date column with data, standardize it
    if has_date_data and date_col:
        df['Date'] = df[date_col].apply(standardize_date_format)
    else:
        # No date data available (aggregated results)
        print(f"  ℹ {data_name}: No date data found - skipping date-based filtering (aggregated results)")
        df['Date'] = None

    if 'Noise_Level' in df.columns:
        df['Noise_Level_Numeric'] = pd.to_numeric(df['Noise_Level'], errors='coerce')

    return df


def get_best_config(df, method_name, config_cols):
    """Find best configuration by mean ex-post profit."""
    if 'Data_Type' not in df.columns:
        print(f"Warning: Data_Type column not found in {method_name}")
        return None, pd.DataFrame()

    random_samples_df = df[df['Data_Type'] == 'random_samples'].copy()
    max_iter_col = config_cols[-1] if config_cols else None

    if max_iter_col and max_iter_col in random_samples_df.columns:
        rs_valid = random_samples_df[random_samples_df[max_iter_col] > 1]
    else:
        rs_valid = random_samples_df

    if len(rs_valid) > 0 and config_cols and all(col in rs_valid.columns for col in config_cols):
        rs_config_means = rs_valid.groupby(config_cols)['Ex_post_Profit'].mean()
        best_config = rs_config_means.idxmax()
        print(f"\nBest {method_name} configuration: {best_config}")
        print(f"  Mean ex-post profit: €{rs_config_means[best_config]:.2f}")
        return best_config, random_samples_df

    return None, random_samples_df


def load_dfl_data():
    """Load DFL validation results and extract best configuration for both GL and PW."""
    df = load_and_prepare_df(DFL_VALIDATION_PATH, 'DFL')
    if df is None:
        return pd.DataFrame(), pd.DataFrame(), None

    max_iter_col = find_column(df, ['Max_Iterations', 'Max_Iter', 'max_iteration', 'max_iter'])
    method_type_col = find_column(df, ['Method_Type', 'method_type'])

    if not max_iter_col:
        print(f"Warning: Max_Iterations column not found")
        return pd.DataFrame(), pd.DataFrame(), None

    # Filter for iteration 7 (best configuration for GL and PW)
    df_best = df[df[max_iter_col] == 7].copy()

    if df_best.empty:
        print(f"Warning: No data found for iteration 7")
        return pd.DataFrame(), pd.DataFrame(), None

    # SEPARATE GL-based and PW-based DFL
    df_gl = pd.DataFrame()
    df_pw = pd.DataFrame()
    df_no_nn = pd.DataFrame()

    if not method_type_col or method_type_col not in df_best.columns:
        print(f"  Method_Type column not found, inferring from Database and Architecture...")
        df_best = df_best.copy()
        df_best['Method_Type'] = None

        if 'Architecture' in df_best.columns:
            no_nn_mask = df_best['Architecture'].astype(str).str.lower() == 'nonn'
            df_best.loc[no_nn_mask, 'Method_Type'] = 'No-NN-RS'

        if 'Database' in df_best.columns:
            db_lower = df_best['Database'].astype(str).str.lower()
            gl_mask = db_lower.str.contains('dfl_gl|dfl-gl|miqp_linear', na=False)
            pw_mask = db_lower.str.contains('dfl_pw|dfl-pw|dfl_piecewise|miqp_piecewise', na=False)
            df_best.loc[gl_mask, 'Method_Type'] = 'DFL-GL-RS'
            df_best.loc[pw_mask & df_best['Method_Type'].isna(), 'Method_Type'] = 'DFL-RS'

        method_type_col = 'Method_Type'

    # Try Method_Type column first
    if method_type_col and method_type_col in df_best.columns:
        # Separate by Method_Type column
        if 'DFL-GL-RS' in df_best[method_type_col].values:
            df_gl = df_best[df_best[method_type_col] == 'DFL-GL-RS'].copy()
            df_gl['Method'] = 'DFL-GL-RS'

        # Check for both 'DFL-RS' and 'DFL-PW-RS'
        if 'DFL-RS' in df_best[method_type_col].values:
            df_pw = df_best[df_best[method_type_col] == 'DFL-RS'].copy()
            df_pw['Method'] = 'DFL-RS'
        elif 'DFL-PW-RS' in df_best[method_type_col].values:
            df_pw = df_best[df_best[method_type_col] == 'DFL-PW-RS'].copy()
            df_pw['Method'] = 'DFL-RS'

        if 'No-NN-RS' in df_best[method_type_col].values:
            df_no_nn = df_best[df_best[method_type_col] == 'No-NN-RS'].copy()
            df_no_nn['Method'] = 'No-NN-RS'

        # Fallback: if GL not found by exact name, try prefix matching
        if df_gl.empty:
            gl_mask = df_best[method_type_col].str.contains('GL', case=False, na=False)
            if gl_mask.any():
                df_gl = df_best[gl_mask].copy()
                df_gl['Method'] = 'DFL-GL-RS'

        # Fallback: if PW still not found, try prefix matching
        if df_pw.empty:
            pw_mask = df_best[method_type_col].str.contains('PW', case=False, na=False)
            if pw_mask.any():
                df_pw = df_best[pw_mask].copy()
                df_pw['Method'] = 'DFL-RS'
    # Log what was found
    print(f"  DFL-GL-RS records found: {len(df_gl)}")
    print(f"  DFL-RS (PW-based) records found: {len(df_pw)}")
    if not df_no_nn.empty:
        print(f"  No-NN-RS records found: {len(df_no_nn)}")

    # If no GL or PW found, use No-NN as fallback for visualization
    if df_gl.empty and df_pw.empty:
        if not df_no_nn.empty:
            print(f"  ℹ Using No-NN results as fallback...")
            df_pw = df_no_nn.copy()
        else:
            return pd.DataFrame(), pd.DataFrame(), None

    return df_gl, df_pw, {'arch': 'LSTM', 'layers': 3, 'iters': 7}


def load_comprehensive_comparison():
    """Load comprehensive comparison data from results/tables."""
    if not os.path.exists(COMPREHENSIVE_COMPARISON_PATH):
        print(f"Warning: Comprehensive comparison file not found: {COMPREHENSIVE_COMPARISON_PATH}")
        return None

    df = pd.read_csv(COMPREHENSIVE_COMPARISON_PATH)
    print(f"Loaded comprehensive comparison data: {len(df)} records")
    return df


def get_method_style(method_name, use_ablation_colors=False):
    """Get linestyle and label for a method."""
    style_map = {
        'DFL-GL-RS': (LINESTYLES['DFL-GL'], 'DFL (GL-based)'),
        'DFL-RS': (LINESTYLES['DFL'], 'DFL (PW-based)'),
        'MIQP-Linear': (LINESTYLES['MIQP-Linear'], 'MIQP-GL'),
        'MIQP-Piecewise': (LINESTYLES['MIQP-PW'], 'MIQP-PW'),
        'No-NN-RS': (LINESTYLES['No-NN'], 'DFL (No-NN)'),
        'DFL-RS-NoRec': (LINESTYLES['DFL-NR'], 'DFL (No-Rec)')
    }

    if use_ablation_colors:
        ablation_labels = {
            'DFL-RS': 'DFL (PW-based)',
            'No-NN-RS': 'DFL (PW-no-NN)',
            'DFL-RS-NoRec': 'DFL (PW-no-Rec)'
        }
        return '-', ablation_labels.get(method_name, method_name)

    return style_map.get(method_name, ('-', method_name))


# ============================================================================
# Plot 1: Profit Density - Main Contribution
# ============================================================================

def plot_profit_density_main_contribution(methods_data_left, methods_data_right, output_path,
                                        figure_width=7.16, figure_height=2.0, fill_alpha=0.2, bw_factor=0.5):
    """Create two side-by-side density plots for profit distribution comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figure_width, figure_height))

    # Define explicit color mappings for each method
    color_map_left = {
        'DFL-GL-RS': 'C0',      # Blue for DFL (GL-based)
        'MIQP-Linear': 'C1'     # Orange for MIQP-GL
    }
    color_map_right = {
        'DFL-RS': 'C2',         # Green for DFL (PW-based)
        'MIQP-Piecewise': 'C3'  # Red for MIQP-PW
    }

    # Plot left subplot (GL-based) - DFL first, then MIQP
    legend_labels_left, legend_handles_left = [], []
    plot_order_left = ['DFL-GL-RS', 'MIQP-Linear']
    
    for method_name in plot_order_left:
        if method_name not in methods_data_left:
            continue
        df = methods_data_left[method_name]
        if df.empty or 'Ex_post_Profit' not in df.columns:
            continue

        profit = df['Ex_post_Profit'].dropna()
        if len(profit) < 2:
            continue

        mean_profit, std_profit = profit.mean(), profit.std()
        ls, label_short = get_method_style(method_name, use_ablation_colors=False)

        kde = gaussian_kde(profit, bw_method='scott')
        kde.set_bandwidth(kde.factor * bw_factor)
        x_range = np.linspace(profit.min(), profit.max(), 500)
        density = kde(x_range)

        color = color_map_left[method_name]
        line = ax1.plot(x_range, density, color=color, linestyle=ls, linewidth=1.0, alpha=0.95)[0]
        ax1.fill_between(x_range, density, alpha=fill_alpha, color=color)
        ax1.axvline(mean_profit, color=color, linestyle='--', linewidth=1.0, alpha=0.7)

        legend_label = f"{label_short} (€{mean_profit:.0f}±{std_profit:.0f})"
        legend_labels_left.append(legend_label)
        legend_handles_left.append(line)

    # Plot right subplot (PW-based) - DFL first, then MIQP
    legend_labels_right, legend_handles_right = [], []
    plot_order_right = ['DFL-RS', 'MIQP-Piecewise']
    
    for method_name in plot_order_right:
        if method_name not in methods_data_right:
            continue
        df = methods_data_right[method_name]
        if df.empty or 'Ex_post_Profit' not in df.columns:
            continue

        profit = df['Ex_post_Profit'].dropna()
        if len(profit) < 2:
            continue

        mean_profit, std_profit = profit.mean(), profit.std()
        ls, label_short = get_method_style(method_name, use_ablation_colors=False)

        kde = gaussian_kde(profit, bw_method='scott')
        kde.set_bandwidth(kde.factor * bw_factor)
        x_range = np.linspace(profit.min(), profit.max(), 500)
        density = kde(x_range)

        color = color_map_right[method_name]
        line = ax2.plot(x_range, density, color=color, linestyle=ls, linewidth=1.0, alpha=0.95)[0]
        ax2.fill_between(x_range, density, alpha=fill_alpha, color=color)
        ax2.axvline(mean_profit, color=color, linestyle='--', linewidth=1.0, alpha=0.7)

        legend_label = f"{label_short} (€{mean_profit:.0f}±{std_profit:.0f})"
        legend_labels_right.append(legend_label)
        legend_handles_right.append(line)

    # Formatting
    for ax in [ax1, ax2]:
        ax.set_xlabel('Ex-post Profit (€)')
        ax.set_ylabel('Density')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax1.legend(legend_handles_left, legend_labels_left, loc='best', frameon=True,
                framealpha=0.95, edgecolor='black', fancybox=False)
    ax2.legend(legend_handles_right, legend_labels_right, loc='best', frameon=True,
                framealpha=0.95, edgecolor='black', fancybox=False)

    ax1.set_title('(a) GL-based Methods', pad=5)
    ax2.set_title('(b) PW-based Methods', pad=5)

    plt.tight_layout(pad=0.3, w_pad=0.5)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Plot 2: Noise Robustness - DFL vs MIQP
# ============================================================================

def plot_noise_robustness_dfl_vs_miqp(comp_df, output_path):
    """Create line plot showing DFL performance vs MIQP across noise levels."""
    noise_levels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', 'RS']
    x_positions = np.arange(len(noise_levels))

    # Extract data from comprehensive comparison dataframe
    def get_noise_series(comp_df, method_name, noise_values):
        """Extract profit values for given method across noise levels."""
        profits = []
        for noise in noise_values:
            # Convert noise to string for comparison
            noise_str = str(noise)
            row = comp_df[(comp_df['Method'] == method_name) & (comp_df['Noise'] == noise_str)]
            if not row.empty:
                profits.append(row['Ex_post_Profit'].values[0])
            else:
                profits.append(np.nan)
        return profits

    # Use string values for noise levels to match CSV format
    noise_map = ['10', '20', '30', '40', '50', '60', '70', '80', 'RS']

    # DFL methods (solid lines with markers)
    dfl_gl = get_noise_series(comp_df, 'DFL (GL-based)', noise_map)
    dfl_pw = get_noise_series(comp_df, 'DFL (PW-based)', noise_map)

    # Noisy MIQP baselines (dashed lines with dots)
    miqp_gl_noised = get_noise_series(comp_df, 'MIQP-GL-noised', noise_map)
    miqp_pw_noised = get_noise_series(comp_df, 'MIQP-PW-noised', noise_map)

    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    # Plot DFL methods with solid lines
    line_dfl_gl = ax.plot(x_positions, dfl_gl, 's-', linewidth=1.5,
            markersize=4, label='DFL (GL-based)')[0]

    # Plot noisy MIQP-GL with dots on dashed line
    ax.plot(x_positions, miqp_gl_noised, 'o--', color=line_dfl_gl.get_color(), linewidth=1.5,
            markersize=3, label='MIQP-GL-noised', alpha=0.8)

    line_dfl_pw = ax.plot(x_positions, dfl_pw, '^-', linewidth=1.5,
            markersize=4, label='DFL (PW-based)')[0]

    # Plot noisy MIQP-PW with dots on dashed line
    ax.plot(x_positions, miqp_pw_noised, 'o--', color=line_dfl_pw.get_color(), linewidth=1.5,
            markersize=3, label='MIQP-PW-noised', alpha=0.8)

    # Fill between to show DFL improvement over noisy MIQP
    ax.fill_between(x_positions, miqp_pw_noised, dfl_pw,
                    where=np.array(dfl_pw) >= np.array(miqp_pw_noised),
                    alpha=0.15, color=line_dfl_pw.get_color(), label='DFL-PW gain')
    ax.fill_between(x_positions, miqp_gl_noised, dfl_gl,
                    where=np.array(dfl_gl) >= np.array(miqp_gl_noised),
                    alpha=0.15, color=line_dfl_gl.get_color(), label='DFL-GL gain')

    ax.set_xlabel('Perturbation Level')
    ax.set_ylabel('Ex-post Profit (€)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(noise_levels)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='black', fancybox=False)

    plt.tight_layout(pad=0.2)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Plot 3: Noise Robustness - Ablation Study
# ============================================================================

def plot_noise_robustness_ablation(comp_df, output_path):
    """Create line plot for ablation study across noise levels."""
    noise_levels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', 'RS']
    x_positions = np.arange(len(noise_levels))

    # Extract data from comprehensive comparison dataframe
    def get_noise_series(comp_df, method_name, noise_values):
        """Extract profit values for given method across noise levels."""
        profits = []
        for noise in noise_values:
            # Convert noise to string for comparison
            noise_str = str(noise)
            row = comp_df[(comp_df['Method'] == method_name) & (comp_df['Noise'] == noise_str)]
            if not row.empty:
                profits.append(row['Ex_post_Profit'].values[0])
            else:
                profits.append(np.nan)
        return profits

    # Use string values for noise levels to match CSV format
    noise_map = ['10', '20', '30', '40', '50', '60', '70', '80', 'RS']

    # Get baseline values for MIQP methods (using '--' or 'RS' noise level)
    miqp_pw_row = comp_df[(comp_df['Method'] == 'MIQP-PW') & (comp_df['Noise'] == '--')]
    miqp_gl_row = comp_df[(comp_df['Method'] == 'MIQP-GL') & (comp_df['Noise'] == '--')]
    miqp_pw_rs = miqp_pw_row['Ex_post_Profit'].values[0] if not miqp_pw_row.empty else np.nan
    miqp_gl_rs = miqp_gl_row['Ex_post_Profit'].values[0] if not miqp_gl_row.empty else np.nan

    dfl_pw = get_noise_series(comp_df, 'DFL (PW-based)', noise_map)
    dfl_gl = get_noise_series(comp_df, 'DFL (GL-based)', noise_map)
    dfl_pw_no_nn = get_noise_series(comp_df, 'DFL (PW-no-NN)', noise_map)
    dfl_pw_no_rec = get_noise_series(comp_df, 'DFL (PW-no-Rec)', noise_map)

    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    line_dfl_gl = ax.plot(x_positions, dfl_gl, 's-', linewidth=1.5,
            markersize=4, label='DFL (GL-based)')[0]
    ax.axhline(y=miqp_gl_rs, color=line_dfl_gl.get_color(),
                linestyle='--', linewidth=1.5, label='MIQP-GL', alpha=0.8)

    line_dfl_pw = ax.plot(x_positions, dfl_pw, 'o-', linewidth=1.5,
            markersize=4, label='DFL (PW-based)')[0]
    ax.axhline(y=miqp_pw_rs, color=line_dfl_pw.get_color(),
               linestyle='--', linewidth=1.5, label='MIQP-PW', alpha=0.8)

    ax.plot(x_positions, dfl_pw_no_nn, '^-', linewidth=1.5,
            markersize=4, label='DFL (PW-no-NN)')
    ax.plot(x_positions, dfl_pw_no_rec, 'd-', linewidth=1.5,
            markersize=4, label='DFL (PW-no-Rec)')

    ax.set_xlabel('Perturbation Level')
    ax.set_ylabel('Ex-post Profit (€)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(noise_levels)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.05), frameon=True, framealpha=0.95,
                edgecolor='black', fancybox=False)

    plt.tight_layout(pad=0.2)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Plot 4: Profit vs Penalties Trade-off (PW-based Ablation)
# ============================================================================

def plot_profit_vs_penalties_ablation(comp_df, output_path, figure_width=7.16, figure_height=2.0):
    """Create two side-by-side scatter plots comparing profit vs. penalties for PW-based methods."""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_dfl_pw = colors[0]
    color_miqp_pw = colors[1]
    color_dfl_no_nn = colors[2]
    color_dfl_no_rec = colors[3]

    # Extract data from comprehensive comparison dataframe
    def get_noise_series(comp_df, method_name, noise_values, column):
        """Extract values for given method across noise levels."""
        values = []
        for noise in noise_values:
            # Convert noise to string for comparison
            noise_str = str(noise)
            row = comp_df[(comp_df['Method'] == method_name) & (comp_df['Noise'] == noise_str)]
            if not row.empty:
                values.append(row[column].values[0])
            else:
                values.append(np.nan)
        return values

    # Use string values for noise levels to match CSV format
    noise_map = ['10', '20', '30', '40', '50', '60', '70', '80', 'RS']

    # Get MIQP-PW baseline data (using '--' noise level)
    miqp_pw_row = comp_df[(comp_df['Method'] == 'MIQP-PW') & (comp_df['Noise'] == '--')]
    miqp_pw_profit = miqp_pw_row['Ex_post_Profit'].values[0] if not miqp_pw_row.empty else np.nan
    miqp_pw_vol_penalty = miqp_pw_row['Volume_Penalty'].values[0] if not miqp_pw_row.empty else np.nan
    miqp_pw_si_penalty = miqp_pw_row['SI_Penalty'].values[0] if not miqp_pw_row.empty else np.nan

    # Get DFL method data
    dfl_pw_profits = get_noise_series(comp_df, 'DFL (PW-based)', noise_map, 'Ex_post_Profit')
    dfl_pw_vol_penalties = get_noise_series(comp_df, 'DFL (PW-based)', noise_map, 'Volume_Penalty')
    dfl_pw_si_penalties = get_noise_series(comp_df, 'DFL (PW-based)', noise_map, 'SI_Penalty')

    dfl_pw_no_nn_profits = get_noise_series(comp_df, 'DFL (PW-no-NN)', noise_map, 'Ex_post_Profit')
    dfl_pw_no_nn_vol_penalties = get_noise_series(comp_df, 'DFL (PW-no-NN)', noise_map, 'Volume_Penalty')
    dfl_pw_no_nn_si_penalties = get_noise_series(comp_df, 'DFL (PW-no-NN)', noise_map, 'SI_Penalty')

    dfl_pw_no_rec_profits = get_noise_series(comp_df, 'DFL (PW-no-Rec)', noise_map, 'Ex_post_Profit')
    dfl_pw_no_rec_vol_penalties = get_noise_series(comp_df, 'DFL (PW-no-Rec)', noise_map, 'Volume_Penalty')
    dfl_pw_no_rec_si_penalties = get_noise_series(comp_df, 'DFL (PW-no-Rec)', noise_map, 'SI_Penalty')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figure_width, figure_height))

    # Left subplot: Volume Penalty
    ax1.scatter(miqp_pw_vol_penalty, miqp_pw_profit, c=color_miqp_pw, marker='X',
                label='MIQP-PW', s=80, zorder=10,
                edgecolor='black', linewidth=0.8)
    ax1.scatter(dfl_pw_vol_penalties, dfl_pw_profits, c=color_dfl_pw, marker='o',
                label='DFL (PW-based)', s=25, alpha=0.7, linewidths=0.5, edgecolors='black')
    ax1.scatter(dfl_pw_no_nn_vol_penalties, dfl_pw_no_nn_profits, c=color_dfl_no_nn, marker='^',
                label='DFL (PW-no-NN)', s=25, alpha=0.7, linewidths=0.5, edgecolors='black')
    ax1.scatter(dfl_pw_no_rec_vol_penalties, dfl_pw_no_rec_profits, c=color_dfl_no_rec, marker='d',
                label='DFL (PW-no-Rec)', s=25, alpha=0.7, linewidths=0.5, edgecolors='black')

    # Right subplot: SI Penalty
    ax2.scatter(miqp_pw_si_penalty, miqp_pw_profit, c=color_miqp_pw, marker='X',
                label='MIQP-PW', s=80, zorder=10,
                edgecolor='black', linewidth=0.8)
    ax2.scatter(dfl_pw_si_penalties, dfl_pw_profits, c=color_dfl_pw, marker='o',
                label='DFL (PW-based)', s=25, alpha=0.7, linewidths=0.5, edgecolors='black')
    ax2.scatter(dfl_pw_no_nn_si_penalties, dfl_pw_no_nn_profits, c=color_dfl_no_nn, marker='^',
                label='DFL (PW-no-NN)', s=25, alpha=0.7, linewidths=0.5, edgecolors='black')
    ax2.scatter(dfl_pw_no_rec_si_penalties, dfl_pw_no_rec_profits, c=color_dfl_no_rec, marker='d',
                label='DFL (PW-no-Rec)', s=25, alpha=0.7, linewidths=0.5, edgecolors='black')

    # Formatting
    ax1.set_xlabel('Volume Penalty (€)')
    ax1.set_ylabel('Ex-post Profit (€)')
    ax1.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='lower left', bbox_to_anchor=(0.3, 0.0), frameon=True, framealpha=0.95,
                edgecolor='black', fancybox=False)
    ax1.set_title('(a) Profit vs. Volume Penalty', pad=5)

    ax2.set_xlabel('System Imbalance Penalty (€)')
    ax2.set_ylabel('Ex-post Profit (€)')
    ax2.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='lower left', bbox_to_anchor=(0.3, 0.0), frameon=True, framealpha=0.95,
                edgecolor='black', fancybox=False)
    ax2.set_title('(b) Profit vs. SI Penalty', pad=5)

    plt.tight_layout(pad=0.3, w_pad=0.5)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Generate publication-quality plots."""
    print("=" * 80)
    print("Generating Publication-Quality Plots for UPHES Optimization")
    print("=" * 80)
    print(f"\nRepository root: {REPO_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for density plots
    print("\nLoading data...")
    miqp_linear_df = load_miqp_data(MIQP_PATHS['MIQP-Linear'], 'MIQP-Linear')
    miqp_piecewise_df = load_miqp_data(MIQP_PATHS['MIQP-Piecewise'], 'MIQP-Piecewise')
    dfl_gl_df, dfl_pw_df, best_config = load_dfl_data()
    comp_df = load_comprehensive_comparison()

    print("\nData Summary:")
    print(f"  MIQP-Linear: {len(miqp_linear_df)} records")
    print(f"  MIQP-Piecewise: {len(miqp_piecewise_df)} records")
    print(f"  DFL-GL (iteration 7): {len(dfl_gl_df)} records")
    print(f"  DFL-PW (iteration 7): {len(dfl_pw_df)} records")
    if best_config:
        print(f"  Best config: {best_config}")
    if comp_df is not None:
        print(f"  Comprehensive comparison: {len(comp_df)} records")

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Plots (4 of 4)...")
    print("=" * 80)

    # Plot 1: Profit Density - Main Contribution (GL-based vs PW-based)
    print("\n[1/4] Profit Density - Main Contribution")
    for ext in ['pdf', 'png']:
        # Prepare data for both sides: Left = GL-based, Right = PW-based
        methods_left = {'MIQP-Linear': miqp_linear_df}
        methods_right = {'MIQP-Piecewise': miqp_piecewise_df}

        # Filter DFL data to ONLY Random Samples (RS) for density plot
        # Add GL-based DFL to left side if available
        if not dfl_gl_df.empty and len(dfl_gl_df) > 0:
            dfl_gl_rs = dfl_gl_df[(dfl_gl_df['Noise_Level'] == 'random') | (dfl_gl_df['Noise_Level'] == 'N/A')].copy()
            if len(dfl_gl_rs) > 0:
                methods_left['DFL-GL-RS'] = dfl_gl_rs
                print(f"    ✓ GL-based DFL (RS only) added to left subplot ({len(dfl_gl_rs)} records)")
            else:
                print(f"    ⚠ Warning: GL-based DFL RS data not found")
        else:
            print(f"    ⚠ Warning: GL-based DFL data not found")

        # Add PW-based DFL to right side if available
        if not dfl_pw_df.empty and len(dfl_pw_df) > 0:
            dfl_pw_rs = dfl_pw_df[(dfl_pw_df['Noise_Level'] == 'random') | (dfl_pw_df['Noise_Level'] == 'N/A')].copy()
            if len(dfl_pw_rs) > 0:
                methods_right['DFL-RS'] = dfl_pw_rs
                print(f"    ✓ PW-based DFL (RS only) added to right subplot ({len(dfl_pw_rs)} records)")
            else:
                print(f"    ⚠ Warning: PW-based DFL RS data not found")
        else:
            print(f"    ⚠ Warning: PW-based DFL data not found")

        plot_profit_density_main_contribution(
            methods_data_left=methods_left,
            methods_data_right=methods_right,
            output_path=OUTPUT_DIR / f'profit_density_main_contribution.{ext}'
        )

    # Plot 2: Noise Robustness - DFL vs MIQP
    print("\n[2/4] Noise Robustness - DFL vs MIQP")
    if comp_df is not None:
        for ext in ['pdf', 'png']:
            plot_noise_robustness_dfl_vs_miqp(
                comp_df=comp_df,
                output_path=OUTPUT_DIR / f'noise_robustness_dfl_vs_miqp.{ext}'
            )
    else:
        print("  ⚠ Skipping: comprehensive comparison data not available")

    # Plot 3: Noise Robustness - Ablation Study
    print("\n[3/4] Noise Robustness - Ablation Study")
    if comp_df is not None:
        for ext in ['pdf', 'png']:
            plot_noise_robustness_ablation(
                comp_df=comp_df,
                output_path=OUTPUT_DIR / f'noise_robustness_ablation_study.{ext}'
            )
    else:
        print("  ⚠ Skipping: comprehensive comparison data not available")

    # Plot 4: Profit vs Penalties - Ablation
    print("\n[4/4] Profit vs Penalties - Ablation")
    if comp_df is not None:
        for ext in ['pdf', 'png']:
            plot_profit_vs_penalties_ablation(
                comp_df=comp_df,
                output_path=OUTPUT_DIR / f'profit_vs_penalties_ablation.{ext}'
            )
    else:
        print("  ⚠ Skipping: comprehensive comparison data not available")

    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
