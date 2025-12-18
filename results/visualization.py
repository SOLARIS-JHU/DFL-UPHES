#%%!/usr/bin/env python3
"""
Publication-Quality Plots for UPHES Optimization Methods
Generate IEEE-style density plots, noise robustness analysis, and trade-off visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
import os

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

# File paths
MIQP_PATHS = {
    'MIQP-Linear': r'..\MIQP\MIQP_linear\MILP_global_linear_benchmark.csv',
    'MIQP-Piecewise': r'..\MIQP\MIQP_piecewise\MIQP_piecewise_benchmark.csv'
}
DFL_VALIDATION_PATH = r'..\DFL_noise\validation_results\comprehensive\master_validation_benchmarks.csv'
ABLATION_BENCHMARK_PATH = r'..\no_NN_ablation\validation_results\ablation_study\ablation_benchmarks.csv'
DFL_GL_ABLATION_PATH = r'..\DFL_GL_ablation\validation_results\comprehensive\master_validation_benchmarks.csv'
EXTREME_DATE = '2024-12-12'


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


def filter_extreme_date(df, method_name=""):
    """Filter out EXTREME_DATE records."""
    initial_count = len(df)
    df = df[df['Date'] != EXTREME_DATE].copy()
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} {method_name} records with EXTREME_DATE")
    return df


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
        df = filter_extreme_date(df, method_name)

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

    date_col = 'New_Date' if 'New_Date' in df.columns else 'Date'
    if date_col not in df.columns:
        print(f"Warning: No date column found in {data_name}")
        return None

    df['Date'] = df[date_col].apply(standardize_date_format)
    df = filter_extreme_date(df, data_name)

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
        rs_valid = random_samples_df[random_samples_df[max_iter_col] != 1]
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
    """Load DFL validation results and extract best configuration."""
    df = load_and_prepare_df(DFL_VALIDATION_PATH, 'DFL')
    if df is None:
        return pd.DataFrame(), pd.DataFrame(), None

    max_iter_col = find_column(df, ['Max_Iterations', 'Max_Iter', 'max_iteration', 'max_iter'])
    arch_col = find_column(df, ['Architecture', 'architecture', 'arch'])
    layers_col = find_column(df, ['Num_Layers', 'num_layers', 'layers'])

    if not all([max_iter_col, arch_col, layers_col]):
        print(f"Warning: Missing config columns: max_iter={max_iter_col}, arch={arch_col}, layers={layers_col}")
        return pd.DataFrame(), pd.DataFrame(), None

    config_cols = [arch_col, layers_col, max_iter_col]
    best_config, random_samples_df = get_best_config(df, 'DFL-RS', config_cols)

    if best_config is None or len(random_samples_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), None

    # Get best configuration data
    mask = (random_samples_df[arch_col] == best_config[0]) & \
           (random_samples_df[layers_col] == best_config[1]) & \
           (random_samples_df[max_iter_col] == best_config[2])
    best_rs_df = random_samples_df[mask].copy()
    best_rs_df['Method'] = 'DFL-RS'

    # Get NoRec version (max_iter=1)
    mask_norec = (random_samples_df[arch_col] == best_config[0]) & \
                 (random_samples_df[layers_col] == best_config[1]) & \
                 (random_samples_df[max_iter_col] == 1)
    norec_rs_df = random_samples_df[mask_norec].copy()
    norec_rs_df['Method'] = 'DFL-RS-NoRec'

    return best_rs_df, norec_rs_df, best_config


def load_dfl_gl_data():
    """Load DFL-GL ablation results and extract best configuration."""
    df = load_and_prepare_df(DFL_GL_ABLATION_PATH, 'DFL-GL')
    if df is None:
        print("  Error: Could not load DFL-GL data file")
        return pd.DataFrame()

    max_iter_col = find_column(df, ['Max_Iterations', 'Max_Iteration', 'Max_Iter', 'max_iteration', 'max_iter'])
    arch_col = find_column(df, ['Architecture', 'architecture', 'arch'])
    layers_col = find_column(df, ['Num_Layers', 'num_layers', 'layers'])

    if not all([max_iter_col, arch_col, layers_col]):
        print(f"  Error: Missing config columns - max_iter: {max_iter_col}, arch: {arch_col}, layers: {layers_col}")
        return pd.DataFrame()

    config_cols = [arch_col, layers_col, max_iter_col]
    best_config, random_samples_df = get_best_config(df, 'DFL-GL-RS', config_cols)

    if best_config is None or len(random_samples_df) == 0:
        print("  Error: Could not determine best configuration or no random samples found")
        return pd.DataFrame()

    mask = (random_samples_df[arch_col] == best_config[0]) & \
           (random_samples_df[layers_col] == best_config[1]) & \
           (random_samples_df[max_iter_col] == best_config[2])
    best_gl_rs_df = random_samples_df[mask].copy()
    best_gl_rs_df['Method'] = 'DFL-GL-RS'

    print(f"  Successfully loaded {len(best_gl_rs_df)} DFL-GL-RS records")
    return best_gl_rs_df


def load_ablation_data():
    """Load No-NN ablation study results."""
    df = load_and_prepare_df(ABLATION_BENCHMARK_PATH, 'No-NN')
    if df is None:
        return pd.DataFrame()

    max_iter_col = find_column(df, ['Max_Iterations', 'Max_Iteration', 'Max_Iter', 'max_iteration', 'max_iter'])
    if max_iter_col is None:
        return pd.DataFrame()

    config_cols = [max_iter_col]
    best_config, random_samples_df = get_best_config(df, 'No-NN-RS', config_cols)

    if best_config is None or len(random_samples_df) == 0:
        return pd.DataFrame()

    best_nn_rs_df = random_samples_df[random_samples_df[max_iter_col] == best_config].copy()
    best_nn_rs_df['Method'] = 'No-NN-RS'

    return best_nn_rs_df


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

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_iter = iter(colors)

    # Plot left subplot (GL-based)
    legend_labels_left, legend_handles_left = [], []
    for method_name, df in methods_data_left.items():
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

        color = next(color_iter)
        line = ax1.plot(x_range, density, color=color, linestyle=ls, linewidth=1.0, alpha=0.95)[0]
        ax1.fill_between(x_range, density, alpha=fill_alpha, color=color)
        ax1.axvline(mean_profit, color=color, linestyle='--', linewidth=1.0, alpha=0.7)

        legend_label = f"{label_short} (€{mean_profit:.0f}±{std_profit:.0f})"
        legend_labels_left.append(legend_label)
        legend_handles_left.append(line)

    # Plot right subplot (PW-based)
    legend_labels_right, legend_handles_right = [], []
    for method_name, df in methods_data_right.items():
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

        color = next(color_iter)
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
# Plot 2: Profit Density - Ablation Study
# ============================================================================

def plot_profit_density_ablation(methods_data, output_path, figure_width=3.5, figure_height=2.0,
                                fill_alpha=0.2, bw_factor=0.5):
    """Create density plot for ablation study."""
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    legend_labels, legend_handles = [], []

    for method_name, df in methods_data.items():
        if df.empty or 'Ex_post_Profit' not in df.columns:
            continue

        profit = df['Ex_post_Profit'].dropna()
        if len(profit) < 2:
            continue

        mean_profit, std_profit = profit.mean(), profit.std()
        ls, label_short = get_method_style(method_name, use_ablation_colors=True)

        kde = gaussian_kde(profit, bw_method='scott')
        kde.set_bandwidth(kde.factor * bw_factor)
        x_range = np.linspace(profit.min(), profit.max(), 500)
        density = kde(x_range)

        line = ax.plot(x_range, density, linestyle=ls, linewidth=1.0, alpha=0.9)[0]
        color = line.get_color()

        ax.fill_between(x_range, density, alpha=fill_alpha, color=color)
        ax.axvline(mean_profit, color=color, linestyle='--', linewidth=1.0, alpha=0.6)

        legend_label = f"{label_short} (€{mean_profit:.0f}±{std_profit:.0f})"
        legend_labels.append(legend_label)
        legend_handles.append(line)

    ax.set_xlabel('Ex-post Profit (€)')
    ax.set_ylabel('Density')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(legend_handles, legend_labels, loc='best', frameon=True,
                framealpha=0.95, edgecolor='black', fancybox=False)

    plt.tight_layout(pad=0.2)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Plot 3: Noise Robustness - DFL vs MIQP
# ============================================================================

def plot_noise_robustness_dfl_vs_miqp(output_path):
    """Create line plot showing DFL performance vs MIQP across noise levels."""
    noise_levels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', 'RS']
    x_positions = np.arange(len(noise_levels))

    miqp_pw_baseline = [3837, 3780, 3735, 3716, 3643, 3587, 3546, 3514, 3365]
    miqp_gl_baseline = [3624, 3563, 3451, 3325, 3253, 3136, 3118, 2948, 2810]
    dfl_pw = [3872.41, 3870.92, 3864.53, 3872.47, 3871.29, 3868.49, 3866.57, 3879.12, 3890.00]
    dfl_gl = [3727.59, 3733.36, 3731.93, 3735.49, 3725.12, 3710.25, 3718.57, 3719.21, 3718.35]

    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    # Use consistent color scheme with ablation study
    line_dfl_gl = ax.plot(x_positions, dfl_gl, 's-', linewidth=1.5,
            markersize=4, label='DFL (GL-based)')[0]
    ax.plot(x_positions, miqp_gl_baseline, 's--', color=line_dfl_gl.get_color(), linewidth=1.5,
            markersize=4, label='MIQP-GL-noised', alpha=0.8)

    line_dfl_pw = ax.plot(x_positions, dfl_pw, 'o-', linewidth=1.5,
            markersize=4, label='DFL (PW-based)')[0]
    ax.plot(x_positions, miqp_pw_baseline, 'o--', color=line_dfl_pw.get_color(), linewidth=1.5,
            markersize=4, label='MIQP-PW-noised', alpha=0.8)

    ax.fill_between(x_positions, miqp_pw_baseline, dfl_pw,
                    where=np.array(dfl_pw) >= np.array(miqp_pw_baseline),
                    alpha=0.15, color=line_dfl_pw.get_color(), label='DFL-PW gain')
    ax.fill_between(x_positions, miqp_gl_baseline, dfl_gl,
                    where=np.array(dfl_gl) >= np.array(miqp_gl_baseline),
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
# Plot 4: Noise Robustness - Ablation Study
# ============================================================================

def plot_noise_robustness_ablation(output_path):
    """Create line plot for ablation study across noise levels."""
    noise_levels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', 'RS']
    x_positions = np.arange(len(noise_levels))

    miqp_pw_rs = 3849.42
    miqp_gl_rs = 3409.11
    dfl_pw = [3872.41, 3870.92, 3864.53, 3872.47, 3871.29, 3868.49, 3866.57, 3879.12, 3890.00]
    dfl_gl = [3727.59, 3733.36, 3731.93, 3735.49, 3725.12, 3710.25, 3718.57, 3719.21, 3718.35]
    dfl_pw_no_nn = [3764.13, 3798.99, 3776.93, 3813.68, 3788.56, 3810.25, 3798.56, 3797.99, 3803.94]
    dfl_pw_no_rec = [3821.41, 3831.55, 3841.09, 3847.67, 3848.38, 3869.27, 3855.90, 3849.15, 3851.07]

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

    # Position legend slightly higher
    fig_obj = ax.get_figure()
    axes_bbox = ax.get_position()
    axes_height_inches = fig_obj.get_size_inches()[1] * axes_bbox.height
    delta = 5.0 / (axes_height_inches * 25.4) if axes_height_inches > 0 else 0.02
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0 + 0.3*delta), bbox_transform=ax.transAxes,
                frameon=True, framealpha=0.95, edgecolor='black', fancybox=False)

    plt.tight_layout(pad=0.2)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Plot 5: Profit vs Penalties Trade-off (PW-based Ablation)
# ============================================================================

def plot_profit_vs_penalties_ablation(output_path, figure_width=7.16, figure_height=2.0):
    """Create two side-by-side scatter plots comparing profit vs. penalties for PW-based methods."""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_dfl_pw = colors[0]
    color_miqp_pw = colors[1]
    color_dfl_no_nn = colors[2]
    color_dfl_no_rec = colors[3]


    # Common profit data
    miqp_pw_profit = 3849.42
    dfl_pw_profits = [3872.41, 3870.92, 3864.53, 3872.47, 3871.29,
                      3868.49, 3866.57, 3879.12, 3890.00]
    dfl_pw_no_nn_profits = [3764.13, 3798.99, 3776.93, 3813.68, 3788.56,
                            3810.25, 3798.56, 3797.99, 3803.94]
    dfl_pw_no_rec_profits = [3821.41, 3831.55, 3841.09, 3847.67, 3848.38,
                             3869.27, 3855.90, 3849.15, 3851.07]

    # Volume penalty data
    miqp_pw_vol_penalty = 119.59
    dfl_pw_vol_penalties = [277.65, 245.47, 234.73, 267.47, 273.54,
                            277.43, 261.85, 233.89, 271.72]
    dfl_pw_no_nn_vol_penalties = [534.44, 500.31, 518.72, 494.60, 521.15,
                                   494.47, 518.65, 519.97, 532.53]
    dfl_pw_no_rec_vol_penalties = [195.04, 214.28, 236.24, 237.14, 286.32,
                                    284.50, 338.80, 362.53, 453.13]

    # SI penalty data
    miqp_pw_si_penalty = -19.15
    dfl_pw_si_penalties = [-25.75, -15.18, -17.58, -19.69, -23.59,
                           -22.09, -21.38, -19.75, -31.59]
    dfl_pw_no_nn_si_penalties = [12.68, 7.04, 16.82, 6.48, 8.77,
                                  15.56, 7.57, 15.66, 7.86]
    dfl_pw_no_rec_si_penalties = [-23.70, -25.63, -28.02, -24.02, -26.93,
                                   -25.72, -24.20, -20.08, -2.94]

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
    """Generate all publication-quality plots."""
    print("=" * 80)
    print("Generating Publication-Quality Plots for UPHES Optimization")
    print("=" * 80)

    output_dir = Path("./figures")
    output_dir.mkdir(exist_ok=True)

    # Load data for density plots
    print("\nLoading data...")
    miqp_linear_df = load_miqp_data(MIQP_PATHS['MIQP-Linear'], 'MIQP-Linear')
    miqp_piecewise_df = load_miqp_data(MIQP_PATHS['MIQP-Piecewise'], 'MIQP-Piecewise')
    dfl_rs_df, dfl_rs_norec_df, _ = load_dfl_data()
    dfl_gl_rs_df = load_dfl_gl_data()
    no_nn_rs_df = load_ablation_data()

    print("\nData Summary:")
    print(f"  MIQP-Linear: {len(miqp_linear_df)} records")
    print(f"  MIQP-Piecewise: {len(miqp_piecewise_df)} records")
    print(f"  DFL-RS: {len(dfl_rs_df)} records")
    print(f"  DFL-GL-RS: {len(dfl_gl_rs_df)} records")
    print(f"  No-NN-RS: {len(no_nn_rs_df)} records")
    print(f"  DFL-RS-NoRec: {len(dfl_rs_norec_df)} records")

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Plots...")
    print("=" * 80)

    # Plot 1: Profit Density - Main Contribution
    print("\n[1/5] Profit Density - Main Contribution")
    for ext in ['pdf', 'png']:
        plot_profit_density_main_contribution(
            methods_data_left={'DFL-GL-RS': dfl_gl_rs_df, 'MIQP-Linear': miqp_linear_df},
            methods_data_right={'DFL-RS': dfl_rs_df, 'MIQP-Piecewise': miqp_piecewise_df},
            output_path=output_dir / f'profit_density_main_contribution.{ext}'
        )

    # Plot 2: Profit Density - Ablation Study
    print("\n[2/5] Profit Density - Ablation Study")
    for ext in ['pdf', 'png']:
        plot_profit_density_ablation(
            methods_data={
                'DFL-RS': dfl_rs_df,
                'No-NN-RS': no_nn_rs_df,
                'DFL-RS-NoRec': dfl_rs_norec_df
            },
            output_path=output_dir / f'profit_density_ablation_study.{ext}'
        )

    # Plot 3: Noise Robustness - DFL vs MIQP
    print("\n[3/5] Noise Robustness - DFL vs MIQP")
    for ext in ['pdf', 'png']:
        plot_noise_robustness_dfl_vs_miqp(
            output_path=output_dir / f'noise_robustness_dfl_vs_miqp.{ext}'
        )

    # Plot 4: Noise Robustness - Ablation Study
    print("\n[4/5] Noise Robustness - Ablation Study")
    for ext in ['pdf', 'png']:
        plot_noise_robustness_ablation(
            output_path=output_dir / f'noise_robustness_ablation_study.{ext}'
        )

    # Plot 5: Profit vs Penalties - Ablation
    print("\n[5/5] Profit vs Penalties - Ablation")
    for ext in ['pdf', 'png']:
        plot_profit_vs_penalties_ablation(
            output_path=output_dir / f'profit_vs_penalties_ablation.{ext}'
        )

    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
