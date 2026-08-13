#!/usr/bin/env python3
"""
Generate comprehensive comparison table for UPHES optimization methods.
Produces a single unified LaTeX table combining all methods and configurations.
Works from any directory - automatically finds repo root.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

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

#%% Database file paths and configurations
MIQP_PATHS = {
    'MIQP-Linear': REPO_ROOT / 'MIQP' / 'MIQP_linear' / 'MILP_global_linear_benchmark.csv',
    'MIQP-Piecewise': REPO_ROOT / 'MIQP' / 'MIQP_piecewise' / 'MIQP_piecewise_benchmark.csv'
}

# Centralized DFL output paths
DFL_VALIDATION_PATH = REPO_ROOT / 'DFL' / 'outputs' / 'validation_results' / 'comprehensive' / 'master_validation_benchmarks.csv'

OUTPUT_DIR = REPO_ROOT / 'results' / 'tables'

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

    return df

def load_dfl_validation_data():
    """Load DFL validation results and extract best configurations."""
    if not os.path.exists(DFL_VALIDATION_PATH):
        print(f"Warning: DFL validation file not found: {DFL_VALIDATION_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(DFL_VALIDATION_PATH)
    print(f"Loaded {len(df)} DFL validation records")

    # Check if date columns exist and have data
    has_date_data = False
    date_col = None

    if 'New_Date' in df.columns:
        non_null_count = df['New_Date'].notna().sum()
        if non_null_count > 0:
            date_col = 'New_Date'
            has_date_data = True
            print(f"  Using date column: {date_col} ({non_null_count} non-null values)")

    if not has_date_data and 'Closest_Historical_Date' in df.columns:
        non_null_count = df['Closest_Historical_Date'].notna().sum()
        if non_null_count > 0:
            date_col = 'Closest_Historical_Date'
            has_date_data = True
            print(f"  Using date column: {date_col} ({non_null_count} non-null values)")

    # If we have date data, standardize date format
    if has_date_data and date_col:
        df['Date_Standardized'] = df[date_col].apply(standardize_date_format)

        # Debug: Show sample standardized dates (non-null only)
        sample_dates = df['Date_Standardized'].dropna().unique()[:5]
        print(f"  Sample standardized dates: {sample_dates}")
    else:
        print(f"  ℹ No date columns with data found - skipping date-based filtering (aggregated results)")
        print(f"    Available date columns: New_Date={('New_Date' in df.columns)}, Closest_Historical_Date={('Closest_Historical_Date' in df.columns)}")

    # Convert noise level to numeric
    df['Noise_Level_Numeric'] = pd.to_numeric(df['Noise_Level'], errors='coerce')

    return df

def generate_comprehensive_table(df_miqp_gl, df_miqp_pw, df_dfl):
    """Generate single comprehensive LaTeX table with all methods and noise levels."""

    print("\n" + "="*80)
    print("Generating Comprehensive Performance Comparison Table")
    print("="*80)

    # Calculate mean values per method and noise level
    rows = []

    # ===== MIQP Baselines =====
    if not df_miqp_gl.empty:
        miqp_gl_stats = df_miqp_gl.groupby('Method').agg({
            'Ex_post_Profit': ['mean', 'std', 'count'],
            'Expected_Profit': 'mean',
            'SI_Penalty': 'mean',
            'Volume_Penalty': 'mean',
            'Operating_Cost': 'mean',
            'Processing_Time_Seconds': 'mean'
        }).iloc[0]

        rows.append({
            'Method': 'MIQP-GL',
            'Noise': '--',
            'Samples': int(miqp_gl_stats[('Ex_post_Profit', 'count')]),
            'Ex_post_Profit': miqp_gl_stats[('Ex_post_Profit', 'mean')],
            'Ex_post_Std': miqp_gl_stats[('Ex_post_Profit', 'std')],
            'Expected_Profit': miqp_gl_stats[('Expected_Profit', 'mean')],
            'SI_Penalty': miqp_gl_stats[('SI_Penalty', 'mean')],
            'Volume_Penalty': miqp_gl_stats[('Volume_Penalty', 'mean')],
            'Operating_Cost': miqp_gl_stats[('Operating_Cost', 'mean')],
            'Processing_Time_Seconds': miqp_gl_stats[('Processing_Time_Seconds', 'mean')]
        })

    if not df_miqp_pw.empty:
        miqp_pw_stats = df_miqp_pw.groupby('Method').agg({
            'Ex_post_Profit': ['mean', 'std', 'count'],
            'Expected_Profit': 'mean',
            'SI_Penalty': 'mean',
            'Volume_Penalty': 'mean',
            'Operating_Cost': 'mean',
            'Processing_Time_Seconds': 'mean'
        }).iloc[0]

        rows.append({
            'Method': 'MIQP-PW',
            'Noise': '--',
            'Samples': int(miqp_pw_stats[('Ex_post_Profit', 'count')]),
            'Ex_post_Profit': miqp_pw_stats[('Ex_post_Profit', 'mean')],
            'Ex_post_Std': miqp_pw_stats[('Ex_post_Profit', 'std')],
            'Expected_Profit': miqp_pw_stats[('Expected_Profit', 'mean')],
            'SI_Penalty': miqp_pw_stats[('SI_Penalty', 'mean')],
            'Volume_Penalty': miqp_pw_stats[('Volume_Penalty', 'mean')],
            'Operating_Cost': miqp_pw_stats[('Operating_Cost', 'mean')],
            'Processing_Time_Seconds': miqp_pw_stats[('Processing_Time_Seconds', 'mean')]
        })

    # ===== DFL Methods AND Noisy MIQP Methods =====
    if not df_dfl.empty:
        # Filter for max_iterations == 7 (best configuration for GL and PW) OR Max_Iterations == 0 (noisy MIQP)
        df_best = df_dfl[(df_dfl['Max_Iterations'] == 7) | (df_dfl['Max_Iterations'] == 0)].copy()

        # Get unique method types (DFL-GL-RS, DFL-RS, No-NN-RS, etc.)
        method_types = []
        use_method_type = True
        use_database_path = False

        if 'Method_Type' in df_best.columns:
            method_types = sorted(df_best['Method_Type'].dropna().unique())
        elif 'Database' in df_best.columns:
            print(f"  Inferring DFL method types from Database names and architecture...")
            df_best = df_best.copy()
            df_best['Method_Type'] = None

            if 'Architecture' in df_best.columns:
                no_nn_mask = df_best['Architecture'].astype(str).str.lower() == 'nonn'
                df_best.loc[no_nn_mask, 'Method_Type'] = 'No-NN-RS'

            db_lower = df_best['Database'].astype(str).str.lower()
            gl_mask = db_lower.str.contains('miqp_linear', na=False)
            pw_mask = db_lower.str.contains('miqp_piecewise', na=False)

            df_best.loc[gl_mask, 'Method_Type'] = 'DFL-GL-RS'
            df_best.loc[pw_mask & df_best['Method_Type'].isna(), 'Method_Type'] = 'DFL-RS'

            method_types = sorted(df_best['Method_Type'].dropna().unique())
        else:
            # Fall back to architecture-based grouping
            method_types = sorted(df_best['Architecture'].unique())
            use_method_type = False

        print(f"\nDFL Methods found: {[mt[0] if isinstance(mt, tuple) else mt for mt in method_types]}")

        for method_type_info in method_types:
            # Handle both tuple (method_type, pattern) and string (method_type) formats
            if isinstance(method_type_info, tuple):
                method_type, db_pattern = method_type_info
            else:
                method_type = method_type_info
                db_pattern = None

            if use_method_type:
                method_data = df_best[df_best['Method_Type'] == method_type]
            elif use_database_path:
                method_data = df_best[df_best['Database'].str.lower().str.contains(db_pattern, na=False)]
            else:
                method_data = df_best[df_best['Architecture'] == method_type]

            # Create label from method type
            if method_type.startswith('DFL-GL'):
                method_label = 'DFL (GL-based)'
            elif method_type == 'DFL-PW-RS' or method_type == 'DFL-RS':
                method_label = 'DFL (PW-based)'
            elif method_type == 'DFL-PW-no-Rec':
                method_label = 'DFL (PW-no-Rec)'
            elif method_type == 'DFL-PW-no-NN' or method_type.startswith('No-NN'):
                method_label = 'DFL (PW-no-NN)'
            elif method_type == 'IPOPT-NLP-PW':
                method_label = 'IPOPT-NLP (PW)'
            elif method_type == 'IPOPT-NLP-GL':
                method_label = 'IPOPT-NLP (GL)'
            elif method_type == 'MIQP-GL-noised':
                method_label = 'MIQP-GL-noised'
            elif method_type == 'MIQP-PW-noised':
                method_label = 'MIQP-PW-noised'
            else:
                method_label = f'DFL ({method_type}-based)' if use_method_type else f"DFL ({method_type})"

            print(f"  Processing {method_label}: {len(method_data)} records")

            # Get noise levels
            noise_levels = sorted([nl for nl in method_data['Noise_Level_Numeric'].unique() if pd.notna(nl)])

            # Add rows for each noise level
            for noise_level in noise_levels:
                noise_data = method_data[method_data['Noise_Level_Numeric'] == noise_level]
                if not noise_data.empty:
                    stats = noise_data.agg({
                        'Ex_post_Profit': ['mean', 'std', 'count'],
                        'Expected_Profit': 'mean',
                        'SI_Penalty': 'mean',
                        'Volume_Penalty': 'mean',
                        'Operating_Cost': 'mean',
                        'Processing_Time_Seconds': 'mean'
                    })

                    noise_pct = int(noise_level * 100)
                    # Stats has aggregation functions as rows, columns as metrics
                    sample_count = int(stats['Ex_post_Profit']['count'])
                    ex_post_mean = stats['Ex_post_Profit']['mean']
                    ex_post_std = stats['Ex_post_Profit']['std']

                    rows.append({
                        'Method': method_label,
                        'Noise': str(noise_pct),
                        'Samples': sample_count,
                        'Ex_post_Profit': ex_post_mean,
                        'Ex_post_Std': ex_post_std,
                        'Expected_Profit': stats['Expected_Profit']['mean'] if 'Expected_Profit' in stats.columns else float('nan'),
                        'SI_Penalty': stats['SI_Penalty']['mean'] if 'SI_Penalty' in stats.columns else float('nan'),
                        'Volume_Penalty': stats['Volume_Penalty']['mean'] if 'Volume_Penalty' in stats.columns else float('nan'),
                        'Operating_Cost': stats['Operating_Cost']['mean'] if 'Operating_Cost' in stats.columns else float('nan'),
                        'Processing_Time_Seconds': stats['Processing_Time_Seconds']['mean'] if 'Processing_Time_Seconds' in stats.columns else float('nan')
                    })

            # Add random samples row
            random_data = method_data[(method_data['Noise_Level'] == 'random') | (method_data['Noise_Level'] == 'N/A')]
            if not random_data.empty:
                stats = random_data.agg({
                    'Ex_post_Profit': ['mean', 'std', 'count'],
                    'Expected_Profit': 'mean',
                    'SI_Penalty': 'mean',
                    'Volume_Penalty': 'mean',
                    'Operating_Cost': 'mean',
                    'Processing_Time_Seconds': 'mean'
                })

                sample_count = int(stats['Ex_post_Profit']['count'])

                rows.append({
                    'Method': method_label,
                    'Noise': 'RS',
                    'Samples': sample_count,
                    'Ex_post_Profit': stats['Ex_post_Profit']['mean'],
                    'Ex_post_Std': stats['Ex_post_Profit']['std'],
                    'Expected_Profit': stats['Expected_Profit']['mean'] if 'Expected_Profit' in stats.columns else float('nan'),
                    'SI_Penalty': stats['SI_Penalty']['mean'] if 'SI_Penalty' in stats.columns else float('nan'),
                    'Volume_Penalty': stats['Volume_Penalty']['mean'] if 'Volume_Penalty' in stats.columns else float('nan'),
                    'Operating_Cost': stats['Operating_Cost']['mean'] if 'Operating_Cost' in stats.columns else float('nan'),
                    'Processing_Time_Seconds': stats['Processing_Time_Seconds']['mean'] if 'Processing_Time_Seconds' in stats.columns else float('nan')
                })

        # ===== Process DFL (PW-no-Rec) - DFL-RS with Max_Iterations == 1 =====
        print("\n  Processing DFL (PW-no-Rec) - single iteration ablation study...")
        df_no_rec = df_dfl[(df_dfl['Max_Iterations'] == 1) &
                           ((df_dfl['Method_Type'] == 'DFL-RS') |
                            (df_dfl['Database'].astype(str).str.lower().str.contains('miqp_piecewise', na=False)))].copy()

        if not df_no_rec.empty:
            print(f"  Processing DFL (PW-no-Rec): {len(df_no_rec)} records")

            # Get noise levels
            df_no_rec['Noise_Level_Numeric'] = pd.to_numeric(df_no_rec['Noise_Level'], errors='coerce')
            noise_levels = sorted([nl for nl in df_no_rec['Noise_Level_Numeric'].unique() if pd.notna(nl)])

            # Add rows for each noise level
            for noise_level in noise_levels:
                noise_data = df_no_rec[df_no_rec['Noise_Level_Numeric'] == noise_level]
                if not noise_data.empty:
                    stats = noise_data.agg({
                        'Ex_post_Profit': ['mean', 'std', 'count'],
                        'Expected_Profit': 'mean',
                        'SI_Penalty': 'mean',
                        'Volume_Penalty': 'mean',
                        'Operating_Cost': 'mean',
                        'Processing_Time_Seconds': 'mean'
                    })

                    noise_pct = int(noise_level * 100)
                    sample_count = int(stats['Ex_post_Profit']['count'])
                    ex_post_mean = stats['Ex_post_Profit']['mean']
                    ex_post_std = stats['Ex_post_Profit']['std']

                    rows.append({
                        'Method': 'DFL (PW-no-Rec)',
                        'Noise': str(noise_pct),
                        'Samples': sample_count,
                        'Ex_post_Profit': ex_post_mean,
                        'Ex_post_Std': ex_post_std,
                        'Expected_Profit': stats['Expected_Profit']['mean'] if 'Expected_Profit' in stats.columns else float('nan'),
                        'SI_Penalty': stats['SI_Penalty']['mean'] if 'SI_Penalty' in stats.columns else float('nan'),
                        'Volume_Penalty': stats['Volume_Penalty']['mean'] if 'Volume_Penalty' in stats.columns else float('nan'),
                        'Operating_Cost': stats['Operating_Cost']['mean'] if 'Operating_Cost' in stats.columns else float('nan'),
                        'Processing_Time_Seconds': stats['Processing_Time_Seconds']['mean'] if 'Processing_Time_Seconds' in stats.columns else float('nan')
                    })

            # Add random samples row
            random_data = df_no_rec[(df_no_rec['Noise_Level'] == 'random') | (df_no_rec['Noise_Level'] == 'N/A')]
            if not random_data.empty:
                stats = random_data.agg({
                    'Ex_post_Profit': ['mean', 'std', 'count'],
                    'Expected_Profit': 'mean',
                    'SI_Penalty': 'mean',
                    'Volume_Penalty': 'mean',
                    'Operating_Cost': 'mean',
                    'Processing_Time_Seconds': 'mean'
                })

                sample_count = int(stats['Ex_post_Profit']['count'])

                rows.append({
                    'Method': 'DFL (PW-no-Rec)',
                    'Noise': 'RS',
                    'Samples': sample_count,
                    'Ex_post_Profit': stats['Ex_post_Profit']['mean'],
                    'Ex_post_Std': stats['Ex_post_Profit']['std'],
                    'Expected_Profit': stats['Expected_Profit']['mean'] if 'Expected_Profit' in stats.columns else float('nan'),
                    'SI_Penalty': stats['SI_Penalty']['mean'] if 'SI_Penalty' in stats.columns else float('nan'),
                    'Volume_Penalty': stats['Volume_Penalty']['mean'] if 'Volume_Penalty' in stats.columns else float('nan'),
                    'Operating_Cost': stats['Operating_Cost']['mean'] if 'Operating_Cost' in stats.columns else float('nan'),
                    'Processing_Time_Seconds': stats['Processing_Time_Seconds']['mean'] if 'Processing_Time_Seconds' in stats.columns else float('nan')
                })
        else:
            print(f"  ⚠ Warning: No DFL (PW-no-Rec) data found with Max_Iterations == 1")

    results_df = pd.DataFrame(rows)
    return results_df

def generate_latex_table(results_df):
    """Generate comprehensive LaTeX table with sample counts and std dev."""

    latex = r"""\begin{table*}[!t]
\centering
% \footnotesize
\caption{Comprehensive Performance Comparison Across All Methods and Noise Levels}
\label{tab:method_comparison_updated}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{ccccccccccc}
\toprule
\textbf{Method} & \textbf{Noise} & \textbf{N} & \textbf{Ex-post} & \textbf{Std} & \textbf{Expected} & \textbf{SI} & \textbf{Volume} & \textbf{Operating} & \textbf{Time} \\
 & \textbf{(\%)} &  & \textbf{Profit (€)} & \textbf{Dev} & \textbf{Profit (€)} & \textbf{Penalty (€)} & \textbf{Penalty (€)} & \textbf{Cost (€)} & \textbf{(s)} \\
\midrule
"""

    # Add MIQP rows first
    miqp_rows = results_df[results_df['Method'].str.contains('MIQP')]
    for _, row in miqp_rows.iterrows():
        n = int(row['Samples']) if 'Samples' in row and pd.notna(row['Samples']) else 1
        std_val = row['Ex_post_Std'] if 'Ex_post_Std' in row and pd.notna(row['Ex_post_Std']) else 0
        latex += f"{row['Method']} & {row['Noise']} & {n} & {row['Ex_post_Profit']:.2f} & {std_val:.2f} & {row['Expected_Profit']:.2f} & {row['SI_Penalty']:.2f} & {row['Volume_Penalty']:.2f} & {row['Operating_Cost']:.2f} & {row['Processing_Time_Seconds']:.2f} \\\\\n"

    latex += r"\midrule" + "\n"

    # Add DFL rows grouped by method
    dfl_rows = results_df[~results_df['Method'].str.contains('MIQP')].copy()
    current_method = None
    method_groups = []
    current_group = []

    for _, row in dfl_rows.iterrows():
        if row['Method'] != current_method:
            if current_group:
                method_groups.append((current_method, current_group))
            current_method = row['Method']
            current_group = [row]
        else:
            current_group.append(row)

    if current_group:
        method_groups.append((current_method, current_group))

    # Add grouped DFL rows with multirow
    for method, group in method_groups:
        # Extract method name - handle different formats
        if 'GL-based' in method:
            method_short = 'GL-based'
        elif 'PW-based' in method:
            method_short = 'PW-based'
        elif 'PW-no-NN' in method:
            method_short = 'PW-no-NN'
        elif 'PW-no-Rec' in method:
            method_short = 'PW-no-Rec'
        else:
            method_short = method.replace('DFL (', '').replace('-based)', '').replace('DFL ', '')

        for i, row in enumerate(group):
            n = int(row['Samples']) if 'Samples' in row and pd.notna(row['Samples']) else 1
            std_val = row['Ex_post_Std'] if 'Ex_post_Std' in row and pd.notna(row['Ex_post_Std']) else 0
            if i == 0:
                latex += f"\\multirow{{{len(group)}}}{{*}}{{\\shortstack{{{method_short}}}}} & {row['Noise']} & {n} & {row['Ex_post_Profit']:.2f} & {std_val:.2f} & {row['Expected_Profit']:.2f} & {row['SI_Penalty']:.2f} & {row['Volume_Penalty']:.2f} & {row['Operating_Cost']:.2f} & {row['Processing_Time_Seconds']:.2f} \\\\\n"
            else:
                latex += f" & {row['Noise']} & {n} & {row['Ex_post_Profit']:.2f} & {std_val:.2f} & {row['Expected_Profit']:.2f} & {row['SI_Penalty']:.2f} & {row['Volume_Penalty']:.2f} & {row['Operating_Cost']:.2f} & {row['Processing_Time_Seconds']:.2f} \\\\\n"

        # Add separator between method groups (but not after last group)
        if method != method_groups[-1][0]:
            latex += r"\midrule" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""

    return latex

def main():
    """Main execution."""
    print("="*80)
    print("UPHES Optimization Methods - Comprehensive Table Generation")
    print("="*80)
    print(f"\nRepository root: {REPO_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df_miqp_gl = load_miqp_data(MIQP_PATHS['MIQP-Linear'], 'MIQP-GL')
    df_miqp_pw = load_miqp_data(MIQP_PATHS['MIQP-Piecewise'], 'MIQP-PW')
    df_dfl = load_dfl_validation_data()

    print(f"\nData loaded:")
    print(f"  MIQP-GL: {len(df_miqp_gl)} records")
    print(f"  MIQP-PW: {len(df_miqp_pw)} records")
    print(f"  DFL: {len(df_dfl)} records")

    # Generate comprehensive table
    results_df = generate_comprehensive_table(df_miqp_gl, df_miqp_pw, df_dfl)

    # Generate LaTeX
    latex_table = generate_latex_table(results_df)

    # Save LaTeX table
    table_file = OUTPUT_DIR / 'comprehensive_comparison.tex'
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print(f"\n✓ LaTeX table saved to: {table_file}")

    # Also save CSV for reference
    csv_file = OUTPUT_DIR / 'comprehensive_comparison.csv'
    results_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"✓ CSV summary saved to: {csv_file}")

    print("\n" + "="*80)
    print("Table generation completed!")
    print("="*80)

if __name__ == "__main__":
    main()
