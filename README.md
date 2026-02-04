# DFL-for-UPHES

This repository contains implementations of various optimization methodologies for Underground Pumped Hydroelectric Energy Storage scheduling, including Mixed-Integer Quadratic Programming (MIQP) approaches and Decision-Focused Learning (DFL) techniques.

## Project Status / Which Folder To Use

This repo currently contains both a refactored (structured) DFL implementation and several legacy implementations:

- `DFL/`: refactored, configuration-driven, modular DFL framework (actively being cleaned up; may contain bugs).
- `DFL_GL-based/`, `DFL_PW-based/`, `DFL_no-NN/`: legacy experiment code (less structured, but runs end-to-end and is the recommended starting point if you want reproducible results quickly).

## Prerequisites

**Important**: Before running any scripts, execute `preprocessing.py` in the root directory to update `preprocess.pkl`. This ensures compatibility across different versions of the dill library, as newer versions may not support pickle files created with older syntax.

```bash
python preprocessing.py
```

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Some MIQP scripts require a working Gurobi installation + license (`gurobipy`).

Common entry points (run from the repo root):

```bash
# MIQP baselines
python MIQP/MIQP_linear/MIQP_global_linear.py
python MIQP/MIQP_nn/MIQP_nn.py
python MIQP/MIQP_piecewise/MIQP_piecewise.py

# DFL (legacy, stable)
python DFL_GL-based/DFL_pretraining.py
python DFL_GL-based/DFL_validation.py
python DFL_PW-based/DFL_pretraining.py
python DFL_PW-based/DFL_validation.py
python DFL_no-NN/NN_ablation.py
```

## Complete Workflow: Preprocessing → MIQP → DFL

This section provides the complete command sequence to run the entire pipeline from data preprocessing through DFL training and validation.

### Step 1: Preprocessing

First, update the preprocessed pickle file (required before running any other scripts):

```bash
python preprocessing.py
```

This generates `preprocess.pkl` and ensures compatibility with your dill library version.

### Step 2: Generate MIQP Baselines

Run the two main MIQP baseline methods. Both scripts work from the repo root or their own directories:

**Global Linearization MIQP:**
```bash
python MIQP/MIQP_linear/MIQP_global_linear.py
```
Outputs: `MIQP/MIQP_linear/MILP_global_linear_results.csv`, `MILP_global_linear_benchmark.csv`

**Piecewise Linearization MIQP (with SOS2 constraints):**
```bash
python MIQP/MIQP_piecewise/MIQP_piecewise.py
```
Outputs: `MIQP/MIQP_piecewise/MIQP_piecewise_results.csv`, `MIQP_piecewise_benchmark.csv`

*Note: These scripts require Gurobi and may take several hours to complete depending on the number of dates in the price data.*

### Step 3: Generate Noisy Training Data

Generate noisy datasets for DFL training (run both variants):

```bash
# GL variant (noise levels 10%-80% + random samples)
python DFL/scripts/generate_noisy_data.py --variant GL --noise-levels "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8" --random-samples

# PW variant (noise levels 10%-80% + random samples)
python DFL/scripts/generate_noisy_data.py --variant PW --noise-levels "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8" --random-samples
```

**Note:** The refactored DFL excludes 0% noise level (original MIQP results), focusing on noisy training data (10%-80%) plus random samples for robustness testing.

This creates training datasets in `DFL/outputs/noisy_data/`:
- `MIQP_linear_results_relative_noise_{10-80}pct.csv` and `MIQP_linear_results_random_samples.csv` (GL)
- `MIQP_piecewise_results_relative_noise_{10-80}pct.csv` and `MIQP_piecewise_results_random_samples.csv` (PW)

### Step 4: Train DFL Models

Train DFL models on the noisy datasets. Choose one or both:

**Global Linear (GL) variant:**
```bash
python DFL/scripts/run_pretraining_gl.py
```

**Piecewise (PW) variant:**
```bash
python DFL/scripts/run_pretraining_pw.py
```

**Piecewise No-Recursion (PW-no-Rec) variant (Ablation Study):**
```bash
python DFL/scripts/run_pretraining_pw_norec.py
```

These scripts train the neural network-based optimization models. GL and PW variants use 7 recursive linearization iterations (optimized through validation), while PW-no-Rec uses a single iteration to test the impact of recursive refinement. All save the best model checkpoints.

### Step 5: Validate DFL Models

Run validation on test data:

**Global Linear variant:**
```bash
python DFL/scripts/run_validation_gl.py
```

**Piecewise variant:**
```bash
python DFL/scripts/run_validation_pw.py
```

**Piecewise No-Recursion (PW-no-Rec) variant:**
```bash
python DFL/scripts/run_validation_pw_norec.py
```

These generate validation metrics and compare DFL performance against MIQP baselines.

### Step 5C: Fixed-Weight Baseline (Optional)

Run the fixed-weight baseline to validate the impact of the neural network weight predictor:

```bash
python DFL/scripts/run_ablation_study.py
```

This baseline approach uses fixed weights (w_p=0.1, w_q=0.01, w_h=0.05) with 7 recursive linearization iterations (same as GL and PW variants) instead of learned weights from the neural network. This allows measurement of how much improvement the neural network weight predictor provides.

### Step 6: Aggregate Validation Results

After running all 4 validation scripts, aggregate the results into a comprehensive master file:

```bash
python results/aggregate_validation_results.py
```

This script collects validation results from all 4 variants:
- **DFL-GL-RS**: GL-based (7 iterations, LSTM)
- **DFL-PW-RS**: PW-based (7 iterations, LSTM)
- **DFL-PW-no-Rec**: PW no-recursion (1 iteration, LSTM)
- **DFL-PW-no-NN**: PW no-neural-network (7 iterations, fixed weights)

Output: `DFL/outputs/validation_results/comprehensive/master_validation_benchmarks.csv`

For detailed configuration information on all 4 workflow variants, see [WORKFLOW_CONFIGURATION.md](WORKFLOW_CONFIGURATION.md).

### Step 6: Generate Tables and Visualizations

Generate publication-quality tables and plots comparing all methods:

**Generate comprehensive comparison tables:**
```bash
python results/print_tables.py
```
Outputs:
- `results/tables/comprehensive_comparison.tex` - LaTeX table for papers
- `results/tables/comprehensive_comparison.csv` - CSV summary for reference

**Generate publication-quality visualizations:**
```bash
python results/visualization.py
```
Outputs to `results/figures/`:
- `profit_density_main_contribution.{pdf,png}` - Profit distribution comparisons (GL vs PW)
- `noise_robustness_dfl_vs_miqp.{pdf,png}` - DFL performance vs MIQP across noise levels
- `noise_robustness_ablation_study.{pdf,png}` - Ablation study robustness analysis
- `profit_vs_penalties_ablation.{pdf,png}` - Profit-penalty trade-off visualizations

### Full Workflow (Bash Script)

To run the entire pipeline automatically:

```bash
#!/bin/bash
set -e  # Exit on first error

echo "=== Cleanup Previous Outputs ==="
rm -rf DFL/outputs/noisy_data \
  DFL/outputs/trained_models \
  DFL/outputs/validation_results \
  results/tables \
  results/figures

echo "=== Step 1: Preprocessing ==="
python preprocessing.py

echo "=== Step 2: MIQP Baselines ==="
python MIQP/MIQP_linear/MIQP_global_linear.py
python MIQP/MIQP_piecewise/MIQP_piecewise.py

echo "=== Step 3: Generate Noisy Data ==="
python DFL/scripts/generate_noisy_data.py --variant GL --noise-levels "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8" --random-samples
python DFL/scripts/generate_noisy_data.py --variant PW --noise-levels "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8" --random-samples

echo "=== Step 4: Train DFL Models ==="
python DFL/scripts/run_pretraining_gl.py
python DFL/scripts/run_pretraining_pw.py
python DFL/scripts/run_pretraining_pw_norec.py

echo "=== Step 5: Validate DFL Models ==="
python DFL/scripts/run_validation_gl.py
python DFL/scripts/run_validation_pw.py
python DFL/scripts/run_validation_pw_norec.py

echo "=== Step 5C: Fixed-Weight Baseline ==="
python DFL/scripts/run_ablation_study.py

echo "=== Step 6: Aggregate Results ==="
python results/aggregate_validation_results.py

echo "=== Step 7: Generate Tables ==="
python results/print_tables.py

echo "=== Step 8: Generate Visualizations ==="
python results/visualization.py

echo "=== Pipeline Complete ==="
```

Save as `run_full_pipeline.sh`, then execute:
```bash
bash run_full_pipeline.sh
```

## Repository Structure

### Data
The `Data/` folder contains day-ahead electricity price data and historical operational datasets. Unit Performance Curve (UPC) data is located in `./Data/UPCs/`, which includes visualization Python scripts and Origin files for UPC analysis.

### Linearization Error Analysis
The `linearization_error/` directory contains accuracy assessments for different MIQP approximation methods. These analyses evaluate the precision of each approximation technique for both UPC relationships and volume-head dynamics using preliminary experimental results.

### MIQP Implementations
The `MIQP/` folder includes three distinct MIQP approximation approaches:
- Global linearization (`MIQP/MIQP_linear/MIQP_global_linear.py`)
- Neural network-informed optimization (`MIQP/MIQP_nn/MIQP_nn.py`)
- Piecewise linearization with SOS2 constraints (`MIQP/MIQP_piecewise/MIQP_piecewise.py`)

Each method has corresponding benchmark results and outputs in their respective subfolders. Note that the December 12, 2024 dataset represents an extreme price event and will be excluded from comparative analyses.

### Decision-Focused Learning
DFL implementations live in multiple places:

- **Refactored framework (WIP)**: `DFL/` (see `DFL/README.md` and `DFL/scripts/` for example entry points).
- **Legacy (stable) experiments**:
  - `DFL_GL-based/`: Global Linear training-data variant (`DFL_pretraining.py`, `DFL_validation.py`)
  - `DFL_PW-based/`: Piecewise training-data variant (`DFL_pretraining.py`, `DFL_validation.py`)
  - `DFL_no-NN/`: ablation without the neural network component (`NN_ablation.py`, `DFL_pretraining.py`)
