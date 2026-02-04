# DFL4UPHES - Decision-Focused Learning for Underground Pumped Hydro Energy Storage

A modular implementation of the decision-focused learning (DFL) framework for Underground Pumped Hydro Energy Storage (UPHES) day-ahead scheduling. This code accompanies our paper: **"Accelerating Underground Pumped Hydro Energy Storage Scheduling with Decision-Focused Learning"**.

## Overview

The DFL framework uses neural networks to predict penalty weights that guide recursive linearization, transforming the intractable Mixed-Integer Nonlinear Programming (MINLP) problem into a sequence of convex quadratic programs trained end-to-end.

**Three Variants:**
- **GL (Global Linear)**: Global linear approximation - fastest for real-time
- **PW (Piecewise)**: Piecewise SOS2 approximation - highest accuracy
- **No-NN (Ablation)**: Fixed penalty weights - baseline without neural network

## Prerequisites

The following must exist in the repository:
- `MIQP/MIQP_linear/MILP_global_linear_results.csv` - GL baseline optimization results
- `MIQP/MIQP_piecewise/MIQP_piecewise_results.csv` - PW baseline optimization results
- `Library/` - Portfolio and system configuration files
- `preprocess.pkl` - Preprocessed UPC functions and coefficients (both GL and PW variants)
- `Data/price_data_2024.csv` - Validation price scenarios (for validation step)

## Quick Start

**All commands run from repository root.**

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Generate training data
#    Creates: DFL/outputs/noisy_data/
python DFL/scripts/generate_noisy_data.py --variant GL --random-samples
python DFL/scripts/generate_noisy_data.py --variant PW --random-samples

# 2. Train models
#    Creates: DFL/outputs/trained_models/
python DFL/scripts/run_pretraining_gl.py
python DFL/scripts/run_pretraining_pw.py

# 3. Validate models
#    Creates: DFL/outputs/validation_results/
python DFL/scripts/run_validation_gl.py
python DFL/scripts/run_validation_pw.py

# 4. Run ablation study (optional)
#    Creates: DFL/outputs/validation_results/ablation_study/
python DFL/scripts/run_ablation_study.py
```

**All outputs are automatically organized in `DFL/outputs/` for easy experiment management.**

## Methodology

The DFL framework integrates four differentiable components (detailed in Section IV of the paper):

1. **Neural Penalty Predictor** (`core/models.py`): LSTM network predicts time-varying penalty weights
2. **Local Linearization Layer** (`core/layers.py`): First-order Taylor approximations of nonlinear constraints
3. **Differentiable Convex Optimizer** (`core/layers.py`): CVXPYLayers for QP solving
4. **Physical Simulator** (`core/layers.py`): Validates schedules under true nonlinear dynamics

**Complete Pipeline:** `core/pipeline.py` orchestrates recursive refinement through K iterations.

## Repository Structure

```
DFL/
├── config/          # Configuration classes for GL/PW/Ablation variants
├── core/            # DFL components (models, layers, pipeline)
├── data/            # Data loading and noise injection
├── training/        # End-to-end training procedures
├── validation/      # Model evaluation and benchmarking
├── utils/           # Device setup, data helpers
└── scripts/         # CLI commands for training and validation
```

## Configuration

### Key Hyperparameters

**Neural Network:**
- Architecture: LSTM, 3 layers, 128 hidden units
- Dropout: 0.2

**Training:**
- Optimizer: Adam, learning rate 0.001
- Epochs: 100, early stopping patience 20
- Recursive linearization: max_iterations = 1-5, penalty growth rate = 1.5

**Penalty Weights:**
- Power (w_p): [0.01, 10.0], init=0.05
- Flow (w_q): [0.01, 5.0], init=0.05
- Head (w_h): [0.01, 5.0], init=0.05

**Solver (ECOS):**
- Max iterations: 200,000
- Tolerances: reltol=1e-5, abstol=1e-5, feastol=1e-5

### Variant Differences

- **GL**: Uses `MIQP_linear_results_*.csv`
- **PW**: Uses `MIQP_piecewise_results_*.csv`
- **No-NN**: Fixed weights (w_p=0.6, w_q=0.02, w_h=0.1)

## Output Structure

All outputs are organized in a centralized `DFL/outputs/` directory with three subdirectories:

```
DFL/outputs/
├── noisy_data/                          # Generated noisy training data
│   ├── MIQP_linear_results_relative_noise_10pct.csv    (GL variant)
│   ├── MIQP_linear_results_relative_noise_20pct.csv
│   ├── ...
│   ├── MIQP_linear_results_relative_noise_80pct.csv
│   ├── MIQP_linear_results_random_samples.csv
│   ├── MIQP_piecewise_results_relative_noise_10pct.csv (PW variant)
│   ├── MIQP_piecewise_results_relative_noise_20pct.csv
│   ├── ...
│   ├── MIQP_piecewise_results_relative_noise_80pct.csv
│   └── MIQP_piecewise_results_random_samples.csv
│
├── trained_models/                      # Trained neural network models
│   ├── MIQP_linear_results_relative_noise_10pct/       (GL variant)
│   │   └── LSTM_3layer_1iter/
│   │       └── YYYY-MM-DD_HHMMSS/
│   │           ├── model.pt
│   │           └── training_log.csv
│   ├── MIQP_piecewise_results_relative_noise_10pct/    (PW variant)
│   │   └── LSTM_3layer_1iter/
│   │       └── YYYY-MM-DD_HHMMSS/
│   │           ├── model.pt
│   │           └── training_log.csv
│   └── ...
│
└── validation_results/                  # Validation and test results
    ├── comprehensive/
    │   ├── master_validation_benchmarks.csv
    │   └── best_configurations.json
    ├── MIQP_linear_results_relative_noise_10pct/       (GL variant)
    │   └── LSTM_3layer_1iter/
    │       └── scheduling_benchmarks.csv
    ├── MIQP_piecewise_results_relative_noise_10pct/    (PW variant)
    │   └── LSTM_3layer_1iter/
    │       └── scheduling_benchmarks.csv
    ├── ablation_study/
    │   ├── MIQP_linear_results_relative_noise_10pct/
    │   │   └── NoNN_1iter/
    │   │       └── scheduling_benchmarks.csv
    │   ├── MIQP_piecewise_results_relative_noise_10pct/
    │   │   └── NoNN_1iter/
    │   │       └── scheduling_benchmarks.csv
    │   └── ...
    └── ...
```

## Understanding Results

**Key Metrics:**
- **Ex-post Profit (€)**: Revenue minus costs and penalties
- **System Imbalance**: Penalty for power deviations
- **Volume Violations**: Penalty for reservoir violations
- **Computation Time (s)**: Wall-clock time

**Validation Benchmarks:**
- View results in `validation_results/comprehensive/master_validation_benchmarks.csv`
- Per-scenario details in subdirectories
- Results should match within solver tolerance

## Advanced Usage

### Data Generation Options

```bash
# Generate both GL and PW variants (separate directories, no overwriting)
python DFL/scripts/generate_noisy_data.py --variant GL --random-samples
python DFL/scripts/generate_noisy_data.py --variant PW --random-samples

# Generate specific noise levels only (instead of 10-80%)
python DFL/scripts/generate_noisy_data.py --variant GL --noise-levels "0.1,0.2,0.3"

# Generate without random samples dataset
python DFL/scripts/generate_noisy_data.py --variant PW

# Data is saved to separate directories:
#   GL:  ./DFL/outputs/noisy_data/MIQP_linear_results_*.csv
#   PW:  ./DFL/outputs/noisy_data/MIQP_piecewise_results_*.csv
```

### Training Options

```bash
# Reduce parallel workers for debugging or memory constraints (default: 20)
python DFL/scripts/run_pretraining_gl.py --n-jobs 4

# Use single worker for sequential processing (best for debugging)
python DFL/scripts/run_pretraining_gl.py --n-jobs 1

# Use all available CPU cores
python DFL/scripts/run_pretraining_gl.py --n-jobs -1
```

### Validation Options

```bash
# Use custom price data file instead of default
python DFL/scripts/run_validation_gl.py --price-file ./custom_prices.csv

# Use custom price data with ablation study
python DFL/scripts/run_ablation_study.py --price-file ./my_prices.csv
```

### Modify Configuration

Edit configuration files:
- `DFL/config/gl_config.py` - GL variant settings
- `DFL/config/pw_config.py` - PW variant settings
- `DFL/config/ablation_config.py` - Ablation settings

Common modifications:
- `max_iterations`: Number of recursive linearization steps
- `num_epochs`, `patience`: Training duration
- `learning_rate`: Optimizer learning rate
- Penalty weight bounds: `w_p_min`, `w_p_max`, etc.

### Parallel Processing

Scripts use joblib with 20 parallel workers by default. To adjust:
- Edit `n_jobs` parameter in training scripts
- Set to -1 for all CPU cores
- Set to 1 for sequential processing (debugging)

## Troubleshooting

**CVXPY Solver Issues:**
- Ensure ECOS is installed: `pip install ecos`
- Check solver tolerances in config files
- Some scenarios may require tighter tolerances

**Memory Issues:**
- Reduce parallel workers: `python DFL/scripts/run_pretraining_gl.py --n-jobs 4`
- Process fewer noise levels: `python DFL/scripts/generate_noisy_data.py --variant GL --noise-levels "0.1,0.2"`

**Missing Outputs:**
- All outputs are saved in `DFL/outputs/` directory
- Ensure this directory and subdirectories are created: `mkdir -p DFL/outputs/{noisy_data,trained_models,validation_results}`
- Verify sufficient disk space is available for generated data

**File Not Found Errors:**
- Noisy data should be in: `DFL/outputs/noisy_data/`
- Trained models should be in: `DFL/outputs/trained_models/`
- Validation results saved to: `DFL/outputs/validation_results/`

**Path and Dependencies:**
- All scripts must be run from repository root: `cd /path/to/DFL-for-UPHES`
- Scripts need access to: `./MIQP/`, `./Library/`, and `./preprocess.pkl`
- Outputs directory must be writable: `DFL/outputs/`

## Paper Reference

This implementation accompanies the paper:

**"Accelerating Underground Pumped Hydro Energy Storage Scheduling with Decision-Focused Learning"**

See: `DFL/submission-arXiv.tex`

### Citation

```bibtex
@article{dfl-uphes-2025,
  title={Accelerating Underground Pumped Hydro Energy Storage Scheduling with Decision-Focused Learning},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## License

[License information to be added]

## Contact

For questions or issues, please open an issue on the repository.
