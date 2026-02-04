# DFL Full Execution Scripts

This folder contains the full execution scripts that reproduce results from `DFL_GL-based/`, `DFL_PW-based/`, and `DFL_no-NN/` using the refactored DFL framework.

## Available Scripts

### Data Generation (Run First - Both Variants)

0. **`generate_noisy_data.py`** - Generate noisy training datasets
   - **GL variant**: `python DFL/scripts/generate_noisy_data.py --variant GL --random-samples`
   - **PW variant**: `python DFL/scripts/generate_noisy_data.py --variant PW --random-samples`
   - Noise levels: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%
   - Plus random samples dataset
   - GL data saved to: `DFL/outputs/noisy_data/MIQP_linear_results_*.csv`
   - PW data saved to: `DFL/outputs/noisy_data/MIQP_piecewise_results_*.csv`
   - **Note**: Both variants save to the same directory with different prefixes (no overwriting)

### GL (Global Linear) Variant Training & Validation

1. **`run_pretraining_gl.py`** - Full pretraining for GL variant
   - Data: `MIQP_linear_results_*` (from data generation step)
   - Noise levels: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%
   - Plus random samples dataset
   - Architecture: LSTM, 3 layers
   - Max iterations: 7
   - Parallel: 20 workers

2. **`run_validation_gl.py`** - Validation for GL variant
   - Validates all trained GL models on 2024 price scenarios
   - Tests all noise levels and configurations

### PW (Piecewise) Variant Training & Validation

3. **`run_pretraining_pw.py`** - Full pretraining for PW variant
   - Data: `MIQP_piecewise_results_*` (from data generation step)
   - Same configuration as GL
   - Uses piecewise approximation

4. **`run_validation_pw.py`** - Validation for PW variant
   - Validates all trained PW models

### Ablation Study (No Neural Network)

5. **`run_ablation_study.py`** - Baseline without NN
   - Uses fixed weights instead of learned weights
   - Tests impact of neural network component

## Usage

**All commands run from repository root**

### Complete Pipeline (Recommended)

```bash
# 1. Generate data for both variants (all from repo root)
python DFL/scripts/generate_noisy_data.py --variant GL --random-samples
python DFL/scripts/generate_noisy_data.py --variant PW --random-samples

# 2. Train models for both variants
python DFL/scripts/run_pretraining_gl.py
python DFL/scripts/run_pretraining_pw.py

# 3. Validate both variants
python DFL/scripts/run_validation_gl.py
python DFL/scripts/run_validation_pw.py

# 4. Optional: Run ablation study
python DFL/scripts/run_ablation_study.py
```

### Individual Steps

```bash
# GL variant only
python DFL/scripts/generate_noisy_data.py --variant GL --random-samples
python DFL/scripts/run_pretraining_gl.py
python DFL/scripts/run_validation_gl.py

# PW variant only
python DFL/scripts/generate_noisy_data.py --variant PW --random-samples
python DFL/scripts/run_pretraining_pw.py
python DFL/scripts/run_validation_pw.py
```

## Algorithm Fidelity

The refactored scripts use **EXACTLY the same algorithm and parameters** as the original implementations:

### Solver Configuration (ECOS)
- `solve_method`: "ECOS"
- `max_iters`: 200,000
- `reltol`: 1e-5
- `abstol`: 1e-5
- `feastol`: 1e-5
- `verbose`: True

### Weight Initialization
- `init_w_p`: 0.05
- `init_w_q`: 0.05
- `init_w_h`: 0.05

### Weight Bounds
- `w_p_min`: 0.01, `w_p_max`: 10.0
- `w_q_min`: 0.01, `w_q_max`: 5.0
- `w_h_min`: 0.01, `w_h_max`: 5.0

### Network Architecture
- Architecture: LSTM
- Hidden size: 128
- Layers: 3
- Dropout: 0.2

### Training Parameters
- Learning rate: 0.001
- Epochs: 100
- Patience: 20 (early stopping)
- Penalty growth rate: 1.5

### System Parameters
- Time horizon: 24 hours
- Sampling rate: 50
- δ_p: 0.5
- δ_h: 1.0
- δ_q: 0.5
- Operational cost: 0.4

## Configuration: Max Iterations

The refactored code uses optimized iteration counts selected through validation:
- GL and PW variants: max_iterations = 7 (optimal for neural network-based DFL)
- Ablation study (no-NN): max_iterations = 1 (no recursive refinement)

All other algorithm details, parameters, and processing steps are **identical** to the original implementations to ensure reproducible results.

## Output Structure

Results are saved to:
- Trained models: `./trained_models/{data_source}/{config_name}/{date}/`
- Validation results: `./validation_results/{data_source}/{config_name}/`
- Comprehensive results: `./validation_results/comprehensive/`

## Notes

- Scripts use joblib for parallel processing (20 workers by default)
- All dates from the original datasets are processed
- Models are saved with best validation performance
- Comprehensive benchmarking CSV files are generated
- Compatible with existing data file structure
