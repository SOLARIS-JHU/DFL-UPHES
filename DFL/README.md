# DFL - Deep Learning Framework for Hydropower Economic Scheduling

A refactored, modular framework for training and validating neural network-based optimization methods for pumped-storage hydropower systems.

## Overview

This framework implements recursive linearization with neural network-based penalty weight prediction for hydropower economic scheduling. It supports multiple variants (Global Linear, Piecewise, Ablation) through a configuration-driven design.

## Architecture

```
DFL/
├── config/         # Configuration classes for variants
├── core/           # Core components (parameters, layers, models, pipelines)
├── data/           # Data loading and preprocessing
├── training/       # Training and pretraining logic
├── validation/     # Validation and benchmarking
├── utils/          # Utility functions
└── scripts/        # Example entry scripts
```

### Key Components

- **HydroParameters**: Container for all hydropower system parameters
- **TaylorRegressionLayer**: First-order Taylor approximation for nonlinear functions
- **OptiLayer**: CVXPY-based convex optimization layer
- **SimulationLayer**: Physical simulation with operational constraints
- **BoundedLogWeightPredictor**: LSTM/RNN/FC network for penalty weight prediction
- **RecursiveLinearizationPipeline**: Orchestrates recursive linearization with neural network
- **BaselineRecursiveLinearization**: Ablation version with fixed weights

## Variants

### Global Linear (GL)
Uses global linear approximation MIQP results as training data.
```python
from DFL.config.gl_config import GLConfig
config = GLConfig()
```

### Piecewise (PW)
Uses piecewise linear approximation MIQP results as training data.
```python
from DFL.config.pw_config import PWConfig
config = PWConfig()
```

### Ablation (No-NN)
Tests recursive linearization without neural network (fixed penalty weights).
```python
from DFL.config.ablation_config import AblationConfig
config = AblationConfig()
```

<!-- ## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CVXPY
- cvxpylayers
- pandas, numpy, dill, joblib

### Setup
```bash
# Clone the repository
cd DFL-for-UPHES

# Install dependencies
pip install torch cvxpy cvxpylayers pandas numpy dill joblib
``` -->

## Usage

### 1. Pretraining

Train models on historical data with different noise levels:

```python
from DFL.config.gl_config import GLConfig
from DFL.training.pretraining import pretraining_single_noise_level
from DFL.utils.helpers import setup_device, load_portfolio_data, load_preprocessed_data
from DFL.core.parameters import HydroParameters

# Setup
device = setup_device()
portfolio = load_portfolio_data()
preprocess_data = load_preprocessed_data()

# Create configuration
config = GLConfig()

# Initialize parameters
params = HydroParameters(
    # ... (see example scripts for full initialization)
    device=device
)

# Run pretraining for 10% noise level
pretraining_single_noise_level(
    config=config,
    params=params,
    device=device,
    noise_level=0.1
)
```

### 2. Validation

Validate trained models on new price scenarios:

```python
from DFL.config.pw_config import PWConfig
from DFL.validation.validator import comprehensive_validation

# Setup (same as pretraining)
config = PWConfig()
params = HydroParameters(...)  # Initialize as above

# Run validation
comprehensive_validation(
    config=config,
    params=params,
    device=device,
    new_price_file="../Data/price_data_2024.csv"
)
```

### 3. Ablation Study

Test without neural network component:

```python
from DFL.config.ablation_config import AblationConfig

config = AblationConfig()
# config.use_neural_network == False
# Uses fixed weights: w_p=0.6, w_q=0.02, w_h=0.1

# Run training (same as above)
pretraining_single_noise_level(config, params, device, random_samples=True)
```

## Example Scripts

Three complete example scripts are provided in `DFL/scripts/`:

1. **example_pretraining_gl.py** - GL variant pretraining
2. **example_validation_pw.py** - PW variant validation
3. **example_ablation.py** - Ablation study

Run them from the scripts directory:
```bash
cd DFL/scripts
python example_pretraining_gl.py
```

## Configuration

### Base Configuration Parameters

All variants inherit from `DFLConfig` with these common parameters:

```python
# Neural Network Settings
architecture = 'LSTM'  # 'LSTM', 'RNN', 'FC'
num_layers = 3
hidden_size = 128
dropout = 0.2

# Training Settings
max_iterations = 3  # Recursive linearization iterations
penalty_growth_rate = 1.5
learning_rate = 0.001
num_epochs = 100
patience = 20  # Early stopping

# Weight Bounds
init_w_p = 0.05
w_p_min = 0.01
w_p_max = 10.0
# ... (similar for w_q, w_h)
```

### Variant-Specific Settings

Override methods in subclasses:
```python
class GLConfig(DFLConfig):
    def get_miqp_file_path(self):
        return "../MIQP/MIQP_linear/MILP_global_linear_results.csv"

    def get_data_file_pattern(self, noise_level=None, random_samples=False):
        # Returns appropriate filename for GL variant
        ...
```

## Output Structure

### Pretraining Output
```
trained_models/
└── MIQP_linear_results_relative_noise_10pct/
    └── LSTM_3layer_3iter/
        └── 2024-01-01/
            └── model.pt
```

### Validation Output
```
validation_results/
├── comprehensive/
│   ├── master_validation_benchmarks.csv
│   └── best_configurations.json
└── MIQP_piecewise_results_random_samples/
    └── LSTM_3layer_5iter/
        ├── scheduling_benchmarks.csv
        └── 2024-01-01/
            └── results.npy
```


## Workflow

### Training Workflow
1. Load historical data (power, head, flow, price)
2. Initialize weight prediction network (or fixed weights)
3. For each epoch:
   - Predict penalty weights
   - Run recursive linearization (multiple iterations)
   - Optimize with CVXPY
   - Simulate operation
   - Calculate profit loss
   - Backpropagate and update weights
4. Save best model

### Validation Workflow
1. Load new price scenarios
2. Load historical training data
3. For each new price scenario:
   - Find closest historical price profile
   - Load corresponding pretrained model
   - Run recursive linearization with new prices
   - Simulate and calculate ex-post profit
4. Generate benchmark reports
<!-- 
## Citation

If you use this framework, please cite:

```
[Your paper citation here]
```

## License

[Your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your contact info]. -->
