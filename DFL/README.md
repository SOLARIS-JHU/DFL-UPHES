# DFL4UPHES - Decision-Focused Learning for Underground Pumped Hydro Energy Storage

This repository provides a modular, end-to-end implementation of the decision-focused learning (DFL) framework for Underground Pumped Hydro Energy Storage (UPHES) day-ahead scheduling presented in our paper: **"Accelerating Underground Pumped Hydro Energy Storage Scheduling with Decision-Focused Learning"**.

## Overview

The DFL framework addresses the computational-accuracy trade-off in UPHES scheduling by employing neural networks to predict penalty weights that guide recursive linearization, transforming the intractable Mixed-Integer Nonlinear Programming (MINLP) problem into a sequence of convex quadratic programs trained end-to-end via differentiable optimization layers. The framework supports three variants corresponding to different baseline approximation methods:

- **Global Linear (GL)**: Uses global linear approximation of Unit Performance Curves (UPCs) as initialization
- **Piecewise (PW)**: Uses piecewise bilinear SOS2 approximation for high-fidelity initialization
- **Ablation (No-NN)**: Baseline recursive linearization with fixed penalty weights (no neural network)

## Methodology

The DFL framework (detailed in Section IV of the paper) integrates four differentiable components in an end-to-end pipeline:

![DFL Pipeline](../figs/DFL.pdf)

### 1. Neural Penalty Predictor (Section IV-A)

Predicts time-varying regularization weights that establish adaptive trust regions for local linearization validity. The LSTM-based network processes day-ahead prices and initial noisy trajectories to output penalty weights $\mathbf{w}_p$, $\mathbf{w}_q$, $\mathbf{w}_h$ for power, flow, and head constraints.

**Implementation:** `core/models.py` - `BoundedLogWeightPredictor`

### 2. Local Linearization Layer (Section IV-B)

Constructs first-order Taylor approximations of nonlinear UPC mappings and volume-head relationships around current operating points, converting polynomial constraints into affine form.

**Implementation:** `core/layers.py` - `TaylorRegressionLayer`

### 3. Differentiable Convex Optimizer (Section IV-C)

Solves convex quadratic programs with penalty-guided regularization using CVXPYLayers. Mode-locking eliminates integer variables while preserving discrete operational structure from initialization.

**Implementation:** `core/layers.py` - `OptiLayer`

### 4. Physical Simulator (Section IV-D)

Validates schedules under true nonlinear dynamics, clamping power to head-dependent feasibility bounds. Computes ex-post profit including system imbalance and volume violation penalties.

**Implementation:** `core/layers.py` - `SimulationLayer`

### Complete Pipeline

The `RecursiveLinearizationPipeline` orchestrates these components through $K$ iterations with exponentially growing penalty weights ($\gamma^k \mathbf{w}^{(0)}$), progressively refining solutions from noisy initializations.

**Implementation:** `core/pipeline.py` - `RecursiveLinearizationPipeline`, `BaselineRecursiveLinearization`

## Repository Structure

```
DFL/
├── config/         # Configuration classes for GL/PW/Ablation variants
├── core/           # Core DFL components
│   ├── parameters.py      # HydroParameters system specification
│   ├── layers.py          # TaylorRegressionLayer, OptiLayer, SimulationLayer
│   ├── models.py          # BoundedLogWeightPredictor (LSTM/RNN/FC)
│   └── pipeline.py        # RecursiveLinearizationPipeline
├── data/           # Data loading and noise injection
│   ├── loaders.py         # Portfolio/price/MIQP data loading
│   └── noise.py           # Training data perturbation (10%-80%, random)
├── training/       # End-to-end training procedures
│   ├── pretraining.py     # Single noise level training with multiprocessing
│   └── trainer.py         # Training loop with early stopping
├── validation/     # Model evaluation and benchmarking
│   └── validator.py       # Comprehensive validation on new price scenarios
├── utils/          # Device setup, data helpers
└── scripts/        # Usage examples
```

## Variants

The framework supports three configurations corresponding to different baseline approximation methods described in Section III (Preliminaries):

### 1. Global Linear (GL) - Section III-A

Uses global linear approximation of UPCs (Eq. 10-12) for computational efficiency. Single affine functions fitted over entire operational domain provide fast MIQP initialization, trading accuracy for speed.

```python
from DFL.config.gl_config import GLConfig
config = GLConfig()
# Initializes with global linearization baseline
# Fastest training, suitable for real-time applications
```

### 2. Piecewise (PW) - Section III-B

Uses piecewise bilinear SOS2 approximation (Eq. 13-15) for high-fidelity initialization. Discretized grid with bilinear interpolation achieves MAPE < 0.21% on UPC mappings.

```python
from DFL.config.pw_config import PWConfig
config = PWConfig()
# Initializes with piecewise SOS2 baseline
# Highest accuracy, best for profit maximization
```

### 3. Ablation (No-NN) - Section V-C

Recursive linearization with fixed heuristic penalty weights (no neural network). Tests baseline performance without learned penalty prediction.

```python
from DFL.config.ablation_config import AblationConfig
config = AblationConfig()
# config.use_neural_network == False
# Fixed weights: w_p=0.6, w_q=0.02, w_h=0.1
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

### 1. Pretraining (Section IV-E)

The training procedure follows Algorithm 1 in the paper. Models are trained offline using noisy MIQP solutions generated by perturbing baseline results with 10%-80% noise or random sampling.

**Multiprocessing Acceleration:** The pretraining implementation uses Python's multiprocessing to parallelize training across different noise levels and configurations, significantly accelerating the offline training phase without affecting inference time during validation.

```python
from DFL.config.gl_config import GLConfig
from DFL.training.pretraining import pretraining_single_noise_level
from DFL.utils.helpers import setup_device, load_portfolio_data, load_preprocessed_data
from DFL.core.parameters import HydroParameters

# Setup
device = setup_device()
portfolio = load_portfolio_data()
preprocess_data = load_preprocessed_data()

# Create configuration (GL, PW, or Ablation)
config = GLConfig()

# Initialize UPHES parameters (Section V-A)
params = HydroParameters(
    portfolio=portfolio,
    preprocess_data=preprocess_data,
    device=device
    # See scripts/ for complete initialization
)

# Train on noisy MIQP solutions
# Noise levels: 0.1, 0.2, ..., 0.8, or random_samples=True
pretraining_single_noise_level(
    config=config,
    params=params,
    device=device,
    noise_level=0.1  # 10% perturbation
)
```

**Loss Function (Eq. 21):** Training maximizes ex-post profit $\Pi$ computed by the simulation layer, incorporating system imbalance and volume violation penalties.

### 2. Validation (Section V)

Validate trained models on 19 representative Belgian electricity market scenarios (k-medoids clustered from 2024 Elia day-ahead prices).

```python
from DFL.config.pw_config import PWConfig
from DFL.validation.validator import comprehensive_validation

# Setup (same as pretraining)
config = PWConfig()
params = HydroParameters(...)  # Initialize as above

# Evaluate on new price scenarios
comprehensive_validation(
    config=config,
    params=params,
    device=device,
    new_price_file="../Data/price_data_2024.csv"
)
```

**Metrics:** Ex-post profit, system imbalance (SI), volume violations (Vol), and computation time.

### 3. Ablation Study (Section V-C)

Evaluate contribution of neural network and recursive linearization components:

```python
from DFL.config.ablation_config import AblationConfig

config = AblationConfig()
# config.use_neural_network == False
# Fixed heuristic weights: w_p=0.6, w_q=0.02, w_h=0.1

# Train baseline without neural penalty predictor
pretraining_single_noise_level(config, params, device, random_samples=True)
```

## Example Scripts

Complete usage examples are provided in `DFL/scripts/` (see scripts directory for executable examples):

- **GL variant training:** Global linear baseline for real-time applications
- **PW variant validation:** Piecewise SOS2 baseline for profit maximization
- **Ablation study:** Fixed penalty weights without neural network

```bash
cd DFL/scripts
python generate_noisy_data.py  # Generate training data with noise injection
```

## Configuration

### Hyperparameters (Section V-A)

All variants inherit from `DFLConfig` with the following settings used in the paper:

```python
# Neural Network Architecture (Section IV-A)
architecture = 'LSTM'  # LSTM for temporal dependencies in price patterns
num_layers = 3
hidden_size = 128
dropout = 0.2

# Recursive Linearization (Algorithm 1)
max_iterations = 3  # K iterations in inner loop
penalty_growth_rate = 1.5  # γ > 1 penalty growth factor

# Training (Section IV-E)
learning_rate = 0.001  # Adam optimizer
num_epochs = 500
patience = 20  # Early stopping
batch_size = 1  # Full batch training

# Penalty Weight Bounds (Eq. 14)
# Log-domain parameterization ensures w > 0
init_w_p = 0.6    # Power penalty initialization
init_w_q = 0.02   # Flow penalty initialization
init_w_h = 0.1    # Head penalty initialization
w_p_min = 0.1     # Minimum power penalty
w_p_max = 3.0     # Maximum power penalty
w_q_min = 0.001   # Minimum flow penalty
w_q_max = 0.2     # Maximum flow penalty
w_h_min = 0.01    # Minimum head penalty
w_h_max = 5.0     # Maximum head penalty
```

### Variant-Specific Configuration

Each variant specifies its baseline approximation method:

```python
class GLConfig(DFLConfig):
    """Global Linear configuration (Section III-A)"""
    def get_miqp_file_path(self):
        return "../MIQP/MIQP_linear/MILP_global_linear_results.csv"

class PWConfig(DFLConfig):
    """Piecewise SOS2 configuration (Section III-B)"""
    def get_miqp_file_path(self):
        return "../MIQP/MIQP_piecewise/results.csv"

class AblationConfig(DFLConfig):
    """No neural network baseline (Section V-C)"""
    use_neural_network = False
```

## Output Structure

### Pretraining Outputs

Trained models are saved with noise level and architecture specification:

```
trained_models/
└── MIQP_linear_results_relative_noise_10pct/  # GL variant, 10% noise
    └── LSTM_3layer_3iter/                      # Architecture_layers_K
        └── YYYY-MM-DD/                         # Training date
            ├── model.pt                        # Trained neural network θ*
            └── training_log.csv                # Loss trajectory
```

### Validation Outputs

Evaluation results include ex-post profit, penalties, and computation time:

```
validation_results/
├── comprehensive/
│   ├── master_validation_benchmarks.csv       # Aggregated metrics
│   └── best_configurations.json               # Optimal hyperparameters
└── MIQP_piecewise_results_random_samples/     # PW variant validation
    └── LSTM_3layer_5iter/
        ├── scheduling_benchmarks.csv          # Per-scenario results
        └── YYYY-MM-DD/
            ├── results.npy                    # Optimized trajectories
            └── simulated_results.npy          # Simulated trajectories
```

## Workflow

### Training Workflow (Algorithm 1)

End-to-end decision-focused learning with ex-post profit optimization:

1. **Data Loading:** Load 19 price scenarios and noisy MIQP initializations $\bar{\mathbf{x}}_i$
2. **Initialize Network:** Random initialization of LSTM parameters $\theta^{(0)}$
3. **For each epoch $e = 1$ to $E$:**
   - **For each sample** $(\boldsymbol{\lambda}_i^{\text{DA}}, \bar{\mathbf{x}}_i)$:
     - **Neural Prediction:** $\mathbf{w}_i^{(0)} \gets \exp(\mathcal{N}_{\theta}([\boldsymbol{\lambda}_i^{\text{DA}}, \bar{\mathbf{x}}_i]))$ (Eq. 14)
     - **For $k = 0$ to $K-1$** (Recursive Linearization):
       - Scale penalties: $\mathbf{w}_i^{(k)} \gets \gamma^k \mathbf{w}_i^{(0)}$
       - Linearize at $\hat{\mathbf{x}}_i^{(k)}$ to obtain $\boldsymbol{\xi}_i^{(k)}$ (Eq. 15-18)
       - Solve QP (Eq. 19) with CVXPYLayers to get $\hat{\mathbf{x}}_i^{(k+1)}$
     - **Simulation:** $\tilde{\mathbf{x}}_i \gets \textsc{Simulate}(\hat{\mathbf{x}}_i^{(K)})$ (Algorithm 2)
     - **Loss:** $\mathcal{L}_i \gets -\Pi_i(\tilde{\mathbf{x}}_i)$ (Eq. 20)
   - **Update:** $\theta^{(e+1)} \gets \theta^{(e)} - \eta \nabla_\theta \mathcal{L}(\theta^{(e)})$ via Adam
4. **Save:** Best model $\theta^*$ based on validation ex-post profit

### Validation Workflow (Section V)

Test trained models on representative market scenarios:

1. **Load Scenarios:** 19 representative Belgian day-ahead price profiles (k-medoids clustering)
2. **Load Model:** Select trained model based on noise level and architecture
3. **For each price scenario:**
   - **Initialization:** Obtain $\bar{\mathbf{x}}$ from baseline MIQP (GL or PW)
   - **Refinement:** Run trained DFL pipeline to get $\hat{\mathbf{x}}^{(K)}$
   - **Simulation:** Validate under true nonlinear dynamics $\tilde{\mathbf{x}}$
   - **Metrics:** Compute ex-post profit $\Pi$, SI penalty, Vol penalty, time
4. **Benchmarking:** Compare against MIQP-GL and MIQP-PW baselines
