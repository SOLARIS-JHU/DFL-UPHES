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
