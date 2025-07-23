# DFL-for-UPHES

This repository contains implementations of various optimization methodologies for Underground Pumped Hydroelectric Energy Storage scheduling, including Mixed-Integer Quadratic Programming (MIQP) approaches and Decision-Focused Learning (DFL) techniques.

## Prerequisites

**Important**: Before running any scripts, execute `preprocessing.py` in the root directory to update `preprocess.pkl`. This ensures compatibility across different versions of the dill library, as newer versions may not support pickle files created with older syntax.

```bash
python preprocessing.py
```

## Repository Structure

### Data
The `Data/` folder contains day-ahead electricity price data and historical operational datasets. Unit Performance Curve (UPC) data is located in `./Data/UPCs/`, which includes visualization Python scripts and Origin files for UPC analysis.

### Linearization Error Analysis
The `linearization_error/` directory contains accuracy assessments for different MIQP approximation methods. These analyses evaluate the precision of each approximation technique for both UPC relationships and volume-head dynamics using preliminary experimental results.

### MIQP Implementations
The `MIQP/` folder includes three distinct MIQP approximation approaches:
- Global linearization (`MIQP_global_linear.py`)
- Neural network-informed optimization (`MIQP_nn.py`) 
- Piecewise linearization with SOS2 constraints (`MIQP_piecewise.py`)

Each method has corresponding benchmark results and outputs in their respective subfolders. Note that the December 12, 2024 dataset represents an extreme price event and will be excluded from comparative analyses.

### Decision-Focused Learning
DFL-related code and corresponding historical operational datasets are currently being processed and organized.
