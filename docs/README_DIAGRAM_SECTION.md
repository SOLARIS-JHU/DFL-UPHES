# Repository Diagram Section for README

Add this section to your main README.md file. The Mermaid diagrams will render automatically on GitHub!

---

## Repository Architecture

### Pipeline Workflow

The complete DFL-for-UPHES pipeline follows this workflow:

```mermaid
flowchart TD
    subgraph Canvas1[" "]
        direction TD
        A[preprocessing.py] --> B{MIQP Baselines}
        B --> C[Global Linear<br/>MIQP/MIQP_linear/]
        B --> D[Piecewise<br/>MIQP/MIQP_piecewise/]

        C --> E[Generate Noisy Data<br/>DFL/scripts/generate_noisy_data.py]
        D --> E

        E --> F[Train DFL-GL<br/>run_pretraining_gl.py]
        E --> G[Train DFL-PW<br/>run_pretraining_pw.py]

        F --> H[Validate GL<br/>run_validation_gl.py]
        G --> I[Validate PW<br/>run_validation_pw.py]

        F --> J[Ablation Study<br/>run_ablation_study.py]
        G --> J

        H --> K{Results Analysis}
        I --> K
        J --> K

        K --> L[Generate Tables<br/>results/print_tables.py]
        K --> M[Generate Visualizations<br/>results/visualization.py]
    end

    style Canvas1 fill:#F7F7F7,stroke:#DDDDDD,stroke-width:1px
    style A fill:#E6F3FF
    style B fill:#87CEEB
    style E fill:#DDA0DD
    style K fill:#87CEEB
    style L fill:#98FB98
    style M fill:#98FB98
```

### Directory Structure

```
DFL-for-UPHES/
│
├── 📊 Data/                      # Input data and UPC information
│   ├── UPCs/                     # Unit Performance Curves
│   └── price_data_2024.csv       # Day-ahead electricity prices
│
├── 🧠 DFL/                        # Main DFL Framework (Refactored)
│   ├── config/                   # Configuration classes (GL/PW/Ablation)
│   ├── core/                     # Core DFL components
│   │   ├── models.py             # Neural penalty predictor (LSTM)
│   │   ├── layers.py             # Linearization, solver, simulator
│   │   └── pipeline.py           # Recursive refinement orchestrator
│   ├── data/                     # Data loaders and noise injection
│   ├── training/                 # End-to-end training procedures
│   ├── validation/               # Model evaluation
│   ├── utils/                    # Helper utilities
│   ├── scripts/                  # CLI entry points
│   │   ├── generate_noisy_data.py
│   │   ├── run_pretraining_gl.py
│   │   ├── run_pretraining_pw.py
│   │   ├── run_validation_gl.py
│   │   ├── run_validation_pw.py
│   │   └── run_ablation_study.py
│   └── outputs/                  # All generated outputs
│       ├── noisy_data/           # Training data (0-80% noise)
│       ├── trained_models/       # Neural network checkpoints
│       └── validation_results/   # Performance benchmarks
│
├── 🔢 MIQP/                       # MIQP Baseline Methods
│   ├── MIQP_linear/              # Global linearization baseline
│   └── MIQP_piecewise/           # Piecewise SOS2 baseline
│
├── 📦 Legacy/                     # Stable legacy implementations
│   ├── DFL_GL-based/             # GL training-data variant
│   ├── DFL_PW-based/             # PW training-data variant
│   └── DFL_no-NN/                # Ablation study baseline
│
├── 📈 results/                    # Publication outputs
│   ├── tables/                   # LaTeX & CSV comparison tables
│   ├── figures/                  # PDF & PNG visualizations
│   ├── print_tables.py           # Table generation script
│   └── visualization.py          # Visualization script
│
├── 🔬 linearization_error/        # Approximation accuracy analysis
├── 📚 Library/                    # System configuration files
├── 📄 preprocessing.py            # Preprocessing script
└── 📄 preprocess.pkl              # Preprocessed UPC data
```

### Component Architecture

The DFL framework consists of four differentiable components trained end-to-end:

```mermaid
flowchart LR
    subgraph Canvas2[" "]
        direction LR
        subgraph DFL["DFL Framework"]
            direction TB
            A[Neural Penalty<br/>Predictor<br/>LSTM] --> B[Local<br/>Linearization<br/>Layer]
            B --> C[Differentiable<br/>QP Solver<br/>CVXPYLayers]
            C --> D[Physical<br/>Simulator]
            D -.Recursive<br/>Feedback.-> B
        end

        Input[Price Data] --> DFL
        MIQP[MIQP Results] --> DFL
        DFL --> Output[Optimal Schedule]
    end

    style Canvas2 fill:#F7F7F7,stroke:#DDDDDD,stroke-width:1px
    style A fill:#DDA0DD
    style B fill:#87CEEB
    style C fill:#98FB98
    style D fill:#F0E68C
    style Input fill:#E6F3FF
    style Output fill:#FFE4B5
```

### Key Features

- **🚀 Fast**: GL-based variant achieves near-real-time scheduling (<5s per day)
- **🎯 Accurate**: PW variant matches MIQP quality with 100x speedup
- **🔄 End-to-End**: Differentiable pipeline for gradient-based optimization
- **📊 Comprehensive**: Extensive validation across noise levels and scenarios
- **🧪 Reproducible**: Modular codebase with clear separation of concerns

<!-- ---

## Alternative Visualizations

For publication-quality diagrams, you can generate PNG/PDF versions:

```bash
# Install graphviz (optional)
pip install graphviz

# Generate diagrams
python docs/generate_repo_diagram.py
```

This creates:
- `docs/workflow_diagram.png` - Complete pipeline workflow
- `docs/directory_structure.png` - Directory tree visualization
- `docs/architecture_diagram.png` - Component architecture
 -->
