# Repository Graphs for DFL-for-UPHES

## 1. Pipeline Workflow

```mermaid
flowchart TD
    Start([Start]) --> Prep[preprocessing.py<br/>Generate preprocess.pkl]

    Prep --> MIQP{MIQP Baselines}

    MIQP --> GL[Global Linear MIQP<br/>MIQP/MIQP_linear/]
    MIQP --> PW[Piecewise MIQP<br/>MIQP/MIQP_piecewise/]

    GL --> GLOut[MILP_global_linear_results.csv]
    PW --> PWOut[MIQP_piecewise_results.csv]

    GLOut --> Noise[Generate Noisy Data<br/>DFL/scripts/generate_noisy_data.py]
    PWOut --> Noise

    Noise --> NoiseGL[GL: 0-80% noise levels<br/>DFL/outputs/noisy_data/]
    Noise --> NoisePW[PW: 0-80% noise levels<br/>DFL/outputs/noisy_data/]

    NoiseGL --> TrainGL[Train DFL-GL<br/>run_pretraining_gl.py]
    NoisePW --> TrainPW[Train DFL-PW<br/>run_pretraining_pw.py]

    TrainGL --> ModelGL[GL Models<br/>DFL/outputs/trained_models/]
    TrainPW --> ModelPW[PW Models<br/>DFL/outputs/trained_models/]

    ModelGL --> ValGL[Validate GL<br/>run_validation_gl.py]
    ModelPW --> ValPW[Validate PW<br/>run_validation_pw.py]

    ValGL --> Results[Validation Results<br/>DFL/outputs/validation_results/]
    ValPW --> Results

    ModelGL --> Ablation[Ablation Study<br/>run_ablation_study.py]
    ModelPW --> Ablation
    Ablation --> Results

    Results --> Analysis{Results Analysis}

    Analysis --> Tables[Generate Tables<br/>results/print_tables.py]
    Analysis --> Viz[Generate Visualizations<br/>results/visualization.py]

    Tables --> Final[results/tables/<br/>LaTeX & CSV outputs]
    Viz --> Final2[results/figures/<br/>PDF & PNG plots]

    Final --> End([End])
    Final2 --> End

    style Start fill:#90EE90
    style End fill:#FFB6C1
    style MIQP fill:#87CEEB
    style Analysis fill:#87CEEB
    style Noise fill:#DDA0DD
    style Results fill:#F0E68C
```

## 2. Directory Structure

```mermaid
graph TB
    Root[DFL-for-UPHES/] --> Data[📁 Data/]
    Root --> DFL[📁 DFL/]
    Root --> MIQP[📁 MIQP/]
    Root --> Legacy[📁 Legacy DFL/]
    Root --> Results[📁 results/]
    Root --> Lib[📁 Library/]
    Root --> LinErr[📁 linearization_error/]
    Root --> Files[📄 Root Files]

    Data --> UPCs[📁 UPCs/<br/>Unit Performance Curves]
    Data --> PriceData[📄 price_data_2024.csv]

    DFL --> Config[📁 config/<br/>GL/PW/Ablation configs]
    DFL --> Core[📁 core/<br/>Models, Layers, Pipeline]
    DFL --> DFLData[📁 data/<br/>Loaders & Noise]
    DFL --> Training[📁 training/<br/>Pretraining & Trainer]
    DFL --> Validation[📁 validation/<br/>Validator]
    DFL --> Scripts[📁 scripts/<br/>CLI entry points]
    DFL --> Utils[📁 utils/<br/>Helpers]
    DFL --> Outputs[📁 outputs/<br/>Generated data & results]

    Outputs --> NoisyData[📁 noisy_data/]
    Outputs --> TrainedModels[📁 trained_models/]
    Outputs --> ValResults[📁 validation_results/]

    MIQP --> MIQPLinear[📁 MIQP_linear/<br/>Global linearization]
    MIQP --> MIQPPW[📁 MIQP_piecewise/<br/>SOS2 constraints]

    Legacy --> DFLGL[📁 DFL_GL-based/<br/>Stable GL experiments]
    Legacy --> DFLPW[📁 DFL_PW-based/<br/>Stable PW experiments]
    Legacy --> DFLNoNN[📁 DFL_no-NN/<br/>Ablation baseline]

    Results --> Tables[📁 tables/<br/>LaTeX & CSV]
    Results --> Figures[📁 figures/<br/>PDF & PNG]
    Results --> Scripts2[📄 print_tables.py<br/>📄 visualization.py]

    LinErr --> PrelimResults[📁 preliminary_results/<br/>Accuracy assessments]

    Files --> Preprocess[📄 preprocessing.py<br/>📄 preprocess.pkl]
    Files --> Requirements[📄 requirements.txt]
    Files --> README[📄 README.md]

    style Root fill:#FFE4B5
    style DFL fill:#87CEEB
    style MIQP fill:#98FB98
    style Legacy fill:#DDA0DD
    style Results fill:#FFB6C1
    style Outputs fill:#F0E68C
```

## 3. Component Architecture

```mermaid
flowchart LR
    subgraph Input["🔵 Input Data"]
        Price[Price Data<br/>Data/price_data_2024.csv]
        UPC[UPC Data<br/>Data/UPCs/]
        Preproc[Preprocessed<br/>preprocess.pkl]
    end

    subgraph MIQP["🟢 MIQP Baselines"]
        GL[Global Linear<br/>MIQP/MIQP_linear/]
        PW[Piecewise SOS2<br/>MIQP/MIQP_piecewise/]
    end

    subgraph DFLCore["🟣 DFL Framework"]
        direction TB
        LSTM[Neural Penalty<br/>Predictor<br/>core/models.py]
        Local[Local Linearization<br/>Layer<br/>core/layers.py]
        Solver[Differentiable QP<br/>Solver<br/>core/layers.py]
        Sim[Physical Simulator<br/>core/layers.py]
        Pipeline[Recursive Pipeline<br/>core/pipeline.py]

        LSTM --> Local
        Local --> Solver
        Solver --> Sim
        Sim -.Feedback.-> Local
        Pipeline -.Orchestrates.-> LSTM
    end

    subgraph Training["🟡 Training"]
        NoiseGen[Noise Injection<br/>data/noise.py]
        Trainer[End-to-End Trainer<br/>training/trainer.py]
        Models[Trained Models<br/>outputs/trained_models/]
    end

    subgraph Eval["🔴 Evaluation"]
        Validator[Validator<br/>validation/validator.py]
        Benchmarks[Benchmarks<br/>outputs/validation_results/]
    end

    subgraph Output["🟠 Results"]
        Tables[LaTeX Tables<br/>results/tables/]
        Viz[Visualizations<br/>results/figures/]
    end

    Price --> MIQP
    UPC --> Preproc
    Preproc --> MIQP

    MIQP --> NoiseGen
    NoiseGen --> Trainer
    Trainer --> DFLCore
    DFLCore --> Models

    Models --> Validator
    Price --> Validator
    Validator --> Benchmarks

    Benchmarks --> Tables
    Benchmarks --> Viz

    style Input fill:#E6F3FF
    style MIQP fill:#E6FFE6
    style DFLCore fill:#F3E6FF
    style Training fill:#FFFACD
    style Eval fill:#FFE6E6
    style Output fill:#FFE6CC
```

## 4. Simplified Repository Tree

```
DFL-for-UPHES/
├── 📊 Data/
│   ├── UPCs/                    # Unit Performance Curves
│   └── price_data_2024.csv      # Day-ahead prices
│
├── 🧠 DFL/                       # Refactored DFL Framework
│   ├── config/                  # GL/PW/Ablation configurations
│   ├── core/                    # Models, layers, pipeline
│   ├── data/                    # Loaders & noise injection
│   ├── training/                # Pretraining & trainer
│   ├── validation/              # Validator
│   ├── utils/                   # Helpers
│   ├── scripts/                 # CLI entry points
│   │   ├── generate_noisy_data.py
│   │   ├── run_pretraining_gl.py
│   │   ├── run_pretraining_pw.py
│   │   ├── run_validation_gl.py
│   │   ├── run_validation_pw.py
│   │   └── run_ablation_study.py
│   └── outputs/                 # All generated outputs
│       ├── noisy_data/          # Training data with noise
│       ├── trained_models/      # Neural network checkpoints
│       └── validation_results/  # Benchmarks & metrics
│
├── 🔢 MIQP/                      # Mixed-Integer QP Baselines
│   ├── MIQP_linear/             # Global linearization
│   └── MIQP_piecewise/          # SOS2 piecewise
│
├── 📦 Legacy DFL/                # Stable legacy experiments
│   ├── DFL_GL-based/            # GL training-data variant
│   ├── DFL_PW-based/            # PW training-data variant
│   └── DFL_no-NN/               # Ablation without NN
│
├── 📈 results/                   # Publication outputs
│   ├── tables/                  # LaTeX & CSV tables
│   ├── figures/                 # PDF & PNG plots
│   ├── print_tables.py          # Generate comparison tables
│   └── visualization.py         # Generate plots
│
├── 🔬 linearization_error/       # Approximation accuracy
├── 📚 Library/                   # Portfolio & system config
├── 📄 preprocessing.py           # Generate preprocess.pkl
├── 📄 preprocess.pkl             # Preprocessed UPC data
└── 📄 requirements.txt           # Python dependencies
```
