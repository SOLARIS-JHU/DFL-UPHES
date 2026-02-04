"""
Generate publication-quality repository structure diagram for DFL-for-UPHES
Requires: pip install graphviz matplotlib
"""

import os
from graphviz import Digraph


def create_workflow_diagram(output_path="docs/workflow_diagram"):
    """Create the complete pipeline workflow diagram"""
    dot = Digraph(comment='DFL-for-UPHES Workflow', format='png')
    dot.attr(rankdir='TB', size='12,16')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')

    # Preprocessing
    dot.node('start', 'Start', fillcolor='#90EE90', shape='ellipse')
    dot.node('prep', 'preprocessing.py\nGenerate preprocess.pkl', fillcolor='#E6F3FF')

    # MIQP Baselines
    dot.node('miqp', 'MIQP Baselines', fillcolor='#87CEEB', shape='diamond')
    dot.node('gl_miqp', 'Global Linear MIQP\nMIQP/MIQP_linear/', fillcolor='#98FB98')
    dot.node('pw_miqp', 'Piecewise MIQP\nMIQP/MIQP_piecewise/', fillcolor='#98FB98')
    dot.node('gl_out', 'MILP_global_linear\n_results.csv', fillcolor='#F0E68C')
    dot.node('pw_out', 'MIQP_piecewise\n_results.csv', fillcolor='#F0E68C')

    # Noise Generation
    dot.node('noise', 'Generate Noisy Data\ngenerate_noisy_data.py', fillcolor='#DDA0DD')
    dot.node('noise_gl', 'GL: 0-80% noise\nDFL/outputs/noisy_data/', fillcolor='#F0E68C')
    dot.node('noise_pw', 'PW: 0-80% noise\nDFL/outputs/noisy_data/', fillcolor='#F0E68C')

    # Training
    dot.node('train_gl', 'Train DFL-GL\nrun_pretraining_gl.py', fillcolor='#87CEEB')
    dot.node('train_pw', 'Train DFL-PW\nrun_pretraining_pw.py', fillcolor='#87CEEB')
    dot.node('model_gl', 'GL Models\nDFL/outputs/trained_models/', fillcolor='#FFE4B5')
    dot.node('model_pw', 'PW Models\nDFL/outputs/trained_models/', fillcolor='#FFE4B5')

    # Validation
    dot.node('val_gl', 'Validate GL\nrun_validation_gl.py', fillcolor='#FFB6C1')
    dot.node('val_pw', 'Validate PW\nrun_validation_pw.py', fillcolor='#FFB6C1')
    dot.node('ablation', 'Ablation Study\nrun_ablation_study.py', fillcolor='#FFB6C1')
    dot.node('results', 'Validation Results\nDFL/outputs/validation_results/', fillcolor='#F0E68C')

    # Analysis
    dot.node('analysis', 'Results Analysis', fillcolor='#87CEEB', shape='diamond')
    dot.node('tables', 'Generate Tables\nprint_tables.py', fillcolor='#98FB98')
    dot.node('viz', 'Generate Visualizations\nvisualization.py', fillcolor='#98FB98')
    dot.node('final', 'results/tables/\nLaTeX & CSV', fillcolor='#FFE4B5')
    dot.node('final2', 'results/figures/\nPDF & PNG', fillcolor='#FFE4B5')
    dot.node('end', 'End', fillcolor='#FFB6C1', shape='ellipse')

    # Edges
    dot.edge('start', 'prep')
    dot.edge('prep', 'miqp')
    dot.edge('miqp', 'gl_miqp')
    dot.edge('miqp', 'pw_miqp')
    dot.edge('gl_miqp', 'gl_out')
    dot.edge('pw_miqp', 'pw_out')
    dot.edge('gl_out', 'noise')
    dot.edge('pw_out', 'noise')
    dot.edge('noise', 'noise_gl')
    dot.edge('noise', 'noise_pw')
    dot.edge('noise_gl', 'train_gl')
    dot.edge('noise_pw', 'train_pw')
    dot.edge('train_gl', 'model_gl')
    dot.edge('train_pw', 'model_pw')
    dot.edge('model_gl', 'val_gl')
    dot.edge('model_pw', 'val_pw')
    dot.edge('model_gl', 'ablation')
    dot.edge('model_pw', 'ablation')
    dot.edge('val_gl', 'results')
    dot.edge('val_pw', 'results')
    dot.edge('ablation', 'results')
    dot.edge('results', 'analysis')
    dot.edge('analysis', 'tables')
    dot.edge('analysis', 'viz')
    dot.edge('tables', 'final')
    dot.edge('viz', 'final2')
    dot.edge('final', 'end')
    dot.edge('final2', 'end')

    # Render
    dot.render(output_path, cleanup=True)
    print(f"Workflow diagram saved to {output_path}.png")


def create_directory_structure(output_path="docs/directory_structure"):
    """Create the directory structure diagram"""
    dot = Digraph(comment='DFL-for-UPHES Directory Structure', format='png')
    dot.attr(rankdir='LR', size='16,12')
    dot.attr('node', shape='folder', style='filled', fontname='Arial')

    # Root
    dot.node('root', 'DFL-for-UPHES/', fillcolor='#FFE4B5', shape='box3d')

    # Main directories
    dot.node('data', '📊 Data/', fillcolor='#E6F3FF')
    dot.node('dfl', '🧠 DFL/', fillcolor='#87CEEB')
    dot.node('miqp', '🔢 MIQP/', fillcolor='#98FB98')
    dot.node('legacy', '📦 Legacy DFL/', fillcolor='#DDA0DD')
    dot.node('results', '📈 results/', fillcolor='#FFB6C1')
    dot.node('lib', '📚 Library/', fillcolor='#F0E68C')
    dot.node('linerr', '🔬 linearization_error/', fillcolor='#FFE4B5')

    # Data subdirectories
    dot.node('upcs', 'UPCs/', fillcolor='#E6F3FF')
    dot.node('price', 'price_data_2024.csv', shape='note', fillcolor='#F0E68C')

    # DFL subdirectories
    dot.node('config', 'config/', fillcolor='#B0E0E6')
    dot.node('core', 'core/', fillcolor='#B0E0E6')
    dot.node('dfl_data', 'data/', fillcolor='#B0E0E6')
    dot.node('training', 'training/', fillcolor='#B0E0E6')
    dot.node('validation', 'validation/', fillcolor='#B0E0E6')
    dot.node('scripts', 'scripts/', fillcolor='#B0E0E6')
    dot.node('utils', 'utils/', fillcolor='#B0E0E6')
    dot.node('outputs', 'outputs/', fillcolor='#FFFACD')

    # Outputs subdirectories
    dot.node('noisy', 'noisy_data/', fillcolor='#FFFACD')
    dot.node('trained', 'trained_models/', fillcolor='#FFFACD')
    dot.node('val_res', 'validation_results/', fillcolor='#FFFACD')

    # MIQP subdirectories
    dot.node('miqp_lin', 'MIQP_linear/', fillcolor='#C1FFC1')
    dot.node('miqp_pw', 'MIQP_piecewise/', fillcolor='#C1FFC1')

    # Legacy subdirectories
    dot.node('dfl_gl', 'DFL_GL-based/', fillcolor='#E6D5FF')
    dot.node('dfl_pw', 'DFL_PW-based/', fillcolor='#E6D5FF')
    dot.node('dfl_nonn', 'DFL_no-NN/', fillcolor='#E6D5FF')

    # Results subdirectories
    dot.node('tables', 'tables/', fillcolor='#FFD5D5')
    dot.node('figures', 'figures/', fillcolor='#FFD5D5')

    # Edges
    dot.edge('root', 'data')
    dot.edge('root', 'dfl')
    dot.edge('root', 'miqp')
    dot.edge('root', 'legacy')
    dot.edge('root', 'results')
    dot.edge('root', 'lib')
    dot.edge('root', 'linerr')

    dot.edge('data', 'upcs')
    dot.edge('data', 'price')

    dot.edge('dfl', 'config')
    dot.edge('dfl', 'core')
    dot.edge('dfl', 'dfl_data')
    dot.edge('dfl', 'training')
    dot.edge('dfl', 'validation')
    dot.edge('dfl', 'scripts')
    dot.edge('dfl', 'utils')
    dot.edge('dfl', 'outputs')

    dot.edge('outputs', 'noisy')
    dot.edge('outputs', 'trained')
    dot.edge('outputs', 'val_res')

    dot.edge('miqp', 'miqp_lin')
    dot.edge('miqp', 'miqp_pw')

    dot.edge('legacy', 'dfl_gl')
    dot.edge('legacy', 'dfl_pw')
    dot.edge('legacy', 'dfl_nonn')

    dot.edge('results', 'tables')
    dot.edge('results', 'figures')

    # Render
    dot.render(output_path, cleanup=True)
    print(f"Directory structure diagram saved to {output_path}.png")


def create_architecture_diagram(output_path="docs/architecture_diagram"):
    """Create the component architecture diagram"""
    dot = Digraph(comment='DFL Component Architecture', format='png')
    dot.attr(rankdir='TB', size='14,10')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')

    # Input Data
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Data', style='filled', color='#E6F3FF')
        c.node('price_in', 'Price Data\nData/price_data_2024.csv', fillcolor='#B0D0FF')
        c.node('upc_in', 'UPC Data\nData/UPCs/', fillcolor='#B0D0FF')
        c.node('prep_in', 'Preprocessed\npreprocess.pkl', fillcolor='#B0D0FF')

    # MIQP Baselines
    with dot.subgraph(name='cluster_miqp') as c:
        c.attr(label='MIQP Baselines', style='filled', color='#E6FFE6')
        c.node('miqp_gl', 'Global Linear\nMIQP/MIQP_linear/', fillcolor='#98FB98')
        c.node('miqp_pw', 'Piecewise SOS2\nMIQP/MIQP_piecewise/', fillcolor='#98FB98')

    # DFL Core
    with dot.subgraph(name='cluster_dfl') as c:
        c.attr(label='DFL Framework', style='filled', color='#F3E6FF')
        c.node('lstm', 'Neural Penalty\nPredictor\ncore/models.py', fillcolor='#DDA0DD')
        c.node('local', 'Local Linearization\nLayer\ncore/layers.py', fillcolor='#DDA0DD')
        c.node('solver', 'Differentiable QP\nSolver\ncore/layers.py', fillcolor='#DDA0DD')
        c.node('sim', 'Physical Simulator\ncore/layers.py', fillcolor='#DDA0DD')
        c.node('pipeline', 'Recursive Pipeline\ncore/pipeline.py', fillcolor='#C8A2C8')

    # Training
    with dot.subgraph(name='cluster_train') as c:
        c.attr(label='Training', style='filled', color='#FFFACD')
        c.node('noise_gen', 'Noise Injection\ndata/noise.py', fillcolor='#F0E68C')
        c.node('trainer', 'End-to-End Trainer\ntraining/trainer.py', fillcolor='#F0E68C')
        c.node('models', 'Trained Models\noutputs/trained_models/', fillcolor='#FFE4B5')

    # Evaluation
    with dot.subgraph(name='cluster_eval') as c:
        c.attr(label='Evaluation', style='filled', color='#FFE6E6')
        c.node('validator', 'Validator\nvalidation/validator.py', fillcolor='#FFB6C1')
        c.node('benchmarks', 'Benchmarks\noutputs/validation_results/', fillcolor='#FFD5D5')

    # Results
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Results', style='filled', color='#FFE6CC')
        c.node('tables_out', 'LaTeX Tables\nresults/tables/', fillcolor='#FFD699')
        c.node('viz_out', 'Visualizations\nresults/figures/', fillcolor='#FFD699')

    # Data flow edges
    dot.edge('price_in', 'miqp_gl')
    dot.edge('price_in', 'miqp_pw')
    dot.edge('upc_in', 'prep_in')
    dot.edge('prep_in', 'miqp_gl')
    dot.edge('prep_in', 'miqp_pw')

    dot.edge('miqp_gl', 'noise_gen')
    dot.edge('miqp_pw', 'noise_gen')
    dot.edge('noise_gen', 'trainer')

    # DFL internal flow
    dot.edge('trainer', 'lstm')
    dot.edge('lstm', 'local')
    dot.edge('local', 'solver')
    dot.edge('solver', 'sim')
    dot.edge('sim', 'local', style='dashed', label='feedback')
    dot.edge('pipeline', 'lstm', style='dotted', label='orchestrates')

    dot.edge('sim', 'models')
    dot.edge('models', 'validator')
    dot.edge('price_in', 'validator')
    dot.edge('validator', 'benchmarks')

    dot.edge('benchmarks', 'tables_out')
    dot.edge('benchmarks', 'viz_out')

    # Render
    dot.render(output_path, cleanup=True)
    print(f"Architecture diagram saved to {output_path}.png")


if __name__ == '__main__':
    # Create output directory
    os.makedirs('docs', exist_ok=True)

    print("Generating repository diagrams...")
    print("=" * 60)

    try:
        create_workflow_diagram()
        print("✓ Workflow diagram generated")
    except Exception as e:
        print(f"✗ Error generating workflow diagram: {e}")

    try:
        create_directory_structure()
        print("✓ Directory structure diagram generated")
    except Exception as e:
        print(f"✗ Error generating directory structure: {e}")

    try:
        create_architecture_diagram()
        print("✓ Architecture diagram generated")
    except Exception as e:
        print(f"✗ Error generating architecture diagram: {e}")

    print("=" * 60)
    print("All diagrams generated successfully!")
    print("\nGenerated files:")
    print("  - docs/workflow_diagram.png")
    print("  - docs/directory_structure.png")
    print("  - docs/architecture_diagram.png")
    print("\nYou can now add these to your README.md")
