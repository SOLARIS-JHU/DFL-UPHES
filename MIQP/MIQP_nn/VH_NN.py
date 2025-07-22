# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
import dill as pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import pandas as pd
from datetime import datetime

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HeadVolumeNN(nn.Module):
    """Neural network for volume-head relationship modeling"""
    def __init__(self, input_size=1, hidden_sizes=[64, 32], output_size=1):
        super(HeadVolumeNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
        # Count total parameters
        self.total_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        return self.model(x)

def generate_volume_head_data(n_samples=10000, v_low_min=0, v_low_max=None):
    """
    Generate data points mapping lower reservoir volume to head.
    """
    try:
        # Try to import from the Library
        sys.path.append('/Library')
        from Library.V_H_relations import load_portfolio_data, gross_head
        load_portfolio_data()
        from Library.V_H_relations import max_vol_low
        
        if v_low_max is None:
            v_low_max = max_vol_low
            
        # Generate volume values
        v_low_range = np.linspace(v_low_min, v_low_max, n_samples)
        
        # Calculate head values
        h_values = np.array([gross_head(v_low=v) for v in v_low_range])
        
        return v_low_range, h_values
        
    except ImportError:
        print("Could not import from Library. Using data from preprocessing.py approach...")
        # If we can't import directly, try to get data from the preprocessing approach
        try:
            with open('preprocess.pkl', 'rb') as f:
                preprocessed_data = pickle.load(f)
            
            # Extract the volumne-head polynomial from the loaded data
            v_low_h_poly = preprocessed_data[3]  # Based on the save order in preprocessing.py
            
            # Generate synthetic data based on the polynomial function
            v_low_range = np.linspace(0, 1000000, n_samples)  # Sample volume range
            h_values = np.array([v_low_h_poly(v) for v in v_low_range])
            
            return v_low_range, h_values
            
        except (FileNotFoundError, IndexError):
            print("Could not load from preprocess.pkl. Using synthetic data...")
            # If preprocessing data isn't available either, use synthetic data
            if v_low_max is None:
                v_low_max = 1000000  # Some reasonable maximum volume
                
            v_low_range = np.linspace(v_low_min, v_low_max, n_samples)
            
            # A synthetic model: head decreases as volume increases with slight non-linearity
            h_values = 100 - 0.00005 * v_low_range - 0.0000000001 * v_low_range**2 + np.random.normal(0, 0.5, n_samples)
            
            return v_low_range, h_values

def train_model(model, train_loader, test_loader, learning_rate=0.001, epochs=200, early_stopping_patience=20):
    """
    Train the PyTorch model with early stopping
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    
    best_test_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    train_start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}')
        
        # Check for improvement
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    train_time = time.time() - train_start_time
    
    # Load the best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, train_losses, test_losses, train_time

def evaluate_model(model, X, y, norm_params=None):
    """
    Evaluate model performance with multiple metrics
    """
    # Start eval time measurement
    eval_start_time = time.time()
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Normalize if needed
    if norm_params:
        X_normalized = (X_tensor - norm_params['v_low_mean']) / norm_params['v_low_std']
        with torch.no_grad():
            y_pred_normalized = model(X_normalized)
        # Denormalize predictions
        y_pred = y_pred_normalized * norm_params['h_std'] + norm_params['h_mean']
    else:
        # No normalization applied
        with torch.no_grad():
            y_pred = model(X_tensor)
    
    # Convert to numpy for metrics calculation
    y_pred_np = y_pred.cpu().numpy().flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred_np)
    r2 = r2_score(y, y_pred_np)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y - y_pred_np) / (y + 1e-10))) * 100  # Mean Absolute Percentage Error
    max_error = np.max(np.abs(y - y_pred_np))  # Maximum absolute error
    
    # Inference time per sample
    eval_time = time.time() - eval_start_time
    inference_time_per_sample = eval_time / len(X)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2,
        'MAPE': mape,
        'Max_Error': max_error,
        'Inference_Time_Per_Sample': inference_time_per_sample,
        'Total_Params': model.total_params
    }
    
    return metrics, y_pred_np

def plot_architecture_comparison(architectures_results, model_type="Volume to Head"):
    """Plot comparison of different architectures"""
    # Prepare data for plotting
    arch_names = [f"{arch}" for arch in architectures_results.keys()]
    rmse_values = [results['metrics']['RMSE'] for results in architectures_results.values()]
    r2_values = [results['metrics']['R^2'] for results in architectures_results.values()]
    param_counts = [results['metrics']['Total_Params'] for results in architectures_results.values()]
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot RMSE
    bar_positions = np.arange(len(arch_names))
    bars1 = ax1.bar(bar_positions, rmse_values, width=0.6)
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(arch_names, rotation=45, ha='right')
    ax1.set_title(f"{model_type} Model - RMSE by Architecture")
    ax1.set_ylabel("RMSE")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{height:.5f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Plot R²
    bars2 = ax2.bar(bar_positions, r2_values, width=0.6)
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(arch_names, rotation=45, ha='right')
    ax2.set_title(f"{model_type} Model - R² by Architecture")
    ax2.set_ylabel("R²")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{height:.5f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Add parameter count annotation
    for i, params in enumerate(param_counts):
        ax1.text(bar_positions[i], 0.0001, f'params: {params}', ha='center', va='bottom', 
                rotation=90, fontsize=8, color='darkred')
    
    plt.tight_layout()
    filename = f'architecture_comparison_{model_type.replace(" ", "_").lower()}.png'
    plt.savefig(f'models/{filename}', dpi=300)
    return fig

def main():
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    models_dir = Path('models')
    
    # Generate or load volume-head data
    v_low, h_values = generate_volume_head_data(n_samples=20000)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        v_low, h_values, test_size=0.2, random_state=42
    )
    
    # Normalize data
    v_mean, v_std = np.mean(X_train), np.std(X_train)
    h_mean, h_std = np.mean(y_train), np.std(y_train)
    
    X_train_normalized = (X_train - v_mean) / v_std
    y_train_normalized = (y_train - h_mean) / h_std
    X_test_normalized = (X_test - v_mean) / v_std
    y_test_normalized = (y_test - h_mean) / h_std
    
    norm_params = {
        'v_low_mean': v_mean,
        'v_low_std': v_std,
        'h_mean': h_mean,
        'h_std': h_std
    }
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_normalized, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(y_train_normalized, dtype=torch.float32).reshape(-1, 1)
    )
    
    test_dataset = TensorDataset(
        torch.tensor(X_test_normalized, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(y_test_normalized, dtype=torch.float32).reshape(-1, 1)
    )
    
    # Same for head to volume data (swapped inputs/outputs)
    train_dataset_inverse = TensorDataset(
        torch.tensor(y_train_normalized, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(X_train_normalized, dtype=torch.float32).reshape(-1, 1)
    )
    
    test_dataset_inverse = TensorDataset(
        torch.tensor(y_test_normalized, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(X_test_normalized, dtype=torch.float32).reshape(-1, 1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    train_loader_inverse = DataLoader(train_dataset_inverse, batch_size=64, shuffle=True)
    test_loader_inverse = DataLoader(test_dataset_inverse, batch_size=64, shuffle=False)
    
    # Define architectures to test for each model type
    hidden_layer_configurations = [
        [4],              # 1 layer, 4 neurons
        [8],              # 1 layer, 8 neurons
        [16],             # 1 layer, 16 neurons
        [4, 4],           # 2 layers, 4 neurons each
        [8, 8],           # 2 layers, 8 neurons each
        [16, 8],          # 2 layers, 16 and 8 neurons
        [32, 16],         # 2 layers, 32 and 16 neurons
        [64, 32],         # 2 layers, 64 and 32 neurons
        [16, 8, 4],       # 3 layers, 16, 8, and 4 neurons
        [32, 16, 8],      # 3 layers, 32, 16, and 8 neurons
        [64, 32, 16]      # 3 layers, 64, 32, and 16 neurons
    ]
    
    # Volume to Head Model training and evaluation
    v_to_h_results = {}
    
    print("Training and evaluating Volume->Head models with different architectures...\n")
    
    for hidden_layers in hidden_layer_configurations:
        architecture_name = str(hidden_layers)
        print(f"\nTraining Volume->Head model with architecture: {architecture_name}")
        
        # Create model
        model = HeadVolumeNN(input_size=1, hidden_sizes=hidden_layers, output_size=1).to(device)
        
        # Train model
        model, train_losses, test_losses, train_time = train_model(
            model, train_loader, test_loader, learning_rate=0.001, epochs=300
        )
        
        # Evaluate model
        metrics, predictions = evaluate_model(model, X_test, y_test, norm_params)
        
        # Save model
        model_filename = f"v_to_h_model_{'-'.join(map(str, hidden_layers))}.pt"
        torch.save(model.state_dict(), models_dir / model_filename)
        
        # Store results
        v_to_h_results[tuple(hidden_layers)] = {
            'model': model,
            'metrics': metrics,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'predictions': predictions,
            'train_time': train_time,
            'model_filename': model_filename
        }
        
        # Print metrics
        print(f"Architecture: {architecture_name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
    
    # Head to Volume Model training and evaluation
    h_to_v_results = {}
    
    print("\nTraining and evaluating Head->Volume models with different architectures...\n")
    
    for hidden_layers in hidden_layer_configurations:
        architecture_name = str(hidden_layers)
        print(f"\nTraining Head->Volume model with architecture: {architecture_name}")
        
        # Create model
        model = HeadVolumeNN(input_size=1, hidden_sizes=hidden_layers, output_size=1).to(device)
        
        # Train model
        model, train_losses, test_losses, train_time = train_model(
            model, train_loader_inverse, test_loader_inverse, learning_rate=0.001, epochs=300
        )
        
        # Evaluate model - note swapped inputs/outputs for h_to_v
        h_to_v_norm_params = {
            'v_low_mean': h_mean, 
            'v_low_std': h_std,
            'h_mean': v_mean,
            'h_std': v_std
        }
        metrics, predictions = evaluate_model(model, y_test, X_test, h_to_v_norm_params)
        
        # Save model
        model_filename = f"h_to_v_model_{'-'.join(map(str, hidden_layers))}.pt"
        torch.save(model.state_dict(), models_dir / model_filename)
        
        # Store results
        h_to_v_results[tuple(hidden_layers)] = {
            'model': model,
            'metrics': metrics,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'predictions': predictions,
            'train_time': train_time,
            'model_filename': model_filename
        }
        
        # Print metrics
        print(f"Architecture: {architecture_name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
    
    # Plot architecture comparison
    v_to_h_comparison_fig = plot_architecture_comparison(v_to_h_results, "Volume to Head")
    h_to_v_comparison_fig = plot_architecture_comparison(h_to_v_results, "Head to Volume")
    
    # Find best models based on RMSE
    best_v_to_h_arch = min(v_to_h_results.keys(), key=lambda x: v_to_h_results[x]['metrics']['RMSE'])
    best_h_to_v_arch = min(h_to_v_results.keys(), key=lambda x: h_to_v_results[x]['metrics']['RMSE'])
    
    # Copy best models to standard names for easy reference
    best_v_to_h_model_path = models_dir / v_to_h_results[best_v_to_h_arch]['model_filename']
    best_h_to_v_model_path = models_dir / h_to_v_results[best_h_to_v_arch]['model_filename']
    
    best_v_to_h_model = v_to_h_results[best_v_to_h_arch]['model']
    best_h_to_v_model = h_to_v_results[best_h_to_v_arch]['model']
    
    # Save best models with standard names
    torch.save(best_v_to_h_model.state_dict(), models_dir / 'v_to_h_model_best.pt')
    torch.save(best_h_to_v_model.state_dict(), models_dir / 'h_to_v_model_best.pt')
    
    # Save normalization parameters
    with open(models_dir / 'norm_params.pkl', 'wb') as f:
        pickle.dump(norm_params, f)
    
    # Save complete results
    results_data = {
        'v_to_h': {
            arch: {
                'metrics': data['metrics'],
                'train_losses': data['train_losses'],
                'test_losses': data['test_losses'],
                'train_time': data['train_time'],
                'model_filename': data['model_filename']
            } for arch, data in v_to_h_results.items()
        },
        'h_to_v': {
            arch: {
                'metrics': data['metrics'],
                'train_losses': data['train_losses'],
                'test_losses': data['test_losses'],
                'train_time': data['train_time'],
                'model_filename': data['model_filename']
            } for arch, data in h_to_v_results.items()
        },
        'best_v_to_h_arch': best_v_to_h_arch,
        'best_h_to_v_arch': best_h_to_v_arch,
        'norm_params': norm_params,
        'h_to_v_norm_params': {
            'v_low_mean': h_mean, 
            'v_low_std': h_std,
            'h_mean': v_mean,
            'h_std': v_std
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(models_dir / 'vh_model_performance.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    # Create comparison tables for markdown report
    v_to_h_table = []
    for arch, data in v_to_h_results.items():
        metrics = data['metrics']
        v_to_h_table.append({
            'Architecture': str(list(arch)),
            'RMSE': metrics['RMSE'],
            'R²': metrics['R^2'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'Inference_Time': metrics['Inference_Time_Per_Sample'],
            'Parameters': metrics['Total_Params'],
            'Train_Time': data['train_time']
        })
    
    h_to_v_table = []
    for arch, data in h_to_v_results.items():
        metrics = data['metrics']
        h_to_v_table.append({
            'Architecture': str(list(arch)),
            'RMSE': metrics['RMSE'],
            'R²': metrics['R^2'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'Inference_Time': metrics['Inference_Time_Per_Sample'],
            'Parameters': metrics['Total_Params'],
            'Train_Time': data['train_time']
        })
    
    # Convert to DataFrames for easier handling
    v_to_h_df = pd.DataFrame(v_to_h_table)
    h_to_v_df = pd.DataFrame(h_to_v_table)
    
    # Sort by RMSE (ascending)
    v_to_h_df = v_to_h_df.sort_values('RMSE')
    h_to_v_df = h_to_v_df.sort_values('RMSE')
    
    # Create markdown report
    with open(models_dir / 'architecture_comparison_report.md', 'w') as f:
        f.write("# Volume-Head Relationship Neural Network Architecture Comparison\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Volume to Head Model Comparison\n\n")
        f.write(v_to_h_df.to_markdown(index=False, floatfmt='.6f'))
        
        f.write("\n\n## Head to Volume Model Comparison\n\n")
        f.write(h_to_v_df.to_markdown(index=False, floatfmt='.6f'))
        
        f.write("\n\n## Best Architectures\n\n")
        f.write(f"* **Best Volume to Head Model**: {list(best_v_to_h_arch)}\n")
        f.write(f"* **Best Head to Volume Model**: {list(best_h_to_v_arch)}\n\n")
        
        f.write("## Testing Best Models\n\n")
        
        # Test the best models on some sample data
        sample_volumes = np.array([100000, 300000, 500000, 700000, 900000])
        sample_heads = np.array([90, 80, 70, 60, 50])
        
        # Define prediction functions for best models
        def predict_h_from_v(v_values, model):
            model.eval()
            v_tensor = torch.tensor(v_values, dtype=torch.float32).reshape(-1, 1).to(device)
            v_normalized = (v_tensor - norm_params['v_low_mean']) / norm_params['v_low_std']
            with torch.no_grad():
                h_normalized = model(v_normalized)
            h_values = h_normalized * norm_params['h_std'] + norm_params['h_mean']
            return h_values.cpu().numpy()
        
        def predict_v_from_h(h_values, model):
            model.eval()
            h_tensor = torch.tensor(h_values, dtype=torch.float32).reshape(-1, 1).to(device)
            h_normalized = (h_tensor - results_data['h_to_v_norm_params']['v_low_mean']) / results_data['h_to_v_norm_params']['v_low_std']
            with torch.no_grad():
                v_normalized = model(h_normalized)
            v_values = v_normalized * results_data['h_to_v_norm_params']['h_std'] + results_data['h_to_v_norm_params']['h_mean']
            return v_values.cpu().numpy()
        
        # Make predictions with best models
        pred_heads = predict_h_from_v(sample_volumes, best_v_to_h_model)
        pred_volumes = predict_v_from_h(sample_heads, best_h_to_v_model)
        
        # Write sample predictions to the report
        f.write("### Sample Volume → Head Predictions (Best Model)\n\n")
        f.write("| Volume | Predicted Head |\n")
        f.write("|--------|---------------|\n")
        for v, h in zip(sample_volumes, pred_heads.flatten()):
            f.write(f"| {v:.2f} | {h:.4f} |\n")
        
        f.write("\n### Sample Head → Volume Predictions (Best Model)\n\n")
        f.write("| Head | Predicted Volume |\n")
        f.write("|------|------------------|\n")
        for h, v in zip(sample_heads, pred_volumes.flatten()):
            f.write(f"| {h:.2f} | {v:.4f} |\n")
    
    # Save comparison tables as CSV
    v_to_h_df.to_csv(models_dir / 'v_to_h_comparison.csv', index=False)
    h_to_v_df.to_csv(models_dir / 'h_to_v_comparison.csv', index=False)
    
    # Print final summary
    print("\n=== Architecture Comparison Complete ===")
    print(f"Best Volume->Head Architecture: {list(best_v_to_h_arch)}")
    print(f"  RMSE: {v_to_h_results[best_v_to_h_arch]['metrics']['RMSE']:.6f}")
    print(f"  R²: {v_to_h_results[best_v_to_h_arch]['metrics']['R^2']:.6f}")
    
    print(f"\nBest Head->Volume Architecture: {list(best_h_to_v_arch)}")
    print(f"  RMSE: {h_to_v_results[best_h_to_v_arch]['metrics']['RMSE']:.6f}")
    print(f"  R²: {h_to_v_results[best_h_to_v_arch]['metrics']['R^2']:.6f}")
    
    print("\nAll models and results saved to the 'models' directory")
    print(f"Detailed comparison report: models/architecture_comparison_report.md")

if __name__ == "__main__":
    main()
# %%
