# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
import time
import csv

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class UPCNet(nn.Module):
    """Neural network for UPC function approximation"""
    def __init__(self, hidden_sizes):
        super(UPCNet, self).__init__()
    
        layers = []
        input_size = 2  # Power (p) and head (h)
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer (flow rate q)
        layers.append(nn.Linear(input_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # x should be tensor of shape (..., 2) containing [p, h] pairs
        return self.model(x)

def load_and_prepare_upc_data(file_path, test_size=0.2, random_state=42):
    """Load UPC data from Excel file with power in columns and head in rows"""
    data = pd.read_excel(file_path)
    
    # Extract power values from columns (skip first column which has head values)
    p_values = np.array(data.columns[1:], dtype=float)
    
    # Extract head values from first column
    h_values = np.array(data.iloc[:, 0], dtype=float)
    
    # Create dataset
    inputs = []   # Will contain [p, h] pairs
    outputs = []  # Will contain q values
    
    for i, h in enumerate(h_values):
        for j, p in enumerate(p_values):
            q = data.iloc[i, j+1]
            if not np.isnan(q):  # Skip NaN values
                inputs.append([p, h])
                outputs.append([q])
    
    # Convert to numpy arrays
    original_inputs = np.array(inputs)
    original_outputs = np.array(outputs).flatten()
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        original_inputs, original_outputs, test_size=test_size, random_state=random_state
    )
    
    # Generate additional test points using bilinear interpolation
    if len(original_inputs) > 10:  # Make sure we have enough points to interpolate
        p_min, p_max = original_inputs[:, 0].min(), original_inputs[:, 0].max()
        h_min, h_max = original_inputs[:, 1].min(), original_inputs[:, 1].max()
        
        # Create a grid of points for interpolation
        grid_size = 20
        p_grid = np.linspace(p_min, p_max, grid_size)
        h_grid = np.linspace(h_min, h_max, grid_size)
        
        p_mesh, h_mesh = np.meshgrid(p_grid, h_grid)
        interp_points = np.column_stack((p_mesh.flatten(), h_mesh.flatten()))
        
        # Interpolate q values
        interp_values = griddata(original_inputs, original_outputs, interp_points, method='linear')
        
        # Filter out NaN values (points outside the convex hull of the original data)
        valid_indices = ~np.isnan(interp_values)
        X_interp = interp_points[valid_indices]
        y_interp = interp_values[valid_indices]
        
        # Add interpolated points to test set
        X_test = np.vstack((X_test, X_interp))
        y_test = np.concatenate((y_test, y_interp))
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, p_values, h_values

def train_upc_model(model, X_train, y_train, X_val, y_val, epochs=2000, 
                   batch_size=64, lr=0.001, patience=50, model_name="model"):
    """Train the UPC neural network model with early stopping"""
    # Create directories for saving checkpoints and results
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Move model and data to device
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # Mean Squared Error Loss
    
    # Convert data to PyTorch datasets
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # For early stopping
    best_val_loss = float('inf')
    counter = 0
    early_stop = False
    best_model_state = None
    
    # Training loop
    train_loss_history = []
    val_loss_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        if early_stop:
            break
            
        model.train()
        epoch_loss = 0
        
        for x_batch, y_batch in dataloader:
            # Forward pass
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(x_batch)
        
        avg_loss = epoch_loss / len(X_train)
        train_loss_history.append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = loss_fn(val_preds, y_val).item()
            val_loss_history.append(val_loss)
        
        # Print progress periodically
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
            # Save best model checkpoint
            torch.save(model.state_dict(), f"models/{model_name}_best.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                early_stop = True
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save final model
    torch.save(model.state_dict(), f"models/{model_name}_final.pt")
    
    # Save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Loss Curves for {model_name}')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f"results/{model_name}_loss_curves.png")
    plt.close()
    
    return model, train_loss_history, val_loss_history, best_val_loss

def predict_q_nn(p, h, turbine_model, pump_model):
    """
    Direct replacement for predict_q_poly using neural networks
    
    Args:
        p: torch tensor of power values 
        h: torch tensor of head values (same shape as p)
        turbine_model: trained neural network for turbine mode (p > 0)
        pump_model: trained neural network for pump mode (p < 0)
        
    Returns:
        torch tensor of predicted flow values (same shape as inputs)
    """
    # Ensure inputs are torch tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    if not isinstance(h, torch.Tensor):
        h = torch.tensor(h, dtype=torch.float32)
    
    # Move tensors to appropriate device
    p = p.to(device)
    h = h.to(device)
    
    # Store original shape for reshaping output
    original_shape = p.shape
    
    # Flatten tensors for processing
    p_flat = p.reshape(-1)
    h_flat = h.reshape(-1)
    
    # Create masks for different modes
    mask_turbine = (p_flat > 0)
    mask_pump = (p_flat < 0)
    mask_idle = (p_flat == 0)
    
    # Initialize output tensor
    q = torch.zeros_like(p_flat, device=device)
    
    # Apply turbine model where p > 0
    if torch.any(mask_turbine):
        # Extract turbine data points
        p_turbine = p_flat[mask_turbine]
        h_turbine = h_flat[mask_turbine]
        
        # Prepare input for neural network
        turbine_input = torch.stack([p_turbine, h_turbine], dim=1)
        
        # Get predictions
        with torch.no_grad():
            q_turbine = turbine_model(turbine_input).squeeze()
        
        # Assign predictions to output
        q[mask_turbine] = q_turbine
    
    # Apply pump model where p < 0
    if torch.any(mask_pump):
        # Extract pump data points
        p_pump = p_flat[mask_pump]
        h_pump = h_flat[mask_pump]
        
        # Prepare input for neural network
        pump_input = torch.stack([p_pump, h_pump], dim=1)
        
        # Get predictions
        with torch.no_grad():
            q_pump = pump_model(pump_input).squeeze()
        
        # Assign predictions to output
        q[mask_pump] = q_pump
    
    # For idle mode (p == 0), flow rate is 0 (already initialized)
    
    # Reshape output to match input shape
    return q.reshape(original_shape)

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data and return metrics"""
    model.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        y_pred = model(X_test)
        mse = ((y_pred - y_test) ** 2).mean().item()
        rmse = np.sqrt(mse)
        mae = (y_pred - y_test).abs().mean().item()
        
        # R-squared calculation
        y_mean = y_test.mean().item()
        total_sum_squares = ((y_test - y_mean) ** 2).sum().item()
        residual_sum_squares = ((y_test - y_pred) ** 2).sum().item()
        r_squared = 1 - (residual_sum_squares / total_sum_squares)
        
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'predictions': y_pred.cpu().numpy(),
        'actuals': y_test.cpu().numpy()
    }

def plot_predictions(actuals, predictions, title, save_path):
    """Plot actual vs predicted values with a perfect prediction line"""
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Flow Rate (q)')
    plt.ylabel('Predicted Flow Rate (q)')
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Load UPC data
    print("Loading UPC data...")
    turbine_X_train, turbine_X_test, turbine_y_train, turbine_y_test, turbine_p, turbine_h = load_and_prepare_upc_data('./Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx')
    pump_X_train, pump_X_test, pump_y_train, pump_y_test, pump_p, pump_h = load_and_prepare_upc_data('./Data/UPCs/temp/Mod_Francis_pump_temp.xlsx')
    
    print(f"Turbine data: {len(turbine_X_train)} training samples, {len(turbine_X_test)} testing samples")
    print(f"Pump data: {len(pump_X_train)} training samples, {len(pump_X_test)} testing samples")
    
    # Define model architectures to test
    architectures = []
    
    # Single hidden layer architectures
    for neurons in [8, 16, 32]:
        architectures.append([neurons])
    
    # Two hidden layer architectures
    for neurons1 in [16, 32]:
        for neurons2 in [8, 16]:
            architectures.append([neurons1, neurons2])
    
    # Three hidden layer architectures
    for neurons1 in [32, 64]:
        for neurons2 in [16, 32]:
            for neurons3 in [8, 16]:
                architectures.append([neurons1, neurons2, neurons3])
    
    # Results storage
    results = []
    
    # Train and test all model configurations
    for mode_name, (X_train, X_test, y_train, y_test) in [
        ("turbine", (turbine_X_train, turbine_X_test, turbine_y_train, turbine_y_test)),
        ("pump", (pump_X_train, pump_X_test, pump_y_train, pump_y_test))
    ]:
        for arch in architectures:
            # Create model name
            arch_str = '_'.join(map(str, arch))
            model_name = f"{mode_name}_{len(arch)}layers_{arch_str}"
            print(f"\nTraining {model_name}...")
            
            # Create and train model
            model = UPCNet(hidden_sizes=arch)
            trained_model, train_losses, val_losses, best_val = train_upc_model(
                model, X_train, y_train, X_test, y_test, 
                epochs=2000, batch_size=64, lr=0.001, patience=50,
                model_name=model_name
            )
            
            # Evaluate model
            metrics = evaluate_model(trained_model, X_test, y_test)
            
            # Plot predictions vs actuals
            plot_predictions(
                metrics['actuals'].flatten(), 
                metrics['predictions'].flatten(), 
                f"{mode_name.capitalize()} Model ({len(arch)} layers, {arch_str})\nMSE: {metrics['mse']:.6f}, R²: {metrics['r_squared']:.4f}",
                f"results/{model_name}_predictions.png"
            )
            
            # Record results
            results.append({
                'mode': mode_name,
                'layers': len(arch),
                'architecture': arch_str,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r_squared': metrics['r_squared'],
                'best_val_loss': best_val,
                'final_train_loss': train_losses[-1],
                'epochs_trained': len(train_losses),
                'model_file': f"models/{model_name}_best.pt"
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_performance.csv", index=False)
    print("\nResults saved to results/model_performance.csv")
    
    # Find best models
    best_turbine_idx = results_df[results_df['mode'] == 'turbine']['mse'].idxmin()
    best_pump_idx = results_df[results_df['mode'] == 'pump']['mse'].idxmin()
    
    best_turbine = results_df.iloc[best_turbine_idx]
    best_pump = results_df.iloc[best_pump_idx]
    
    print(f"\nBest turbine model: {best_turbine['model_file']}")
    print(f"  Architecture: {best_turbine['layers']} layers, {best_turbine['architecture']}")
    print(f"  MSE: {best_turbine['mse']:.8f}, R²: {best_turbine['r_squared']:.4f}")
    
    print(f"\nBest pump model: {best_pump['model_file']}")
    print(f"  Architecture: {best_pump['layers']} layers, {best_pump['architecture']}")
    print(f"  MSE: {best_pump['mse']:.8f}, R²: {best_pump['r_squared']:.4f}")
    
    # Load best models for testing
    best_turbine_arch = list(map(int, best_turbine['architecture'].split('_')))
    best_pump_arch = list(map(int, best_pump['architecture'].split('_')))
    
    best_turbine_model = UPCNet(hidden_sizes=best_turbine_arch)
    best_turbine_model.load_state_dict(torch.load(best_turbine['model_file']))
    best_turbine_model.to(device)
    
    best_pump_model = UPCNet(hidden_sizes=best_pump_arch)
    best_pump_model.load_state_dict(torch.load(best_pump['model_file']))
    best_pump_model.to(device)
    
    # Test combined model on some sample points
    print("\nTesting predict_q_nn function with best models...")
    
    # Generate test data spanning both pump and turbine modes
    test_p = torch.tensor([-5.0, -2.5, 0.0, 3.0, 7.5], dtype=torch.float32)
    test_h = torch.tensor([70.0, 75.0, 80.0, 85.0, 90.0], dtype=torch.float32)
    
    # Use predict_q_nn
    predicted_q = predict_q_nn(test_p, test_h, best_turbine_model, best_pump_model)
    
    # Print results
    print("\nSample prediction results:")
    print("Power (p) | Head (h) | Flow (q)")
    print("-" * 30)
    for p, h, q in zip(test_p.cpu().numpy(), test_h.cpu().numpy(), predicted_q.cpu().numpy()):
        print(f"{p:8.2f} | {h:7.2f} | {q:8.4f}")

# %% Quick Test Block
def quick_test():
    """Run a quick test of the training process with a single architecture"""
    print("Running quick test...")
    
    # 1. Load UPC data (smaller subset for faster testing)
    print("Loading UPC data...")
    turbine_X_train, turbine_X_test, turbine_y_train, turbine_y_test, _, _ = load_and_prepare_upc_data(
        './Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx', test_size=0.3
    )
    pump_X_train, pump_X_test, pump_y_train, pump_y_test, _, _ = load_and_prepare_upc_data(
        './Data/UPCs/temp/Mod_Francis_pump_temp.xlsx', test_size=0.3
    )
    
    print(f"Turbine data: {len(turbine_X_train)} training samples, {len(turbine_X_test)} testing samples")
    print(f"Pump data: {len(pump_X_train)} training samples, {len(pump_X_test)} testing samples")
    
    # 2. Define a single architecture to test
    arch = [32, 16]  # Simple 2-layer architecture
    
    # 3. Create and train models with fewer epochs and no file saving
    model_results = []
    
    for mode_name, (X_train, X_test, y_train, y_test) in [
        ("turbine", (turbine_X_train, turbine_X_test, turbine_y_train, turbine_y_test)),
        ("pump", (pump_X_train, pump_X_test, pump_y_train, pump_y_test))
    ]:
        print(f"\nTesting {mode_name} model with {len(arch)} layers...")
        
        # Create model
        model = UPCNet(hidden_sizes=arch)
        
        # Move to device
        model = model.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        # Create dataloader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Simplified training loop (fewer epochs, no early stopping, no saving)
        train_losses = []
        val_losses = []
        
        for epoch in range(200):  # Reduced epochs for testing
            # Training
            model.train()
            epoch_loss = 0
            
            for x_batch, y_batch in dataloader:
                # Forward pass
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(x_batch)
            
            avg_loss = epoch_loss / len(X_train)
            train_losses.append(avg_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test)
                val_loss = loss_fn(val_preds, y_test).item()
                val_losses.append(val_loss)
            
            # Print progress every 20 epochs
            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}/200, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        print(f"\n{mode_name.capitalize()} Model Metrics:")
        print(f"  MSE: {metrics['mse']:.8f}")
        print(f"  RMSE: {metrics['rmse']:.8f}")
        print(f"  MAE: {metrics['mae']:.8f}")
        print(f"  R²: {metrics['r_squared']:.4f}")
        
        # Test prediction
        example_input = torch.tensor([[5.0, 75.0]] if mode_name == 'turbine' else [[-5.0, 75.0]], 
                                    dtype=torch.float32).to(device)
        with torch.no_grad():
            example_output = model(example_input)
        print(f"  Sample prediction: Power={example_input[0,0].item()}, Head={example_input[0,1].item()}, Flow={example_output[0,0].item():.4f}")
        
        model_results.append({
            'mode': mode_name,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'mse': metrics['mse'],
            'r_squared': metrics['r_squared']
        })
    
    # 4. Test the prediction function with both models
    print("\nTesting predict_q_nn function...")
    
    # Move models to appropriate device
    turbine_model = model_results[0]['model'].to(device) if 'model' in model_results[0] else None
    pump_model = model_results[1]['model'].to(device) if 'model' in model_results[1] else None
    
    # Create test sample points
    test_p = torch.tensor([-5.0, -2.5, 0.0, 3.0, 7.5], dtype=torch.float32)
    test_h = torch.tensor([70.0, 75.0, 80.0, 85.0, 90.0], dtype=torch.float32)
    
    if turbine_model is not None and pump_model is not None:
        predicted_q = predict_q_nn(test_p, test_h, turbine_model, pump_model)
        
        print("\nSample prediction results:")
        print("Power (p) | Head (h) | Flow (q)")
        print("-" * 30)
        for p, h, q in zip(test_p.cpu().numpy(), test_h.cpu().numpy(), predicted_q.cpu().numpy()):
            print(f"{p:8.2f} | {h:7.2f} | {q:8.4f}")
    else:
        print("Models not available for testing predict_q_nn")

# Run the quick test if this file is executed directly
# if __name__ == "__main__":
    # Uncomment the line below to run the full training
    # main()
    
    # Run quick test instead
    # quick_test()
# %%
# if __name__ == "__main__":
#     main()
# %%
def train_2x2_architecture():
    """Train and evaluate only the [2, 2] architecture for both pump and turbine modes"""
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Load UPC data
    print("Loading UPC data...")
    turbine_X_train, turbine_X_test, turbine_y_train, turbine_y_test, turbine_p, turbine_h = load_and_prepare_upc_data('./Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx')
    pump_X_train, pump_X_test, pump_y_train, pump_y_test, pump_p, pump_h = load_and_prepare_upc_data('./Data/UPCs/temp/Mod_Francis_pump_temp.xlsx')
    
    print(f"Turbine data: {len(turbine_X_train)} training samples, {len(turbine_X_test)} testing samples")
    print(f"Pump data: {len(pump_X_train)} training samples, {len(pump_X_test)} testing samples")
    
    # Define the specific architecture to test
    arch = [2, 2]  # 2 hidden layers with 2 neurons each
    
    # Results storage
    results = []
    
    # Train and test the specified model configuration
    for mode_name, (X_train, X_test, y_train, y_test) in [
        ("turbine", (turbine_X_train, turbine_X_test, turbine_y_train, turbine_y_test)),
        ("pump", (pump_X_train, pump_X_test, pump_y_train, pump_y_test))
    ]:
        # Create model name
        arch_str = '_'.join(map(str, arch))
        model_name = f"{mode_name}_{len(arch)}layers_{arch_str}"
        print(f"\nTraining {model_name}...")
        
        # Create and train model
        model = UPCNet(hidden_sizes=arch)
        trained_model, train_losses, val_losses, best_val = train_upc_model(
            model, X_train, y_train, X_test, y_test, 
            epochs=2000, batch_size=64, lr=0.001, patience=50,
            model_name=model_name
        )
        
        # Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test)
        
        # Plot predictions vs actuals
        plot_predictions(
            metrics['actuals'].flatten(), 
            metrics['predictions'].flatten(), 
            f"{mode_name.capitalize()} Model ({len(arch)} layers, {arch_str})\nMSE: {metrics['mse']:.6f}, R²: {metrics['r_squared']:.4f}",
            f"results/{model_name}_predictions.png"
        )
        
        # Record results
        results.append({
            'mode': mode_name,
            'layers': len(arch),
            'architecture': arch_str,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r_squared': metrics['r_squared'],
            'best_val_loss': best_val,
            'final_train_loss': train_losses[-1],
            'epochs_trained': len(train_losses),
            'model_file': f"models/{model_name}_best.pt"
        })
    
    # Check if the CSV file exists
    csv_file = "results/model_performance.csv"
    if os.path.exists(csv_file):
        # Load existing results
        existing_results = pd.read_csv(csv_file)
        # Concatenate with new results
        all_results = pd.concat([existing_results, pd.DataFrame(results)], ignore_index=True)
    else:
        all_results = pd.DataFrame(results)
    
    # Save updated results to CSV
    all_results.to_csv(csv_file, index=False)
    print(f"\nResults appended to {csv_file}")
    
    # Print results
    for result in results:
        print(f"\n{result['mode'].capitalize()} Model with {result['layers']} layers ({result['architecture']}):")
        print(f"  MSE: {result['mse']:.8f}, R²: {result['r_squared']:.4f}")
        print(f"  Model saved to: {result['model_file']}")

if __name__ == "__main__":
    train_2x2_architecture()

# %% 
def train_8x4_architecture():
    """Train and evaluate only the [8, 4] architecture for both pump and turbine modes"""
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Load UPC data
    print("Loading UPC data...")
    turbine_X_train, turbine_X_test, turbine_y_train, turbine_y_test, turbine_p, turbine_h = load_and_prepare_upc_data('./Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx')
    pump_X_train, pump_X_test, pump_y_train, pump_y_test, pump_p, pump_h = load_and_prepare_upc_data('./Data/UPCs/temp/Mod_Francis_pump_temp.xlsx')
    
    print(f"Turbine data: {len(turbine_X_train)} training samples, {len(turbine_X_test)} testing samples")
    print(f"Pump data: {len(pump_X_train)} training samples, {len(pump_X_test)} testing samples")
    
    # Define the specific architecture to test
    arch = [8, 4]  # 8 neurons in first hidden layer, 4 neurons in second hidden layer
    
    # Results storage
    results = []
    
    # Train and test the specified model configuration
    for mode_name, (X_train, X_test, y_train, y_test) in [
        ("turbine", (turbine_X_train, turbine_X_test, turbine_y_train, turbine_y_test)),
        ("pump", (pump_X_train, pump_X_test, pump_y_train, pump_y_test))
    ]:
        # Create model name
        arch_str = '_'.join(map(str, arch))
        model_name = f"{mode_name}_{len(arch)}layers_{arch_str}"
        print(f"\nTraining {model_name}...")
        
        # Create and train model
        model = UPCNet(hidden_sizes=arch)
        trained_model, train_losses, val_losses, best_val = train_upc_model(
            model, X_train, y_train, X_test, y_test, 
            epochs=2000, batch_size=64, lr=0.001, patience=50,
            model_name=model_name
        )
        
        # Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test)
        
        # Plot predictions vs actuals
        plot_predictions(
            metrics['actuals'].flatten(), 
            metrics['predictions'].flatten(), 
            f"{mode_name.capitalize()} Model ({len(arch)} layers, {arch_str})\nMSE: {metrics['mse']:.6f}, R²: {metrics['r_squared']:.4f}",
            f"results/{model_name}_predictions.png"
        )
        
        # Record results
        results.append({
            'mode': mode_name,
            'layers': len(arch),
            'architecture': arch_str,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r_squared': metrics['r_squared'],
            'best_val_loss': best_val,
            'final_train_loss': train_losses[-1],
            'epochs_trained': len(train_losses),
            'model_file': f"models/{model_name}_best.pt"
        })
    
    # Check if the CSV file exists
    csv_file = "results/model_performance.csv"
    if os.path.exists(csv_file):
        # Load existing results
        existing_results = pd.read_csv(csv_file)
        # Concatenate with new results
        all_results = pd.concat([existing_results, pd.DataFrame(results)], ignore_index=True)
    else:
        all_results = pd.DataFrame(results)
    
    # Save updated results to CSV
    all_results.to_csv(csv_file, index=False)
    print(f"\nResults appended to {csv_file}")
    
    # Print results
    for result in results:
        print(f"\n{result['mode'].capitalize()} Model with {result['layers']} layers ({result['architecture']}):")
        print(f"  MSE: {result['mse']:.8f}, R²: {result['r_squared']:.4f}")
        print(f"  Model saved to: {result['model_file']}")
    
    return results

if __name__ == "__main__":
    train_8x4_architecture()