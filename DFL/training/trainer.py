"""
Training functions for DFL models.

This module contains the core training logic for weight prediction networks
with recursive linearization.
"""

import torch
import torch.nn as nn
import time
from pathlib import Path

from ..core.models import BoundedLogWeightPredictor, FixedWeightConfig
from ..core.pipeline import RecursiveLinearizationPipeline, BaselineRecursiveLinearization


def train_recursive_linearization(weight_network, params, optimizer_layer, regression_layer,
                                   historical_data, config, num_epochs=100, learning_rate=0.001,
                                   patience=10):
    """
    Train the log-domain weight predictor with recursive linearization.
    Uses simulated profit as the loss function.

    Args:
        weight_network: BoundedLogWeightPredictor or None (for ablation)
        params: HydroParameters instance
        optimizer_layer: OptiLayer instance
        regression_layer: TaylorRegressionLayer instance
        historical_data: Dictionary of historical data by date
        config: DFLConfig instance
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience

    Returns:
        tuple: (trained_network or None, history dictionary)
    """
    # Get device
    if weight_network is not None:
        device = next(weight_network.parameters()).device
        weight_network.train()

        # Create optimizer
        optimizer = torch.optim.Adam(weight_network.parameters(), lr=learning_rate)
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        # Create the pipeline with neural network
        pipeline = RecursiveLinearizationPipeline(
            weight_network, params, optimizer_layer, regression_layer, historical_data,
            max_iterations=config.max_iterations,
            penalty_growth_rate=config.penalty_growth_rate
        )
    else:
        # Ablation mode: no neural network
        device = torch.device("cpu")
        weight_config = FixedWeightConfig(
            w_p=config.fixed_w_p,
            w_q=config.fixed_w_q,
            w_h=config.fixed_w_h,
            time_horizon=params.time_horizon,
            device=device
        )

        pipeline = BaselineRecursiveLinearization(
            weight_config, params, optimizer_layer, regression_layer,
            max_iterations=config.max_iterations,
            penalty_growth_rate=config.penalty_growth_rate
        )

    # Select a single date for training
    train_date = list(historical_data.keys())[0]

    # Get original data
    date_data = historical_data[train_date]
    power_orig = date_data['power']
    head_orig = date_data['head']
    flow_orig = params.predict_q_poly(power_orig, head_orig)

    # Initialize history tracking
    history = {
        'epoch': [],
        'loss': [],
        'profit': [],
        'simulated_profit': [],
        'SI_penalty': [],
        'volume_penalty': [],
        'operating_cost': [],
        'p_opt': [],
        'h_opt': [],
        'q_opt': [],
        'p_sim': [],
        'q_sim': [],
        'h_sim': [],
        'v_sim': [],
        'p_orig': power_orig.cpu().numpy(),
        'h_orig': head_orig.cpu().numpy(),
        'q_orig': flow_orig.cpu().numpy(),
        'iterations': []
    }

    # Add weight tracking if using neural network
    if weight_network is not None:
        history.update({
            'log_w_p': [],
            'log_w_q': [],
            'log_w_h': [],
            'w_p': [],
            'w_q': [],
            'w_h': []
        })

    # Initialize early stopping
    best_profit = float('-inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(num_epochs):
        if weight_network is not None:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with neural network
            simulated_profit, optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
            p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty, operating_cost, \
            (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h), c, d, e, a, b, iter_results = pipeline.forward(train_date)

            # Compute loss
            loss = -simulated_profit

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(weight_network.parameters(), max_norm=1.0)

            optimizer.step()

            # Update learning rate scheduler
            scheduler.step(simulated_profit)

            # Record weight info
            history['log_w_p'].append(log_w_p.detach().cpu().numpy())
            history['log_w_q'].append(log_w_q.detach().cpu().numpy())
            history['log_w_h'].append(log_w_h.detach().cpu().numpy())
            history['w_p'].append(w_p.detach().cpu().numpy())
            history['w_q'].append(w_q.detach().cpu().numpy())
            history['w_h'].append(w_h.detach().cpu().numpy())

        else:
            # Forward pass without neural network (ablation)
            (simulated_profit, optimized_profit, p_opt, q_opt, h_opt, v_opt,
             p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty,
             operating_cost, iter_results) = pipeline.forward(
                date_data['price'], power_orig, head_orig, flow_orig
            )

            loss = -simulated_profit

        # Record iteration details
        history['iterations'].append(iter_results)

        # Record common history
        history['epoch'].append(epoch)
        history['loss'].append(loss.item())
        history['profit'].append(optimized_profit.item())
        history['simulated_profit'].append(simulated_profit.item())
        history['SI_penalty'].append(SI_penalty.item())
        history['volume_penalty'].append(volume_penalty.item())
        history['operating_cost'].append(operating_cost.item())
        history['p_opt'].append(p_opt.detach().cpu().numpy())
        history['h_opt'].append(h_opt.detach().cpu().numpy())
        history['q_opt'].append(q_opt.detach().cpu().numpy())
        history['p_sim'].append(p_sim.detach().cpu().numpy())
        history['q_sim'].append(q_sim.detach().cpu().numpy())
        history['h_sim'].append(h_sim.detach().cpu().numpy())
        history['v_sim'].append(v_low_sim.detach().cpu().numpy())

        # Early stopping check
        if simulated_profit.item() > best_profit:
            best_profit = simulated_profit.item()
            if weight_network is not None:
                best_weights = weight_network.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best weights
    if weight_network is not None and best_weights is not None:
        weight_network.load_state_dict(best_weights)

    return weight_network, history


def train_single_model(config, architecture, num_layers, max_iterations, date_str, date_data,
                       params, device):
    """
    Train a single model configuration for a specific date.

    Args:
        config: DFLConfig instance
        architecture: Network architecture ('LSTM', 'RNN', 'FC')
        num_layers: Number of network layers
        max_iterations: Number of recursive linearization iterations
        date_str: Date string for this training instance
        date_data: Data dictionary for this date
        params: HydroParameters instance
        device: PyTorch device

    Returns:
        dict: Training results
    """
    try:
        from ..core.layers import TaylorRegressionLayer, OptiLayer

        # Initialize layers
        regression_layer = TaylorRegressionLayer(params)
        optimizer_layer = OptiLayer(params)

        # Create output directory
        config_name = f"{architecture}_{num_layers}layer_{max_iterations}iter"
        output_dir = Path(config.output_base_dir) / config_name / date_str
        output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize network based on config
        if config.use_neural_network:
            weight_network = BoundedLogWeightPredictor(
                input_size=4,
                hidden_size=config.hidden_size,
                num_layers=num_layers,
                dropout=config.dropout,
                time_horizon=params.time_horizon,
                archetype=architecture,
                init_w_p=config.init_w_p,
                init_w_q=config.init_w_q,
                init_w_h=config.init_w_h,
                w_p_min=config.w_p_min,
                w_p_max=config.w_p_max,
                w_q_min=config.w_q_min,
                w_q_max=config.w_q_max,
                w_h_min=config.w_h_min,
                w_h_max=config.w_h_max
            ).to(device)
        else:
            weight_network = None

        # Train
        start_time = time.time()
        trained_network, history = train_recursive_linearization(
            weight_network=weight_network,
            params=params,
            optimizer_layer=optimizer_layer,
            regression_layer=regression_layer,
            historical_data={date_str: date_data},
            config=config,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            patience=config.patience
        )
        training_time = time.time() - start_time

        # Save model weights if using neural network
        if trained_network is not None:
            torch.save(trained_network.state_dict(), output_dir / "model.pt")

        return {'success': True, 'training_time': training_time}

    except Exception as e:
        print(f"Error training model for {date_str}: {e}")
        return {'success': False, 'error': str(e)}
