"""
Recursive linearization pipelines for hydropower optimization.

This module contains pipelines that orchestrate the recursive linearization process:
- RecursiveLinearizationPipeline: With neural network weight prediction
- BaselineRecursiveLinearization: With fixed weights (for ablation studies)
"""

import torch

from .layers import SimulationLayer


class RecursiveLinearizationPipeline:
    """
    Pipeline that recursively updates linearization coefficients using optimization results.
    Works with log-domain weights and exponentiates them for the optimizer.
    Also includes simulation to calculate realistic profit.
    """

    def __init__(self, weight_network, params, optimizer, regression, historical_data,
                 max_iterations=3, penalty_growth_rate=1.5):
        """
        Initialize RecursiveLinearizationPipeline.

        Args:
            weight_network: BoundedLogWeightPredictor instance
            params: HydroParameters instance
            optimizer: OptiLayer instance
            regression: TaylorRegressionLayer instance
            historical_data: Dictionary of historical operational data
            max_iterations: Number of recursive linearization iterations
            penalty_growth_rate: Factor by which penalties grow each iteration
        """
        self.weight_network = weight_network
        self.params = params
        self.optimizer = optimizer
        self.regression = regression
        self.historical_data = historical_data
        self.max_iterations = max_iterations
        self.simulator = SimulationLayer(params)
        self.penalty_growth_rate = penalty_growth_rate

    def forward(self, date_str):
        """
        Run recursive linearization pipeline for a specific date.

        Args:
            date_str: Date string key for historical_data dictionary

        Returns:
            Tuple containing all optimization and simulation results
        """
        # Get the data for this date
        date_data = self.historical_data[date_str]
        power_init = date_data['power'].clone()
        head_init = date_data['head'].clone()
        price = date_data['price'].clone()

        # Predict initial flow from (p,h)
        # Note: predict_q_poly expects batch dimension, so we unsqueeze, call, then squeeze
        flow_init = self.params.predict_q_poly(power_init.unsqueeze(0), head_init.unsqueeze(0)).squeeze(0)

        # Get input features for the weight predictor
        x = torch.stack([price, power_init, flow_init, head_init], dim=1)  # [time_horizon, 4]

        # Run weight prediction with gradient tracking
        log_w_p, log_w_q, log_w_h = self.weight_network(x)

        # Exponentiate to get initial weights (with gradient tracking)
        w_p_initial = torch.exp(log_w_p)
        w_q_initial = torch.exp(log_w_q)
        w_h_initial = torch.exp(log_w_h)

        # Initialize parameters for first iteration
        p_current = power_init.clone().detach()
        h_current = head_init.clone().detach()
        flow_current = flow_init.clone().detach()

        # Store results from each iteration
        iter_results = []

        # Recursive linearization loop
        for iteration in range(self.max_iterations):
            # Apply growth to penalty weights based on iteration number
            growth_factor = self.penalty_growth_rate ** iteration
            w_p = w_p_initial * growth_factor
            w_q = w_q_initial * growth_factor
            w_h = w_h_initial * growth_factor

            # Compute linearization coefficients based on current power and head
            c, d, e, a, b = self.regression.run_regression(p_current, h_current, flow_current)

            # Initialize the OptiLayer with current values before optimization
            self.optimizer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())

            # Run optimization with current coefficients and growing weights
            p_opt, q_opt, h_opt, v_opt, optimized_profit, optimized_objective = self.optimizer.forward(
                price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                w_p.cpu(), w_h.cpu(), w_q.cpu()
            )

            # Store results from this iteration
            iter_result = {
                'iteration': iteration,
                'optimized_profit': optimized_profit.item(),
                'optimized_objective': optimized_objective.item(),
                'p_opt': p_opt.detach().cpu().numpy(),
                'q_opt': q_opt.detach().cpu().numpy(),
                'h_opt': h_opt.detach().cpu().numpy(),
                'c': c.detach().cpu().numpy(),
                'd': d.detach().cpu().numpy(),
                'e': e.detach().cpu().numpy(),
                'a': a.detach().cpu().numpy(),
                'b': b.detach().cpu().numpy(),
                'growth_factor': growth_factor,
                'w_p': w_p.detach().cpu().numpy(),
                'w_q': w_q.detach().cpu().numpy(),
                'w_h': w_h.detach().cpu().numpy()
            }
            iter_results.append(iter_result)

            # If not the last iteration, update current power, head, and flow for next iteration
            if iteration < self.max_iterations - 1:
                p_current = p_opt.clone().detach().to(device=power_init.device)
                h_current = h_opt.clone().detach().to(device=head_init.device)
                flow_current = q_opt.clone().detach().to(device=flow_init.device)

        # Get device from initial data
        device = power_init.device

        # After optimization, run simulation with the final p_opt, q_opt, h_opt
        p_sim, q_sim, h_sim, v_low_sim = self.simulator.simulate_operation(
            p_opt.to(device), q_opt.to(device), h_opt.to(device)
        )

        # Calculate the simulated profit
        simulated_profit, SI_penalty, volume_penalty, operating_cost = self.simulator.calc_profit(
            p_sim, p_opt.to(device), v_low_sim, price.to(device)
        )

        # Return both optimized and simulated results
        return simulated_profit, optimized_profit, optimized_objective, p_opt, q_opt, h_opt, v_opt, \
               p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty, operating_cost, \
               (log_w_p, log_w_q, log_w_h), (w_p_initial, w_q_initial, w_h_initial), c, d, e, a, b, iter_results


class BaselineRecursiveLinearization:
    """
    Baseline pipeline using fixed weights instead of LSTM predictions.
    Applies the same recursive linearization and penalty growth.
    """

    def __init__(self, weight_config, params, optimizer, regression,
                 max_iterations=3, penalty_growth_rate=1.5):
        """
        Initialize BaselineRecursiveLinearization.

        Args:
            weight_config: FixedWeightConfig instance
            params: HydroParameters instance
            optimizer: OptiLayer instance
            regression: TaylorRegressionLayer instance
            max_iterations: Number of recursive linearization iterations
            penalty_growth_rate: Factor by which penalties grow each iteration
        """
        self.weight_config = weight_config
        self.params = params
        self.optimizer = optimizer
        self.regression = regression
        self.max_iterations = max_iterations
        self.penalty_growth_rate = penalty_growth_rate
        self.simulator = SimulationLayer(params)

    def forward(self, price, power_init, head_init, flow_init):
        """
        Run recursive linearization with fixed weights.

        Args:
            price: Price tensor
            power_init: Initial power schedule
            head_init: Initial head schedule
            flow_init: Initial flow schedule

        Returns:
            Tuple containing all optimization and simulation results
        """
        # Get fixed weights
        w_p_base, w_q_base, w_h_base = self.weight_config.get_weights()

        # Initialize for first iteration
        p_current = power_init.clone().detach()
        h_current = head_init.clone().detach()
        flow_current = flow_init.clone().detach()

        # Store iteration results
        iter_results = []

        # Recursive linearization loop
        for iteration in range(self.max_iterations):
            # Apply growth to penalty weights
            growth_factor = self.penalty_growth_rate ** iteration
            w_p = w_p_base * growth_factor
            w_q = w_q_base * growth_factor
            w_h = w_h_base * growth_factor

            # Compute linearization coefficients
            c, d, e, a, b = self.regression.run_regression(p_current, h_current, flow_current)

            # Initialize OptiLayer
            self.optimizer.initialize_layer(p_current.cpu(), h_current.cpu(), flow_current.cpu())

            # Run optimization
            p_opt, q_opt, h_opt, v_opt, expected_profit, optimized_objective = self.optimizer.forward(
                price.cpu(), c.cpu(), d.cpu(), e.cpu(), a.cpu(), b.cpu(),
                p_current.cpu(), h_current.cpu(), flow_current.cpu(),
                w_p.cpu(), w_h.cpu(), w_q.cpu()
            )

            # Store iteration results
            iter_results.append({
                'iteration': iteration,
                'expected_profit': expected_profit.item(),
                'optimized_objective': optimized_objective.item(),
                'p_opt': p_opt.detach().cpu().numpy(),
                'q_opt': q_opt.detach().cpu().numpy(),
                'h_opt': h_opt.detach().cpu().numpy(),
                'growth_factor': growth_factor,
                'w_p_mean': w_p.mean().item(),
                'w_q_mean': w_q.mean().item(),
                'w_h_mean': w_h.mean().item()
            })

            # Update for next iteration
            if iteration < self.max_iterations - 1:
                p_current = p_opt.clone().detach().to(device=power_init.device)
                h_current = h_opt.clone().detach().to(device=head_init.device)
                flow_current = q_opt.clone().detach().to(device=flow_init.device)

        # Get device from initial data
        device = power_init.device

        # Run simulation with final optimized schedule
        p_sim, q_sim, h_sim, v_low_sim = self.simulator.simulate_operation(
            p_opt.to(device), q_opt.to(device), h_opt.to(device)
        )

        # Calculate ex-post profit
        ex_post_profit, SI_penalty, volume_penalty, operating_cost = self.simulator.calc_profit(
            p_sim, p_opt.to(device), v_low_sim, price.to(device)
        )

        return (ex_post_profit, expected_profit, p_opt, q_opt, h_opt, v_opt,
                p_sim, q_sim, h_sim, v_low_sim, SI_penalty, volume_penalty,
                operating_cost, iter_results)
