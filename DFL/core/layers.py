"""
Neural network layers for hydropower optimization.

This module contains the core computational layers:
- TaylorRegressionLayer: First-order Taylor approximation for nonlinear functions
- OptiLayer: CVXPY-based convex optimization layer
- SimulationLayer: Physical simulation with operational constraints
"""

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from .parameters import HydroParameters


class TaylorRegressionLayer:
    """
    Performs first-order Taylor approximation on nonlinear hydropower functions.

    Linearizes the flow-power-head relationship and volume-head relationship
    around operational points to enable convex optimization.
    """

    def __init__(self, params: HydroParameters):
        """
        Initialize TaylorRegressionLayer.

        Args:
            params: HydroParameters instance with system specifications
        """
        self.params = params

    def calculate_gradients(self, func, x, create_graph=False, retain_graph=False):
        """
        Calculate gradients of a function with respect to its inputs.

        Args:
            func: A callable that computes the output
            x: Input tensor with requires_grad=True
            create_graph: Whether to create a computational graph for the gradient
            retain_graph: Whether to retain the computational graph

        Returns:
            Gradient of func with respect to x
        """
        try:
            y = func(x)

            # Ensure y is a scalar for gradient computation
            if y.numel() == 1:
                grad_output = torch.ones_like(y)
            else:
                grad_output = torch.ones_like(y)

            grad = torch.autograd.grad(
                outputs=y,
                inputs=x,
                create_graph=create_graph,
                retain_graph=retain_graph,
                grad_outputs=grad_output,
                allow_unused=True
            )[0]

            # Handle case where gradient is None (no connection between input and output)
            if grad is None:
                grad = torch.zeros_like(x)

            return grad
        except Exception as e:
            print(f"Error in gradient calculation: {e}")
            print(f"  Input shape: {x.shape}, Input value: {x.item() if x.numel() == 1 else x}")
            # Return zero gradient on error
            return torch.zeros_like(x, requires_grad=False)

    def run_regression(self, power, head, flow=None):
        """
        Run Taylor regression to linearize the nonlinear functions
        at each operational point.

        Args:
            power (torch.Tensor): Power values [time_horizon]
            head (torch.Tensor): Head values [time_horizon]
            flow (torch.Tensor, optional): Flow values [time_horizon]

        Returns:
            tuple: (c, d, e, a, b) tensors for the linearized equations:
                  q = c*p + d*h + e
                  v_low = a*h + b
        """
        TH = self.params.time_horizon
        device = power.device
        c_list, d_list, e_list = [], [], []
        a_list, b_list = [], []

        # Process each time step individually
        for t in range(TH):
            try:
                # Get operational point
                p0 = power[t].detach().clone().requires_grad_(True)
                h0 = head[t].detach().clone().requires_grad_(True)

                # Skip computation for idle mode
                if abs(p0.item()) < 0.01:  # Close to zero power
                    c_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                    d_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                    e_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                else:
                    # Create function to compute q given p with fixed h0
                    def q_given_p(p):
                        return self.params.predict_q_poly(p.unsqueeze(0), h0.unsqueeze(0)).squeeze(0)

                    # Create function to compute q given h with fixed p0
                    def q_given_h(h):
                        return self.params.predict_q_poly(p0.unsqueeze(0), h.unsqueeze(0)).squeeze(0)

                    # Compute gradients (partial derivatives)
                    dq_dp = self.calculate_gradients(q_given_p, p0, retain_graph=True)
                    dq_dh = self.calculate_gradients(q_given_h, h0, retain_graph=True)

                    # Compute q0 at the operating point
                    q0 = self.params.predict_q_poly(p0.unsqueeze(0), h0.unsqueeze(0)).squeeze(0).detach()

                    # Compute Taylor coefficients for q = c*p + d*h + e
                    # Using first-order Taylor expansion around (p0, h0):
                    # q(p,h) ≈ q(p0,h0) + (∂q/∂p)(p-p0) + (∂q/∂h)(h-h0)
                    # Rearranged as: q(p,h) ≈ (∂q/∂p)*p + (∂q/∂h)*h + [q(p0,h0) - (∂q/∂p)*p0 - (∂q/∂h)*h0]
                    c = dq_dp.detach()  # corresponds to ∂q/∂p
                    d = dq_dh.detach()  # corresponds to ∂q/∂h
                    e = q0 - c * p0.detach() - d * h0.detach()  # constant term

                    c_list.append(c)
                    d_list.append(d)
                    e_list.append(e)

                # Create function to compute v_low given h
                def v_low_given_h(h):
                    return self.params.h_to_v_low_fitted(h)

                # Compute derivative dv_low/dh at h0
                dv_low_dh = self.calculate_gradients(v_low_given_h, h0, retain_graph=False)

                # Compute v_low0 at the operating point
                v_low0 = self.params.h_to_v_low_fitted(h0).detach()

                # Compute Taylor coefficients for v_low = a*h + b
                # Using first-order Taylor expansion around h0:
                # v_low(h) ≈ v_low(h0) + (dv_low/dh)(h-h0)
                # Rearranged as: v_low(h) ≈ (dv_low/dh)*h + [v_low(h0) - (dv_low/dh)*h0]
                a = dv_low_dh.detach()  # corresponds to dv_low/dh
                b = v_low0 - a * h0.detach()  # constant term

                a_list.append(a)
                b_list.append(b)

            except Exception as e:
                # On error, add default values to maintain consistent list lengths
                print(f"Error at time step {t}: {e}")
                c_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                d_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                e_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                a_list.append(torch.tensor(0.0, device=device, requires_grad=True))
                b_list.append(torch.tensor(0.0, device=device, requires_grad=True))

        try:
            # Stack results with gradient tracking
            c_tensor = torch.stack(c_list)
            d_tensor = torch.stack(d_list)
            e_tensor = torch.stack(e_list)
            a_tensor = torch.stack(a_list)
            b_tensor = torch.stack(b_list)
        except RuntimeError as e:
            # Handle stacking errors
            print(f"Stacking error: {e}. Attempting to create tensors manually.")

            # Convert to simple Python floats and recreate tensors
            c_values = [float(c.item()) for c in c_list]
            d_values = [float(d.item()) for d in d_list]
            e_values = [float(e.item()) for e in e_list]
            a_values = [float(a.item()) for a in a_list]
            b_values = [float(b.item()) for b in b_list]

            c_tensor = torch.tensor(c_values, device=device, requires_grad=True)
            d_tensor = torch.tensor(d_values, device=device, requires_grad=True)
            e_tensor = torch.tensor(e_values, device=device, requires_grad=True)
            a_tensor = torch.tensor(a_values, device=device, requires_grad=True)
            b_tensor = torch.tensor(b_values, device=device, requires_grad=True)

        return c_tensor, d_tensor, e_tensor, a_tensor, b_tensor


class OptiLayer:
    """
    CVXPY-based optimization layer for convex hydropower scheduling.

    Constructs and solves a convex optimization problem with penalty terms
    for deviation from previous iteration's solution (for recursive linearization).
    """

    def __init__(self, params: HydroParameters):
        """
        Initialize OptiLayer.

        Args:
            params: HydroParameters instance with system specifications
        """
        self.params = params.to_cpu()
        self.layer = None
        self.power_init = None
        self.head_init = None
        self.flow_init = None

    def initialize_layer(self, power, head, flow):
        """
        Only build the CVXPY problem if needed.
        """
        if (self.layer is None
            or not torch.allclose(self.power_init, power)
            or not torch.allclose(self.head_init, head)):

            self.power_init = power.detach().cpu()
            self.head_init = head.detach().cpu()
            self.flow_init = flow.detach().cpu()
            self.layer = self._build_cvxpy()

    def _build_cvxpy(self):
        TH = self.params.time_horizon
        # Define CVXPY variables
        p_var = cp.Variable(TH)
        q_var = cp.Variable(TH)
        h_var = cp.Variable(TH)
        v_low_var = cp.Variable(TH)

        # Define CVXPY parameters
        DA_price_param = cp.Parameter(TH)
        c_param = cp.Parameter(TH)
        d_param = cp.Parameter(TH)
        e_param = cp.Parameter(TH)
        a_param = cp.Parameter(TH)
        b_param = cp.Parameter(TH)
        w_p_param = cp.Parameter(TH, nonneg=True)
        w_h_param = cp.Parameter(TH, nonneg=True)
        w_q_param = cp.Parameter(TH, nonneg=True)

        # Warm starts
        p_var.value = self.power_init.tolist()
        h_var.value = self.head_init.tolist()

        # Objective
        revenue = DA_price_param @ p_var
        cost = self.params.operational_cost * cp.sum_squares(p_var)

        # Penalty terms: element-wise multiply then sum
        power_dev_pen = cp.sum(cp.multiply(w_p_param, cp.square(p_var - self.power_init)))
        head_dev_pen = cp.sum(cp.multiply(w_h_param, cp.square(h_var - self.head_init)))
        flow_dev_pen = cp.sum(cp.multiply(w_q_param, cp.square(q_var - self.flow_init)))

        objective = cp.Maximize(
            revenue
            - cost
            - power_dev_pen
            - head_dev_pen
            - flow_dev_pen
        )

        # Constraints
        constraints = []
        for t in range(TH):
            # Mode constraints based on sign of power_init
            if self.power_init[t] == 0:
                constraints += [p_var[t] == 0, q_var[t] == 0]
            elif self.power_init[t] > 0:  # Turbine
                constraints += [
                    p_var[t] >= self.params.pos_min_fit[0] * h_var[t] + self.params.pos_min_fit[1],
                    p_var[t] <= self.params.pos_max_fit[0] * h_var[t] + self.params.pos_max_fit[1],
                    q_var[t] == c_param[t] * p_var[t] + d_param[t]*h_var[t] + e_param[t],
                ]
            else:  # Pump
                constraints += [
                    p_var[t] >= self.params.neg_min_fit[0] * h_var[t] + self.params.neg_min_fit[1],
                    p_var[t] <= self.params.neg_max_fit[0] * h_var[t] + self.params.neg_max_fit[1],
                    q_var[t] == c_param[t] * p_var[t] + d_param[t]*h_var[t] + e_param[t],
                ]

            # Head and volume constraints
            constraints += [
                h_var[t] >= self.params.head_min,
                h_var[t] <= self.params.head_max,
                v_low_var[t] == a_param[t] * h_var[t] + b_param[t],
            ]

            # Volume balance
            if t == 0:
                constraints += [v_low_var[0] == self.params.v_low_init + q_var[0] * 3600]
            else:
                constraints += [v_low_var[t] == v_low_var[t-1] + q_var[t] * 3600]

        # Final volume constraint
        constraints += [v_low_var[TH-1] <= self.params.target_vol_low]

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        layer = CvxpyLayer(
            problem,
            parameters=[DA_price_param, c_param, d_param, e_param, a_param, b_param, w_p_param, w_h_param, w_q_param],
            variables=[p_var, q_var, h_var, v_low_var]
        )
        return layer

    def forward(self, DA_prices, c, d, e, a, b, power, head, flow, w_p, w_h, w_q):
        self.initialize_layer(power, head, flow)

        DA_prices_cpu = DA_prices.cpu()
        c_cpu = c.cpu()
        d_cpu = d.cpu()
        e_cpu = e.cpu()
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        w_p_cpu = w_p.cpu()
        w_h_cpu = w_h.cpu()
        w_q_cpu = w_q.cpu()

        try:
            # Solve with more robust settings
            (p_opt, q_opt, h_opt, v_opt) = self.layer(
                DA_prices_cpu, c_cpu, d_cpu, e_cpu, a_cpu, b_cpu,
                w_p_cpu, w_h_cpu, w_q_cpu,
                solver_args={
                    "solve_method": "ECOS",
                    "max_iters": 200000,  # Increased iterations
                    "reltol": 1e-5,       # Tighter tolerances
                    "abstol": 1e-5,
                    "feastol": 1e-5,
                    "verbose": True
                }
            )
        except Exception as er:
            print(f"\n⚠️ Solver error: {er}")
            print("Problematic parameters:")
            print(f"DA_prices: {DA_prices.detach().cpu().numpy().round(2)}")
            print(f"c: {c.detach().cpu().numpy().round(2)}")
            print(f"d: {d.detach().cpu().numpy().round(2)}")
            print(f"e: {e.detach().cpu().numpy().round(2)}")
            print(f"a: {a.detach().cpu().numpy().round(2)}")
            print(f"b: {b.detach().cpu().numpy().round(2)}")
            print(f"w_p: {w_p.detach().cpu().numpy().round(2)}")
            print(f"w_h: {w_h.detach().cpu().numpy().round(2)}")
            print(f"w_q: {w_q.detach().cpu().numpy().round(2)}\n")
            raise

        # Check for numerical issues
        if any(torch.isnan(tensor).any() for tensor in [p_opt, q_opt, h_opt, v_opt]):
            print("\n❌ NaN detected in solution. Parameters:")
            print(f"c[0]: {c[0].item():.2f}, d[0]: {d[0].item():.2f}, e[0]: {e[0].item():.2f}")
            print(f"w_p[0]: {w_p[0].item():.2f}, w_h[0]: {w_h[0].item():.2f}, w_q[0]: {w_q[0].item():.2f}")

        # Threshold processing - adjust values close to zero to exactly 0
        threshold = 0.1
        p_opt_thresholded = torch.where(torch.abs(p_opt) < threshold, torch.zeros_like(p_opt), p_opt)
        q_opt_thresholded = torch.where(torch.abs(q_opt) < threshold, torch.zeros_like(q_opt), q_opt)

        # Calculate profit from the optimization
        revenue = torch.sum(DA_prices_cpu * p_opt_thresholded)
        operating_cost = self.params.operational_cost * torch.sum(p_opt_thresholded**2)

        # Calculate the penalty terms from the objective function
        power_dev_pen = torch.sum(w_p_cpu * torch.square(p_opt_thresholded - self.power_init))
        head_dev_pen = torch.sum(w_h_cpu * torch.square(h_opt - self.head_init))
        flow_dev_pen = torch.sum(w_q_cpu * torch.square(q_opt_thresholded - self.flow_init))

        # Calculate the complete objective function value
        optimized_objective = revenue - operating_cost - power_dev_pen - head_dev_pen - flow_dev_pen

        # Calculate the profit without penalties (for comparison with simulator)
        optimized_profit = revenue - operating_cost

        return p_opt_thresholded, q_opt_thresholded, h_opt, v_opt, optimized_profit, optimized_objective


class SimulationLayer:
    """
    Physical simulation layer with operational constraints.

    Simulates hourly operation respecting physical feasibility constraints
    and calculates ex-post profit accounting for imbalances and penalties.
    """

    def __init__(self, params):
        """
        Initialize SimulationLayer.

        Args:
            params: HydroParameters instance with system specifications
        """
        self.params = params

    def simulate_operation(self, p, q, h):
        """
        Simulate hourly operation with physical constraints.

        Args:
            p (torch.Tensor): Hourly power schedule [time_horizon]
            q (torch.Tensor): Hourly flow schedule [time_horizon] (not directly used, recalculated)
            h (torch.Tensor): Hourly head schedule [time_horizon] (from optimization, for reference)

        Returns:
            tuple: Calibrated hourly (p, q, h, v_low) schedules.
        """
        TH = self.params.time_horizon

        # Initialize lists for each state
        p_list = []
        q_list = []
        h_list = []
        v_list = []

        # Start states - use initial conditions
        v_current = self.params.v_low_init  # Initial reservoir volume
        h_current = self.params.head_init   # Initial head value

        v_list.append(v_current)
        h_list.append(h_current)  # Store initial head

        for i in range(TH):
            p_current = p[i]

            # a) Base: idle => q=0
            q_candidate = torch.zeros_like(p_current)
            p_clamped = p_current

            # b) For turbine mode (p_current>0), clamp p between pos_min(h) and pos_max(h)
            #    then get q via polynomial using CURRENT head (not optimized head)
            if p_current > 0.5:  # Turbine mode
                p_min_turb = self.params.pos_min(h_current)  # Use current head
                p_max_turb = self.params.pos_max(h_current)  # Use current head
                p_clamped = torch.clamp(p_current, min=p_min_turb, max=p_max_turb)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)

            # c) For pump mode (p_current<0), clamp p between neg_min(h) and neg_max(h)
            elif p_current < -0.5:  # Pump mode
                p_min_pump = self.params.neg_min(h_current)  # Use current head
                p_max_pump = self.params.neg_max(h_current)  # Use current head
                p_clamped = torch.clamp(p_current, min=p_min_pump, max=p_max_pump)
                q_candidate = self.params.predict_q_poly(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)

            # Update volume: v_next = v_current + q * 3600 (seconds in an hour)
            v_next = v_current + q_candidate * 3600

            # Check if volume is within bounds
            out_of_bounds = (v_next > self.params.max_vol_up) | (v_next < self.params.min_vol_low)

            # If out of bounds, set to idle mode
            if out_of_bounds:
                p_final = torch.zeros_like(p_current)
                q_final = torch.zeros_like(q_candidate)
                v_next = v_current  # No change to volume
                h_next = h_current  # No change to head
            else:
                p_final = p_clamped if p_current != 0 else torch.zeros_like(p_current)
                q_final = q_candidate
                # Update head based on new volume
                h_next = self.params.v_low_to_h_fitted(v_next)

            # Append states for this hour
            p_list.append(p_final)
            q_list.append(q_final)

            # Update current states for next iteration
            v_current = v_next
            h_current = h_next  # Important: update h_current for next iteration

            v_list.append(v_current.item())
            h_list.append(h_current)

        # Convert lists to tensors
        p_sim = torch.stack(p_list)
        q_sim = torch.stack(q_list)
        h_sim = torch.stack(h_list[:-1])  # Remove the extra head value (we have TH+1 heads)
        v_low_sim = torch.tensor(v_list[:-1], dtype=torch.float32)  # Remove extra volume

        return p_sim, q_sim, h_sim, v_low_sim

    def calc_profit(self, p_sim, p_opt, v_low_sim, DA_price):
        """
        Calculate the daily profit from the hourly simulation.
        """
        # Calculate energy per hour (MWh)
        e_sim = p_sim  # Already in MW, and we're using hourly intervals

        # Calculate revenue
        revenue = torch.sum(DA_price * e_sim)

        # Determine the System Imbalance (SI) price
        surplus_penalty_multiplier = -0.5
        shortage_penalty_multiplier = -2.0

        SI_price = torch.where(
            e_sim < p_opt,  # Shortage in simulation
            shortage_penalty_multiplier * DA_price,  # Lower output penalty
            surplus_penalty_multiplier * DA_price  # Higher output penalty
        )

        # Calculate imbalance penalty
        imbalance = e_sim - p_opt
        penalty = imbalance * SI_price
        SI_penalty = penalty.sum()

        # Volume penalty - if final volume exceeds target
        volume_deficit = max(0, v_low_sim[-1] - self.params.target_vol_low)
        energy_loss = self.params.rho * volume_deficit * self.params.g * self.params.target_head * self.params.mu / 3.6e9  # Convert J to MWh
        volume_penalty = energy_loss * torch.median(DA_price)

        # Operating cost
        operating_cost = self.params.operational_cost * torch.sum(p_sim**2)

        # Total profit
        total_profit = revenue - operating_cost - SI_penalty - volume_penalty

        return total_profit, SI_penalty, volume_penalty, operating_cost
