%% Define pipeline class(draft)
class Pipeline(nn.Module):
    def __init__(
            self,
            time_horizon=24, # number of time periods
            UPC_sampling_rate=400, # number of samples for UPC regression
            δp=5,  # MW, power trust region
            δh=20, # m, head trust region
            δq=7,  # m^3/s, flow trust region
            operational_cost=3.8, # EUR/MWh
            rho=1000, # kg/m^3
            g=9.81, # m/s^2
            mu=0.9  # efficiency
        ):

        super(Pipeline, self).__init__() # Initialize parent class
        
        # Store parameters from imported data
        self.head_min = head_min
        self.head_max = head_max
        self.max_vol_up = max_vol_up
        self.min_vol_low = min_vol_low
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.target_head = target_head
        self.target_vol_low = target_vol_low
        self.head_init = head_init
        self.v_low_init = v_low_init
        
        # Store other parameters
        self.time_horizon = time_horizon
        self.UPC_sampling_rate = UPC_sampling_rate
        self.operational_cost = operational_cost
        self.rho = rho
        self.g = g
        self.mu = mu
        
        # Trust region parameters
        self.δp = δp
        self.δh = δh
        self.δq = δq

        # UPC bounds
        self.neg_min_fit = neg_min_fit
        self.neg_max_fit = neg_max_fit
        self.pos_min_fit = pos_min_fit
        self.pos_max_fit = pos_max_fit
        
        # Store imported functions
        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.predict_q_poly = predict_q_poly
        self.gross_head = gross_head

    def least_squares_UPC_torch(self, p_samples, h_samples, q_values):
        """
        Perform least squares regression for UPC using PyTorch operations
        
        Args:
            p_samples (torch.Tensor): Power samples
            h_samples (torch.Tensor): Head samples
            q_values (torch.Tensor): Flow values
            
        Returns:
            torch.Tensor: Regression coefficients [c, d, e] for q = c*p + d*h + e
        """
        X = torch.stack([p_samples, h_samples, torch.ones_like(p_samples)], dim=1)
        y = q_values.unsqueeze(1)
        XTX = torch.matmul(X.t(), X)
        XTX_inv = torch.inverse(XTX)
        XTy = torch.matmul(X.t(), y)
        beta = torch.matmul(XTX_inv, XTy)
        return beta.squeeze()

    def least_squares_v_low_torch(self, h_samples, v_low_samples):
        """
        Perform least squares regression for v_low using PyTorch operations
        
        Args:
            h_samples (torch.Tensor): Head samples
            v_low_samples (torch.Tensor): Lower reservoir volume samples
            
        Returns:
            torch.Tensor: Regression coefficients [a, b] for v_low = a*h + b
        """
        X = torch.stack([h_samples, torch.ones_like(h_samples)], dim=1)
        y = v_low_samples.unsqueeze(1)
        XTX = torch.matmul(X.t(), X)
        XTX_inv = torch.inverse(XTX)
        XTy = torch.matmul(X.t(), y)
        beta = torch.matmul(XTX_inv, XTy)
        return beta.squeeze()

    def regression_layer(self, power, head):
        """
        Perform regression analysis for UPC and v_low relationships
        
        Args:
            power (torch.Tensor): Power schedule [time_horizon]
            head (torch.Tensor): Head schedule [time_horizon]
            
        Returns:
            tuple: Tensors of regression coefficients (c, d, e) for UPC and (a, b) for v_low
        """
        c, d, e = {}, {}, {}  # UPC regression coefficients
        a, b = {}, {}         # v_low regression coefficients
        
        for t in range(self.time_horizon):
            # UPC regression
            h_samples = torch.linspace(
                max(self.head_min, head[t] - self.δh), 
                min(self.head_max, head[t] + self.δh), 
                self.UPC_sampling_rate
            )
            p_samples = torch.linspace(
                power[t] - self.δp, 
                power[t] + self.δp, 
                self.UPC_sampling_rate
            )
            
            # Create meshgrid of power and head samples
            p_mesh, h_mesh = torch.meshgrid(p_samples, h_samples, indexing='ij')
            p_flat = p_mesh.flatten()
            h_flat = h_mesh.flatten()
            
            # Create mask for valid points using imported fit coefficients
            mask = ((self.neg_min_fit[0] * h_flat + self.neg_min_fit[1] <= p_flat) &
                    (p_flat <= self.neg_max_fit[0] * h_flat + self.neg_max_fit[1])) | \
                ((self.pos_min_fit[0] * h_flat + self.pos_min_fit[1] <= p_flat) &
                    (p_flat <= self.pos_max_fit[0] * h_flat + self.pos_max_fit[1]))
            
            # Get valid points
            p_valid = p_flat[mask]
            h_valid = h_flat[mask]
            
            if p_valid.numel() > 0:
                # Calculate q values using imported predict_q_poly function
                q_values = torch.tensor([
                    self.predict_q_poly(p.item(), h.item()) 
                    for p, h in zip(p_valid, h_valid)
                ], dtype=torch.float32)
                
                # Perform UPC regression
                beta = self.least_squares_UPC_torch(p_valid, h_valid, q_values)
                c[t], d[t], e[t] = beta.tolist()
            else:
                c[t], d[t], e[t] = 0, 0, 0  # Default values if no valid points
            
            # v_low regression
            h_samples = torch.linspace(
                max(self.head_min, head[t] - self.δh),
                min(self.head_max, head[t] + self.δh),
                self.UPC_sampling_rate
            )
            
            # Calculate v_low samples using imported h_to_v_low_fitted function
            v_low_samples = torch.tensor([
                self.h_to_v_low_fitted(h.item()) 
                for h in h_samples
            ], dtype=torch.float32)
            
            # Perform v_low regression
            beta = self.least_squares_v_low_torch(h_samples, v_low_samples)
            a[t], b[t] = beta.tolist()
        
        # Convert coefficient dictionaries to tensors (require grad?)
        c_tensor = torch.tensor([c[t] for t in range(self.time_horizon)], dtype=torch.float32)
        d_tensor = torch.tensor([d[t] for t in range(self.time_horizon)], dtype=torch.float32)
        e_tensor = torch.tensor([e[t] for t in range(self.time_horizon)], dtype=torch.float32)
        a_tensor = torch.tensor([a[t] for t in range(self.time_horizon)], dtype=torch.float32)
        b_tensor = torch.tensor([b[t] for t in range(self.time_horizon)], dtype=torch.float32)
        
        return c_tensor, d_tensor, e_tensor, a_tensor, b_tensor

    def optimization_layer(self, c, d, e, a, b, power, head, DA_prices):
        """
        Set up and solve the CVXPY optimization problem
        
        Args:
            c, d, e (torch.Tensor): UPC regression coefficients for q = c*p + d*h + e
            a, b (torch.Tensor): v_low regression coefficients for v_low = a*h + b
            power (torch.Tensor): Current power schedule [time_horizon]
            head (torch.Tensor): Current head schedule [time_horizon]
            DA_prices (torch.Tensor): Day-ahead prices [time_horizon]
            
        Returns:
            tuple: Optimized power and flow schedules
        """
        # Define variables
        p = cp.Variable(self.time_horizon)
        q = cp.Variable(self.time_horizon)
        h = cp.Variable(self.time_horizon)
        v_low = cp.Variable(self.time_horizon)
        
        # Define parameters
        DA_price = cp.Parameter(self.time_horizon)
        DA_price.value = DA_prices
        
        # objective function: revenue - operational costs
        revenue = DA_price @ p
        operational_costs = self.operational_cost * cp.sum_squares(p)
        objective = cp.Maximize(revenue - operational_costs)
        
        # Define constraints
        constraints = []
        
        # Add constraints for each time period
        for t in range(self.time_horizon):
            
            # Power bounds based on mode
            if power[t] == 0:
                constraints += [
                    p[t] == 0,
                    q[t] == 0
                ]
            elif power[t] >= 0:
                constraints += [
                    p[t] >= self.pos_min_fit[0] * h[t] + self.pos_min_fit[1],
                    p[t] <= self.pos_max_fit[0] * h[t] + self.pos_max_fit[1]
                ]
            elif power[t] <= 0:
                constraints += [
                    p[t] >= self.neg_min_fit[0] * h[t] + self.neg_min_fit[1],
                    p[t] <= self.neg_max_fit[0] * h[t] + self.neg_max_fit[1]
                ]
            
            constraints += [
                q[t] == c[t] * p[t] + d[t] * h[t] + e[t],
                v_low[t] == a[t] * h[t] + b[t],
                h[t] >= self.head_min,
                h[t] <= self.head_max,
                p[t] <= power[t] + self.δp,
                p[t] >= power[t] - self.δp,
                q[t] <= q[t] + self.δq,
                q[t] >= q[t] - self.δq,
                h[t] <= head[t] + self.δh,
                h[t] >= head[t] - self.δh
            ]

            # Volume balance constraints
            if t > 0:
                constraints += [
                    v_low[t] == v_low[t-1] + q[t] * 3600  # Convert flow rate (m³/s) to volume (m³)
                ]
            else:
                # For the first hour, use the initial volume calculated from h_init
                constraints += [
                    v_low[0] == self.v_low_init + q[0] * 3600  # Convert flow rate (m³/s) to volume (m³)
                ]
            
        
        # Final volume constraint
        constraints =+ [v_low[self.time_horizon-1] <= self.target_vol_low]
        
        # Create and solve the problem
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        
        # Create CVXPY layer
        cvxpy_layer = CvxpyLayer(problem, parameters=[DA_price], variables=[p, q, h, v_low])

        # Call the layer with DA_prices
        p_opt, q_opt, h_opt, v_low_opt = cvxpy_layer(DA_prices)

        return p_opt, q_opt, h_opt, v_low_opt

    def simulate_operation(self, p, q, h):
        """
        Simulate system operation with optimized schedules.

        Args:
            p (torch.Tensor): Optimized power schedule from optimization_layer (shape [time_horizon])
            q (torch.Tensor): Optimized flow schedule from optimization_layer (shape [time_horizon])
            h (torch.Tensor): Optimized head schedule from optimization_layer (shape [time_horizon])

        Returns:
            tuple: Calibrated power, flow, head, and volume trajectories (minute-wise)
        """
        # Repeat the hourly schedules to minute-wise resolution
        p_sim = p.repeat_interleave(60)
        q_sim = q.repeat_interleave(60)
        h_sim = h.repeat_interleave(60)

        num_minutes = len(p_sim)
        
        # Initialize calibrated simulation arrays
        p_sim_clb = p_sim.clone()
        q_sim_clb = torch.zeros(num_minutes + 1)
        h_sim_clb = torch.zeros(num_minutes + 1)
        v_low_clb = torch.zeros(num_minutes + 1)

        # Add the new element to the tensor (time 0 next day)
        p_sim_clb = torch.cat([p_sim_clb, p_sim_clb[-1].unsqueeze(0)])

        # Idle for 1 min before changing modes
        for i in range(num_minutes - 1, 0, -1):
            if p_sim_clb[i] * p_sim_clb[i - 1] < 0:
                p_sim_clb[i - 1] = 0

        # Iterate backwards through the day, adjusting power values
        for hour in range(self.time_horizon-1, -1, -1):  # from 23 to 0
            hour_start = hour * 60
            hour_end = hour_start + 60

            # Ensure the first minute of the hour matches p[hour]
            p_sim_clb[hour_start] = p[hour]

            # Backward adjustment for the rest of the hour
            for i in range(hour_end - 1, hour_start, -1):
                if p_sim_clb[i] - p_sim_clb[i - 1] > self.ramp_down:
                    p_sim_clb[i - 1] = p_sim_clb[i] - self.ramp_down
                elif p_sim_clb[i] - p_sim_clb[i - 1] < -self.ramp_up:
                    p_sim_clb[i - 1] = p_sim_clb[i] + self.ramp_up

        # Initialize the first elements
        q_sim_clb[0] = q_sim[0]
        h_sim_clb[0] = h_sim[0]
        v_low_clb[0] = self.h_to_v_low_fitted(h_sim_clb[0])

        # Calibrate with real reservoir properties
        for i in range(num_minutes):
            # Turbine mode
            if p_sim_clb[i] > 0:
                pos_min_value = self.pos_min(h_sim_clb[i])
                pos_max_value = self.pos_max(h_sim_clb[i])

                # Predict the flow based on the current state
                if pos_min_value <= p_sim_clb[i] <= pos_max_value:
                    q_sim_clb[i] = self.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
                elif p_sim_clb[i] < pos_min_value:
                    p_sim_clb[i] = pos_min_value
                    q_sim_clb[i] = self.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
                elif p_sim_clb[i] > pos_max_value:
                    p_sim_clb[i] = pos_max_value
                    q_sim_clb[i] = self.predict_q_poly(p_sim_clb[i], h_sim_clb[i])

            # Pump mode
            elif p_sim_clb[i] < 0:
                neg_min_value = self.neg_min(h_sim_clb[i])
                neg_max_value = self.neg_max(h_sim_clb[i])

                if neg_min_value <= p_sim_clb[i] <= neg_max_value:
                    q_sim_clb[i] = self.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
                elif p_sim_clb[i] < neg_min_value:
                    p_sim_clb[i] = neg_min_value
                    q_sim_clb[i] = self.predict_q_poly(p_sim_clb[i], h_sim_clb[i])
                elif p_sim_clb[i] > neg_max_value:
                    p_sim_clb[i] = neg_max_value
                    q_sim_clb[i] = self.predict_q_poly(p_sim_clb[i], h_sim_clb[i])

            else:
                # Idle mode
                q_sim_clb[i] = 0

            # Update the volume of the lower reservoir
            v_low_clb[i + 1] = v_low_clb[i] + q_sim_clb[i] * 60  # Convert flow to volume (assuming q in m^3/s)

            # Check if v_low_clb is within limits
            if v_low_clb[i + 1] > self.max_vol_up or v_low_clb[i + 1] < self.min_vol_low:
                # Set to idle mode if out of bounds
                p_sim_clb[i] = 0
                q_sim_clb[i] = 0
                h_sim_clb[i + 1] = h_sim_clb[i]
                v_low_clb[i + 1] = v_low_clb[i]  # No change in volume
            else:
                # Update head for valid volume
                h_sim_clb[i + 1] = self.gross_head(v_low_clb[i + 1])

        # Return the calibrated minute-wise trajectories
        return p_sim_clb[:-1], q_sim_clb[:-1], h_sim_clb[:-1], v_low_clb[:-1]


    def calculate_profit(self, p_sim_clb, p_opt, v_low_clb, DA_price_quarter):
        """
        Calculate profit from simulation results.

        Args:
            p_sim_clb (torch.Tensor): Calibrated minute-wise power trajectory (length 1441)
            p_opt (torch.Tensor): Optimized hourly power schedule from optimization_layer (length 24)
            v_low_clb (torch.Tensor): Simulated volume trajectory (minute-wise, length 1441)
            DA_price_quarter (torch.Tensor): Day-ahead prices (quarter-hourly, length 96)

        Returns:
            tuple: Total daily profit, System Imbalance penalty, simulated and optimized energy per quarter-hour
        """
        # Truncate the last element of p_sim_clb to match the 1440 minutes in a day
        p_sim_clb = p_sim_clb[:-1]  # Remove the last element

        # Expand p_opt from hourly to minute-wise by repeating each value 60 times
        p_opt_minute = p_opt.repeat_interleave(60)

        # Sum every 15 minutes to aggregate the minute-wise data to quarter-hourly totals
        e_sim_quarter = p_sim_clb.view(-1, 15).sum(dim=1) * 0.25  # Convert MW to MWh for each quarter-hour
        e_opt_quarter = p_opt_minute.view(-1, 15).sum(dim=1) * 0.25  # Convert MW to MWh for each quarter-hour

        # Calculate the revenue for each quarter-hour
        revenue_per_quarter = DA_price_quarter * e_sim_quarter  # Revenue calculation in EUR

        # Determine the System Imbalance (SI) price
        surplus_penalty_multiplier = -0.5
        shortage_penalty_multiplier = -2

        SI_price = torch.where(e_sim_quarter < e_opt_quarter,  # Shortage in simulation
                            shortage_penalty_multiplier * DA_price_quarter,  # Lower output penalty
                            surplus_penalty_multiplier * DA_price_quarter)  # Higher output penalty

        # Calculate the penalty for each quarter-hour
        penalty_per_quarter = (e_sim_quarter - e_opt_quarter) * SI_price  # Penalty calculation adjusted for MWh

        # Sum the penalties over all quarter-hours to get the total penalty
        SI_penalty = penalty_per_quarter.sum()

        # Calculate volume penalty
        volume_deficit = max(0, v_low_clb[-1] - self.target_vol_low)  # Ensure no penalty if above target
        energy_loss = self.rho * volume_deficit * self.g * self.target_head * self.mu / 3.6e9  # Convert from J to MWh
        volume_penalty = energy_loss * torch.max(DA_price_quarter)

        # Calculate the operating cost
        operating_cost = self.operational_cost * torch.sum(p_sim_clb ** 2) / 60  # Operating cost in EUR

        # Calculate total daily profit
        total_daily_profit = revenue_per_quarter.sum() - SI_penalty - volume_penalty - operating_cost

        return total_daily_profit


    def forward(self, power, head, DA_prices):
        """
        Forward pass through the pipeline
        
        Args:
            power (torch.Tensor): Initial power schedule [time_horizon]
            head (torch.Tensor): Initial head schedule [time_horizon]
            DA_prices (torch.Tensor): Day-ahead prices [time_horizon]
            
        Returns:
            tuple: Profit and various schedules (power, flow, head, volume)
        """
        # Perform regression analysis
        c, d, e, a, b = self.regression_layer(power, head)
        
        # Optimize schedules
        p_opt, q_opt = self.optimization_layer(c, d, e, a, b, power, head, DA_prices)
        
        # Simulate operation
        p_sim, q_sim, h_sim, v_low = self.simulate_operation(p_opt, q_opt)
        
        # Calculate profit
        profit = self.calculate_profit(p_sim, q_sim, h_sim, v_low, DA_prices)
        
        return profit, p_opt, q_opt, p_sim, q_sim, h_sim, v_low
    


