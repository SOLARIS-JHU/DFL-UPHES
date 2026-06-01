"""
HydroParameters class for storing hydropower system parameters.

This module contains the HydroParameters class which encapsulates all
physical and operational parameters of the pumped-storage hydropower system.
"""

import torch


class HydroParameters:
    """
    Container for all hydropower system parameters.

    Stores physical constants, operational limits, target values, and
    fitted functions for the pumped-storage hydropower optimization problem.
    """

    def __init__(
        self,
        time_horizon=24, # number of time periods
        sampling_rate=50, # number of samples for regression
        δ_p=0.5,
        δ_h=1,
        δ_q=0.5,
        operational_cost=0.4,
        si_shortage_mult=-2.0,
        si_surplus_mult=-0.5,
        vol_water_value_mult=1.0,
        rho=1000,
        g=9.81,
        mu=0.9,
        head_min=None,
        head_max=None,
        max_vol_up=None,
        min_vol_low=None,
        ramp_up=None,
        ramp_down=None,
        target_head=None,
        target_vol_low=None,
        head_init=None,
        v_low_init=None,
        neg_min_fit=None,
        neg_max_fit=None,
        pos_min_fit=None,
        pos_max_fit=None,
        neg_min=None,
        neg_max=None,
        pos_min=None,
        pos_max=None,
        predict_q_poly=None,
        h_to_v_low_fitted=None,
        gross_head=None,
        v_low_to_h_fitted=None,
        device=None,
    ):
        """
        Initialize HydroParameters with system specifications.

        Args:
            time_horizon: Number of time periods (default 24 hours)
            sampling_rate: Number of samples for Taylor regression
            δ_p, δ_h, δ_q: Penalty weight parameters
            operational_cost: Cost per unit of operation
            rho: Water density (kg/m³)
            g: Gravitational acceleration (m/s²)
            mu: Efficiency coefficient
            head_min, head_max: Min/max head constraints
            max_vol_up: Maximum upper reservoir volume
            min_vol_low: Minimum lower reservoir volume
            ramp_up, ramp_down: Ramping rate limits
            target_head, target_vol_low: Target end-of-horizon values
            head_init, v_low_init: Initial conditions
            neg_min_fit, neg_max_fit: Fitted power bounds (turbine mode)
            pos_min_fit, pos_max_fit: Fitted power bounds (pump mode)
            neg_min, neg_max: Power bounds (turbine mode)
            pos_min, pos_max: Power bounds (pump mode)
            predict_q_poly: Polynomial flow prediction function
            h_to_v_low_fitted: Head to lower volume fitted function
            gross_head: Gross head calculation function
            v_low_to_h_fitted: Lower volume to head fitted function
            device: PyTorch device (cpu or cuda)
        """
        if device is None:
            device = torch.device("cpu")

        self.time_horizon = time_horizon
        self.sampling_rate = sampling_rate
        self.operational_cost = operational_cost
        self.si_shortage_mult = si_shortage_mult
        self.si_surplus_mult = si_surplus_mult
        self.vol_water_value_mult = vol_water_value_mult
        self.device = device

        self.δ_p = torch.tensor(δ_p, dtype=torch.float32, device=device)
        self.δ_h = torch.tensor(δ_h, dtype=torch.float32, device=device)
        self.δ_q = torch.tensor(δ_q, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rho, dtype=torch.float32, device=device)
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)

        self.head_min = torch.tensor(head_min, dtype=torch.float32, device=device) if head_min is not None else None
        self.head_max = torch.tensor(head_max, dtype=torch.float32, device=device) if head_max is not None else None
        self.max_vol_up = torch.tensor(max_vol_up, dtype=torch.float32, device=device) if max_vol_up is not None else None
        self.min_vol_low = torch.tensor(min_vol_low, dtype=torch.float32, device=device) if min_vol_low is not None else None
        self.ramp_up = torch.tensor(ramp_up, dtype=torch.float32, device=device) if ramp_up is not None else None
        self.ramp_down = torch.tensor(ramp_down, dtype=torch.float32, device=device) if ramp_down is not None else None

        self.target_head = torch.tensor(target_head, dtype=torch.float32, device=device) if target_head is not None else None
        self.target_vol_low = torch.tensor(target_vol_low, dtype=torch.float32, device=device) if target_vol_low is not None else None

        if head_init is not None:
            self.head_init = head_init.clone().detach().to(device=device, dtype=torch.float32) if isinstance(head_init, torch.Tensor) else torch.tensor(head_init, dtype=torch.float32, device=device)
        else:
            self.head_init = None

        if v_low_init is not None:
            self.v_low_init = v_low_init.clone().detach().to(device=device, dtype=torch.float32) if isinstance(v_low_init, torch.Tensor) else torch.tensor(v_low_init, dtype=torch.float32, device=device)
        else:
            self.v_low_init = None

        self.neg_min_fit = torch.tensor(neg_min_fit, dtype=torch.float32, device=device) if neg_min_fit is not None else None
        self.neg_max_fit = torch.tensor(neg_max_fit, dtype=torch.float32, device=device) if neg_max_fit is not None else None
        self.pos_min_fit = torch.tensor(pos_min_fit, dtype=torch.float32, device=device) if pos_min_fit is not None else None
        self.pos_max_fit = torch.tensor(pos_max_fit, dtype=torch.float32, device=device) if pos_max_fit is not None else None

        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.predict_q_poly = predict_q_poly
        self.h_to_v_low_fitted = h_to_v_low_fitted
        self.gross_head = gross_head
        self.v_low_to_h_fitted = v_low_to_h_fitted

    def to_cpu(self):
        """Move all PyTorch tensors to CPU"""
        for attr_name in dir(self):
            # Skip private attributes, methods, and callable attributes
            if attr_name.startswith('_') or callable(getattr(self, attr_name)):
                continue

            try:
                attr = getattr(self, attr_name)

                # Handle tensors
                if isinstance(attr, torch.Tensor):
                    setattr(self, attr_name, attr.cpu())

                # Handle lists of tensors
                elif isinstance(attr, list):
                    new_list = []
                    for item in attr:
                        if isinstance(item, torch.Tensor):
                            new_list.append(item.cpu())
                        else:
                            new_list.append(item)
                    setattr(self, attr_name, new_list)

                # Handle dictionaries containing tensors
                elif isinstance(attr, dict):
                    new_dict = {}
                    for key, value in attr.items():
                        if isinstance(value, torch.Tensor):
                            new_dict[key] = value.cpu()
                        else:
                            new_dict[key] = value
                    setattr(self, attr_name, new_dict)
            except: # Skip attributes that cannot be accessed or operated on
                pass

        return self  # Return self to support method chaining
