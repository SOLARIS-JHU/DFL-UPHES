# tests/test_penalty_parameterization.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DFL.core.parameters import HydroParameters


def test_penalty_fields_default_to_current_values():
    p = HydroParameters()
    assert p.si_shortage_mult == -2.0
    assert p.si_surplus_mult == -0.5
    assert p.vol_water_value_mult == 1.0


def test_penalty_fields_are_overridable():
    p = HydroParameters(si_shortage_mult=-1.0, si_surplus_mult=-1.0,
                        vol_water_value_mult=0.8)
    assert p.si_shortage_mult == -1.0
    assert p.si_surplus_mult == -1.0
    assert p.vol_water_value_mult == 0.8


import torch
from DFL.core.layers import SimulationLayer


def _make_sim(**penalty_overrides):
    # Minimal params: calc_profit needs operational_cost, rho, g, mu,
    # target_head, target_vol_low, and the penalty fields.
    p = HydroParameters(
        operational_cost=0.4, rho=1000, g=9.81, mu=0.9,
        target_head=75.0, target_vol_low=300000.0,
        **penalty_overrides,
    )
    return SimulationLayer(p)


def test_calc_profit_symmetric_si_zeroes_asymmetry():
    # Equal multipliers => SI_price magnitude identical for surplus and shortage.
    sim = _make_sim(si_shortage_mult=-1.0, si_surplus_mult=-1.0)
    DA = torch.tensor([10.0, 10.0])
    p_opt = torch.tensor([1.0, 1.0])
    p_sim = torch.tensor([2.0, 0.0])           # +1 surplus, -1 shortage
    v_low = torch.tensor([300000.0, 300000.0]) # no volume deficit
    _, si, _, _ = sim.calc_profit(p_sim, p_opt, v_low, DA)
    # imbalance*-1*DA summed: (+1*-1*10) + (-1*-1*10) = -10 + 10 = 0
    assert abs(si.item()) < 1e-4


def test_calc_profit_default_si_matches_legacy_constants():
    sim = _make_sim()  # defaults -2.0 / -0.5
    DA = torch.tensor([10.0, 10.0])
    p_opt = torch.tensor([1.0, 1.0])
    p_sim = torch.tensor([2.0, 0.0])
    v_low = torch.tensor([300000.0, 300000.0])
    _, si, _, _ = sim.calc_profit(p_sim, p_opt, v_low, DA)
    # surplus hour: +1 * (-0.5*10) = -5 ; shortage hour: -1 * (-2.0*10) = +20
    assert abs(si.item() - 15.0) < 1e-4
