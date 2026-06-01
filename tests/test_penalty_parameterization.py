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


def test_calc_profit_volume_penalty_scales_with_multiplier():
    # Same trajectory with a terminal-volume DEFICIT; only the multiplier differs.
    DA = torch.tensor([10.0, 10.0])
    p_opt = torch.tensor([1.0, 1.0])
    p_sim = torch.tensor([1.0, 1.0])                  # no SI imbalance, isolate volume
    v_low = torch.tensor([300000.0, 360000.0])        # final volume 60000 above target
    _, _, vol_base, _ = _make_sim(vol_water_value_mult=1.0).calc_profit(
        p_sim, p_opt, v_low, DA)
    _, _, vol_scaled, _ = _make_sim(vol_water_value_mult=2.0).calc_profit(
        p_sim, p_opt, v_low, DA)
    # Non-zero penalty under a real deficit, and exactly doubled by a 2x multiplier.
    assert vol_base.item() > 0
    assert abs(vol_scaled.item() - 2.0 * vol_base.item()) < 1e-4


from DFL.scripts.rescore_miqp_penalties import rescore_schedule


def test_rescore_schedule_matches_calc_profit_revenue_and_opcost():
    # A fully feasible schedule (sim == opt) under default penalties.
    DA = [10.0, 20.0, 5.0]
    power = [1.0, -2.0, 0.0]
    out = rescore_schedule(
        power=power, DA_price=DA,
        si_shortage_mult=-2.0, si_surplus_mult=-0.5, vol_water_value_mult=1.0,
        operational_cost=0.4,
        final_volume=300000.0, target_vol_low=300000.0,  # no deficit
        rho=1000.0, g=9.81, mu=0.9, target_head=75.0,
    )
    # revenue = 10*1 + 20*-2 + 5*0 = -30 ; op = 0.4*(1+4+0)=2.0 ; SI=0 ; vol=0
    assert abs(out["revenue"] - (-30.0)) < 1e-6
    assert abs(out["operating_cost"] - 2.0) < 1e-6
    assert abs(out["SI_penalty"]) < 1e-6
    assert abs(out["volume_penalty"]) < 1e-6
    assert abs(out["ex_post_profit"] - (-32.0)) < 1e-6
