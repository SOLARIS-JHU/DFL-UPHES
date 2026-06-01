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
