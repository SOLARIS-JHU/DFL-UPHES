#!/usr/bin/env python3
"""
Re-score MIQP-PW schedules under alternative penalty settings (Reviewer 3, Issue 2).

MIQP-PW has no SI/volume terms in its objective; its schedule is feasible, so the
ex-post simulated power equals the optimized power. We therefore re-score directly
from the saved `power` and `price` columns, mirroring SimulationLayer.calc_profit:

    profit = revenue - operating_cost - SI_penalty - volume_penalty

This is exact (no re-optimization): MIQP does not respond to these penalties.
"""
import argparse
import numpy as np
import pandas as pd


def rescore_schedule(power, DA_price, si_shortage_mult, si_surplus_mult,
                     vol_water_value_mult, operational_cost, final_volume,
                     target_vol_low, rho, g, mu, target_head):
    """Re-score a single day's MIQP schedule. Mirrors calc_profit exactly.

    For a feasible MIQP schedule, simulated power == optimized power, so the
    imbalance is zero and SI_penalty is zero regardless of multipliers; we still
    compute it via the same algebra for parity with the DFL side.
    """
    p = np.asarray(power, dtype=float)
    da = np.asarray(DA_price, dtype=float)
    p_opt = p  # feasible: sim == opt

    revenue = float(np.sum(da * p))
    operating_cost = float(operational_cost * np.sum(p ** 2))

    si_price = np.where(p < p_opt, si_shortage_mult * da, si_surplus_mult * da)
    imbalance = p - p_opt  # == 0 for MIQP
    SI_penalty = float(np.sum(imbalance * si_price))

    volume_deficit = max(0.0, float(final_volume) - float(target_vol_low))
    energy_loss = rho * volume_deficit * g * target_head * mu / 3.6e9
    volume_penalty = float(energy_loss * vol_water_value_mult * np.median(da))

    ex_post_profit = revenue - operating_cost - SI_penalty - volume_penalty
    return {
        "revenue": revenue,
        "operating_cost": operating_cost,
        "SI_penalty": SI_penalty,
        "volume_penalty": volume_penalty,
        "ex_post_profit": ex_post_profit,
    }


def rescore_miqp_file(results_csv, params, cell, test_dates=None):
    """Re-score every (or selected) date in an MIQP results CSV under `cell`.

    Args:
        results_csv: path to MIQP_piecewise_results*.csv (date,hour,power,...,price)
        params: HydroParameters (for operational_cost, rho, g, mu, target_head, target_vol_low)
        cell: dict with si_shortage_mult, si_surplus_mult, vol_water_value_mult
        test_dates: optional iterable of date strings to restrict to.
    Returns:
        pandas.DataFrame: one row per date with the re-scored components.
    """
    df = pd.read_csv(results_csv)
    rows = []
    for date, g_df in df.groupby("date"):
        if test_dates is not None and str(date) not in set(map(str, test_dates)):
            continue
        g_df = g_df.sort_values("hour")
        final_volume = float(g_df["volume"].iloc[-1])
        out = rescore_schedule(
            power=g_df["power"].to_numpy(),
            DA_price=g_df["price"].to_numpy(),
            si_shortage_mult=cell["si_shortage_mult"],
            si_surplus_mult=cell["si_surplus_mult"],
            vol_water_value_mult=cell["vol_water_value_mult"],
            operational_cost=params.operational_cost,
            final_volume=final_volume,
            target_vol_low=float(params.target_vol_low),
            rho=float(params.rho),
            g=float(params.g),
            mu=float(params.mu),
            target_head=float(params.target_head),
        )
        out["date"] = str(date)
        rows.append(out)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="MIQP/MIQP_piecewise/MIQP_piecewise_results.csv")
    ap.add_argument("--si-shortage", type=float, default=-2.0)
    ap.add_argument("--si-surplus", type=float, default=-0.5)
    ap.add_argument("--vol-mult", type=float, default=1.0)
    args = ap.parse_args()
    # Standalone smoke run requires building params; see run_penalty_sensitivity.py
    # for the wired-up path. This branch is intentionally minimal.
    print("Use run_penalty_sensitivity.py to drive re-scoring with real params.")
