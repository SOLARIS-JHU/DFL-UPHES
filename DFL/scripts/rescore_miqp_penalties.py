#!/usr/bin/env python3
"""
Re-score MIQP-PW schedules under alternative penalty settings (Reviewer 3, Issue 2).

MIQP-PW has no SI/volume terms in its objective, so it does not respond to these
penalties (no re-optimization). But its planned schedule is NOT perfectly feasible
under the true nonlinear dynamics: the production path is to run the planned
(p, q, h) through SimulationLayer.simulate_operation, which recomputes a physically
feasible trajectory `p_sim` that DEVIATES from the plan (real SI imbalance) and a
real terminal volume `v_low_sim` (real volume deficit), then call calc_profit. This
mirrors exactly how the paper evaluates MIQP-PW (see MIQP/MIQP_piecewise/
MIQP_piecewise.py), giving an honest comparison: both DFL and MIQP go through
simulate -> calc_profit under the same cell's penalties.

`rescore_schedule` below is a simplified numpy helper that ASSUMES a perfectly
feasible schedule (sim == opt, raw final volume), which zeroes both penalties. It is
retained only for the feasible-identity unit test and is NOT the production path.
"""
import argparse
import numpy as np
import pandas as pd
import torch


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
    imbalance = p - p_opt  # imbalance == 0 for MIQP; si_price computed for structural parity, no effect
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

    Mirrors the paper's MIQP-PW evaluation: per date, build tensors from the planned
    schedule, run SimulationLayer.simulate_operation to get a physically feasible
    trajectory (p_sim deviating from the plan -> real imbalance; v_low_sim -> real
    terminal deficit), then calc_profit under the cell's penalties. `params` must
    already carry the cell's penalty multipliers (built by build_context); we trust
    that here rather than re-injecting `cell`.

    Args:
        results_csv: path to MIQP_piecewise_results*.csv (date,hour,power,head,...,flow,price)
        params: HydroParameters carrying this cell's penalty multipliers + dynamics.
        cell: dict with si_shortage_mult, si_surplus_mult, vol_water_value_mult
            (used only for the assertion that params reflects the cell).
        test_dates: optional iterable of date strings to restrict to.
    Returns:
        pandas.DataFrame: one row per date with the re-scored components.
    """
    from DFL.core.layers import SimulationLayer

    # Trust, but verify, that params reflects the requested cell.
    assert float(params.si_shortage_mult) == float(cell["si_shortage_mult"]), (
        f"params.si_shortage_mult={params.si_shortage_mult} != cell={cell['si_shortage_mult']}")
    assert float(params.si_surplus_mult) == float(cell["si_surplus_mult"]), (
        f"params.si_surplus_mult={params.si_surplus_mult} != cell={cell['si_surplus_mult']}")
    assert float(params.vol_water_value_mult) == float(cell["vol_water_value_mult"]), (
        f"params.vol_water_value_mult={params.vol_water_value_mult} != cell={cell['vol_water_value_mult']}")

    device = getattr(params, "device", torch.device("cpu"))
    simulator = SimulationLayer(params)

    df = pd.read_csv(results_csv)
    rows = []
    for date, day_df in df.groupby("date"):
        if test_dates is not None and str(date) not in set(map(str, test_dates)):
            continue
        day_df = day_df.sort_values("hour")
        try:
            p = torch.tensor(day_df["power"].to_numpy(), dtype=torch.float32, device=device)
            q = torch.tensor(day_df["flow"].to_numpy(), dtype=torch.float32, device=device)
            h = torch.tensor(day_df["head"].to_numpy(), dtype=torch.float32, device=device)
            da = torch.tensor(day_df["price"].to_numpy(), dtype=torch.float32, device=device)

            with torch.no_grad():
                p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p, q, h)
                n = len(p_sim)
                profit, si, vol, op = simulator.calc_profit(
                    p_sim, p[:n], v_low_sim, da[:n])
                revenue = float((da[:n] * p_sim).sum())

            rows.append({
                "date": str(date),
                "ex_post_profit": float(profit),
                "SI_penalty": float(si),
                "volume_penalty": float(vol),
                "operating_cost": float(op),
                "revenue": revenue,
            })
        except Exception as exc:  # noqa: BLE001 - skip a bad date, keep the sweep alive
            print(f"WARNING: rescore_miqp_file skipping date {date}: {exc!r}")
            continue

    if not rows:
        raise ValueError(
            f"rescore_miqp_file produced no rows from {results_csv} "
            f"(test_dates={test_dates!r}); all dates skipped or no matching dates. "
            f"Check date formats match the CSV.")
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
