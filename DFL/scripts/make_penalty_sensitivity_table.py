#!/usr/bin/env python3
"""Build the penalty-sensitivity LaTeX table from summary.csv (Reviewer 3 Issue 2).

Shows DFL-PW vs MIQP-PW ex-post profit, their gap, and the decomposed SI/volume
penalties for BOTH methods under each penalty setting (both scored identically via
simulate->calc_profit). Demonstrates the ranking/margin is invariant to the penalty
choices the reviewer flagged.
"""
import os
import sys
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

SUMMARY = "DFL/outputs/penalty_sensitivity/summary.csv"
OUT_TEX = "results/tables/penalty_sensitivity.tex"
OUT_CSV = "results/tables/penalty_sensitivity.csv"

LABELS = {
    "baseline": "Current (2.0/0.5, 1.0$\\times$)",
    "si_symmetric": "SI symmetric (1.0/1.0)",
    "si_mild": "SI mild (1.5/0.75)",
    "vol_low": "Water value 0.8$\\times$",
    "vol_high": "Water value 1.2$\\times$",
}
ORDER = ["baseline", "si_symmetric", "si_mild", "vol_low", "vol_high"]


def main():
    df = pd.read_csv(SUMMARY).set_index("cell").loc[ORDER].reset_index()
    out = pd.DataFrame({
        "Setting": df["cell"].map(LABELS),
        "DFL-PW (EUR)": df["dfl_profit_mean"].round(0).astype(int),
        "MIQP-PW (EUR)": df["miqp_profit_mean"].round(0).astype(int),
        "Gap (%)": df["gap_pct"].round(2),
        "DFL SI": df["dfl_si"].round(1),
        "MIQP SI": df["miqp_si"].round(1),
        "DFL Vol": df["dfl_vol"].round(1),
        "MIQP Vol": df["miqp_vol"].round(1),
    })
    out.to_csv(OUT_CSV, index=False)

    lines = [
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Penalty setting & DFL-PW & MIQP-PW & Gap & DFL SI & MIQP SI & DFL Vol & MIQP Vol \\",
        r" & (EUR) & (EUR) & (\%) & (EUR) & (EUR) & (EUR) & (EUR) \\", r"\midrule",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"{r['Setting']} & {r['DFL-PW (EUR)']} & {r['MIQP-PW (EUR)']} & "
            f"{r['Gap (%)']} & {r['DFL SI']} & {r['MIQP SI']} & "
            f"{r['DFL Vol']} & {r['MIQP Vol']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(out.to_string(index=False))
    print(f"\nWrote {OUT_TEX} and {OUT_CSV}")


if __name__ == "__main__":
    main()
