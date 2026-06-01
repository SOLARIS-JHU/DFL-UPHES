#!/usr/bin/env python3
"""
Plot mode-perturbation robustness results (Reviewer 3, Round 2, Issue 1).

Reads the per-flip-type CSVs produced by run_mode_perturbation.py and produces:
  * paper/figs/mode_perturbation_robustness.pdf  (profit vs rho, all flip types)
  * a printed LaTeX-ready summary table.

The natural mode-disagreement rate (idle<->active) from the audit is drawn as a
vertical marker to anchor the realistic perturbation regime.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'figure.figsize': (3.5, 2.8), 'axes.linewidth': 0.8, 'grid.linewidth': 0.4,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
    'legend.framealpha': 0.95, 'legend.edgecolor': 'black',
})

OUT_DIR = "./DFL/outputs"
FIG_PATH = "./paper/figs/mode_perturbation_robustness.pdf"

# (csv suffix, label, marker, color)
SERIES = [
    ("idle2active", r"Idle$\rightarrow$active (commit)", "o", "#1f77b4"),
    ("active2idle", r"Active$\rightarrow$idle (de-commit)", "s", "#2ca02c"),
    ("sign",        r"Turbine$\leftrightarrow$pump (sign)", "^", "#d62728"),
]

# Natural disagreement rate (idle<->active) from run_mode_disagreement_audit.py
NATURAL_RATE = 2.74


def load(suffix):
    path = os.path.join(OUT_DIR, f"mode_perturbation_{suffix}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # average over successful solves only; failures tracked separately
    g = df.groupby('rho')['ex_post'].apply(lambda s: np.nanmean(s))
    fails = df.groupby('rho')['failed'].sum()
    return g, fails


def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    baseline = None
    summary = {}
    fails_by = {}
    for suffix, label, marker, color in SERIES:
        loaded = load(suffix)
        if loaded is None:
            print(f"[warn] missing {suffix}")
            continue
        g, fails = loaded
        if baseline is None:
            baseline = g.loc[0]
        ax.plot(g.index, g.values, marker=marker, color=color, label=label)
        summary[label] = g
        fails_by[label] = fails

    if baseline is not None:
        ax.axhline(baseline, color='gray', ls=':', lw=0.8,
                   label='Correct warm-start')

    ax.axvline(NATURAL_RATE, color='black', ls='--', lw=0.8)
    ymin, ymax = ax.get_ylim()
    ax.text(NATURAL_RATE + 0.08, ymax, 'realistic\nrate', fontsize=6,
            va='top', ha='left')

    ax.set_xlabel("Corrupted warm-start modes (hours)")
    ax.set_ylabel("Mean ex-post profit (EUR)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0))
    fig.savefig(FIG_PATH)
    print(f"Wrote figure to {FIG_PATH}")

    # Summary table (% change vs baseline)
    print("\n=== Mean ex-post profit (EUR) and % change vs correct warm-start ===")
    rhos = sorted(set().union(*[set(g.index) for g in summary.values()]))
    header = "rho  " + "  ".join(f"{lbl[:18]:>20}" for lbl in summary)
    print(header)
    for r in rhos:
        cells = []
        for lbl, g in summary.items():
            if r in g.index:
                pct = 100 * (g.loc[r] - baseline) / baseline
                nf = int(fails_by[lbl].loc[r]) if r in fails_by[lbl].index else 0
                cells.append(f"{g.loc[r]:7.1f} ({pct:+.1f}%, f={nf})")
            else:
                cells.append(" " * 24)
        print(f"{r:>3}  " + "  ".join(f"{c:>24}" for c in cells))


if __name__ == "__main__":
    main()
