#!/usr/bin/env python3
"""
Plot mode-perturbation robustness results (Reviewer 3, Round 2, Issue 1).

Reads the six per-transition CSVs produced by run_mode_perturbation.py and
produces a 2x3 small-multiples figure (one mini-panel per directed mode error).
Each panel shows:
  * DFL-refined ex-post profit   (solid + markers, left axis)
  * raw perturbed warm-start     (dashed, no DFL, left axis)
  * infeasibility rate           (shaded, right axis: % of 19 days infeasible)

Profit is averaged over feasible days only; the shaded infeasibility band makes
the survivor bias explicit.

Also prints a LaTeX-ready summary table including the DFL-over-raw improvement.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'grid.linewidth': 0.4,
    'lines.linewidth': 1.1, 'lines.markersize': 3.5,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'legend.framealpha': 0.95, 'legend.edgecolor': 'black',
})

OUT_DIR = "./DFL/outputs"
FIG_PATH = "./paper/figs/mode_perturbation_robustness.pdf"

# (suffix, title, category color) grouped: blue=commit, green=de-commit, red=sign
PANELS = [
    ("idle2turbine",  r"Idle$\rightarrow$turbine",  "#1f77b4"),
    ("idle2pump",     r"Idle$\rightarrow$pump",      "#1f77b4"),
    ("turbine2idle",  r"Turbine$\rightarrow$idle",   "#2ca02c"),
    ("pump2idle",     r"Pump$\rightarrow$idle",      "#2ca02c"),
    ("turbine2pump",  r"Turbine$\rightarrow$pump",   "#d62728"),
    ("pump2turbine",  r"Pump$\rightarrow$turbine",   "#d62728"),
]

N_DAYS = 19


def load(suffix):
    path = os.path.join(OUT_DIR, f"mode_perturbation_{suffix}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    g = df.groupby('rho').agg(
        dfl=('ex_post', lambda s: np.nanmean(s)),
        raw=('raw_ex_post', lambda s: np.nanmean(s)),
        n_fail=('failed', lambda s: int(s.sum())),
        n=('failed', 'size'),
    )
    g['infeas_pct'] = 100.0 * g['n_fail'] / g['n']
    return g


def main():
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 2.7), sharex=True)
    axes = axes.ravel()

    summary = {}
    baseline = None
    for ax, (suffix, title, color) in zip(axes, PANELS):
        g = load(suffix)
        if g is None:
            print(f"[warn] missing {suffix}")
            continue
        summary[suffix] = g
        if baseline is None:
            baseline = g['dfl'].loc[0]

        # left axis: profit
        ax.plot(g.index, g['dfl'], marker='o', color=color, ls='-',
                label='DFL-refined')
        ax.plot(g.index, g['raw'], marker='x', color='0.45', ls='--',
                label='Raw warm-start')
        ax.axhline(baseline, color='gray', ls=':', lw=0.7)
        ax.set_title(title, color=color)
        ax.set_ylim(-6500, 4500)
        ax.yaxis.set_major_locator(MultipleLocator(2000))
        ax.yaxis.set_minor_locator(MultipleLocator(1000))
        ax.axhline(0, color='0.7', lw=0.5)
        ax.grid(True, which='major', alpha=0.35)
        ax.grid(True, which='minor', alpha=0.15)

        # right axis: infeasibility rate (shaded)
        ax2 = ax.twinx()
        ax2.fill_between(g.index, 0, g['infeas_pct'], color=color, alpha=0.12)
        ax2.set_ylim(0, 100)
        ax2.set_yticks([0, 50, 100])
        ax2.tick_params(labelsize=6)
        # only show right-axis label on rightmost column
        if suffix in ('idle2pump', 'pump2idle', 'pump2turbine'):
            ax2.set_ylabel('infeasible (%)', fontsize=6)
        else:
            ax2.set_yticklabels([])

    # shared labels
    for ax in axes[3:]:
        ax.set_xlabel("Corrupted modes (hours)")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Ex-post profit (EUR)")

    # single shared legend (profit lines + infeasibility patch)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color='0.2', marker='o', ls='-', label='DFL-refined'),
        Line2D([0], [0], color='0.45', marker='x', ls='--', label='Raw warm-start'),
        Line2D([0], [0], color='gray', ls=':', label='Correct warm-start'),
        Patch(facecolor='0.5', alpha=0.2, label='Infeasible (%)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.10), fontsize=6.5)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    fig.savefig(FIG_PATH)
    print(f"Wrote figure to {FIG_PATH}")

    # ---- summary table ----
    print("\n=== DFL-refined vs raw warm-start (mean over feasible days) ===")
    print("transition       rho  DFL_profit  raw_profit  DFL-raw   infeas%")
    for suffix, _, _ in PANELS:
        g = summary.get(suffix)
        if g is None:
            continue
        for rho in g.index:
            row = g.loc[rho]
            print(f"{suffix:14s}  {rho:>3}  {row['dfl']:10.0f}  "
                  f"{row['raw']:10.0f}  {row['dfl']-row['raw']:7.0f}  "
                  f"{row['infeas_pct']:6.0f}")


if __name__ == "__main__":
    main()
