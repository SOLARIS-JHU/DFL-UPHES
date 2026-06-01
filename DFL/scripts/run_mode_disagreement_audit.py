#!/usr/bin/env python3
"""
Natural mode-disagreement audit (Reviewer 3, Round 2, Issue 1, Deliverable A)
=============================================================================

Quantifies how often a cheap warm-start source (MIQP-GL) disagrees on the
discrete operating mode with the high-fidelity MIQP-PW reference, across the 19
representative days. The mode at each hour is the sign of the power schedule
(the same convention DFL locks): p>tol -> turbine, p<-tol -> pump, else idle.

Disagreements are split by type:
  * idle <-> active : commit/de-commit a unit (plausible; low economic leverage)
  * turbine <-> pump: sign reversal (implausible; catastrophic leverage)

This anchors the realistic perturbation magnitude (rho) used in
run_mode_perturbation.py.
"""

import os
import sys
import numpy as np
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

GL_CSV = "./MIQP/MIQP_linear/MILP_global_linear_results.csv"
PW_CSV = "./MIQP/MIQP_piecewise/MIQP_piecewise_results.csv"
TOL = 0.5  # active threshold used throughout the codebase


def mode(power, tol=TOL):
    return np.where(power > tol, 'T', np.where(power < -tol, 'P', 'I'))


def main():
    gl = pd.read_csv(GL_CSV)
    pw = pd.read_csv(PW_CSV)

    tot, ia, tp = [], [], []
    for date in sorted(pw.date.unique()):
        g = gl[gl.date == date].sort_values('hour').power.values[:24]
        p = pw[pw.date == date].sort_values('hour').power.values[:24]
        mg, mp = mode(g), mode(p)
        diff = mg != mp
        n_ia = sum(1 for a, b in zip(mg[diff], mp[diff]) if 'I' in {a, b})
        n_tp = diff.sum() - n_ia
        tot.append(diff.sum()); ia.append(n_ia); tp.append(n_tp)

    tot, ia, tp = map(np.array, (tot, ia, tp))
    print("MIQP-GL vs MIQP-PW mode disagreement across 19 representative days")
    print(f"  total differing hours/day : mean={tot.mean():.2f}  max={tot.max()}  (of 24)")
    print(f"  idle<->active   /day      : mean={ia.mean():.2f}  "
          f"({100*ia.sum()/tot.sum():.0f}% of disagreements)")
    print(f"  turbine<->pump  /day      : mean={tp.mean():.2f}  "
          f"({100*tp.sum()/tot.sum():.0f}% of disagreements)")
    print(f"  total differing hours     : {tot.sum()} / {len(tot)*24}")


if __name__ == "__main__":
    main()
