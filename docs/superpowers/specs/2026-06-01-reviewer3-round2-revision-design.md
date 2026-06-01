# Reviewer 3 (Round 2) Revision — Design

Date: 2026-06-01
Manuscript: `paper/submission-IEEE_TSE.tex`
Response letter: `paper/response_letter.tex`

## Context

Reviewer 3 returned a second-round review. The paper is "promising and close to
publishable" but needs another revision addressing five issues. We address them
step by step, starting with #1.

Execution: experiments run locally in the activated `dfl4uphes` conda env. DFL
validation is **solver-free** (no Gurobi needed at refinement/eval time); 1539
trained per-date models exist under `DFL/outputs/trained_models/`.

Deliverable: both manuscript edits and point-by-point response-letter entries.
**Round-2 manuscript edits must use a distinct color** from round-1 blue `\rev{}`.

## Color scheme for edit tracking

- Round-1 edits already use `\rev{}` (blue) and `\mrev{}` (blue math), toggled by
  `\ifshowrev`.
- Add round-2 commands `\revii{}` (text) and `\mrevii{}` (math) rendering in dark
  green `RGB(0,128,0)`, gated by the same `\ifshowrev` toggle. Round-1 blue is
  left untouched.

---

## Issue 1 — Warm-start mode robustness (EXPERIMENT)

### How mode is encoded (verified in code)
The discrete mode at hour `t` is the **sign of the warm-start power**, hard-coded
in `DFL/core/layers.py` (OptiLayer): `power_init[t]==0` → idle (`p=q=0`);
`>0` → turbine (bounded by head-dependent `pos_min_fit/pos_max_fit`);
`<0` → pump (`neg_min_fit/neg_max_fit`). The mode is **locked** for all K
iterations; DFL cannot correct it. Noise perturbation in `DFL/data/noise.py`
deliberately **preserves** modes. Active power is bounded *below* by
`p_min^m(h) > 0.5`, so power is bimodal: 0 (idle) or within the active envelope.
There is no "small active power" band — the only ambiguity is the discrete
commitment itself.

### Flip taxonomy (by transition type, not |p|)
- **Idle ↔ active**: commit/de-commit a unit. Plausible (real warm-starts
  disagree here, at shoulder hours); modest economic leverage.
- **Turbine ↔ pump**: sign reversal. Implausible (no valid warm-start reverses
  sign at a price extreme); catastrophic leverage (revenue sign flip + water
  displacement → likely terminal-volume penalty).

A naive uniform-random flip blends these and looks catastrophic purely because it
includes corruptions real warm-starts never produce. Stratifying by transition
type tells the honest story.

### Deliverable A — Natural disagreement audit (realistic anchor)
For each of the 19 representative days, compare the locked mode sequence of each
cheap warm-start source (MIQP-GL; historical-DB lookup) against the MIQP-PW
reference, hour by hour. Report mean hours/day that differ, split into
idle↔active vs turbine↔pump counts. Anchors what fraction/type of mode
corruption is realistic.

### Deliverable B — Controlled stratified flip sweep
Starting from the MIQP-PW warm-start per day, flip ρ randomly chosen hours, run
DFL refinement (solver-free), record mean ex-post profit. Two sweeps:
idle↔active only, and turbine↔pump only. ρ ∈ {1,2,3,4,5} hours, 3 random seeds
each (expand if noisy), averaged over the 19 days. When flipping *into* an active
mode, draw power within that mode's head-dependent feasible envelope so the
corrupted warm-start is itself feasible. Output: profit-vs-ρ curves with a
vertical marker at the natural disagreement rate from Deliverable A.

### Implementation
New script `DFL/scripts/run_mode_perturbation.py` reusing the validator's DFL
inner loop (load per-date model → predict weights → K linearization+QP iters →
simulate → ex-post profit). Add a mode-flip step on `power_init` before the loop.

### Result framing
Robust in the realistic regime (idle↔active at measured rate); sensitive only to
sign reversals valid warm-starts don't produce. Ties to the existing "fixed mode
commitment" limitation and future-work item on joint mode learning. Fall back to
discussion-only if a runtime issue blocks the run.

---

## Issue 2 — Penalty sensitivity (SI imbalance + terminal volume)
Both penalties enter ex-post profit `Π`. Re-score existing simulated trajectories
under alternative penalty settings (cheap; reuses `calc_profit`):
- Imbalance asymmetry: symmetric (1×/1×), mild (1.5×/0.75×), current (2×/0.5×).
- Terminal-volume water value: min / median / max DA price.
Show the qualitative conclusion (DFL-PW ≈ or > MIQP-PW; large speedup) is
invariant across settings. Compact table + short paragraph. Both DFL and the
MIQP baseline are re-scored under the *same* penalties, so relative ranking is
the honest comparison. (Breadth TBD with user — default both SI & Vol.)

## Issue 3 — Clear train/val/test description
Add an "Experimental Setup / Data Pipeline" paragraph: explicit sample counts
(19 representative days; 9 perturbation variants each → per-date training set;
19 held-out days for solver-free test); per-date model design (one penalty
predictor per representative date); perturbation generation (sign-preserving
power perturbation, 8 noise levels + random sampling); leakage avoidance
(held-out days never enter training; nearest-neighbor matching uses price only).
Distinguish held-out solver-free test from training/validation.

## Issue 4 — Reproducibility table
Compact table with values pulled from `DFL/config/*`: network arch (LSTM 3-layer,
hidden size, dropout); penalty bounds [w_min,w_max] per channel; K=7 iterations,
growth γ; ECOS tolerances (1e-5) and max_iters; Adam lr, epochs, patience,
gradient clipping.

## Issue 5 — Editing/formatting cleanup
Fix round-1 leftovers (e.g. `\markboth` "Running Title for Header" placeholder,
manuscript date placeholders) plus grammar/spacing. Enumerate findings before
editing.

---

## Sequencing
Issue 1 (experiment + write-up) → 2 → 3 → 4 → 5. Each manuscript edit in
`\revii{}` green; each issue gets a response-letter entry. Issues 3–5 are
text-only and low-risk.
