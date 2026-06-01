# Reviewer 3 (Round 2) — Issue 2: Penalty Sensitivity — Design

Date: 2026-06-01
Manuscript: `paper/submission-IEEE_TSE.tex`
Response letter: `paper/response_letter.tex`
Parent spec: `docs/superpowers/specs/2026-06-01-reviewer3-round2-revision-design.md` (§ "Issue 2")

## Reviewer concern

> The imbalance and terminal-volume penalties are now better justified, but they
> directly affect the reported ex-post profit. The authors should add either a
> small sensitivity analysis or a clearer discussion of how the conclusions
> depend on these penalty choices.

Scope chosen (with user): **sensitivity analysis + a manuscript discussion
paragraph + a response-letter entry**.

## What the manuscript claims and must defend

The penalties enter the ex-post profit metric `Π` (`calc_profit`,
`DFL/core/layers.py:492`). The conclusions tied to these penalty choices are:

- **Headline ranking / margin**: DFL-PW improves ex-post profit by **1.1%** over
  MIQP-PW. `tab:main_results` reports MIQP-PW = `3849 ± 1545` EUR at `1205.79` s;
  DFL-PW sits just above it at negligible runtime.
- **SI-asymmetry mechanism** (`submission-IEEE_TSE.tex:670`): DFL learns to bias
  toward slight overproduction, earning a *negative* mean SI penalty (≈ −20 EUR),
  by exploiting the 2× shortage / 0.5× surplus structure.
- **Volume-penalty mechanism** (`:668`): DFL accepts calculable volume violations
  (≈ 250–280 EUR) during high-revenue hours when net profit improves.

The sensitivity study must show these conclusions are **invariant** across
reasonable penalty settings.

## Where the penalties live (verified in code)

- **SI imbalance**, `DFL/core/layers.py:503–515`: `shortage_penalty_multiplier =
  -2.0`, `surplus_penalty_multiplier = -0.5`, applied as
  `SI_price = where(e_sim < p_opt, mult_short*DA, mult_surplus*DA)`.
- **Terminal volume**, `DFL/core/layers.py:517–520`: `volume_penalty =
  energy_loss * median(DA_price)`, where the median DA price is the water value.
- These constants appear **only in `calc_profit`**, which is the LSTM's training
  loss (`DFL/training/trainer.py:156`, `loss = -simulated_profit`). The DFL QP
  objective (`OptiLayer`) uses the *predicted* deviation weights `w_p/w_h/w_q`,
  not these market penalties.
- **MIQP-PW objective has no SI/volume market-penalty terms**; MIQP produces a
  physical schedule that is evaluated ex-post under `calc_profit`.

## Core design decision — asymmetric treatment

Because the penalties enter MIQP only at evaluation time but enter DFL at
*training* time, the two sides are handled differently:

| Method  | Treatment per penalty cell                                       |
|---------|------------------------------------------------------------------|
| MIQP-PW | **Re-score only** — recompute `calc_profit` on the saved MIQP-PW power trajectory under the cell's penalty. Exact; no Gurobi. |
| DFL-PW  | **Retrain the LSTM** under the cell's penalty (it is the loss), then validate, then score. |

Both sides are scored under the *same* penalty within a cell, so the **relative
ranking and margin are the honest comparison**.

## Penalty grid (one-at-a-time, 6 cells including baseline)

Vary one axis at a time, holding the other at its baseline value.

| Axis            | Settings (baseline in bold)                                  |
|-----------------|--------------------------------------------------------------|
| SI imbalance    | symmetric 1×/1×, mild 1.5×/0.75×, **current 2×/0.5×**         |
| Vol water value | 0.8×median, **median**, 1.2×median DA price                  |

Cells: baseline (2×/0.5×, median) + 2 SI variants + 2 Vol variants = **6 total**.

Volume is a tight ±20% band, so its sensitivity is expected to be mild — framed
honestly as "even a ±20% mis-pricing of stored water leaves the ranking
unchanged," not as a wide stress test.

## Scope

- **Training set: random samples (RS).** Confirmed as the manuscript's headline
  training set: `submission-IEEE_TSE.tex:600` — "All DFL methods reported in
  subsequent tables are trained using the random sampling noise level." The
  baseline cell therefore reconciles directly with `tab:main_results`.
- **Config: `LSTM_3layer_7iter`** (the headline DFL-PW configuration).
- **Cost: 19 representative dates × 6 cells ≈ 114 LSTM retrains** + validation;
  MIQP-PW re-scored on the same test days within each cell. Training matches the
  existing pipeline (Adam lr=0.001, early stopping). Other noise levels are *not*
  retrained — noise-level stability is already established in the main results
  (`fig:noise_robustness`), and we cite that rather than re-prove it.

## Implementation

1. **Parameterize the penalty constants.** Thread the SI multipliers
   (`layers.py:503–504`) and the volume water-value coefficient (`layers.py:520`,
   currently `median(DA_price)` → a configurable multiple) through
   `SimulationLayer` / `SystemParameters` so a cell sets them without editing
   code. Defaults reproduce current behaviour (2×/0.5×, 1.0×median) so existing
   results are unchanged. `calc_profit` reads them from the layer/params.

2. **New script `DFL/scripts/run_penalty_sensitivity.py`.** For each cell:
   - set SI multipliers and volume water-value multiple,
   - retrain DFL-PW (RS, `LSTM_3layer_7iter`) over the 19 dates,
   - validate (solver-free) → DFL-PW ex-post profit + decomposed SI/Vol penalties,
   - re-score MIQP-PW schedules (`MIQP_piecewise_results*.csv`) under the same
     penalty via `calc_profit`,
   - write one CSV row.

3. **Output CSV** (one row per cell): cell id, SI multipliers, Vol multiple,
   DFL-PW profit (mean ± std), DFL-PW SI penalty, DFL-PW Vol penalty, MIQP-PW
   profit, gap (%) = (DFL − MIQP)/MIQP, DFL runtime, MIQP runtime / speedup.

4. **Retrained models / outputs** written under a dedicated subtree (e.g.
   `DFL/outputs/penalty_sensitivity/<cell>/`) so the main `trained_models/` tree
   is untouched.

## Verification

- **Baseline-cell reconciliation:** run the baseline cell first; its DFL-PW and
  MIQP-PW numbers must match `tab:main_results` within sampling noise before
  running the other five cells. This is the correctness gate.
- Confirm the parameterization defaults reproduce the unchanged pre-existing
  validation numbers on at least one date.

## Deliverables

1. The experiment + compact sensitivity table (CSV → LaTeX).
2. **Manuscript paragraph** in round-2 green `\revii{}` (per parent spec's color
   scheme): the DFL-PW ≥ MIQP-PW ranking, the ≈1.1% headline margin, and the
   SI-asymmetry mechanism all hold across the grid; volume framed as ±20%
   robustness. Ties back to the penalty-justification text the reviewer praised.
3. **Response-letter entry** for Issue 2.

## Sequencing

Parameterize penalties → wire `run_penalty_sensitivity.py` → run **baseline cell
first, verify against `tab:main_results`** → run remaining 5 cells → build table →
write manuscript paragraph + response-letter entry.

## Risks / fallback

- If a retrain run hits a solver/numerical issue (the ECOS layer is sensitive),
  isolate the offending date, fall back to the date's existing handling used in
  the main pipeline. If retraining proves infeasible at scale, fall back to the
  reviewer's alternative — a discussion-only paragraph backed by re-scoring both
  sides (DFL re-scored, not retrained) and the existing decomposed penalty
  numbers — and state the caveat explicitly.
