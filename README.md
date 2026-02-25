# sigma-TAP

Paper-first workspace for TAP / biocosmology alignment.

## Contents

- Core references (PDF):
  - `TAPequation-FINAL.pdf`
  - `Applications-of-TAP.pdf`
  - `Paper1-FINAL.pdf`
  - `Paper2-FINAL.pdf`
  - `Long-run patterns in the discovery of the adjacent possible.pdf`
- Review notes:
  - `reviews/adjacent-possible-alignment-review.md`

## Working principle

Use the papers as the starting framework and guidance, then build reproducible analyses that support and extend their program without over-claiming.


## Roadmap

- `TASKS.md` — prioritized implementation tasks derived from the paper-first review.
- `ENGINEERING_PLAN.md` — concrete engineering work packages, interfaces, artifacts, and validation gates.


## Iteration run

- `tools/paper_iteration.py` — executes a concrete paper-vs-workspace iteration check.
- Outputs: `outputs/paper_iteration_report.json` and `outputs/paper_iteration_report.md`.


## Variant-aware runs (paper-aligned)

The simulator now supports TAP variants aligned to the papers:
- `baseline`
- `two_scale`
- `logistic`

Run sweeps with one or more variants via `TAP_VARIANTS`:

```bash
TAP_VARIANTS=baseline,two_scale,logistic python scripts/sweep_mode_b.py > outputs/sweep_mode_b.csv
```
