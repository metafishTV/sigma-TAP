# Model/paper consistency changes applied

This note records direct code changes made after reviewing papers and existing simulator scripts.

## Issues found and fixed

1. **Paper says TAP has multiple variants, code only simulated one kernel.**
   - Added TAP variant support in simulator core (`baseline`, `two_scale`, `logistic`).
   - Implemented variant-aware birth-term computation in `simulator/tap.py` and wired it into runtime simulation.

2. **Paper highlights two-scale behavior (single-object innovation channel), code had no two-scale implementation.**
   - Added `innovation_kernel_two_scale` with `alpha1 * M` + combinatorial term.
   - Extended `ModelParams` to support `tap_variant` + `alpha1`.

3. **Long-run paper emphasizes constraints/search limits, code lacked an explicit constrained variant.**
   - Added logistic-style constraint hook via carrying capacity in `apply_logistic_constraint`.
   - Extended `ModelParams` with `carrying_capacity`.

4. **Sweeps were effectively single-variant, contrary to paper-first comparison requirement.**
   - Updated sweep scripts to support variant loops driven by `TAP_VARIANTS` env var.
   - Added `variant` as an explicit output column in sweep outputs.

5. **Demo script did not compare variants.**
   - Updated `scripts/run_demo.py` to run baseline/two_scale/logistic side-by-side.

## Practical usage

- Single variant (default):
  - `python scripts/sweep_mode_b.py > outputs/sweep_mode_b.csv`
- Multi-variant run:
  - `TAP_VARIANTS=baseline,two_scale,logistic python scripts/sweep_mode_b.py > outputs/sweep_mode_b.csv`

These changes directly iterate the modeling code against paper guidance rather than only adding planning documents.
