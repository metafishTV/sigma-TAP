# Codebase Simplification & Review Design

**Date:** 2026-02-26
**Goal:** Simplify code, remove dead weight, eliminate redundant computation,
and consolidate duplicated logic — all while preserving identical output
fidelity and granularity.
**Approach:** Single atomic PR. All changes are provably behavior-preserving.
Full 273-test suite gates every logical group.

---

## 1. Dead Code Removal

### 1a. Delete `simulator/pressure.py`
Zero callers, zero imports, zero test coverage. Stub for the 9-component
pressure spectrum (old TAPS framework) that was superseded by the 7-mode
two-pole model.

### 1b. Delete `simulator/projection.py`
Zero callers, zero imports, zero test coverage. Three functions
(`project_M`, `birth_death`, `pressure_inverse_inference_allowed`) that
are either reimplemented in `metathetic.py` or covered by
`analysis.identifiability_gate`.

### 1c. Remove `innovation_kernel()` from `tap.py`
The integer-sum form (lines 7-11) is never called anywhere in the
codebase. Only `innovation_kernel_closed` (the algebraic closed form)
is used. Also remove the `from math import comb` import that solely
served this dead function.

### 1d. Relocate `pass_c_additional_runs()` from `analysis.py`
Planning formula called only in `scripts/run_demo.py` as a print
statement. Move the constant to a comment in `run_demo.py` and remove
the function from the analysis module.

---

## 2. Redundant Computation Elimination

### 2a. Wire actual `classify_dM_texture()` into `texture_type`
Currently `classify_step` copies `pressure_regime` into `texture_type`,
producing a duplicate axis that wastes a full eigendecomposition per
sweep point. Replace the alias with the already-implemented
`classify_dM_texture()` function for real independent data.
**Net effect:** increased granularity + saved redundant compute.

### 2b. Eliminate double eigendecomposition in `eigenvalue_analysis`
`eigvals(P)` then `eig(P.T)` computes eigenvalues twice (eigenvalues
of P^T = eigenvalues of P). Compute `eig(P.T)` once, extract both
eigenvalues and the stationary eigenvector from the single call.

### 2c. Cache `_event_deltas` in `compute_all_scores`
Currently called 4x (once per sub-function) on the same trajectory.
Compute once at the top, pass as optional `_deltas` parameter to
`compute_transvolution`, `compute_anopression`, `compute_praxis`,
`compute_syntegration`.

### 2d. Cache `_active_agents()` per step in `metathetic.run()`
Called 3-8x per step via helper methods. Compute once at step boundary,
pass to helpers. Also cache `total_M = sum(a.M_local for a in active)`
which is computed twice in the snapshot block.

### 2e. Cache `_all_type_counts()` per step
Called in `_check_cross_metathesis`, `_convergence_measure`, and temporal
state recording. Compute once after metathesis, pass to consumers.

### 2f. Remove redundant `max(0,...)` in `compute_birth_term`
`innovation_kernel_closed` and `innovation_kernel_two_scale` already
return `max(0,...)`. The outer `max(0, base)` is a tautology. Remove it.

### 2g. Replace O(n^2) expanding window with Welford's algorithm
`detect_adpression` in `predictive.py` recomputes mean/variance from
scratch for each expanding window step. Welford's online algorithm
produces identical population mean and variance in O(n) total.

### 2h. Vectorize `sigma_arr` in `continuous.py`
Post-hoc sigma array reconstruction uses a Python list comprehension.
Replace with `sigma_arr = sigma0 * (1.0 + gamma * Xi_arr)` — pure
NumPy broadcast, identical formula.

---

## 3. Code Consolidation (DRY)

### 3a. Consolidate triple OLS to `scipy.stats.linregress`
Three places implement log-log OLS by hand:
- `longrun.heaps_law_fit` (hand-rolled)
- `empirical.heaps_exponent` (already uses linregress)
- `analysis.innovation_rate_scaling` (hand-rolled)

Make `longrun.heaps_law_fit` call `empirical.heaps_exponent` internally.
Replace hand-rolled OLS in `analysis.innovation_rate_scaling` with
`linregress`. Same slope, same R^2, free p-values.

### 3b. Replace 6x copy-pasted `logspace` with `np.logspace`
Identical function in 6 script files. All scripts already import numpy.
Replace with `np.logspace(np.log10(lo), np.log10(hi), n).tolist()`.

### 3c. Generalize 2D/3D logistic boundary to N-D
`fit_explosive_logistic_boundary` (2D, ~90 lines) and
`fit_explosive_logistic_boundary_3d` (3D, ~95 lines) share ~85% code
including duplicate `sig()` function. Write a single
`_fit_logistic_boundary(X, y, lr, epochs, l2)` for any feature count.
Public functions become thin wrappers.

### 3d. Parameterize H decay in `continuous.py`
`H = max(0.0, 0.02 * Xi)` is hardcoded instead of using the
parameterized value from `hfuncs.py`. Add `eta_decay` field to ODE
params (default 0.02 for identical current behavior). Prevents silent
divergence if defaults change later.

---

## 4. Import Cleanup

### 4a. Move `import math` to top of `analysis.py`
Five deferred `import math` statements inside functions. `math` is
stdlib — no reason to defer. Move to module level alongside existing
`from statistics import median`.

---

## 5. Stage 3 Forward Note

**S3.5 Two-Channel Consummation (beta*B + eta*H):** The old TAPS
document's distinction between manifest births (beta*B) and implicate
densification (eta*H) may inform why the Youn ratio is stuck at 1.0
across all parameter regimes. Currently all cross-metathesis events
classify as novel (exploration). The two-channel framework suggests
that "absorptive" events might map to the eta*H channel — experience
deepening without new feature activation. This could provide a
mechanism for the missing 40% exploitation fraction. Circle back
during Stage 3 design.

---

## Verification Strategy

- Full 273-test suite run after each logical group of changes
- Any test failure immediately identifies which change caused it
- Single branch, single PR — one `git revert` undoes everything
- All changes are mathematically provable identity transformations
  (except 2a which improves granularity)
