# Empirical Parameter Sweep — Design Document

**CLAIM POLICY LABEL: exploratory**

> This document specifies the design for a parameter sweep that runs
> sigma-TAP's empirical validation metrics across a grid of parameters
> to find best-fit regimes and map parameter sensitivity.

**Date:** 2026-02-26
**Status:** Approved — ready for implementation
**Precondition:** Empirical validation (PR #5) merged to main

---

## 1. Motivation

The empirical validation framework (PR #5) compares sigma-TAP output
against four quantitative targets from Youn et al. (2015) and Taalbi
(2025). Default parameters show Heaps' law matches but the other three
metrics diverge. A systematic parameter sweep will:

1. **Find best-fit regimes**: Which parameter combinations best match
   all four empirical targets simultaneously?
2. **Map sensitivity**: Which parameters have the most effect on each
   metric? This guides future calibration efforts.

---

## 2. Architecture

```
scripts/empirical_sweep.py      <- NEW: parameter sweep runner
tests/test_empirical.py         <- MODIFY: add sweep-related tests
docs/empirical_targets.md       <- MODIFY: add sweep results
```

**Dependency flow:**
```
scripts/empirical_sweep.py
  +-- simulator.empirical: youn_ratio, taalbi_linearity, heaps_exponent, power_law_fit
  +-- simulator.empirical: EmpiricalValidationResult
  +-- simulator.metathetic: MetatheticEnsemble
  +-- scripts.empirical_validation: classify_status (via sys.path)
  +-- numpy, itertools.product
```

No changes to simulator/empirical.py — pure metric functions are reused.

---

## 3. Sweep Grid

```python
GRID = {
    "alpha": [1e-3, 3e-3, 5e-3, 1e-2],
    "a":     [2.0, 3.0, 5.0, 8.0],
    "mu":    [0.002, 0.005, 0.01, 0.02],
    "n_agents": [10, 15, 20],
}
SEEDS = [42, 123, 456]
STEPS = 200
VARIANT = "logistic"
```

Total: 4 x 4 x 4 x 3 x 3 = 576 simulations (~10-15 min).

---

## 4. Data Structures

```python
@dataclass
class SweepPoint:
    params: dict                     # {alpha, a, mu, n_agents}
    results: list[EmpiricalValidationResult]  # one per seed
    mean_youn_deviation: float
    mean_linearity_deviation: float
    mean_heaps_exponent: float
    heaps_match: bool
    mean_powerlaw_deviation: float
    composite_score: float           # lower = better

@dataclass
class SweepResult:
    points: list[SweepPoint]
    best: list[SweepPoint]           # top 5 by composite score
    sensitivity: dict[str, dict[str, float]]  # param -> metric -> mean deviation
    grid: dict
    seeds: list[int]
    steps: int
    total_sims: int
```

---

## 5. Composite Score

For each SweepPoint, averaged across seeds:

    score = w_youn * dev_youn + w_linear * dev_linear
          + w_heaps * dev_heaps + w_powerlaw * dev_powerlaw

Where:
- dev_youn = |exploration_fraction - 0.6| (NaN -> penalty 1.0)
- dev_linear = |slope - 1.0| (NaN -> penalty 1.0)
- dev_heaps = 0.0 if exponent < 1.0, else (exponent - 1.0) (NaN -> penalty 1.0)
- dev_powerlaw = |exponent - 2.0| (NaN -> penalty 1.0)
- All weights = 1.0 (equal weighting, configurable)

---

## 6. Sensitivity Analysis

For each parameter p in {alpha, a, mu, n_agents}:
1. Group all SweepPoints by their value of p
2. For each metric, compute mean deviation within each group
3. Sensitivity = max(group_means) - min(group_means) across values of p

The parameter with the largest sensitivity range for a given metric is
flagged as "most sensitive" for that metric.

---

## 7. Output Format

### Best-fit table (top 5):

```
Top 5 Parameter Regimes (by composite score, 576 simulations)
═══════════════════════════════════════════════════════════════════
  Rank  alpha    a      mu      agents  Score  Youn   Slope  Heaps  PL
  1     0.003   5.0    0.005   20      0.42   0.55   0.85   0.08   1.65
  ...
═══════════════════════════════════════════════════════════════════
```

### Sensitivity table:

```
Parameter Sensitivity (range of mean deviation across parameter values)
═══════════════════════════════════════════════════════════════════
  Parameter    Youn      Linearity   Heaps     Power-law
  alpha        0.12      0.35*       0.02      0.08
  a            0.08      0.15        0.04      0.22*
  mu           0.25*     0.18        0.01      0.15
  n_agents     0.05      0.08        0.03      0.41*
═══════════════════════════════════════════════════════════════════
```

---

## 8. Error Handling

| Scenario | Handling |
|----------|----------|
| Simulation crashes for a param combo | Log warning, skip that combo |
| All seeds produce NaN for a metric | Composite score uses penalty |
| No valid points at all | Print error, exit gracefully |

---

## 9. Testing Strategy

~5-6 tests in tests/test_empirical.py:

1. composite_score with known deviations -> correct value
2. composite_score with NaN penalty -> penalty applied
3. rank_sweep_results -> correct ordering by score
4. sensitivity calculation -> identifies dominant parameter
5. small grid integration test -> runs without error
6. classify_status used correctly in output

---

## 10. CLI

```
python scripts/empirical_sweep.py                    # full grid
python scripts/empirical_sweep.py --quick             # 2x2x2x1, 1 seed, 50 steps
python scripts/empirical_sweep.py --steps 500         # override steps
python scripts/empirical_sweep.py --top 10            # show top 10 instead of 5
```
