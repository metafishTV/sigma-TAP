# Empirical Validation — Design Document

**CLAIM POLICY LABEL: exploratory**

> This document specifies the design for an empirical validation module
> within sigma-TAP. The module compares simulation output against four
> quantitative targets from the innovation economics literature.

**Date:** 2026-02-26
**Status:** Approved — ready for implementation planning
**Precondition:** Predictive orientation diagnostic (PR #4) merged to main

---

## 1. Motivation

sigma-TAP's metathetic ensemble produces trajectories with cumulative
innovation counts, type diversity, event counters, and per-agent statistics.
These outputs have direct analogues in empirical innovation data. This
module provides the first quantitative comparison between sigma-TAP
simulation output and real-world findings.

Four empirical targets, extracted from three papers:

1. **Youn et al. (2015) 60:40 invariant**: 60% of US patents 1790-2010
   introduce *new* combinations (exploration), 40% refine *existing*
   combinations (exploitation). Invariant over 220 years. Equation 3.1:
   ΔC = 0.6 × ΔP.

2. **Taalbi (2025) linearity**: Innovation rate scales linearly with
   cumulative innovations. Regression coefficient ≈ 0.997 (Table 3,
   Model 1). dk/dt ∝ k.

3. **Taalbi (2025) Heaps' law**: Product type diversity grows sub-linearly
   with cumulative innovations. dD/dk = νD/(νD + ρk), giving D ~ k^β
   with β < 1.

4. **Taalbi (2025) power law**: Distribution of innovations across
   organizations follows P(k) ~ k^(-α) with α ≈ 2.

---

## 2. Architecture

### 2.1 Module Layout

```
simulator/empirical.py              <- NEW: metric computation functions
scripts/empirical_validation.py     <- NEW: validation runner + output
tests/test_empirical.py             <- NEW: dedicated test file
docs/empirical_targets.md           <- MODIFY: update with results
```

**Dependency flow:**
```
simulator/empirical.py
  +-- numpy (arrays, log operations)
  +-- scipy.stats (linregress)
  +-- no imports from other simulator modules (pure metric functions)

scripts/empirical_validation.py
  +-- imports from empirical.py: all 4 metric functions + result dataclass
  +-- imports from simulator.metathetic: MetatheticEnsemble
```

### 2.2 Design Principle

All metric functions are **pure** — they take arrays/lists as input and
return structured results. No simulation logic inside the metric module.
This makes them independently testable and reusable (e.g., they could
later be applied to real-world data for direct comparison).

---

## 3. Data Structures

```python
from dataclasses import dataclass

@dataclass
class YounRatioResult:
    """Exploration/exploitation ratio comparison."""
    exploration_count: int          # n_novel_cross (final)
    exploitation_count: int         # n_absorptive_cross (final)
    exploration_fraction: float     # n_novel / (n_novel + n_absorptive)
    target: float                   # 0.6 (Youn et al. 2015)
    deviation: float                # |measured - target|
    trajectory_ratios: list[float]  # ratio at each step (stability check)

@dataclass
class TaalbiLinearityResult:
    """Innovation rate linearity test."""
    slope: float                    # OLS slope of log(dk/dt) vs log(k)
    intercept: float
    r_squared: float
    target_slope: float             # ~1.0 (linear scaling)
    p_value: float
    n_points: int

@dataclass
class HeapsLawResult:
    """Sub-linear diversification test."""
    exponent: float                 # beta in D ~ k^beta
    intercept: float
    r_squared: float
    target: str                     # "< 1.0" (sub-linear)
    is_sublinear: bool
    n_points: int

@dataclass
class PowerLawResult:
    """Agent innovation distribution test."""
    exponent: float                 # alpha in P(k) ~ k^(-alpha)
    target_exponent: float          # ~2.0 (Taalbi 2025)
    ks_statistic: float             # Kolmogorov-Smirnov goodness of fit
    p_value: float
    n_agents: int
    k_min: int                      # minimum k for power-law regime

@dataclass
class EmpiricalValidationResult:
    """Full validation output."""
    youn: YounRatioResult
    linearity: TaalbiLinearityResult
    heaps: HeapsLawResult
    power_law: PowerLawResult
    params_used: dict               # simulation parameters
    n_steps: int
    n_agents: int
```

---

## 4. Core Functions

### 4.1 youn_ratio()

```python
def youn_ratio(
    n_novel_cross: list[int],
    n_absorptive_cross: list[int],
) -> YounRatioResult:
```

Takes cumulative event counts from trajectory snapshots. Computes
exploration fraction at each step as:

    ratio_t = n_novel_t / (n_novel_t + n_absorptive_t)

and overall (final step). Youn et al. found ΔC = 0.6 × ΔP — 60% of
patents introduce new combinations. We compare against target = 0.6.

**Edge cases:** If both counts are zero at a step, ratio is NaN (skipped
in trajectory_ratios). If both are zero at final step, exploration_fraction
is NaN and a warning is logged.

### 4.2 taalbi_linearity()

```python
def taalbi_linearity(
    k_total: list[int],
    dt: float = 1.0,
) -> TaalbiLinearityResult:
```

Takes cumulative innovation counts. Computes dk/dt via first differences,
then OLS regression of log(dk/dt) on log(k). Taalbi found slope ≈ 1.0
(Table 3, Model 1), meaning innovation rate scales linearly with
cumulative innovations.

Steps:
1. dk = diff(k_total) / dt
2. Filter out dk <= 0 and k <= 0 (can't take log)
3. OLS: log(dk) = intercept + slope * log(k)
4. Return slope, r², p-value

### 4.3 heaps_exponent()

```python
def heaps_exponent(
    k_total: list[int],
    D_total: list[int],
) -> HeapsLawResult:
```

Takes cumulative innovations and type diversity. OLS of log(D) on log(k).
Heaps' law predicts exponent < 1.0 (sub-linear growth of types relative
to innovations).

Steps:
1. Filter out pairs where k <= 0 or D <= 0
2. OLS: log(D) = intercept + exponent * log(k)
3. Return exponent, r², is_sublinear flag

### 4.4 power_law_fit()

```python
def power_law_fit(
    agent_k_list: list[int],
    k_min: int = 1,
) -> PowerLawResult:
```

Takes per-agent innovation counts from final snapshot. Fits power-law
distribution using MLE (Hill estimator):

    alpha = 1 + n / sum(ln(k_i / k_min))

where k_i >= k_min. Taalbi found alpha ≈ 2 across organizations.

Goodness-of-fit via KS statistic comparing empirical CDF to fitted
power-law CDF.

**No external dependency**: Uses numpy only (no `powerlaw` package).

### 4.5 run_empirical_validation()

```python
def run_empirical_validation(
    trajectory: list[dict],
) -> EmpiricalValidationResult:
```

Thin orchestrator in the script (not in empirical.py). Extracts fields
from trajectory, calls all 4 metric functions, assembles result.

---

## 5. Diagnostics Output

### Summary Table

```
Empirical Validation Summary (10 agents, 150 steps)
═══════════════════════════════════════════════════════════════════
  Metric                Target          Measured     Status
  Youn exploration      0.600           0.583        CLOSE (Δ=0.017)
  Taalbi linearity      slope≈1.0       0.97         MATCH (r²=0.98)
  Heaps' exponent       < 1.0           0.72         MATCH (sub-linear)
  Power-law exponent    ≈ 2.0           1.85         CLOSE (Δ=0.15)
═══════════════════════════════════════════════════════════════════
```

Status categories:
- **MATCH**: within 10% of target (or satisfies inequality constraint)
- **CLOSE**: within 25% of target
- **DIVERGENT**: > 25% deviation

---

## 6. Error Handling & Edge Cases

| Scenario | Handling |
|----------|----------|
| No events (n_novel=0, n_absorptive=0) | Return NaN for ratio, log warning |
| Single-step trajectory | Return empty result, log warning |
| All agents have same k | Power law degenerate, alpha=NaN |
| k_total non-increasing at some steps | Filter zero/negative dk |
| D_total = 0 at any point | Skip those points in Heaps regression |
| Fewer than 3 agents with k >= k_min | Power law unreliable, flag in result |
| log(0) encountered | Filter out zero values before regression |
| Very short trajectory (< 10 steps) | Return result but flag as unreliable |

---

## 7. Testing Strategy

~16-18 tests in `tests/test_empirical.py`:

1. youn_ratio with known counts -> correct fraction
2. youn_ratio 50:50 split -> 0.5 fraction
3. youn_ratio all novel -> 1.0 fraction
4. youn_ratio empty (no events) -> handles gracefully
5. taalbi_linearity with linear growth -> slope ~1.0
6. taalbi_linearity with quadratic growth -> slope ~2.0
7. taalbi_linearity constant k -> handles gracefully
8. heaps_exponent with k^0.5 data -> exponent ~0.5
9. heaps_exponent with linear D=k -> exponent ~1.0
10. heaps_exponent single point -> handles gracefully
11. power_law_fit with known Zipf distribution -> exponent ~2.0
12. power_law_fit with uniform distribution -> exponent far from 2.0
13. power_law_fit all same values -> degenerate
14. power_law_fit single agent -> handles gracefully
15. trajectory_ratios stability in youn_ratio
16. integration: run actual simulation, all 4 metrics valid
17. status classification: MATCH/CLOSE/DIVERGENT thresholds
18. EmpiricalValidationResult structure: all fields populated

---

## 8. Future Work

1. **Parameter sweep validation**: Run metrics across parameter grid
2. **Temporal stability**: Track how metrics evolve over simulation time
3. **Confidence intervals**: Bootstrap over ensemble runs for error bars
4. **Additional targets**: KPSS patents, paleobiology data
5. **Integration with fit_realworld.py**: Combine ODE fitting with
   discrete metric validation

---

## 9. References

- Youn, H., Strumsky, D., Bettencourt, L. M. A., & Lobo, J. (2015).
  "Invention as a combinatorial process." J. R. Soc. Interface,
  12(106), 20150272.
- Taalbi, J. (2025). "Long-run patterns in the discovery of the adjacent
  possible." Industrial and Corporate Change, 35(1), 123-149.
- Taalbi, J. (2025). "Innovation with and without patents — an
  information-theoretic approach." Scientometrics.
- Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). "Power-law
  distributions in empirical data." SIAM Review, 51(4), 661-703.
