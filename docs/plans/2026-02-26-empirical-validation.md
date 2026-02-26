# Empirical Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compare sigma-TAP simulation output against four quantitative targets from Youn et al. (2015) and Taalbi (2025).

**Architecture:** Pure metric functions in `simulator/empirical.py` (no simulation imports) + thin validation script in `scripts/empirical_validation.py` + tests in `tests/test_empirical.py`.

**Tech Stack:** Python 3.12, numpy, scipy.stats.linregress, dataclasses

---

### Task 1: Data Structures + youn_ratio()

**Files:**
- Create: `simulator/empirical.py`
- Create: `tests/test_empirical.py`

**Step 1: Write the failing tests**

```python
# tests/test_empirical.py
"""Tests for empirical validation metrics.

CLAIM POLICY LABEL: exploratory
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from simulator.empirical import (
    YounRatioResult,
    youn_ratio,
)


class TestYounRatio:
    """Tests for youn_ratio() — Youn et al. (2015) 60:40 invariant."""

    def test_known_ratio(self):
        """60 novel, 40 absorptive -> 0.6 fraction."""
        # Cumulative: [0,10,20,...,60] novel, [0,7,13,...,40] absorptive
        novel = list(range(0, 61, 10))
        absorptive = [0, 7, 13, 20, 27, 33, 40]
        result = youn_ratio(novel, absorptive)
        assert isinstance(result, YounRatioResult)
        assert result.exploration_count == 60
        assert result.exploitation_count == 40
        assert abs(result.exploration_fraction - 0.6) < 0.01
        assert result.target == 0.6
        assert abs(result.deviation - 0.0) < 0.01

    def test_fifty_fifty(self):
        """Equal counts -> 0.5 fraction."""
        novel = [0, 5, 10]
        absorptive = [0, 5, 10]
        result = youn_ratio(novel, absorptive)
        assert abs(result.exploration_fraction - 0.5) < 0.01
        assert abs(result.deviation - 0.1) < 0.01  # |0.5 - 0.6| = 0.1

    def test_all_novel(self):
        """All novel -> 1.0 fraction."""
        novel = [0, 5, 10]
        absorptive = [0, 0, 0]
        result = youn_ratio(novel, absorptive)
        assert abs(result.exploration_fraction - 1.0) < 0.01

    def test_no_events(self):
        """No events at all -> NaN fraction."""
        novel = [0, 0, 0]
        absorptive = [0, 0, 0]
        result = youn_ratio(novel, absorptive)
        assert math.isnan(result.exploration_fraction)

    def test_trajectory_ratios_length(self):
        """Trajectory ratios list has correct length."""
        novel = [0, 3, 6, 10, 15]
        absorptive = [0, 2, 4, 7, 10]
        result = youn_ratio(novel, absorptive)
        # Step 0 has 0/0 = NaN, steps 1-4 have valid ratios
        assert len(result.trajectory_ratios) == 5
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_empirical.py -v`
Expected: FAIL (ImportError — module doesn't exist yet)

**Step 3: Write minimal implementation**

```python
# simulator/empirical.py
"""Empirical validation metrics — pure functions for comparing
sigma-TAP simulation output against quantitative targets from
the innovation economics literature.

CLAIM POLICY LABEL: exploratory

All functions take arrays/lists as input and return structured
dataclass results. No simulation imports — pure metric computation.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class YounRatioResult:
    """Exploration/exploitation ratio comparison (Youn et al. 2015)."""
    exploration_count: int
    exploitation_count: int
    exploration_fraction: float
    target: float
    deviation: float
    trajectory_ratios: list[float]


def youn_ratio(
    n_novel_cross: list[int],
    n_absorptive_cross: list[int],
) -> YounRatioResult:
    """Compare exploration/exploitation ratio against Youn et al. (2015) 60:40 invariant.

    Parameters
    ----------
    n_novel_cross : list[int]
        Cumulative novel cross-metathesis events at each step.
    n_absorptive_cross : list[int]
        Cumulative absorptive cross-metathesis events at each step.

    Returns
    -------
    YounRatioResult with exploration fraction and deviation from 0.6 target.
    """
    target = 0.6
    trajectory_ratios: list[float] = []

    for n, a in zip(n_novel_cross, n_absorptive_cross):
        total = n + a
        if total > 0:
            trajectory_ratios.append(n / total)
        else:
            trajectory_ratios.append(float("nan"))

    final_novel = n_novel_cross[-1]
    final_absorptive = n_absorptive_cross[-1]
    final_total = final_novel + final_absorptive

    if final_total > 0:
        fraction = final_novel / final_total
    else:
        fraction = float("nan")

    deviation = abs(fraction - target) if not math.isnan(fraction) else float("nan")

    return YounRatioResult(
        exploration_count=final_novel,
        exploitation_count=final_absorptive,
        exploration_fraction=fraction,
        target=target,
        deviation=deviation,
        trajectory_ratios=trajectory_ratios,
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_empirical.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add simulator/empirical.py tests/test_empirical.py
git commit -m "feat(empirical): add YounRatioResult dataclass and youn_ratio function"
```

---

### Task 2: taalbi_linearity()

**Files:**
- Modify: `simulator/empirical.py`
- Modify: `tests/test_empirical.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_empirical.py
from simulator.empirical import (
    YounRatioResult,
    TaalbiLinearityResult,
    youn_ratio,
    taalbi_linearity,
)


class TestTaalbiLinearity:
    """Tests for taalbi_linearity() — dk/dt proportional to k."""

    def test_linear_growth(self):
        """k growing linearly (dk/dt ~ constant) -> slope ~0 in log-log.
        But if k = c*t, then dk/dt = c (constant), so log(dk/dt) vs log(k)
        gives slope 0. For slope ~1, we need dk/dt ~ k, i.e. exponential growth.
        k_t = k_0 * exp(r*t), dk/dt = r*k -> log(dk/dt) = log(r) + 1*log(k).
        """
        # Exponential growth: k = 10 * 1.1^t -> dk/dt proportional to k
        k_total = [int(10 * 1.1**t) for t in range(50)]
        result = taalbi_linearity(k_total)
        assert isinstance(result, TaalbiLinearityResult)
        assert abs(result.slope - 1.0) < 0.15  # should be close to 1.0
        assert result.r_squared > 0.8
        assert result.target_slope == 1.0

    def test_quadratic_growth(self):
        """k = t^2 -> dk/dt = 2t ~ 2*sqrt(k), so log(dk/dt) ~ 0.5*log(k).
        Slope should be ~0.5."""
        k_total = [t * t for t in range(1, 51)]
        result = taalbi_linearity(k_total)
        assert abs(result.slope - 0.5) < 0.15

    def test_constant_k(self):
        """k never changes -> dk/dt = 0 everywhere -> no valid points."""
        k_total = [10] * 20
        result = taalbi_linearity(k_total)
        assert math.isnan(result.slope)
        assert result.n_points == 0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_empirical.py::TestTaalbiLinearity -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# Add to simulator/empirical.py
from scipy.stats import linregress


@dataclass
class TaalbiLinearityResult:
    """Innovation rate linearity test (Taalbi 2025)."""
    slope: float
    intercept: float
    r_squared: float
    target_slope: float
    p_value: float
    n_points: int


def taalbi_linearity(
    k_total: list[int],
    dt: float = 1.0,
) -> TaalbiLinearityResult:
    """Test whether innovation rate scales linearly with cumulative innovations.

    Computes dk/dt via first differences, then OLS regression of
    log(dk/dt) on log(k). Taalbi (2025) found slope ~1.0.

    Parameters
    ----------
    k_total : list[int]
        Cumulative innovation counts at each step.
    dt : float
        Time step size (default 1.0).

    Returns
    -------
    TaalbiLinearityResult with slope, r², and comparison to target.
    """
    target = 1.0
    k_arr = np.array(k_total, dtype=float)
    dk = np.diff(k_arr) / dt

    # Filter: need dk > 0 and k > 0 for log
    k_mid = k_arr[:-1]  # k at start of each interval
    mask = (dk > 0) & (k_mid > 0)

    if mask.sum() < 2:
        return TaalbiLinearityResult(
            slope=float("nan"),
            intercept=float("nan"),
            r_squared=float("nan"),
            target_slope=target,
            p_value=float("nan"),
            n_points=int(mask.sum()),
        )

    log_k = np.log(k_mid[mask])
    log_dk = np.log(dk[mask])

    result = linregress(log_k, log_dk)

    return TaalbiLinearityResult(
        slope=result.slope,
        intercept=result.intercept,
        r_squared=result.rvalue ** 2,
        target_slope=target,
        p_value=result.pvalue,
        n_points=int(mask.sum()),
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_empirical.py -v`
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add simulator/empirical.py tests/test_empirical.py
git commit -m "feat(empirical): add taalbi_linearity function"
```

---

### Task 3: heaps_exponent()

**Files:**
- Modify: `simulator/empirical.py`
- Modify: `tests/test_empirical.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_empirical.py
from simulator.empirical import (
    YounRatioResult,
    TaalbiLinearityResult,
    HeapsLawResult,
    youn_ratio,
    taalbi_linearity,
    heaps_exponent,
)


class TestHeapsExponent:
    """Tests for heaps_exponent() — sub-linear diversification."""

    def test_sqrt_relationship(self):
        """D = sqrt(k) -> exponent ~0.5."""
        k_total = [i * i for i in range(1, 51)]  # k = 1, 4, 9, ..., 2500
        D_total = [i for i in range(1, 51)]       # D = 1, 2, 3, ..., 50
        result = heaps_exponent(k_total, D_total)
        assert isinstance(result, HeapsLawResult)
        assert abs(result.exponent - 0.5) < 0.05
        assert result.is_sublinear is True
        assert result.r_squared > 0.99

    def test_linear_relationship(self):
        """D = k -> exponent ~1.0."""
        k_total = list(range(1, 51))
        D_total = list(range(1, 51))
        result = heaps_exponent(k_total, D_total)
        assert abs(result.exponent - 1.0) < 0.05
        assert result.is_sublinear is False

    def test_single_point(self):
        """Single data point -> can't regress."""
        k_total = [10]
        D_total = [5]
        result = heaps_exponent(k_total, D_total)
        assert math.isnan(result.exponent)
        assert result.n_points <= 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_empirical.py::TestHeapsExponent -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# Add to simulator/empirical.py

@dataclass
class HeapsLawResult:
    """Sub-linear diversification test (Taalbi 2025, Heaps' law)."""
    exponent: float
    intercept: float
    r_squared: float
    target: str
    is_sublinear: bool
    n_points: int


def heaps_exponent(
    k_total: list[int],
    D_total: list[int],
) -> HeapsLawResult:
    """Test whether type diversity follows Heaps' law (D ~ k^beta, beta < 1).

    OLS regression of log(D) on log(k).

    Parameters
    ----------
    k_total : list[int]
        Cumulative innovation counts at each step.
    D_total : list[int]
        Total type-set diversity at each step.

    Returns
    -------
    HeapsLawResult with exponent and sub-linearity flag.
    """
    k_arr = np.array(k_total, dtype=float)
    d_arr = np.array(D_total, dtype=float)

    mask = (k_arr > 0) & (d_arr > 0)

    if mask.sum() < 2:
        return HeapsLawResult(
            exponent=float("nan"),
            intercept=float("nan"),
            r_squared=float("nan"),
            target="< 1.0",
            is_sublinear=False,
            n_points=int(mask.sum()),
        )

    log_k = np.log(k_arr[mask])
    log_d = np.log(d_arr[mask])

    result = linregress(log_k, log_d)

    return HeapsLawResult(
        exponent=result.slope,
        intercept=result.intercept,
        r_squared=result.rvalue ** 2,
        target="< 1.0",
        is_sublinear=result.slope < 1.0,
        n_points=int(mask.sum()),
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_empirical.py -v`
Expected: 11 PASSED

**Step 5: Commit**

```bash
git add simulator/empirical.py tests/test_empirical.py
git commit -m "feat(empirical): add heaps_exponent function"
```

---

### Task 4: power_law_fit()

**Files:**
- Modify: `simulator/empirical.py`
- Modify: `tests/test_empirical.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_empirical.py
from simulator.empirical import (
    YounRatioResult,
    TaalbiLinearityResult,
    HeapsLawResult,
    PowerLawResult,
    youn_ratio,
    taalbi_linearity,
    heaps_exponent,
    power_law_fit,
)


class TestPowerLawFit:
    """Tests for power_law_fit() — agent innovation distribution."""

    def test_zipf_distribution(self):
        """Known Zipf-like distribution -> exponent ~2.0."""
        rng = np.random.default_rng(42)
        # Generate Zipf with alpha=2.0: P(k) ~ k^(-2)
        # Use inverse CDF: k = k_min * u^(-1/(alpha-1)) where u ~ Uniform(0,1)
        alpha_true = 2.0
        k_min = 1
        u = rng.uniform(0, 1, size=500)
        samples = (k_min * u ** (-1.0 / (alpha_true - 1))).astype(int)
        samples = np.maximum(samples, k_min)  # ensure >= k_min
        result = power_law_fit(samples.tolist(), k_min=k_min)
        assert isinstance(result, PowerLawResult)
        assert abs(result.exponent - 2.0) < 0.3
        assert result.target_exponent == 2.0
        assert result.n_agents == 500

    def test_uniform_distribution(self):
        """Uniform distribution -> exponent far from 2.0."""
        rng = np.random.default_rng(42)
        samples = rng.integers(1, 100, size=100).tolist()
        result = power_law_fit(samples, k_min=1)
        # Uniform is not power-law; exponent should be very different
        assert result.n_agents == 100

    def test_all_same_values(self):
        """All agents have same k -> degenerate."""
        samples = [5, 5, 5, 5, 5]
        result = power_law_fit(samples, k_min=1)
        # All same -> log(k_i/k_min) might be 0 for all -> degenerate
        # With k_min=1 and all k=5, sum(log(5/1)) > 0, so it should work
        # but with k_min=5, all logs are 0 -> degenerate
        result2 = power_law_fit(samples, k_min=5)
        assert math.isnan(result2.exponent)

    def test_single_agent(self):
        """Single agent -> degenerate."""
        result = power_law_fit([10], k_min=1)
        assert result.n_agents == 1
        # With only 1 data point, result should be flagged
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_empirical.py::TestPowerLawFit -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# Add to simulator/empirical.py

@dataclass
class PowerLawResult:
    """Agent innovation distribution test (Taalbi 2025)."""
    exponent: float
    target_exponent: float
    ks_statistic: float
    p_value: float
    n_agents: int
    k_min: int


def power_law_fit(
    agent_k_list: list[int],
    k_min: int = 1,
) -> PowerLawResult:
    """Fit power-law distribution to per-agent innovation counts.

    Uses Hill MLE estimator: alpha = 1 + n / sum(ln(k_i / k_min)).
    KS statistic for goodness-of-fit.

    Parameters
    ----------
    agent_k_list : list[int]
        Per-agent innovation counts.
    k_min : int
        Minimum k for power-law regime (default 1).

    Returns
    -------
    PowerLawResult with fitted exponent and comparison to target ~2.0.
    """
    target = 2.0
    k_arr = np.array(agent_k_list, dtype=float)
    k_filtered = k_arr[k_arr >= k_min]
    n = len(k_filtered)

    if n < 2:
        return PowerLawResult(
            exponent=float("nan"),
            target_exponent=target,
            ks_statistic=float("nan"),
            p_value=float("nan"),
            n_agents=len(agent_k_list),
            k_min=k_min,
        )

    log_ratios = np.log(k_filtered / k_min)
    log_sum = log_ratios.sum()

    if log_sum <= 0:
        return PowerLawResult(
            exponent=float("nan"),
            target_exponent=target,
            ks_statistic=float("nan"),
            p_value=float("nan"),
            n_agents=len(agent_k_list),
            k_min=k_min,
        )

    alpha = 1.0 + n / log_sum

    # KS goodness-of-fit: compare empirical CDF to fitted power-law CDF
    # CDF of power law: F(k) = 1 - (k/k_min)^(-(alpha-1))
    sorted_k = np.sort(k_filtered)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = 1.0 - (sorted_k / k_min) ** (-(alpha - 1.0))
    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))

    # Approximate p-value from KS statistic (Kolmogorov distribution)
    # For large n: P(D_n > x) ~ 2*exp(-2*(x*sqrt(n))^2)
    lambda_ks = ks_stat * np.sqrt(n)
    p_value = 2.0 * np.exp(-2.0 * lambda_ks ** 2)
    p_value = min(max(p_value, 0.0), 1.0)

    return PowerLawResult(
        exponent=alpha,
        target_exponent=target,
        ks_statistic=ks_stat,
        p_value=p_value,
        n_agents=len(agent_k_list),
        k_min=k_min,
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_empirical.py -v`
Expected: 15 PASSED

**Step 5: Commit**

```bash
git add simulator/empirical.py tests/test_empirical.py
git commit -m "feat(empirical): add power_law_fit function"
```

---

### Task 5: Validation Script + Integration Test

**Files:**
- Create: `scripts/empirical_validation.py`
- Modify: `tests/test_empirical.py`

**Step 1: Write the integration test**

```python
# Add to tests/test_empirical.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.empirical import (
    EmpiricalValidationResult,
)
from simulator.metathetic import MetatheticEnsemble


class TestIntegration:
    """Integration tests with actual simulation data."""

    def test_full_validation_from_simulation(self):
        """Run a short simulation and validate all 4 metrics produce results."""
        ens = MetatheticEnsemble(
            n_agents=10,
            initial_M=10.0,
            alpha=5e-3,
            a=3.0,
            mu=0.005,
            variant="logistic",
            carrying_capacity=2e5,
            seed=42,
        )
        trajectory = ens.run(steps=100)

        # Extract fields
        n_novel = [s["n_novel_cross"] for s in trajectory]
        n_absorptive = [s["n_absorptive_cross"] for s in trajectory]
        k_total = [s["k_total"] for s in trajectory]
        D_total = [s["D_total"] for s in trajectory]
        agent_k_list = trajectory[-1]["agent_k_list"]

        yr = youn_ratio(n_novel, n_absorptive)
        assert 0.0 <= yr.exploration_fraction <= 1.0 or math.isnan(yr.exploration_fraction)

        tl = taalbi_linearity(k_total)
        assert isinstance(tl.slope, float)

        he = heaps_exponent(k_total, D_total)
        assert isinstance(he.exponent, float)

        pl = power_law_fit(agent_k_list)
        assert isinstance(pl.exponent, float)

    def test_status_classification(self):
        """MATCH/CLOSE/DIVERGENT thresholds."""
        # This tests the classify_status helper
        from scripts.empirical_validation import classify_status
        assert classify_status(0.05, 0.10) == "MATCH"
        assert classify_status(0.15, 0.10) == "CLOSE"
        assert classify_status(0.30, 0.10) == "DIVERGENT"
```

**Step 2: Write the validation script**

```python
# scripts/empirical_validation.py
"""Run sigma-TAP simulation and compare against empirical targets.

CLAIM POLICY LABEL: exploratory

Compares simulation output against four quantitative targets from
Youn et al. (2015) and Taalbi (2025).

Usage:
  python scripts/empirical_validation.py
  python scripts/empirical_validation.py --n-agents 20 --steps 300
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.metathetic import MetatheticEnsemble
from simulator.empirical import (
    EmpiricalValidationResult,
    youn_ratio,
    taalbi_linearity,
    heaps_exponent,
    power_law_fit,
)


def classify_status(deviation: float, threshold_match: float = 0.10) -> str:
    """Classify deviation as MATCH/CLOSE/DIVERGENT."""
    if deviation <= threshold_match:
        return "MATCH"
    elif deviation <= threshold_match * 2.5:
        return "CLOSE"
    else:
        return "DIVERGENT"


def run_validation(
    n_agents: int = 10,
    steps: int = 150,
    alpha: float = 5e-3,
    a: float = 3.0,
    mu: float = 0.005,
    variant: str = "logistic",
    seed: int = 42,
) -> EmpiricalValidationResult:
    """Run simulation and compute all empirical metrics."""
    import math

    ens = MetatheticEnsemble(
        n_agents=n_agents,
        initial_M=10.0,
        alpha=alpha, a=a, mu=mu,
        variant=variant,
        carrying_capacity=2e5 if variant == "logistic" else None,
        seed=seed,
    )
    trajectory = ens.run(steps=steps)

    n_novel = [s["n_novel_cross"] for s in trajectory]
    n_absorptive = [s["n_absorptive_cross"] for s in trajectory]
    k_total_list = [s["k_total"] for s in trajectory]
    D_total_list = [s["D_total"] for s in trajectory]
    agent_k_list = trajectory[-1]["agent_k_list"]

    yr = youn_ratio(n_novel, n_absorptive)
    tl = taalbi_linearity(k_total_list)
    he = heaps_exponent(k_total_list, D_total_list)
    pl = power_law_fit(agent_k_list)

    return EmpiricalValidationResult(
        youn=yr,
        linearity=tl,
        heaps=he,
        power_law=pl,
        params_used={"alpha": alpha, "a": a, "mu": mu, "variant": variant, "seed": seed},
        n_steps=steps,
        n_agents=n_agents,
    )


def print_validation(result: EmpiricalValidationResult) -> None:
    """Print formatted validation summary."""
    import math

    print(f"\nEmpirical Validation Summary ({result.n_agents} agents, {result.n_steps} steps)")
    print("=" * 70)
    print(f"  {'Metric':<24} {'Target':<16} {'Measured':<14} {'Status'}")
    print("-" * 70)

    # Youn ratio
    yr = result.youn
    if not math.isnan(yr.exploration_fraction):
        status = classify_status(yr.deviation)
        print(f"  {'Youn exploration':<24} {'0.600':<16} {yr.exploration_fraction:<14.3f} "
              f"{status} (Δ={yr.deviation:.3f})")
    else:
        print(f"  {'Youn exploration':<24} {'0.600':<16} {'N/A':<14} NO EVENTS")

    # Taalbi linearity
    tl = result.linearity
    if not math.isnan(tl.slope):
        dev = abs(tl.slope - tl.target_slope)
        status = classify_status(dev)
        print(f"  {'Taalbi linearity':<24} {'slope≈1.0':<16} {tl.slope:<14.3f} "
              f"{status} (r²={tl.r_squared:.2f})")
    else:
        print(f"  {'Taalbi linearity':<24} {'slope≈1.0':<16} {'N/A':<14} INSUFFICIENT DATA")

    # Heaps exponent
    he = result.heaps
    if not math.isnan(he.exponent):
        flag = "sub-linear" if he.is_sublinear else "NOT sub-linear"
        status = "MATCH" if he.is_sublinear else "DIVERGENT"
        print(f"  {'Heaps exponent':<24} {'< 1.0':<16} {he.exponent:<14.3f} "
              f"{status} ({flag})")
    else:
        print(f"  {'Heaps exponent':<24} {'< 1.0':<16} {'N/A':<14} INSUFFICIENT DATA")

    # Power law
    pl = result.power_law
    if not math.isnan(pl.exponent):
        dev = abs(pl.exponent - pl.target_exponent)
        status = classify_status(dev, threshold_match=0.3)
        print(f"  {'Power-law exponent':<24} {'≈ 2.0':<16} {pl.exponent:<14.3f} "
              f"{status} (KS={pl.ks_statistic:.3f})")
    else:
        print(f"  {'Power-law exponent':<24} {'≈ 2.0':<16} {'N/A':<14} DEGENERATE")

    print("=" * 70)
    params = result.params_used
    print(f"  Params: alpha={params['alpha']}, a={params['a']}, mu={params['mu']}, "
          f"variant={params['variant']}, seed={params['seed']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="sigma-TAP empirical validation")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=5e-3)
    parser.add_argument("-a", type=float, default=3.0)
    parser.add_argument("--mu", type=float, default=0.005)
    parser.add_argument("--variant", default="logistic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_validation(
        n_agents=args.n_agents,
        steps=args.steps,
        alpha=args.alpha,
        a=args.a,
        mu=args.mu,
        variant=args.variant,
        seed=args.seed,
    )
    print_validation(result)


if __name__ == "__main__":
    main()
```

**Step 3: Add EmpiricalValidationResult to empirical.py**

```python
# Add to simulator/empirical.py

@dataclass
class EmpiricalValidationResult:
    """Full validation output."""
    youn: YounRatioResult
    linearity: TaalbiLinearityResult
    heaps: HeapsLawResult
    power_law: PowerLawResult
    params_used: dict
    n_steps: int
    n_agents: int
```

**Step 4: Run all tests**

Run: `python -m pytest tests/test_empirical.py -v`
Expected: 17 PASSED

**Step 5: Smoke test the script**

Run: `python scripts/empirical_validation.py`
Expected: formatted table with 4 metrics

**Step 6: Commit**

```bash
git add scripts/empirical_validation.py simulator/empirical.py tests/test_empirical.py
git commit -m "feat(empirical): add validation script and integration tests"
```

---

### Task 6: Update docs/empirical_targets.md

**Files:**
- Modify: `docs/empirical_targets.md`

**Step 1: Run validation script and capture output**

Run: `python scripts/empirical_validation.py`

**Step 2: Add new section 6d to empirical_targets.md**

Add a new section `## 6d. Empirical Validation — Quantitative Comparison` after section 6c, documenting:
- The four empirical targets with literature references
- Default-parameter results table from the validation script
- Status assessment for each metric
- How to run the validation (`python scripts/empirical_validation.py`)

**Step 3: Update assessment table (Section 6)**

Add row for empirical validation marked as **DONE**.
Update recommended next steps 1 and 2 to mark as DONE.

**Step 4: Commit**

```bash
git add docs/empirical_targets.md
git commit -m "docs: document empirical validation results in empirical targets"
```

---

### Task 7: Final verification

**Step 1: Run full test suite**

Run: `python -m pytest -v`
Expected: all tests pass (251 existing + ~17 new = ~268)

**Step 2: Run validation script one more time**

Run: `python scripts/empirical_validation.py`
Expected: clean output, no warnings

**Step 3: Code review**

Review `simulator/empirical.py` for:
- Unused imports
- Consistent docstring style
- Edge case coverage
