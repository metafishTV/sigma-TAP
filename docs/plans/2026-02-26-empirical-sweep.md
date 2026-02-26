# Empirical Parameter Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run empirical validation metrics across a parameter grid to find best-fit regimes and map sensitivity.

**Architecture:** Dedicated sweep script (`scripts/empirical_sweep.py`) importing from existing `simulator/empirical.py` metric functions. Tests added to `tests/test_empirical.py`.

**Tech Stack:** Python 3.12, numpy, itertools, dataclasses

---

### Task 1: SweepPoint dataclass + composite_score + rank functions

**Files:**
- Create: `scripts/empirical_sweep.py`
- Modify: `tests/test_empirical.py`

Create the sweep script with dataclasses and scoring logic (no simulation yet):

```python
# scripts/empirical_sweep.py
"""Parameter sweep for sigma-TAP empirical validation.

CLAIM POLICY LABEL: exploratory

Runs empirical validation metrics across a grid of parameters to find
best-fit regimes and map parameter sensitivity.

Usage:
  python scripts/empirical_sweep.py
  python scripts/empirical_sweep.py --quick
  python scripts/empirical_sweep.py --steps 500 --top 10
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from itertools import product as grid_product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from simulator.metathetic import MetatheticEnsemble
from simulator.empirical import (
    EmpiricalValidationResult,
    youn_ratio,
    taalbi_linearity,
    heaps_exponent,
    power_law_fit,
)

# Import classify_status from sibling script
sys.path.insert(0, os.path.dirname(__file__))
from empirical_validation import classify_status


# ---------------------------------------------------------------------------
# Default grid
# ---------------------------------------------------------------------------

DEFAULT_GRID = {
    "alpha": [1e-3, 3e-3, 5e-3, 1e-2],
    "a": [2.0, 3.0, 5.0, 8.0],
    "mu": [0.002, 0.005, 0.01, 0.02],
    "n_agents": [10, 15, 20],
}
QUICK_GRID = {
    "alpha": [1e-3, 5e-3],
    "a": [3.0, 8.0],
    "mu": [0.005, 0.02],
    "n_agents": [10],
}
DEFAULT_SEEDS = [42, 123, 456]
QUICK_SEEDS = [42]
DEFAULT_STEPS = 200
QUICK_STEPS = 50
NAN_PENALTY = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    params: dict
    results: list[EmpiricalValidationResult]
    mean_youn_deviation: float
    mean_linearity_deviation: float
    mean_heaps_exponent: float
    heaps_match: bool
    mean_powerlaw_deviation: float
    composite_score: float

@dataclass
class SweepResult:
    points: list[SweepPoint]
    best: list[SweepPoint]
    sensitivity: dict[str, dict[str, float]]
    grid: dict
    seeds: list[int]
    steps: int
    total_sims: int
```

Scoring functions:

```python
def compute_composite_score(results: list[EmpiricalValidationResult]) -> SweepPoint:
```

Takes a list of EmpiricalValidationResult (one per seed), computes mean deviations:
- youn: mean of |exploration_fraction - 0.6|, NaN -> NAN_PENALTY
- linearity: mean of |slope - 1.0|, NaN -> NAN_PENALTY
- heaps: 0.0 if mean exponent < 1.0, else (mean_exponent - 1.0), NaN -> NAN_PENALTY
- powerlaw: mean of |exponent - 2.0|, NaN -> NAN_PENALTY
- composite = sum of all 4 deviations

```python
def rank_sweep_results(points: list[SweepPoint], top_n: int = 5) -> list[SweepPoint]:
```
Sort by composite_score ascending, return top_n.

```python
def compute_sensitivity(points: list[SweepPoint], grid: dict) -> dict:
```
For each param in grid, group points by param value, compute mean deviation per group per metric, return range (max - min) as sensitivity.

Tests (add to tests/test_empirical.py):
1. test_composite_score_known: known deviations -> correct score
2. test_composite_score_nan_penalty: NaN -> penalty applied
3. test_rank_results: known scores -> correct ordering
4. test_sensitivity_dominant: known data -> correct most-sensitive param

**Commit:** `feat(sweep): add SweepPoint dataclass and scoring functions`

---

### Task 2: Simulation runner + print functions

**Files:**
- Modify: `scripts/empirical_sweep.py`

Add:

```python
def run_single_point(params: dict, seeds: list[int], steps: int) -> SweepPoint:
```
For each seed: create MetatheticEnsemble, run simulation, extract trajectory fields, call all 4 metrics, collect EmpiricalValidationResult. Then call compute_composite_score. Catch exceptions per seed and log warnings.

```python
def run_sweep(grid: dict, seeds: list[int], steps: int) -> SweepResult:
```
Generate all param combinations via itertools.product, run run_single_point for each, collect SweepPoints, compute sensitivity, rank results.

Print with progress: `[42/576] alpha=0.003 a=5.0 mu=0.005 agents=20 score=0.42`

```python
def print_sweep_results(result: SweepResult, top_n: int = 5) -> None:
```
Print best-fit table and sensitivity table.

```python
def main() -> None:
```
argparse with --quick, --steps, --top, --variant, then run sweep and print.

Add integration test:
5. test_sweep_small_grid: run with QUICK_GRID, 1 seed, 50 steps -> completes without error, returns SweepResult with correct structure

**Commit:** `feat(sweep): add simulation runner and output formatting`

---

### Task 3: Run the full sweep, update docs

**Files:**
- Modify: `docs/empirical_targets.md`

1. Run: `python scripts/empirical_sweep.py` (full grid, ~10-15 min)
2. Capture output
3. Add results to docs/empirical_targets.md section 6d under new subsection "Parameter Sweep Results"
4. Update future work items

**Commit:** `docs: add parameter sweep results to empirical targets`
