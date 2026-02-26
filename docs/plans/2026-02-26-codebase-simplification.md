# Codebase Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate dead code, redundant computation, and duplicated logic across the sigma-TAP codebase while preserving identical output fidelity and all granularity.

**Architecture:** Single atomic branch with 14 findings grouped into 7 tasks. Each task is independently testable. Full test suite (273 tests) gates every commit.

**Tech Stack:** Python 3.12, numpy, scipy, pytest

**Python:** `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`
**Repo:** `C:\Users\user\Documents\New folder\sigma-TAP-repo`
**Test command:** `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`

---

### Task 1: Dead Code Removal

**Files:**
- Delete: `simulator/pressure.py`
- Delete: `simulator/projection.py`
- Modify: `simulator/tap.py:4,7-11` (remove `from math import comb` and `innovation_kernel`)
- Modify: `simulator/analysis.py:47-49` (remove `pass_c_additional_runs`)
- Modify: `scripts/run_demo.py:62` (replace function call with inline comment)

**Step 1: Run baseline tests to confirm 273 pass**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`
Expected: 273 passed

**Step 2: Delete `simulator/pressure.py`**

```bash
git rm simulator/pressure.py
```

**Step 3: Delete `simulator/projection.py`**

```bash
git rm simulator/projection.py
```

**Step 4: Remove `innovation_kernel` and `from math import comb` from `tap.py`**

Remove lines 4 and 7-11 from `simulator/tap.py`. The file should start:
```python
from __future__ import annotations

import math


def innovation_kernel_closed(M: float, alpha: float, a: float) -> float:
```

**Step 5: Remove `pass_c_additional_runs` from `analysis.py`**

Remove lines 47-49 from `simulator/analysis.py`:
```python
def pass_c_additional_runs(k_pressure_params: int, slices: int = 3, lhs_points: int = 150, replicates: int = 24) -> int:
    """Additional run count for pressure Pass C design."""
    return k_pressure_params * slices * lhs_points * replicates
```

**Step 6: Update `scripts/run_demo.py` line 62**

Replace:
```python
print("pass_c_runs_k9", pass_c_additional_runs(9))
```
With:
```python
# Pass C run count for k=9 pressure params: 9 * 3 * 150 * 24 = 97200
```

Also remove `pass_c_additional_runs` from the import at top of `run_demo.py`.

**Step 7: Run tests**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`
Expected: 273 passed

**Step 8: Commit**

```bash
git add -A && git commit -m "refactor: remove dead code (pressure.py, projection.py, innovation_kernel, pass_c_additional_runs)"
```

---

### Task 2: Import Cleanup + Redundant max(0,...) in tap.py

**Files:**
- Modify: `simulator/analysis.py:74,105,160,252,357` (move `import math` to top)
- Modify: `simulator/tap.py:84` (remove outer `max(0.0, base)`)

**Step 1: Move `import math` to top of `analysis.py`**

Add `import math` after line 2 (`from statistics import median`) at the module level. Then remove the 5 deferred `import math` lines inside:
- `classify_regime` (line 74)
- `find_fixed_point` (line 105)
- `fit_explosive_logistic_boundary` (line 160)
- `fit_explosive_logistic_boundary_3d` (line 252)
- `innovation_rate_scaling` (line 357)

NOTE: Line numbers will have shifted after Task 1 removed `pass_c_additional_runs`. Find lines by searching for `import math` inside function bodies.

**Step 2: Remove redundant `max(0.0, base)` in `compute_birth_term`**

In `simulator/tap.py`, replace:
```python
    return max(0.0, base)
```
With:
```python
    return base
```

This is safe because:
- `innovation_kernel_closed` returns `max(0.0, ...)` at line 24
- `innovation_kernel_two_scale` returns `max(0.0, ...)` at line 48
- `apply_logistic_constraint` multiplies non-negative birth_term by `max(0.0, ...)` factor

**Step 3: Run tests**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`
Expected: 273 passed

**Step 4: Commit**

```bash
git add simulator/analysis.py simulator/tap.py && git commit -m "refactor: move deferred math imports to module level, remove redundant max(0) guard"
```

---

### Task 3: Eigenvalue Analysis + texture_type Wiring

**Files:**
- Modify: `simulator/taps_sensitivity.py:97-99` (wire `classify_dM_texture`)
- Modify: `simulator/taps_sensitivity.py:324` (remove redundant `eigvals` call)

**Step 1: Wire `classify_dM_texture` into `classify_step`**

In `simulator/taps_sensitivity.py`, the `classify_step` function currently has (lines 97-99):
```python
    # texture_type: proxy using pressure_regime until WI-3 (Task 4)
    # implements real dM-variance-based texture classification.
    result["texture_type"] = result["pressure_regime"]
```

Replace with:
```python
    # texture_type: classified from dM variance via classify_dM_texture().
    # Steps before the classification window get "unclassified".
    result["texture_type"] = "unclassified"
```

Then, in `build_transition_map`, add dM texture classification. Before the line:
```python
    classifications: list[dict[str, str]] = [
        classify_step(all_scores, ano_scores, rip_result, ratios, step=t)
        for t in range(n_steps)
    ]
```

Add texture computation and inject it into classifications:
```python
    # Compute dM-based texture classification for all steps.
    dM_textures = classify_dM_texture(trajectory, window=10)
    _TEXTURE_LABELS = {0: "unclassified", 1: "placid_randomized",
                       2: "placid_clustered", 3: "disturbed_reactive",
                       4: "turbulent"}

    classifications: list[dict[str, str]] = []
    for t in range(n_steps):
        c = classify_step(all_scores, ano_scores, rip_result, ratios, step=t)
        c["texture_type"] = _TEXTURE_LABELS.get(dM_textures[t], "unclassified")
        classifications.append(c)
```

And remove the old list comprehension that was there.

**Step 2: Remove redundant `eigvals(P)` in `eigenvalue_analysis`**

In `simulator/taps_sensitivity.py`, the `eigenvalue_analysis` function has:
```python
        # --- Eigenvalue decomposition ---
        eigenvalues = np.linalg.eigvals(P)

        # Sort by magnitude descending
        idx_sorted = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx_sorted]
```

Then later:
```python
        evals_T, evecs_T = np.linalg.eig(P.T)
```

Replace the first block and merge with the second. The eigenvalues of P^T equal eigenvalues of P:
```python
        # --- Eigenvalue decomposition ---
        # Eigenvalues of P^T equal eigenvalues of P (transpose preserves spectrum).
        # Compute P^T decomposition once to get both eigenvalues and the
        # stationary distribution eigenvector.
        evals_T, evecs_T = np.linalg.eig(P.T)

        # Sort eigenvalues by magnitude descending
        idx_sorted = np.argsort(-np.abs(evals_T))
        eigenvalues = evals_T[idx_sorted]
```

Then remove the later `evals_T, evecs_T = np.linalg.eig(P.T)` line (it's now done above).

**Step 3: Run tests**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`
Expected: 273 passed. If any tests check for `texture_type == pressure_regime` equality, they will need updating — search tests first.

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/test_taps_sensitivity.py -q -v` to see which tests touch texture_type.

**Step 4: Commit**

```bash
git add simulator/taps_sensitivity.py && git commit -m "refactor: wire classify_dM_texture into texture_type axis, eliminate redundant eigendecomposition"
```

---

### Task 4: Cache `_event_deltas` in TAPS Scoring

**Files:**
- Modify: `simulator/taps.py:53-60,84-101,220-239,293-295,373-393`

**Step 1: Add optional `_deltas` parameter to all four sub-functions**

In `simulator/taps.py`, modify each function signature to accept an optional pre-computed deltas list:

For `compute_transvolution`:
```python
def compute_transvolution(trajectory: Trajectory, *, _deltas: list[dict[str, int]] | None = None) -> dict[str, list[float]]:
    ...
    deltas = _deltas if _deltas is not None else _event_deltas(trajectory)
```

For `compute_anopression`:
```python
def compute_anopression(
    trajectory: Trajectory,
    mu: float = 0.005,
    *,
    _deltas: list[dict[str, int]] | None = None,
) -> dict[str, list[float]]:
    ...
    deltas = _deltas if _deltas is not None else _event_deltas(trajectory)
```

For `compute_praxis`:
```python
def compute_praxis(trajectory: Trajectory, *, _deltas: list[dict[str, int]] | None = None) -> dict[str, list[float]]:
    ...
    deltas = _deltas if _deltas is not None else _event_deltas(trajectory)
```

For `compute_syntegration`:
```python
def compute_syntegration(trajectory: Trajectory, *, _deltas: list[dict[str, int]] | None = None) -> dict[str, list[float]]:
    ...
    deltas = _deltas if _deltas is not None else _event_deltas(trajectory)
```

**Step 2: Cache in `compute_all_scores`**

```python
def compute_all_scores(
    trajectory: Trajectory,
    mu: float = 0.005,
) -> dict[str, list[float]]:
    deltas = _event_deltas(trajectory)
    t_scores = compute_transvolution(trajectory, _deltas=deltas)
    a_scores = compute_anopression(trajectory, mu=mu, _deltas=deltas)
    p_scores = compute_praxis(trajectory, _deltas=deltas)
    s_scores = compute_syntegration(trajectory, _deltas=deltas)

    all_scores = {}
    for d in [t_scores, a_scores, p_scores, s_scores]:
        all_scores.update(d)
    return all_scores
```

**Step 3: Run tests**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/test_taps.py tests/test_taps_sensitivity.py tests/test_predictive.py -q`
Expected: All pass

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`
Expected: 273 passed

**Step 4: Commit**

```bash
git add simulator/taps.py && git commit -m "perf: cache _event_deltas in compute_all_scores, pass to sub-functions"
```

---

### Task 5: Cache Active Agents + Type Counts in Metathetic Engine

**Files:**
- Modify: `simulator/metathetic.py` (run loop ~lines 770-830)

**Step 1: Cache `_active_agents()` at step boundary**

In the `run()` method of `MetatheticEnsemble`, at the start of the step loop (after step 2a `_step_agents`), compute active once:

Find the snapshot block starting at line 783:
```python
            # 5. Record ensemble snapshot.
            active = self._active_agents()
```

This already caches `active` for the snapshot block. Now cache `total_M` to avoid computing it twice. Replace:
```python
            snapshot = {
                "step": step,
                "D_total": self._total_diversity(),
                "k_total": sum(a.k for a in active),
                "total_M": sum(a.M_local for a in active),
                ...
                "innovation_potential": self.env.innovation_potential(
                    sum(a.M_local for a in active)
                ),
```

With:
```python
            total_M = sum(a.M_local for a in active)
            snapshot = {
                "step": step,
                "D_total": self._total_diversity(),
                "k_total": sum(a.k for a in active),
                "total_M": total_M,
                ...
                "innovation_potential": self.env.innovation_potential(total_M),
```

**Step 2: Cache `_all_type_counts()` — compute once after metathesis, reuse**

Currently `_all_type_counts()` is called at:
1. Line 585 inside `_check_cross_metathesis()`
2. Line 489 inside `_convergence_measure()` (called at line 795)
3. Line 822 for temporal state

For the snapshot block (lines 795 + 822), compute once:

Replace:
```python
                "convergence": self._convergence_measure(),
```
...and later...
```python
            type_counts_for_context = self._all_type_counts()
```

With: compute type_counts once before the snapshot, pass to a modified `_convergence_measure`:

```python
            type_counts = self._all_type_counts()
            ...
                "convergence": self._convergence_measure(type_counts=type_counts),
            ...
            temporal_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for a in self.agents:
                ts = a.temporal_state_with_context(type_counts)
                temporal_counts[ts] += 1
```

And modify `_convergence_measure` to accept optional pre-computed counts:

```python
    def _convergence_measure(self, type_counts: dict[int, int] | None = None) -> float:
        counts = type_counts if type_counts is not None else self._all_type_counts()
        if not counts:
            return 0.0
        active = self._active_agents()
        if not active:
            return 0.0
        max_count = max(counts.values())
        return max_count / len(active)
```

NOTE: The `_check_cross_metathesis` call at line 585 computes its own `_all_type_counts()` which is appropriate since it runs BEFORE the snapshot and the type sets may change during metathesis. Do NOT try to share between cross-metathesis and snapshot — they may see different state.

**Step 3: Run tests**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/test_metathetic.py -q -v`
Expected: All pass

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`
Expected: 273 passed

**Step 4: Commit**

```bash
git add simulator/metathetic.py && git commit -m "perf: cache total_M and type_counts in metathetic run loop"
```

---

### Task 6: OLS Consolidation + Welford + Vectorize sigma_arr + H decay

**Files:**
- Modify: `simulator/longrun.py:18-53` (delegate to empirical.heaps_exponent)
- Modify: `simulator/analysis.py:346-391` (replace hand-rolled OLS with linregress)
- Modify: `simulator/predictive.py:172-184` (Welford's algorithm)
- Modify: `simulator/continuous.py:56` (parameterize H decay)
- Modify: `simulator/continuous.py:134` (vectorize sigma_arr)
- Modify: `simulator/state.py` (add `h_decay` field to ModelParams)

**Step 1: Consolidate `longrun.heaps_law_fit` to use `empirical.heaps_exponent`**

In `simulator/longrun.py`, replace the body of `heaps_law_fit` with:
```python
def heaps_law_fit(D_series: list[float], k_series: list[float]) -> dict:
    """Fit D(k) ~ k^beta via log-log OLS.

    Heaps' law predicts beta < 1: the rate of new type discovery declines
    relative to total innovation as the system matures.

    Returns dict with 'beta', 'intercept', 'r_squared', 'n_points'.
    """
    from simulator.empirical import heaps_exponent

    if len(D_series) < 2 or len(k_series) < 2:
        return {"beta": 1.0, "intercept": 0.0, "r_squared": 0.0, "n_points": 0}

    # empirical.heaps_exponent takes (k_total, D_total) — note arg order
    result = heaps_exponent(
        k_total=[int(k) for k in k_series],
        D_total=[int(d) for d in D_series],
    )

    import math
    if math.isnan(result.exponent):
        return {"beta": 1.0, "intercept": 0.0, "r_squared": 0.0, "n_points": result.n_points}

    return {
        "beta": result.exponent,
        "intercept": result.intercept,
        "r_squared": result.r_squared,
        "n_points": result.n_points,
    }
```

Remove `import math` from the top of `longrun.py` only if it has no other uses in the file. Check first — `top_k_share` uses `math.ceil`. Keep `import math` at top level.

**Step 2: Replace hand-rolled OLS in `innovation_rate_scaling`**

In `simulator/analysis.py`, replace the OLS body of `innovation_rate_scaling`:
```python
def innovation_rate_scaling(
    m_traj: list[float],
    dt: float = 1.0,
) -> dict:
    """Fit dk/dt ~ k^sigma using log-log OLS on finite differences."""
    if len(m_traj) < 3:
        return {"exponent": 1.0, "r_squared": 0.0, "n_points": len(m_traj)}

    import numpy as np
    from scipy.stats import linregress

    rates = [(m_traj[i + 1] - m_traj[i]) / dt for i in range(len(m_traj) - 1)]
    midpoints = [0.5 * (m_traj[i] + m_traj[i + 1]) for i in range(len(m_traj) - 1)]

    log_k = []
    log_rate = []
    for k, r in zip(midpoints, rates):
        if k > 0 and r > 0:
            log_k.append(math.log(k))
            log_rate.append(math.log(r))

    n = len(log_k)
    if n < 2:
        return {"exponent": 1.0, "r_squared": 0.0, "n_points": n}

    result = linregress(log_k, log_rate)
    return {
        "exponent": result.slope,
        "r_squared": result.rvalue ** 2,
        "n_points": n,
    }
```

NOTE: `import numpy as np` and `from scipy.stats import linregress` are deferred because `analysis.py` currently has no scipy/numpy imports and we don't want to add heavyweight imports to a module that is otherwise pure-Python for its other functions. Deferred imports for numpy/scipy (optional heavyweight deps) are reasonable unlike deferred `import math`.

**Step 3: Replace O(n^2) expanding window with Welford's algorithm**

In `simulator/predictive.py`, replace lines 172-184 of `detect_adpression`:
```python
    events: list[tuple[int, float, float]] = []
    for i in range(burn_in, len(surprisals)):
        window = surprisals[:i]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        sd = variance ** 0.5
```

With Welford's online algorithm:
```python
    events: list[tuple[int, float, float]] = []

    # Welford's online algorithm for running mean and variance
    count = 0
    mean = 0.0
    M2 = 0.0
    for k in range(burn_in):
        count += 1
        delta = surprisals[k] - mean
        mean += delta / count
        delta2 = surprisals[k] - mean
        M2 += delta * delta2

    for i in range(burn_in, len(surprisals)):
        # Update running stats with surprisals[i-1] if not yet included
        # (burn_in already processed indices 0..burn_in-1)
        # Now compute threshold using stats from surprisals[:i]
        variance = M2 / count if count > 0 else 0.0
        sd = variance ** 0.5
```

Wait — the original code computes stats from `surprisals[:i]` (indices 0 to i-1) for each i. The Welford init should process indices 0 to burn_in-1. Then for each i from burn_in onward, we compute threshold from current stats, check surprisals[i], then update stats to include surprisals[i] for the next iteration.

Corrected:
```python
    events: list[tuple[int, float, float]] = []

    # Welford's online algorithm — initialize with burn-in window
    count = 0
    mean = 0.0
    M2 = 0.0
    for k in range(burn_in):
        count += 1
        delta = surprisals[k] - mean
        mean += delta / count
        delta2 = surprisals[k] - mean
        M2 += delta * delta2

    for i in range(burn_in, len(surprisals)):
        # Stats from surprisals[:i] — exactly matches original window = surprisals[:i]
        variance = M2 / count if count > 0 else 0.0
        sd = variance ** 0.5
        threshold = mean + threshold_sd * sd
        if sd > 0 and surprisals[i] > threshold:
            events.append((i, surprisals[i], threshold))
        elif sd == 0 and surprisals[i] > mean:
            events.append((i, surprisals[i], mean))

        # Update running stats to include surprisals[i]
        count += 1
        delta = surprisals[i] - mean
        mean += delta / count
        delta2 = surprisals[i] - mean
        M2 += delta * delta2
```

**Step 4: Add `h_decay` to `ModelParams` and use in `continuous.py`**

In `simulator/state.py`, add field:
```python
@dataclass
class ModelParams:
    alpha: float
    a: float
    mu: float
    beta: float = 0.0
    eta: float = 0.0
    tap_variant: str = "baseline"
    alpha1: float = 0.0
    carrying_capacity: float | None = None
    h_decay: float = 0.02
```

In `simulator/continuous.py` line 56, replace:
```python
        H = max(0.0, 0.02 * Xi)  # Compression feedback from accumulated Xi
```
With:
```python
        H = max(0.0, params.h_decay * Xi)  # Compression feedback
```

Note: Default `h_decay=0.02` preserves identical behavior for all existing code.

**Step 5: Vectorize `sigma_arr` in `continuous.py`**

In `simulator/continuous.py` lines 133-134, replace:
```python
    sigma_arr = np.array([sigma_linear(xi, sigma0, gamma) for xi in Xi_arr])
```
With:
```python
    sigma_arr = np.maximum(0.0, sigma0 * (1.0 + gamma * Xi_arr))
```

This is the exact formula of `sigma_linear`: `max(0, sigma0 * (1 + gamma * Xi))`.

**Step 6: Run tests**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`
Expected: 273 passed

**Step 7: Commit**

```bash
git add simulator/longrun.py simulator/analysis.py simulator/predictive.py simulator/continuous.py simulator/state.py && git commit -m "refactor: consolidate OLS to linregress, Welford adpression detector, parameterize h_decay, vectorize sigma_arr"
```

---

### Task 7: DRY Logspace + Generalize Logistic Boundary

**Files:**
- Modify: `scripts/sweep_variants.py:21-27`
- Modify: `scripts/sweep_alpha_mu.py:19-26`
- Modify: `scripts/sweep_gamma_threshold.py:13-19`
- Modify: `scripts/sweep_mode_b.py:19-26`
- Modify: `scripts/sweep_sigma_feedback.py:19-26`
- Modify: `scripts/mstar_isocurves.py:10-17`
- Modify: `scripts/sensitivity_analysis.py:79-82`
- Modify: `simulator/analysis.py` (unify logistic boundary functions)

**Step 1: Replace `logspace` in all 6 script files with `np.logspace`**

In each file, remove the `logspace` function definition and replace calls with:
```python
import numpy as np

# Where logspace(lo, hi, n) was called, use:
np.logspace(np.log10(lo), np.log10(hi), n).tolist()
```

All six scripts already import numpy (or can — check each). For `scripts/sensitivity_analysis.py`, replace the inline logspace computation at lines 79-82 with the same pattern.

**Step 2: Generalize logistic boundary fitting**

In `simulator/analysis.py`, add a private general function:
```python
def _fit_logistic_boundary_general(
    X: list[list[float]],
    y: list[float],
    lr: float = 0.05,
    epochs: int = 4000,
    l2: float = 1e-4,
) -> tuple[list[float], list[list[float]], list[float], list[float]]:
    """General N-dimensional logistic boundary via gradient descent.

    Parameters
    ----------
    X : list of feature vectors (each a list of floats)
    y : list of binary labels (0.0 or 1.0)
    lr, epochs, l2 : optimization hyperparameters

    Returns
    -------
    (coefficients, X_standardized, means, stds)
        coefficients[0] = intercept, coefficients[1:] = feature weights
        on standardized features.
    """
    if not X:
        return [], [], [], []

    n_features = len(X[0])
    n_samples = len(X)

    # Standardize
    means = []
    stds = []
    for f in range(n_features):
        vals = [x[f] for x in X]
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 or 1.0
        means.append(m)
        stds.append(s)

    Xs = [[(x[f] - means[f]) / stds[f] for f in range(n_features)] for x in X]

    # Initialize coefficients: b[0] = intercept, b[1:] = feature weights
    b = [0.0] * (n_features + 1)

    def sig(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    n_f = float(n_samples)
    for _ in range(epochs):
        g = [0.0] * (n_features + 1)
        for xs, yi in zip(Xs, y):
            z = b[0] + sum(b[f + 1] * xs[f] for f in range(n_features))
            d = sig(z) - yi
            g[0] += d
            for f in range(n_features):
                g[f + 1] += d * xs[f]
        g[0] /= n_f
        for f in range(n_features):
            g[f + 1] = g[f + 1] / n_f + l2 * b[f + 1]
        for k in range(n_features + 1):
            b[k] -= lr * g[k]

    return b, Xs, means, stds
```

Then rewrite `fit_explosive_logistic_boundary` and `fit_explosive_logistic_boundary_3d` as thin wrappers:

```python
def fit_explosive_logistic_boundary(
    records: list[dict],
    explosive_labels: set[str] | None = None,
    lr: float = 0.05,
    epochs: int = 4000,
    l2: float = 1e-4,
) -> dict:
    """Fit logistic boundary on (log(alpha), log(mu))."""
    explosive_labels = explosive_labels or {"explosive"}
    X = []
    y = []
    for r in records:
        try:
            a, m, label = float(r["alpha"]), float(r["mu"]), str(r["regime"])
        except Exception:
            continue
        if a <= 0 or m <= 0:
            continue
        X.append([math.log(a), math.log(m)])
        y.append(1.0 if label in explosive_labels else 0.0)

    if not X:
        return {"ok": False, "reason": "no_valid_rows"}

    b, Xs, means, stds = _fit_logistic_boundary_general(X, y, lr, epochs, l2)

    def sig(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    correct = 0
    probs = []
    for xs, yi in zip(Xs, y):
        p = sig(b[0] + b[1] * xs[0] + b[2] * xs[1])
        probs.append(p)
        correct += int((1.0 if p >= 0.5 else 0.0) == yi)

    return {
        "ok": True, "n": len(Xs),
        "labels_positive": sum(int(v) for v in y),
        "coef": {"intercept": b[0], "log_alpha": b[1], "log_mu": b[2]},
        "feature_standardization": {
            "log_alpha_mean": means[0], "log_alpha_std": stds[0],
            "log_mu_mean": means[1], "log_mu_std": stds[1],
        },
        "train_accuracy": correct / len(Xs),
        "mean_pred_prob": sum(probs) / len(probs),
    }
```

Similarly for `fit_explosive_logistic_boundary_3d`, extracting `(log(alpha), log(mu), log(m0))` and using `_fit_logistic_boundary_general` with 3 features, then building the same return dict shape with keys `log_m0` added to coef and standardization.

**Step 3: Run tests**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/test_fitting.py tests/test_sensitivity.py -q -v`
Expected: All pass

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`
Expected: 273 passed

**Step 4: Commit**

```bash
git add scripts/ simulator/analysis.py && git commit -m "refactor: deduplicate logspace across scripts, generalize logistic boundary to N-D"
```

---

### Final Verification

**Step 1: Run full test suite**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -q`
Expected: 273 passed

**Step 2: Run smoke tests for affected scripts**

```bash
"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/run_demo.py
"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/empirical_validation.py --n-agents 5 --steps 30
```

**Step 3: Verify no regressions in sweep output**

```bash
"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/empirical_sweep.py --quick
```

**Step 4: Create PR**

Use finishing-a-development-branch skill.
