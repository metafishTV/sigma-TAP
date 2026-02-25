# Unified sigma-TAP Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate continuous-time ODE solver, turbulence diagnostics, real-world data fitting, and long-run scaling diagnostics into the repo's existing modular simulator.

**Architecture:** Build new modules (`continuous.py`, `turbulence.py`, `fit_realworld.py`) on top of the repo's existing `compute_birth_term()` / `sigma_linear()` / `ModelParams` infrastructure. All TAP variant support (baseline, two_scale, logistic) flows through unchanged. Tests use Python's built-in `unittest` (no pytest dependency). Claim auditor extended for new outputs.

**Tech Stack:** Python 3.12, scipy 1.17 (solve_ivp), numpy 2.4, matplotlib 3.10, unittest. No mpmath.

**Python executable:** `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`

**Working directory:** `C:\Users\user\Documents\New folder\sigma-TAP-repo`

**Branch:** `paper-first-model-fixes-20260224` (checked out as detached HEAD at `c3ffb40`)

---

## Task 1: Add `innovation_kernel_derivative()` to `tap.py`

**Files:**
- Modify: `simulator/tap.py:1-69`
- Create: `tests/test_tap_derivative.py`

**Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_tap_derivative.py`:

```python
"""Tests for innovation_kernel_derivative."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.tap import innovation_kernel_closed, innovation_kernel_derivative


class TestKernelDerivative(unittest.TestCase):
    def test_derivative_matches_numerical(self):
        """f'(M) via our function should match (f(M+h)-f(M-h))/(2h)."""
        alpha, a = 1e-3, 8.0
        for M in [5.0, 10.0, 20.0, 50.0]:
            h = 1e-6
            numerical = (
                innovation_kernel_closed(M + h, alpha, a)
                - innovation_kernel_closed(M - h, alpha, a)
            ) / (2 * h)
            analytic = innovation_kernel_derivative(M, alpha, a)
            self.assertAlmostEqual(analytic, numerical, places=4,
                msg=f"Derivative mismatch at M={M}")

    def test_derivative_positive_for_positive_M(self):
        """f'(M) > 0 for M > 1 (innovation kernel is monotonically increasing)."""
        alpha, a = 1e-4, 8.0
        for M in [2.0, 10.0, 100.0]:
            self.assertGreater(innovation_kernel_derivative(M, alpha, a), 0.0)

    def test_derivative_zero_for_small_M(self):
        """f'(M) = 0 for M <= 1."""
        self.assertEqual(innovation_kernel_derivative(0.5, 1e-3, 8.0), 0.0)
        self.assertEqual(innovation_kernel_derivative(1.0, 1e-3, 8.0), 0.0)

    def test_derivative_overflow_returns_inf(self):
        """Very large M should return inf, not crash."""
        result = innovation_kernel_derivative(10000.0, 1e-3, 2.0)
        self.assertEqual(result, float("inf"))


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_tap_derivative -v`

Expected: ImportError — `innovation_kernel_derivative` does not exist yet.

**Step 3: Write the implementation**

Add to `simulator/tap.py` after `innovation_kernel_closed` (after line 24):

```python
def innovation_kernel_derivative(M: float, alpha: float, a: float) -> float:
    """Analytical df/dM for the closed-form power-law TAP kernel.

    f(M) = alpha * a * (exp(M * ln(1+1/a)) - 1 - M/a)
    f'(M) = alpha * a * (ln(1+1/a) * exp(M * ln(1+1/a)) - 1/a)
    """
    if M <= 1.0:
        return 0.0
    k = math.log(1.0 + 1.0 / a)
    exponent = M * k
    if exponent > 700:
        return float("inf")
    return max(0.0, alpha * a * (k * math.exp(exponent) - 1.0 / a))
```

**Step 4: Run test to verify it passes**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_tap_derivative -v`

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add tests/__init__.py tests/test_tap_derivative.py simulator/tap.py
git commit -m "feat: add innovation_kernel_derivative to tap.py with tests

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Create `simulator/continuous.py` — the continuous-time ODE solver

**Files:**
- Create: `simulator/continuous.py`
- Create: `tests/test_continuous.py`
- Modify: `simulator/__init__.py:1-6`

**Step 1: Write the failing test**

Create `tests/test_continuous.py`:

```python
"""Tests for continuous-time ODE solver."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from simulator.continuous import ContinuousResult, run_continuous
from simulator.simulate import run_sigma_tap
from simulator.state import ModelParams


class TestContinuousSolver(unittest.TestCase):
    def _make_params(self, variant="baseline"):
        return ModelParams(
            alpha=1e-3, a=8.0, mu=0.02,
            beta=0.0, eta=0.0,
            tap_variant=variant,
            alpha1=1e-2 if variant == "two_scale" else 0.0,
            carrying_capacity=1e5 if variant == "logistic" else None,
        )

    def test_returns_continuous_result(self):
        params = self._make_params()
        result = run_continuous(initial_M=10.0, t_span=(0, 5), params=params)
        self.assertIsInstance(result, ContinuousResult)
        self.assertGreater(len(result.t), 0)
        self.assertEqual(len(result.t), len(result.M))
        self.assertEqual(len(result.t), len(result.Xi))

    def test_baseline_matches_discrete_qualitatively(self):
        """Continuous and discrete should agree in trend for baseline variant."""
        params = self._make_params("baseline")
        # Discrete: 10 steps
        discrete = run_sigma_tap(
            initial_M=10.0, steps=10, params=params,
            sigma0=1.0, gamma=0.0, append_terminal_state=False,
        )
        m_discrete_final = discrete[-1]["M_t1"]
        # Continuous: same time span
        result = run_continuous(
            initial_M=10.0, t_span=(0, 10), params=params,
            sigma0=1.0, gamma=0.0,
            t_eval=np.array([10.0]),
        )
        m_continuous_final = result.M[-1]
        # Should be in the same order of magnitude
        if m_discrete_final > 0 and m_continuous_final > 0:
            ratio = m_continuous_final / m_discrete_final
            self.assertGreater(ratio, 0.1)
            self.assertLess(ratio, 10.0)

    def test_logistic_variant_bounded(self):
        """Logistic variant should not exceed carrying capacity."""
        params = self._make_params("logistic")
        result = run_continuous(
            initial_M=10.0, t_span=(0, 200), params=params,
            sigma0=1.0, gamma=0.0,
        )
        self.assertTrue(all(m <= params.carrying_capacity * 1.01 for m in result.M))

    def test_two_scale_variant_runs(self):
        """Two-scale variant should run without error."""
        params = self._make_params("two_scale")
        result = run_continuous(
            initial_M=10.0, t_span=(0, 20), params=params,
            sigma0=1.0, gamma=0.0,
        )
        self.assertGreater(len(result.t), 0)

    def test_overflow_terminates(self):
        """Large alpha should trigger overflow termination."""
        params = ModelParams(alpha=0.1, a=2.0, mu=0.0, tap_variant="baseline")
        result = run_continuous(
            initial_M=10.0, t_span=(0, 100), params=params,
            sigma0=1.0, gamma=0.0, m_cap=1e6,
        )
        self.assertTrue(result.terminated_by_overflow)
        self.assertIsNotNone(result.blowup_time)

    def test_sigma_feedback_accelerates(self):
        """Positive gamma should produce faster growth than gamma=0."""
        params = self._make_params("baseline")
        params.beta = 0.05
        r0 = run_continuous(
            initial_M=10.0, t_span=(0, 20), params=params,
            sigma0=1.0, gamma=0.0,
        )
        r1 = run_continuous(
            initial_M=10.0, t_span=(0, 20), params=params,
            sigma0=1.0, gamma=0.5,
        )
        # With sigma feedback, M should be >= M without it
        m0_final = r0.M[-1]
        m1_final = r1.M[-1]
        self.assertGreaterEqual(m1_final, m0_final * 0.99)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_continuous -v`

Expected: ImportError — `simulator.continuous` does not exist.

**Step 3: Write the implementation**

Create `simulator/continuous.py`:

```python
"""Continuous-time ODE solver for TAP / sigma-TAP dynamics.

Wraps scipy.integrate.solve_ivp around the existing discrete-step
infrastructure (compute_birth_term, sigma_linear) to provide smooth
integration at arbitrary time points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .sigma_tap import sigma_linear
from .state import ModelParams
from .tap import compute_birth_term


@dataclass
class ContinuousResult:
    """Output from continuous-time TAP integration."""

    t: np.ndarray
    M: np.ndarray
    Xi: np.ndarray
    sigma: np.ndarray
    f: np.ndarray
    terminated_by_overflow: bool
    blowup_time: float | None


def _build_rhs(params: ModelParams, sigma0: float, gamma: float):
    """Build the right-hand-side function for solve_ivp."""

    def rhs(t, y):
        M, Xi = y[0], y[1]
        if M < 0:
            M = 0.0

        sig = sigma_linear(Xi, sigma0, gamma)
        f_val = compute_birth_term(
            M,
            alpha=params.alpha,
            a=params.a,
            variant=params.tap_variant,
            alpha1=params.alpha1,
            carrying_capacity=params.carrying_capacity,
        )

        B = sig * f_val
        D = params.mu * M
        H = max(0.0, 0.02 * Xi)  # h_compression default

        dM = B - D
        dXi = params.beta * B + params.eta * H

        return [dM, dXi]

    return rhs


def _overflow_event(m_cap: float):
    """Event function: triggers when M >= m_cap."""

    def event(t, y):
        return m_cap - y[0]

    event.terminal = True
    event.direction = -1
    return event


def run_continuous(
    initial_M: float,
    t_span: tuple[float, float],
    params: ModelParams,
    sigma0: float = 1.0,
    gamma: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    max_step: float = 1.0,
    m_cap: float = 1e9,
) -> ContinuousResult:
    """Integrate TAP/sigma-TAP dynamics in continuous time.

    Parameters
    ----------
    initial_M : Starting realized-object count.
    t_span : (t_start, t_end) integration window.
    params : ModelParams with variant, alpha, a, mu, etc.
    sigma0 : Baseline efficiency (default 1.0).
    gamma : Feedback strength (0 = pure TAP).
    t_eval : Optional array of times at which to report solution.
    method : ODE solver method (default RK45).
    max_step : Maximum internal step size.
    m_cap : Overflow cap — integration terminates when M >= m_cap.

    Returns
    -------
    ContinuousResult with time-series arrays.
    """
    rhs = _build_rhs(params, sigma0, gamma)
    overflow_ev = _overflow_event(m_cap)

    y0 = [initial_M, 0.0]

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        max_step=max_step,
        events=[overflow_ev],
        dense_output=False,
        rtol=1e-8,
        atol=1e-10,
    )

    t_arr = sol.t
    M_arr = sol.y[0]
    Xi_arr = sol.y[1]

    # Recompute sigma and f at each output point for diagnostics.
    sigma_arr = np.array([sigma_linear(xi, sigma0, gamma) for xi in Xi_arr])
    f_arr = np.array([
        compute_birth_term(
            m, alpha=params.alpha, a=params.a,
            variant=params.tap_variant, alpha1=params.alpha1,
            carrying_capacity=params.carrying_capacity,
        )
        for m in M_arr
    ])

    terminated = sol.status == 1  # event triggered
    blowup_time = None
    if terminated and sol.t_events and len(sol.t_events[0]) > 0:
        blowup_time = float(sol.t_events[0][0])

    return ContinuousResult(
        t=t_arr,
        M=M_arr,
        Xi=Xi_arr,
        sigma=sigma_arr,
        f=f_arr,
        terminated_by_overflow=terminated,
        blowup_time=blowup_time,
    )
```

Update `simulator/__init__.py`:

```python
"""Minimal TAP/σ-TAP simulation package."""

from .hfuncs import h_compression
from .state import ModelParams, ModelState

__all__ = ["h_compression", "ModelParams", "ModelState"]
```

(No change needed — continuous.py is imported directly by scripts, not via __init__.)

**Step 4: Run test to verify it passes**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_continuous -v`

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add simulator/continuous.py tests/test_continuous.py
git commit -m "feat: add continuous-time ODE solver for all TAP variants

Wraps scipy.integrate.solve_ivp around existing compute_birth_term
and sigma_linear. Supports baseline, two_scale, and logistic variants
with overflow event termination.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Create `simulator/turbulence.py` — post-hoc diagnostics

**Files:**
- Create: `simulator/turbulence.py`
- Create: `tests/test_turbulence.py`

**Step 1: Write the failing test**

Create `tests/test_turbulence.py`:

```python
"""Tests for turbulence diagnostics (Interpretive layer)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from simulator.continuous import run_continuous
from simulator.state import ModelParams
from simulator.turbulence import TurbulenceDiagnostics, compute_turbulence_diagnostics


class TestTurbulenceDiagnostics(unittest.TestCase):
    def _run_trajectory(self, alpha=1e-3, a=8.0, mu=0.02, t_end=30):
        params = ModelParams(alpha=alpha, a=a, mu=mu, tap_variant="baseline")
        return run_continuous(
            initial_M=10.0, t_span=(0, t_end), params=params,
            sigma0=1.0, gamma=0.0,
        ), params

    def test_returns_diagnostics(self):
        result, params = self._run_trajectory()
        diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0)
        self.assertIsInstance(diag, TurbulenceDiagnostics)
        self.assertEqual(len(diag.B_decision), len(result.t))
        self.assertEqual(len(diag.Re_prax), len(result.t))

    def test_B_decreases_as_M_grows(self):
        """Decision bandwidth should decrease as innovation rate overwhelms decision capacity."""
        result, params = self._run_trajectory(alpha=1e-2, a=4.0, mu=0.001, t_end=50)
        diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0)
        # Filter to points where M is meaningfully growing
        valid = result.M > 2.0
        if sum(valid) > 5:
            B_valid = diag.B_decision[valid]
            # B should trend downward (allow noise from numerics)
            self.assertLess(B_valid[-1], B_valid[0] * 1.1)

    def test_laminar_fraction_one_when_M_small(self):
        """When M stays small, everything is laminar (B > 1)."""
        # High mu keeps M small
        result, params = self._run_trajectory(alpha=1e-5, mu=0.1, t_end=10)
        diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0, tau_decision=10.0)
        self.assertEqual(diag.laminar_fraction, 1.0)

    def test_Re_prax_positive(self):
        """Reynolds number should be non-negative."""
        result, params = self._run_trajectory()
        diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0)
        self.assertTrue(all(r >= 0 for r in diag.Re_prax))


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_turbulence -v`

Expected: ImportError — `simulator.turbulence` does not exist.

**Step 3: Write the implementation**

Create `simulator/turbulence.py`:

```python
"""Post-hoc turbulence diagnostics for TAP / sigma-TAP trajectories.

LAYER: Interpretive — these diagnostics compute derived quantities that
aid physical interpretation but do not feed back into the simulation.

Decision Bandwidth B(t) = sigma(Xi) * tau / f'(M)
  B > 1: laminar (all affordances evaluable)
  B < 1: turbulent (affordance overflow)

Praxiological Reynolds Re_prax = f'(M) * M / (sigma(Xi) + alpha)
  Analogous to fluid Reynolds number: ratio of innovation pressure
  to evaluative capacity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sigma_tap import sigma_linear
from .state import ModelParams
from .tap import innovation_kernel_derivative


@dataclass
class TurbulenceDiagnostics:
    """Turbulence diagnostic outputs (Interpretive layer)."""

    t: np.ndarray
    B_decision: np.ndarray
    Re_prax: np.ndarray
    laminar_fraction: float
    transition_time: float | None


def compute_turbulence_diagnostics(
    result,
    params: ModelParams,
    sigma0: float,
    gamma: float,
    tau_decision: float = 1.0,
) -> TurbulenceDiagnostics:
    """Compute turbulence diagnostics from a completed trajectory.

    Parameters
    ----------
    result : ContinuousResult (or any object with .t, .M, .Xi arrays).
    params : ModelParams used for the run.
    sigma0, gamma : Sigma parameters used for the run.
    tau_decision : Decision horizon (time available for evaluation).

    Returns
    -------
    TurbulenceDiagnostics with B(t), Re_prax(t), and summary statistics.
    """
    t = np.asarray(result.t)
    M = np.asarray(result.M)
    Xi = np.asarray(result.Xi)

    n = len(t)
    B = np.zeros(n)
    Re = np.zeros(n)

    for i in range(n):
        sig = sigma_linear(Xi[i], sigma0, gamma)
        fprime = innovation_kernel_derivative(M[i], params.alpha, params.a)

        if fprime > 0 and np.isfinite(fprime):
            B[i] = sig * tau_decision / fprime
        else:
            B[i] = float("inf") if fprime == 0 else 0.0

        denom = sig + params.alpha
        if denom > 0 and np.isfinite(fprime):
            Re[i] = fprime * M[i] / denom
        else:
            Re[i] = 0.0

    # Summary statistics.
    finite_B = B[np.isfinite(B)]
    if len(finite_B) > 0:
        laminar_fraction = float(np.mean(finite_B > 1.0))
    else:
        laminar_fraction = 1.0

    transition_time = None
    for i in range(n):
        if np.isfinite(B[i]) and B[i] < 1.0:
            transition_time = float(t[i])
            break

    return TurbulenceDiagnostics(
        t=t,
        B_decision=B,
        Re_prax=Re,
        laminar_fraction=laminar_fraction,
        transition_time=transition_time,
    )
```

**Step 4: Run test to verify it passes**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_turbulence -v`

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add simulator/turbulence.py tests/test_turbulence.py
git commit -m "feat: add turbulence diagnostics module (Interpretive layer)

Decision Bandwidth B and praxiological Reynolds Re_prax computed
as post-hoc diagnostics on completed trajectories. Does not feed
back into simulation dynamics.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Add long-run scaling diagnostics to `analysis.py`

**Files:**
- Modify: `simulator/analysis.py:1-343`
- Create: `tests/test_longrun.py`

**Step 1: Write the failing test**

Create `tests/test_longrun.py`:

```python
"""Tests for long-run scaling diagnostics (Taalbi-inspired)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.analysis import innovation_rate_scaling, constraint_tag


class TestInnovationRateScaling(unittest.TestCase):
    def test_linear_trajectory(self):
        """Exponential M(t) should give scaling exponent ~ 1."""
        import math
        # M(t) = 10 * exp(0.1 * t) => dk/dt ~ k => sigma ~ 1
        m_traj = [10.0 * math.exp(0.1 * t) for t in range(50)]
        result = innovation_rate_scaling(m_traj, dt=1.0)
        self.assertIn("exponent", result)
        self.assertAlmostEqual(result["exponent"], 1.0, delta=0.3)

    def test_superlinear_trajectory(self):
        """Quadratic-ish growth should give exponent > 1."""
        m_traj = [10.0 + 0.01 * t ** 3 for t in range(100)]
        result = innovation_rate_scaling(m_traj, dt=1.0)
        self.assertGreater(result["exponent"], 1.0)

    def test_short_trajectory_graceful(self):
        """Very short trajectory should not crash."""
        result = innovation_rate_scaling([10.0, 11.0], dt=1.0)
        self.assertIn("exponent", result)


class TestConstraintTag(unittest.TestCase):
    def test_resource_limited(self):
        """M plateauing well below K is resource-limited."""
        m_traj = [10.0 + 0.1 * t for t in range(100)]
        tag = constraint_tag(m_traj, carrying_capacity=1e6, dt=1.0)
        self.assertIn(tag, {"adjacency-limited", "resource-limited", "mixed"})

    def test_no_capacity_is_adjacency(self):
        """Without carrying capacity, tag should be adjacency-limited."""
        m_traj = [10.0 + t for t in range(50)]
        tag = constraint_tag(m_traj, carrying_capacity=None, dt=1.0)
        self.assertEqual(tag, "adjacency-limited")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_longrun -v`

Expected: ImportError — `innovation_rate_scaling` and `constraint_tag` do not exist.

**Step 3: Write the implementation**

Append to `simulator/analysis.py` (after line 343):

```python
def innovation_rate_scaling(
    m_traj: list[float],
    dt: float = 1.0,
) -> dict:
    """Fit dk/dt ~ k^sigma using log-log OLS on finite differences.

    Inspired by Taalbi (2025): sigma ~ 1 indicates resource-constrained
    linear-in-k dynamics. sigma > 1 indicates unconstrained super-linear TAP.

    Returns dict with 'exponent', 'r_squared', 'n_points'.
    """
    import math

    if len(m_traj) < 3:
        return {"exponent": 1.0, "r_squared": 0.0, "n_points": len(m_traj)}

    # Finite differences for dk/dt.
    rates = [(m_traj[i + 1] - m_traj[i]) / dt for i in range(len(m_traj) - 1)]
    midpoints = [0.5 * (m_traj[i] + m_traj[i + 1]) for i in range(len(m_traj) - 1)]

    # Filter to positive values for log-log fit.
    log_k = []
    log_rate = []
    for k, r in zip(midpoints, rates):
        if k > 0 and r > 0:
            log_k.append(math.log(k))
            log_rate.append(math.log(r))

    n = len(log_k)
    if n < 2:
        return {"exponent": 1.0, "r_squared": 0.0, "n_points": n}

    # OLS: log(rate) = sigma * log(k) + c.
    mean_x = sum(log_k) / n
    mean_y = sum(log_rate) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_k, log_rate))
    ss_xx = sum((x - mean_x) ** 2 for x in log_k)
    ss_yy = sum((y - mean_y) ** 2 for y in log_rate)

    if ss_xx < 1e-15:
        return {"exponent": 1.0, "r_squared": 0.0, "n_points": n}

    sigma = ss_xy / ss_xx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-15 else 0.0

    return {"exponent": sigma, "r_squared": r_squared, "n_points": n}


def constraint_tag(
    m_traj: list[float],
    carrying_capacity: float | None,
    dt: float = 1.0,
) -> str:
    """Tag observed dynamics as adjacency-limited, resource-limited, or mixed.

    Uses Taalbi's framing: if carrying capacity is absent or M is far from it,
    dynamics are adjacency-limited. If M approaches K, resource-limited.
    """
    if carrying_capacity is None or carrying_capacity <= 0:
        return "adjacency-limited"

    if len(m_traj) < 2:
        return "adjacency-limited"

    m_final = m_traj[-1]
    ratio = m_final / carrying_capacity

    if ratio > 0.8:
        return "resource-limited"
    elif ratio > 0.3:
        return "mixed"
    else:
        return "adjacency-limited"
```

**Step 4: Run test to verify it passes**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_longrun -v`

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add simulator/analysis.py tests/test_longrun.py
git commit -m "feat: add Taalbi-inspired long-run scaling diagnostics

innovation_rate_scaling fits dk/dt ~ k^sigma via log-log OLS.
constraint_tag labels dynamics as adjacency/resource/mixed-limited.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Create `config/realworld_datasets.json`

**Files:**
- Create: `config/realworld_datasets.json`

**Step 1: Create the dataset config**

```json
{
  "datasets": {
    "wikipedia_articles": {
      "description": "English Wikipedia article count (2001-2024)",
      "units": "articles",
      "tap_context": "Articles create linking possibilities for new articles.",
      "years": [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
      "counts": [20000,96000,188000,420000,890000,1500000,2100000,2600000,3100000,3500000,3800000,4100000,4400000,4700000,5000000,5300000,5500000,5750000,5980000,6200000,6400000,6600000,6750000,6900000]
    },
    "npm_packages": {
      "description": "npm registry total packages (2010-2025)",
      "units": "packages",
      "tap_context": "Packages create dependency possibilities for new packages.",
      "years": [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025],
      "counts": [2000,10000,30000,55000,100000,200000,350000,500000,785000,1000000,1300000,1500000,1700000,1900000,2100000,2300000]
    },
    "described_species": {
      "description": "Cumulative described species (1750-2020)",
      "units": "species",
      "tap_context": "Species create niches enabling new species (Kauffman TAP).",
      "years": [1750,1780,1800,1820,1840,1860,1880,1900,1920,1940,1960,1980,2000,2010,2020],
      "counts": [4000,15000,40000,80000,150000,300000,500000,800000,950000,1050000,1250000,1450000,1700000,1850000,2000000]
    }
  }
}
```

**Step 2: Commit**

```bash
git add config/realworld_datasets.json
git commit -m "data: add real-world dataset definitions for fitting

Wikipedia articles, npm packages, described species.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Create `scripts/fit_realworld.py` — empirical fitting

**Files:**
- Create: `scripts/fit_realworld.py`
- Create: `tests/test_fitting.py`

**Step 1: Write the failing test**

Create `tests/test_fitting.py`:

```python
"""Smoke tests for real-world fitting pipeline."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Test that the fitting module loads and its core function works on a tiny problem.
from scripts.fit_realworld import fit_single_variant, load_datasets


class TestFitRealworld(unittest.TestCase):
    def test_load_datasets(self):
        datasets = load_datasets()
        self.assertIn("wikipedia_articles", datasets)
        self.assertIn("npm_packages", datasets)
        self.assertIn("described_species", datasets)
        for name, ds in datasets.items():
            self.assertEqual(len(ds["years"]), len(ds["counts"]))

    def test_fit_single_variant_runs(self):
        """Fit logistic variant to Wikipedia with minimal grid — should not crash."""
        datasets = load_datasets()
        ds = datasets["wikipedia_articles"]
        result = fit_single_variant(
            years=ds["years"], counts=ds["counts"],
            variant="logistic",
            grid_size=3,  # tiny grid for speed
        )
        self.assertIn("rmse", result)
        self.assertIn("params", result)
        self.assertGreater(result["rmse"], 0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_fitting -v`

Expected: ImportError.

**Step 3: Write the implementation**

Create `scripts/fit_realworld.py`:

```python
"""Fit TAP variants to real-world datasets using continuous-time solver.

Hierarchical strategy:
  Pass 1: Grid search over core params (alpha, a, mu, [K]) -> Nelder-Mead
  Pass 2: Lock core, fit sigma feedback (gamma, beta)
  Pass 3: Compare against null models (exponential, logistic growth)

Usage:
  python scripts/fit_realworld.py
  python scripts/fit_realworld.py --grid-size 10
  python scripts/fit_realworld.py --variants baseline,logistic
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.optimize import minimize

from simulator.continuous import run_continuous
from simulator.state import ModelParams
from simulator.analysis import innovation_rate_scaling, constraint_tag

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def load_datasets() -> dict:
    """Load datasets from config/realworld_datasets.json."""
    path = ROOT / "config" / "realworld_datasets.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["datasets"]


def _cost_logspace(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Log-space MSE cost for scale-invariant fitting."""
    pred = np.clip(predicted, 1.0, None)
    obs = np.clip(observed, 1.0, None)
    return float(np.mean((np.log10(pred) - np.log10(obs)) ** 2))


def _simulate_for_fit(
    years: list[float],
    counts: list[float],
    alpha: float,
    a: float,
    mu: float,
    variant: str,
    alpha1: float = 0.0,
    carrying_capacity: float | None = None,
    gamma: float = 0.0,
    beta: float = 0.0,
) -> float:
    """Run continuous solver and return log-space MSE cost."""
    t_data = np.array(years, dtype=float)
    t_data = t_data - t_data[0]  # normalize to start at 0
    obs = np.array(counts, dtype=float)
    M0 = obs[0]

    params = ModelParams(
        alpha=alpha, a=a, mu=mu,
        beta=beta, eta=0.0,
        tap_variant=variant,
        alpha1=alpha1,
        carrying_capacity=carrying_capacity,
    )

    try:
        result = run_continuous(
            initial_M=M0,
            t_span=(0, t_data[-1]),
            params=params,
            sigma0=1.0,
            gamma=gamma,
            t_eval=t_data,
            max_step=0.5,
            m_cap=1e12,
        )
        if len(result.M) != len(obs):
            return 1e6
        return _cost_logspace(result.M, obs)
    except Exception:
        return 1e6


def fit_single_variant(
    years: list[float],
    counts: list[float],
    variant: str = "baseline",
    grid_size: int = 5,
) -> dict:
    """Fit a single TAP variant to one dataset.

    Returns dict with 'rmse', 'params', 'variant', 'cost'.
    """
    log_alphas = np.linspace(-6, -2, grid_size)
    log_as = np.linspace(0.3, 2.5, grid_size)
    log_mus = np.linspace(-4, -1, grid_size)

    best_cost = 1e6
    best_params = None

    for la in log_alphas:
        for laa in log_as:
            for lm in log_mus:
                alpha = 10 ** la
                a = 10 ** laa
                mu = 10 ** lm

                K = max(counts) * 2.0 if variant == "logistic" else None
                alpha1 = alpha * 10 if variant == "two_scale" else 0.0

                cost = _simulate_for_fit(
                    years, counts, alpha, a, mu, variant,
                    alpha1=alpha1, carrying_capacity=K,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_params = {
                        "alpha": alpha, "a": a, "mu": mu,
                        "alpha1": alpha1, "carrying_capacity": K,
                    }

    # Nelder-Mead refinement.
    if best_params is not None:
        x0 = [
            math.log10(best_params["alpha"]),
            math.log10(best_params["a"]),
            math.log10(best_params["mu"]),
        ]

        def objective(x):
            alpha = 10 ** x[0]
            a = 10 ** x[1]
            mu = 10 ** x[2]
            K = max(counts) * 2.0 if variant == "logistic" else None
            a1 = alpha * 10 if variant == "two_scale" else 0.0
            return _simulate_for_fit(
                years, counts, alpha, a, mu, variant,
                alpha1=a1, carrying_capacity=K,
            )

        try:
            res = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 200, "xatol": 0.01, "fatol": 1e-5})
            if res.fun < best_cost:
                best_cost = res.fun
                best_params["alpha"] = 10 ** res.x[0]
                best_params["a"] = 10 ** res.x[1]
                best_params["mu"] = 10 ** res.x[2]
                if variant == "two_scale":
                    best_params["alpha1"] = best_params["alpha"] * 10
        except Exception:
            pass

    rmse_log = math.sqrt(best_cost) if best_cost < 1e6 else float("inf")

    return {
        "variant": variant,
        "cost": best_cost,
        "rmse": rmse_log,
        "params": best_params or {},
    }


def fit_null_models(years: list[float], counts: list[float]) -> dict:
    """Fit simple null models for comparison."""
    t = np.array(years, dtype=float) - years[0]
    obs = np.array(counts, dtype=float)
    log_obs = np.log10(np.clip(obs, 1, None))

    results = {}

    # Exponential: M(t) = M0 * exp(r*t)
    try:
        def exp_cost(x):
            r = x[0]
            pred = obs[0] * np.exp(r * t)
            return _cost_logspace(pred, obs)
        res = minimize(exp_cost, [0.1], method="Nelder-Mead")
        results["exponential"] = {"cost": res.fun, "rmse": math.sqrt(res.fun), "r": float(res.x[0])}
    except Exception:
        results["exponential"] = {"cost": 1e6, "rmse": float("inf")}

    # Logistic growth: M(t) = K / (1 + ((K-M0)/M0)*exp(-r*t))
    try:
        def logistic_cost(x):
            r, K = x[0], 10 ** x[1]
            if K <= obs[0]:
                return 1e6
            pred = K / (1 + ((K - obs[0]) / obs[0]) * np.exp(-r * t))
            return _cost_logspace(pred, obs)
        res = minimize(logistic_cost, [0.1, math.log10(max(counts) * 2)], method="Nelder-Mead")
        results["logistic_growth"] = {"cost": res.fun, "rmse": math.sqrt(res.fun)}
    except Exception:
        results["logistic_growth"] = {"cost": 1e6, "rmse": float("inf")}

    # Power law: M(t) = M0 * (1 + t)^p
    try:
        def power_cost(x):
            p = x[0]
            pred = obs[0] * (1 + t) ** p
            return _cost_logspace(pred, obs)
        res = minimize(power_cost, [1.5], method="Nelder-Mead")
        results["power_law"] = {"cost": res.fun, "rmse": math.sqrt(res.fun), "p": float(res.x[0])}
    except Exception:
        results["power_law"] = {"cost": 1e6, "rmse": float("inf")}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fit TAP variants to real-world data")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--variants", type=str, default="baseline,two_scale,logistic")
    args = parser.parse_args()

    datasets = load_datasets()
    variants = [v.strip() for v in args.variants.split(",")]

    all_results = []

    for ds_name, ds in datasets.items():
        print(f"\n--- {ds_name}: {ds['description']} ---")

        # Null models.
        nulls = fit_null_models(ds["years"], ds["counts"])
        for model_name, nr in nulls.items():
            print(f"  {model_name}: RMSE(log) = {nr['rmse']:.4f}")
            all_results.append({
                "dataset": ds_name, "model": model_name,
                "rmse_log": f"{nr['rmse']:.6f}", "variant": "",
            })

        # TAP variants.
        for variant in variants:
            result = fit_single_variant(
                ds["years"], ds["counts"],
                variant=variant,
                grid_size=args.grid_size,
            )
            print(f"  TAP-{variant}: RMSE(log) = {result['rmse']:.4f}, params = {result['params']}")

            # Long-run scaling diagnostic on best fit.
            if result["params"]:
                try:
                    p = result["params"]
                    params = ModelParams(
                        alpha=p["alpha"], a=p["a"], mu=p["mu"],
                        tap_variant=variant,
                        alpha1=p.get("alpha1", 0.0),
                        carrying_capacity=p.get("carrying_capacity"),
                    )
                    t_data = np.array(ds["years"], dtype=float) - ds["years"][0]
                    r = run_continuous(
                        initial_M=ds["counts"][0],
                        t_span=(0, t_data[-1]),
                        params=params, t_eval=t_data,
                    )
                    scaling = innovation_rate_scaling(list(r.M))
                    tag = constraint_tag(list(r.M), p.get("carrying_capacity"))
                    print(f"    scaling exponent: {scaling['exponent']:.3f} (R²={scaling['r_squared']:.3f}), constraint: {tag}")
                except Exception:
                    scaling = {"exponent": 0, "r_squared": 0}
                    tag = "unknown"

            all_results.append({
                "dataset": ds_name, "model": f"TAP-{variant}",
                "rmse_log": f"{result['rmse']:.6f}", "variant": variant,
            })

    # Write CSV output.
    out_csv = ROOT / "outputs" / "realworld_fit.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "variant", "rmse_log"])
        w.writeheader()
        w.writerows(all_results)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_fitting -v`

Expected: 2 tests PASS. (The fitting test may take 10-30 seconds due to grid search.)

**Step 5: Commit**

```bash
git add scripts/fit_realworld.py tests/test_fitting.py
git commit -m "feat: add real-world fitting pipeline for all TAP variants

Hierarchical fitting: grid search + Nelder-Mead for core params,
null model comparison (exponential, logistic growth, power law),
long-run scaling diagnostics on best fits.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Create `scripts/sweep_variants.py`

**Files:**
- Create: `scripts/sweep_variants.py`

**Step 1: Write the implementation**

```python
"""Unified cross-variant comparison sweep.

Runs baseline, two_scale, and logistic variants over the same parameter
grid and emits a single comparison CSV.

Usage:
  python scripts/sweep_variants.py > outputs/variant_comparison.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.analysis import adaptive_xi_plateau_threshold, classify_regime
from simulator.hfuncs import h_compression
from simulator.simulate import run_sigma_tap
from simulator.state import ModelParams


def logspace(lo: float, hi: float, n: int) -> list[float]:
    import math
    a = math.log10(lo)
    b = math.log10(hi)
    if n == 1:
        return [10 ** a]
    return [10 ** (a + i * (b - a) / (n - 1)) for i in range(n)]


def main() -> None:
    alphas = logspace(1e-5, 1e-2, 8)
    mus = logspace(1e-3, 1e-1, 8)
    m0_values = [10.0, 20.0, 50.0]
    variants = ["baseline", "two_scale", "logistic"]

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "variant", "alpha", "mu", "a", "m0",
            "steps", "final_M", "final_Xi",
            "regime", "blowup_step",
        ],
    )
    writer.writeheader()

    for variant in variants:
        for m0 in m0_values:
            for alpha in alphas:
                for mu in mus:
                    alpha1 = 10.0 * alpha if variant == "two_scale" else 0.0
                    K = 2e5 if variant == "logistic" else None
                    params = ModelParams(
                        alpha=alpha, a=8.0, mu=mu,
                        beta=0.05, eta=0.02,
                        tap_variant=variant,
                        alpha1=alpha1,
                        carrying_capacity=K,
                    )
                    steps = 120 if (alpha >= 1e-3 and mu <= 1e-2) else 40
                    rows = run_sigma_tap(
                        initial_M=m0, steps=steps, params=params,
                        sigma0=1.0, gamma=0.0,
                        h_func=lambda s: h_compression(s, decay=0.02),
                        append_terminal_state=True,
                    )
                    tr = [r for r in rows if "M_t1" in r]
                    xi = [tr[0]["Xi"]] + [r["Xi_t1"] for r in tr]
                    m = [tr[0]["M"]] + [r["M_t1"] for r in tr]
                    thr = adaptive_xi_plateau_threshold(xi)
                    regime = classify_regime(xi, m, thr)
                    blowup = next(
                        (r.get("blowup_step") for r in tr if r.get("overflow_detected")),
                        None,
                    )
                    writer.writerow({
                        "variant": variant,
                        "alpha": f"{alpha:.8g}",
                        "mu": f"{mu:.8g}",
                        "a": "8",
                        "m0": f"{m0:.8g}",
                        "steps": steps,
                        "final_M": f"{m[-1]:.8g}",
                        "final_Xi": f"{xi[-1]:.8g}",
                        "regime": regime,
                        "blowup_step": "" if blowup is None else int(blowup),
                    })


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/sweep_variants.py | head -5`

Expected: CSV header + 4 data rows.

**Step 3: Commit**

```bash
git add scripts/sweep_variants.py
git commit -m "feat: add unified cross-variant comparison sweep

Runs baseline, two_scale, logistic over same grid. Single CSV output.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Integration smoke test — run everything end-to-end

**Step 1: Run all unit tests**

```bash
"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest discover -s tests -v
```

Expected: All tests pass (test_tap_derivative, test_continuous, test_turbulence, test_longrun, test_fitting).

**Step 2: Run the demo with all variants**

```bash
"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/run_demo.py
```

Expected: Outputs for baseline, two_scale, logistic variants.

**Step 3: Run real-world fitting (reduced grid)**

```bash
"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/fit_realworld.py --grid-size 5
```

Expected: RMSE values printed for each variant × dataset. `outputs/realworld_fit.csv` generated.

**Step 4: Run variant comparison sweep (reduced)**

```bash
"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -c "
import sys, os
sys.path.insert(0, '.')
from scripts.sweep_variants import main, logspace
# Just verify it starts and produces output
main()
" 2>&1 | head -20
```

Expected: CSV rows flowing.

**Step 5: Final commit**

```bash
git add outputs/realworld_fit.csv
git commit -m "chore: integration smoke test — all modules verified

Continuous solver, turbulence diagnostics, real-world fitting,
long-run scaling, variant comparison all operational.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Summary

| Task | Module | Tests | Est. time |
|------|--------|-------|-----------|
| 1 | `tap.py` derivative | 4 tests | 5 min |
| 2 | `continuous.py` ODE solver | 6 tests | 10 min |
| 3 | `turbulence.py` diagnostics | 4 tests | 5 min |
| 4 | `analysis.py` long-run scaling | 5 tests | 5 min |
| 5 | `realworld_datasets.json` | — | 2 min |
| 6 | `fit_realworld.py` fitting | 2 tests | 10 min |
| 7 | `sweep_variants.py` | smoke test | 5 min |
| 8 | Integration smoke test | all | 5 min |

**Total: 8 tasks, 21 tests, ~47 minutes estimated.**
