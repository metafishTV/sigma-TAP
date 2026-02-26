# Predictive Orientation Diagnostic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a step-ahead Markov prediction module that detects parallel matching (usual), surprisal (improbable), and adpressive events (state collapse / annihilation) across TAPS transition axes.

**Architecture:** New `simulator/predictive.py` module with four core functions (predict_step, compute_surprisal, detect_adpression, detect_state_collapse) orchestrated by `run_predictive_diagnostic()`. Imports transition matrices from `taps_sensitivity.build_transition_map()`. Text output via new section in `scripts/taps_diagnostics.py`.

**Tech Stack:** Python 3.12, numpy, dataclasses, unittest

**Design doc:** `docs/plans/2026-02-26-predictive-orientation-diagnostic-design.md`

---

### Task 1: Create predictive.py with data structures and predict_step()

**Files:**
- Create: `simulator/predictive.py`
- Create: `tests/test_predictive.py`

**Step 1: Write the failing tests**

In `tests/test_predictive.py`, write these tests:

```python
"""Tests for predictive orientation diagnostic."""
import os
import sys
import unittest
import math

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestPredictStep(unittest.TestCase):
    """Tests for predict_step()."""

    def test_known_2x2_matrix(self):
        """Given state 'a' in a known 2x2 matrix, predict correct distribution."""
        from simulator.predictive import predict_step

        # P = [[0.7, 0.3], [0.4, 0.6]]
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        states = ["a", "b"]
        result = predict_step("a", P, states, horizon=1)
        self.assertAlmostEqual(result["a"], 0.7, places=5)
        self.assertAlmostEqual(result["b"], 0.3, places=5)

    def test_horizon_2_uses_matrix_power(self):
        """horizon=2 should use P^2 for two-step prediction."""
        from simulator.predictive import predict_step

        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        states = ["a", "b"]
        # P^2 = [[0.61, 0.39], [0.52, 0.48]]
        result = predict_step("a", P, states, horizon=2)
        self.assertAlmostEqual(result["a"], 0.61, places=5)
        self.assertAlmostEqual(result["b"], 0.39, places=5)

    def test_unknown_state_returns_uniform(self):
        """If current_state is not in states, return uniform distribution."""
        from simulator.predictive import predict_step

        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        states = ["a", "b"]
        result = predict_step("unknown", P, states, horizon=1)
        self.assertAlmostEqual(result["a"], 0.5, places=5)
        self.assertAlmostEqual(result["b"], 0.5, places=5)

    def test_distribution_sums_to_one(self):
        """Predicted distribution must sum to 1.0."""
        from simulator.predictive import predict_step

        P = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
        states = ["a", "b", "c"]
        result = predict_step("b", P, states, horizon=1)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=10)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_predictive.py -v`
Expected: FAIL with ImportError (simulator.predictive does not exist)

**Step 3: Write minimal implementation**

Create `simulator/predictive.py`:

```python
"""Predictive Orientation Diagnostic — step-ahead Markov prediction.

CLAIM POLICY LABEL: exploratory

Predicts system mode transitions using post-hoc transition matrices,
computing parallel matching rates (usual case) and surprisal values
(improbable case). Detects candidate adpressive events when surprisal
exceeds an adaptive statistical threshold, and annihilation/extinction
events when axis state diversity collapses.

This is a post-hoc diagnostic tool; it does not alter simulation dynamics.
Online/incremental mode is documented for future work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepPrediction:
    """One prediction per axis per step."""
    step: int
    axis: str
    predicted_distribution: dict[str, float]
    top_predictions: list[tuple[str, float]]
    actual_state: str
    parallel_match: bool
    surprisal: float


@dataclass
class AdpressionEvent:
    """Flagged when surprisal exceeds adaptive threshold or state collapses."""
    step: int
    axis: str
    surprisal: float
    threshold: float
    predicted_top: str
    actual_state: str
    probability_of_actual: float
    event_type: str  # "transition_surprisal" or "state_collapse"


@dataclass
class PredictiveDiagnosticResult:
    """Full diagnostic output."""
    predictions: list[StepPrediction]
    parallel_matching_rate: dict[str, float]
    mean_surprisal: dict[str, float]
    adpression_events: list[AdpressionEvent]
    state_collapse_events: list[AdpressionEvent]
    horizon: int
    grain: str
    step_count: int
    axes_analyzed: list[str]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def predict_step(
    current_state: str,
    transition_matrix: np.ndarray,
    states: list[str],
    horizon: int = 1,
) -> dict[str, float]:
    """Predict probability distribution over next states.

    Parameters
    ----------
    current_state : str
        Current state label (e.g. "entropy").
    transition_matrix : np.ndarray
        Row-stochastic transition matrix (n x n).
    states : list[str]
        State labels aligned with matrix indices.
    horizon : int
        Steps ahead to predict. 1 = next step, k = k steps ahead via P^k.

    Returns
    -------
    dict[str, float]
        Mapping from state label to predicted probability.
        If current_state is unknown, returns uniform distribution.
    """
    n = len(states)
    if n == 0:
        return {}

    # Find index of current state
    try:
        idx = states.index(current_state)
    except ValueError:
        # Unknown state — return uniform
        return {s: 1.0 / n for s in states}

    # Apply matrix power for multi-step look-ahead
    if horizon > 1:
        P_k = np.linalg.matrix_power(transition_matrix, horizon)
    else:
        P_k = transition_matrix

    row = P_k[idx]
    return {states[i]: float(row[i]) for i in range(n)}
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predictive.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add simulator/predictive.py tests/test_predictive.py
git commit -m "feat(predictive): add data structures and predict_step function"
```

---

### Task 2: Add compute_surprisal()

**Files:**
- Modify: `simulator/predictive.py`
- Modify: `tests/test_predictive.py`

**Step 1: Write the failing tests**

Add to `tests/test_predictive.py`:

```python
class TestComputeSurprisal(unittest.TestCase):
    """Tests for compute_surprisal()."""

    def test_known_probability(self):
        """P=0.5 should give exactly 1.0 bit of surprisal."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 0.5, "b": 0.5}, "a")
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_certain_event_zero_surprisal(self):
        """P=1.0 should give 0.0 bits of surprisal."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 1.0}, "a")
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_zero_probability_caps_at_max(self):
        """P=0.0 should cap at max_surprisal (default 10.0 bits)."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 0.0, "b": 1.0}, "a")
        self.assertAlmostEqual(result, 10.0, places=10)

    def test_unknown_state_caps_at_max(self):
        """State not in distribution should cap at max_surprisal."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 0.5, "b": 0.5}, "c")
        self.assertAlmostEqual(result, 10.0, places=10)

    def test_custom_max_surprisal(self):
        """Custom max_surprisal should be respected."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 0.0}, "a", max_surprisal=5.0)
        self.assertAlmostEqual(result, 5.0, places=10)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_predictive.py::TestComputeSurprisal -v`
Expected: FAIL with ImportError (compute_surprisal not found)

**Step 3: Write minimal implementation**

Add to `simulator/predictive.py`:

```python
def compute_surprisal(
    predicted_dist: dict[str, float],
    actual_state: str,
    max_surprisal: float = 10.0,
) -> float:
    """Compute information-theoretic surprisal for an observed transition.

    Parameters
    ----------
    predicted_dist : dict[str, float]
        Predicted probability distribution {state: probability}.
    actual_state : str
        The state that actually occurred.
    max_surprisal : float
        Cap for zero-probability events (default 10.0 bits = P < 1/1024).

    Returns
    -------
    float
        Surprisal in bits: -log2(P(actual_state)).
        Capped at max_surprisal if probability is 0 or state is unknown.
    """
    prob = predicted_dist.get(actual_state, 0.0)
    if prob <= 0.0:
        return max_surprisal
    return -math.log2(prob)
```

Also add `import math` at the top of `simulator/predictive.py`.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predictive.py -v`
Expected: 9 PASS (4 from Task 1 + 5 new)

**Step 5: Commit**

```bash
git add simulator/predictive.py tests/test_predictive.py
git commit -m "feat(predictive): add compute_surprisal function"
```

---

### Task 3: Add detect_adpression() and detect_state_collapse()

**Files:**
- Modify: `simulator/predictive.py`
- Modify: `tests/test_predictive.py`

**Step 1: Write the failing tests**

Add to `tests/test_predictive.py`:

```python
class TestDetectAdpression(unittest.TestCase):
    """Tests for detect_adpression()."""

    def test_spike_detected(self):
        """A single spike in otherwise uniform surprisal should be flagged."""
        from simulator.predictive import detect_adpression

        # 20 uniform values at 1.0, then one spike at 8.0
        surprisals = [1.0] * 20 + [8.0] + [1.0] * 5
        events = detect_adpression(surprisals, threshold_sd=2.0, burn_in=5)
        # The spike at index 20 should be detected
        flagged_steps = [e[0] for e in events]
        self.assertIn(20, flagged_steps)

    def test_burn_in_excludes_early_steps(self):
        """No events should be flagged within the burn-in period."""
        from simulator.predictive import detect_adpression

        # Put a spike in burn-in period
        surprisals = [8.0] + [1.0] * 20
        events = detect_adpression(surprisals, threshold_sd=2.0, burn_in=5)
        flagged_steps = [e[0] for e in events]
        self.assertNotIn(0, flagged_steps)

    def test_uniform_surprisal_no_events(self):
        """Perfectly uniform surprisal should produce no adpression events."""
        from simulator.predictive import detect_adpression

        surprisals = [1.0] * 30
        events = detect_adpression(surprisals, threshold_sd=2.0, burn_in=5)
        self.assertEqual(len(events), 0)

    def test_empty_input(self):
        """Empty surprisal list should return empty events."""
        from simulator.predictive import detect_adpression

        events = detect_adpression([], threshold_sd=2.0, burn_in=5)
        self.assertEqual(len(events), 0)


class TestDetectStateCollapse(unittest.TestCase):
    """Tests for detect_state_collapse()."""

    def test_collapse_detected(self):
        """Axis transitioning from multi-state to single-state should be flagged."""
        from simulator.predictive import detect_state_collapse

        # 15 steps with variety, then 15 steps stuck in one state
        states = (["a", "b", "c"] * 5) + (["a"] * 15)
        events = detect_state_collapse(states, "test_axis", window=10)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0].event_type, "state_collapse")

    def test_always_single_state_no_event(self):
        """An axis that was always single-state should NOT be flagged."""
        from simulator.predictive import detect_state_collapse

        states = ["a"] * 30
        events = detect_state_collapse(states, "test_axis", window=10)
        self.assertEqual(len(events), 0)

    def test_diverse_axis_no_event(self):
        """An axis maintaining diversity should produce no collapse events."""
        from simulator.predictive import detect_state_collapse

        states = (["a", "b", "c"] * 10)
        events = detect_state_collapse(states, "test_axis", window=10)
        self.assertEqual(len(events), 0)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_predictive.py::TestDetectAdpression tests/test_predictive.py::TestDetectStateCollapse -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `simulator/predictive.py`:

```python
def detect_adpression(
    surprisals: list[float],
    threshold_sd: float = 2.0,
    burn_in: int = 5,
) -> list[tuple[int, float, float]]:
    """Detect candidate adpressive events via adaptive surprisal threshold.

    Uses expanding-window running mean and standard deviation. Flags steps
    where surprisal > running_mean + threshold_sd * running_sd.

    Parameters
    ----------
    surprisals : list[float]
        Surprisal values per step.
    threshold_sd : float
        Number of standard deviations above mean for flagging (default 2.0).
    burn_in : int
        Minimum steps before flagging begins (default 5).

    Returns
    -------
    list[tuple[int, float, float]]
        List of (step_index, surprisal_value, threshold_at_step).
    """
    if len(surprisals) <= burn_in:
        return []

    events: list[tuple[int, float, float]] = []
    for i in range(burn_in, len(surprisals)):
        window = surprisals[:i]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        sd = variance ** 0.5
        threshold = mean + threshold_sd * sd
        if surprisals[i] > threshold and sd > 0:
            events.append((i, surprisals[i], threshold))
    return events


def detect_state_collapse(
    trajectory_states: list[str],
    axis: str,
    window: int = 10,
) -> list[AdpressionEvent]:
    """Detect annihilation/extinction events via state diversity collapse.

    Tracks per-axis state diversity using a sliding window. When diversity
    drops from > 1 to exactly 1 (and the axis was not always single-state),
    flags a potential annihilation/extinction adpressive event.

    An extinction is a *created* event — the dynamics produced the
    annihilation; it did not happen passively.

    Parameters
    ----------
    trajectory_states : list[str]
        Sequence of state labels for one axis.
    axis : str
        Axis name for event labeling.
    window : int
        Sliding window size for diversity measurement.

    Returns
    -------
    list[AdpressionEvent]
        Collapse events with event_type="state_collapse".
    """
    n = len(trajectory_states)
    if n < window:
        return []

    # Check if axis was always single-state
    all_states = set(trajectory_states)
    if len(all_states) <= 1:
        return []

    events: list[AdpressionEvent] = []
    prev_diversity = len(set(trajectory_states[:window]))

    for i in range(window, n):
        current_window = trajectory_states[i - window + 1 : i + 1]
        current_diversity = len(set(current_window))
        # Collapse: was diverse, now single-state
        if prev_diversity > 1 and current_diversity == 1:
            events.append(AdpressionEvent(
                step=i,
                axis=axis,
                surprisal=0.0,  # not applicable for state collapse
                threshold=0.0,
                predicted_top="(multiple)",
                actual_state=current_window[0],
                probability_of_actual=0.0,
                event_type="state_collapse",
            ))
        prev_diversity = current_diversity

    return events
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predictive.py -v`
Expected: 16 PASS (9 from Tasks 1-2 + 7 new)

**Step 5: Commit**

```bash
git add simulator/predictive.py tests/test_predictive.py
git commit -m "feat(predictive): add detect_adpression and detect_state_collapse"
```

---

### Task 4: Add run_predictive_diagnostic() orchestrator

**Files:**
- Modify: `simulator/predictive.py`
- Modify: `tests/test_predictive.py`

**Step 1: Write the failing tests**

Add to `tests/test_predictive.py`:

```python
def _make_trajectory(steps=10, with_events=True):
    """Build a synthetic trajectory for testing (same as test_taps_sensitivity.py)."""
    traj = []
    cumul = {"self": 0, "absorptive": 0, "novel": 0, "env": 0, "disint": 0,
             "types_lost": 0, "k_lost": 0.0, "deep_stasis": 0}
    for t in range(steps):
        if with_events and t == 3:
            cumul["self"] += 1
            cumul["absorptive"] += 1
        if with_events and t == 6:
            cumul["novel"] += 1
            cumul["disint"] += 1
        traj.append({
            "step": t,
            "D_total": 10 + t,
            "k_total": 50.0 + t * 5,
            "total_M": 100.0 + t * 10,
            "n_active": 8,
            "n_dormant": 2,
            "agent_k_list": [10.0] * 8,
            "convergence": 0.5,
            "texture_type": 1,
            "a_env": 3.0,
            "K_env": 2e5,
            "innovation_potential": 0.8 - t * 0.01,
            "n_self_metatheses": cumul["self"],
            "n_absorptive_cross": cumul["absorptive"],
            "n_novel_cross": cumul["novel"],
            "n_env_transitions": cumul["env"],
            "n_disintegration_redistributions": cumul["disint"],
            "n_types_lost": cumul["types_lost"],
            "k_lost": cumul["k_lost"],
            "n_deep_stasis": cumul["deep_stasis"],
            "affordance_mean": 0.6,
            "temporal_state_counts": {0: 0, 1: 2, 2: 3, 3: 3, 4: 2},
        })
    return traj


class TestRunPredictiveDiagnostic(unittest.TestCase):
    """Tests for run_predictive_diagnostic()."""

    def _compute_inputs(self, steps=50):
        """Helper to compute TAPS inputs from synthetic trajectory."""
        from simulator.taps import (
            compute_all_scores, compute_anopression, compute_rip,
            pressure_ratio,
        )
        traj = _make_trajectory(steps=steps, with_events=True)
        scores = compute_all_scores(traj, mu=0.005)
        ano_scores = compute_anopression(traj, mu=0.005)
        rip = compute_rip(traj)
        ratios = pressure_ratio(ano_scores)
        return traj, scores, ano_scores, rip, ratios

    def test_coarse_grain_selects_3_axes(self):
        """Coarse grain should analyze syntegration_phase, rip_dominance, ano_dominant."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, grain="coarse")
        self.assertEqual(len(result.axes_analyzed), 3)
        expected = {"syntegration_phase", "rip_dominance", "ano_dominant"}
        self.assertEqual(set(result.axes_analyzed), expected)

    def test_fine_grain_selects_5_axes(self):
        """Fine grain should analyze all 5 transition axes."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, grain="fine")
        self.assertEqual(len(result.axes_analyzed), 5)

    def test_result_has_correct_structure(self):
        """Result should have all expected fields with correct types."""
        from simulator.predictive import (
            run_predictive_diagnostic, PredictiveDiagnosticResult,
        )

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios)
        self.assertIsInstance(result, PredictiveDiagnosticResult)
        self.assertIsInstance(result.predictions, list)
        self.assertIsInstance(result.parallel_matching_rate, dict)
        self.assertIsInstance(result.mean_surprisal, dict)
        self.assertIsInstance(result.adpression_events, list)
        self.assertIsInstance(result.state_collapse_events, list)

    def test_parallel_matching_rate_in_range(self):
        """Parallel matching rate should be between 0.0 and 1.0 for each axis."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios)
        for axis, rate in result.parallel_matching_rate.items():
            self.assertGreaterEqual(rate, 0.0,
                f"Axis '{axis}' match rate should be >= 0")
            self.assertLessEqual(rate, 1.0,
                f"Axis '{axis}' match rate should be <= 1")

    def test_mean_surprisal_non_negative(self):
        """Mean surprisal should be >= 0 for each axis."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios)
        for axis, surp in result.mean_surprisal.items():
            self.assertGreaterEqual(surp, 0.0,
                f"Axis '{axis}' mean surprisal should be >= 0")

    def test_top_k_predictions_ordered(self):
        """Top predictions should be in descending probability order."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, top_k=3)
        for pred in result.predictions:
            probs = [p for _, p in pred.top_predictions]
            for i in range(len(probs) - 1):
                self.assertGreaterEqual(probs[i], probs[i + 1],
                    f"Step {pred.step}, axis {pred.axis}: "
                    f"top predictions not in descending order")

    def test_horizon_changes_predictions(self):
        """Different horizons should (generally) produce different predictions."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs(steps=50)
        r1 = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, horizon=1)
        r3 = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, horizon=3)
        # At least one prediction should differ (for non-degenerate axes)
        # We just verify they produce valid results with different horizons
        self.assertEqual(r1.horizon, 1)
        self.assertEqual(r3.horizon, 3)

    def test_step_count_matches_trajectory(self):
        """step_count should equal len(trajectory) - 1 (first step has no prior)."""
        from simulator.predictive import run_predictive_diagnostic

        steps = 30
        traj, scores, ano_scores, rip, ratios = self._compute_inputs(steps=steps)
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios)
        self.assertEqual(result.step_count, steps - 1)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_predictive.py::TestRunPredictiveDiagnostic -v`
Expected: FAIL with ImportError (run_predictive_diagnostic not found)

**Step 3: Write minimal implementation**

Add to `simulator/predictive.py`:

```python
from simulator.taps_sensitivity import build_transition_map, classify_step
from simulator.taps import (
    compute_all_scores,
    compute_anopression,
    compute_rip,
    pressure_ratio,
)

# Axis groups for grain levels
COARSE_AXES = ["syntegration_phase", "rip_dominance", "ano_dominant"]
FINE_AXES = COARSE_AXES + ["pressure_regime", "transvolution_dir"]


def run_predictive_diagnostic(
    trajectory: list[dict],
    scores: dict,
    ano_scores: dict,
    rip: dict,
    ratios: list[float],
    horizon: int = 1,
    grain: str = "coarse",
    top_k: int = 3,
    threshold_sd: float = 2.0,
) -> PredictiveDiagnosticResult:
    """Run full predictive orientation diagnostic on a simulation trajectory.

    Parameters
    ----------
    trajectory : list[dict]
        Simulation trajectory (list of step dicts).
    scores : dict
        Output of compute_all_scores().
    ano_scores : dict
        Output of compute_anopression().
    rip : dict
        Output of compute_rip().
    ratios : list[float]
        Output of pressure_ratio().
    horizon : int
        Steps ahead for prediction (1 = next step, k = k steps via P^k).
    grain : str
        "coarse" (3 axes) or "fine" (5 axes).
    top_k : int
        Number of top predictions to record per step per axis.
    threshold_sd : float
        Standard deviations above mean for adpression detection.

    Returns
    -------
    PredictiveDiagnosticResult
    """
    # Build transition maps
    t_maps = build_transition_map(scores, ano_scores, rip, ratios, trajectory)

    # Select axes
    axes = COARSE_AXES if grain == "coarse" else FINE_AXES
    # Filter to axes that actually exist in t_maps
    axes = [a for a in axes if a in t_maps]

    # Build row-stochastic matrices
    stochastic: dict[str, tuple[np.ndarray, list[str]]] = {}
    for axis in axes:
        data = t_maps[axis]
        states = data["states"]
        counts = data["counts"].astype(float).copy()
        n = len(states)
        row_sums = counts.sum(axis=1)
        for i in range(n):
            if row_sums[i] > 0:
                counts[i] /= row_sums[i]
            else:
                counts[i, i] = 1.0
        stochastic[axis] = (counts, states)

    # Walk through trajectory step by step
    n_steps = len(trajectory)
    predictions: list[StepPrediction] = []
    surprisals_by_axis: dict[str, list[float]] = {a: [] for a in axes}

    for step_idx in range(1, n_steps):
        for axis in axes:
            P, states = stochastic[axis]
            sequence = t_maps[axis]["sequence"]

            prev_state = sequence[step_idx - 1]
            actual_state = sequence[step_idx]

            # Predict
            pred_dist = predict_step(prev_state, P, states, horizon=horizon)

            # Top-k predictions
            sorted_preds = sorted(pred_dist.items(), key=lambda x: -x[1])
            top_preds = sorted_preds[:top_k]

            # Parallel match
            parallel_match = (top_preds[0][0] == actual_state) if top_preds else False

            # Surprisal
            surp = compute_surprisal(pred_dist, actual_state)
            surprisals_by_axis[axis].append(surp)

            predictions.append(StepPrediction(
                step=step_idx,
                axis=axis,
                predicted_distribution=pred_dist,
                top_predictions=top_preds,
                actual_state=actual_state,
                parallel_match=parallel_match,
                surprisal=surp,
            ))

    # Detect adpression events
    adpression_events: list[AdpressionEvent] = []
    for axis in axes:
        surps = surprisals_by_axis[axis]
        raw_events = detect_adpression(surps, threshold_sd=threshold_sd)
        for (sidx, sval, thresh) in raw_events:
            # sidx is index into surprisals list, which starts at step 1
            step_num = sidx + 1
            # Find the prediction for this step
            pred_dist = {}
            actual = ""
            for p in predictions:
                if p.step == step_num and p.axis == axis:
                    pred_dist = p.predicted_distribution
                    actual = p.actual_state
                    break
            top_pred = max(pred_dist, key=pred_dist.get) if pred_dist else ""
            prob_actual = pred_dist.get(actual, 0.0)
            adpression_events.append(AdpressionEvent(
                step=step_num,
                axis=axis,
                surprisal=sval,
                threshold=thresh,
                predicted_top=top_pred,
                actual_state=actual,
                probability_of_actual=prob_actual,
                event_type="transition_surprisal",
            ))

    # Detect state collapse events
    state_collapse_events: list[AdpressionEvent] = []
    for axis in axes:
        sequence = t_maps[axis]["sequence"]
        collapses = detect_state_collapse(sequence, axis)
        state_collapse_events.extend(collapses)

    # Aggregate statistics
    parallel_matching_rate: dict[str, float] = {}
    mean_surprisal: dict[str, float] = {}
    for axis in axes:
        axis_preds = [p for p in predictions if p.axis == axis]
        if axis_preds:
            matches = sum(1 for p in axis_preds if p.parallel_match)
            parallel_matching_rate[axis] = matches / len(axis_preds)
            mean_surprisal[axis] = (
                sum(p.surprisal for p in axis_preds) / len(axis_preds)
            )
        else:
            parallel_matching_rate[axis] = 0.0
            mean_surprisal[axis] = 0.0

    return PredictiveDiagnosticResult(
        predictions=predictions,
        parallel_matching_rate=parallel_matching_rate,
        mean_surprisal=mean_surprisal,
        adpression_events=adpression_events,
        state_collapse_events=state_collapse_events,
        horizon=horizon,
        grain=grain,
        step_count=n_steps - 1,
        axes_analyzed=axes,
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predictive.py -v`
Expected: 24 PASS (16 from Tasks 1-3 + 8 new)

**Step 5: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v`
Expected: 226 + 24 = 250 PASS

**Step 6: Commit**

```bash
git add simulator/predictive.py tests/test_predictive.py
git commit -m "feat(predictive): add run_predictive_diagnostic orchestrator"
```

---

### Task 5: Add diagnostic output to taps_diagnostics.py

**Files:**
- Modify: `scripts/taps_diagnostics.py:39-42` (imports)
- Modify: `scripts/taps_diagnostics.py:215-301` (print_summary)
- Modify: `scripts/taps_diagnostics.py:331-336` (main, after eigenvalue)

**Step 1: Write a manual smoke test**

We will verify by running the diagnostics script end-to-end. No new unit test for this task — it's a display function.

**Step 2: Modify imports**

At `scripts/taps_diagnostics.py:39-42`, add import:

```python
from simulator.taps_sensitivity import (
    build_transition_map,
    eigenvalue_analysis,
)
from simulator.predictive import run_predictive_diagnostic
```

**Step 3: Add print_predictive_diagnostic function**

After `print_summary()` (around line 301), add a new function:

```python
def print_predictive_diagnostic(result) -> None:
    """Print predictive orientation diagnostic to console."""
    print(f"\n  Predictive Orientation Diagnostic (horizon={result.horizon}, grain={result.grain}):")
    print(f"  {'='*60}")

    # Summary table
    print(f"\n  {'Axis':<24s} {'Match%':>8s}   {'Mean Surprisal':>15s}   {'Adpressions':>12s}")
    print(f"  {'-'*63}")
    for axis in result.axes_analyzed:
        rate = result.parallel_matching_rate.get(axis, 0.0)
        surp = result.mean_surprisal.get(axis, 0.0)
        n_adp = sum(1 for e in result.adpression_events if e.axis == axis)
        n_col = sum(1 for e in result.state_collapse_events if e.axis == axis)
        adp_str = f"{n_adp} event{'s' if n_adp != 1 else ''}"
        if n_col > 0:
            adp_str += f" + {n_col} collapse"
        print(f"  {axis:<24s} {rate*100:>7.1f}%   {surp:>12.2f} bits   {adp_str:>12s}")

    # Adpression events
    if result.adpression_events:
        print(f"\n  Adpression Events (transition surprisal):")
        for e in result.adpression_events:
            print(f"    Step {e.step:>3d}, {e.axis}: "
                  f"predicted={e.predicted_top} -> actual={e.actual_state} "
                  f"(P={e.probability_of_actual:.3f}), "
                  f"surprisal={e.surprisal:.2f} bits, threshold={e.threshold:.2f}")

    # State collapse events
    if result.state_collapse_events:
        print(f"\n  State Collapse Events (annihilation/extinction):")
        for e in result.state_collapse_events:
            print(f"    Step {e.step:>3d}, {e.axis}: "
                  f"collapsed to single state '{e.actual_state}'")

    if not result.adpression_events and not result.state_collapse_events:
        print(f"\n  No adpressive events detected.")

    print(f"\n  [exploratory -- see CLAIM_POLICY.md]")
```

**Step 4: Update main() to run predictive diagnostic**

After the eigenvalue analysis block (around line 333), add:

```python
    # Predictive orientation diagnostic
    pred_result = run_predictive_diagnostic(
        trajectory, scores, ano_scores, rip, ratios,
        horizon=1, grain="coarse",
    )
    print_predictive_diagnostic(pred_result)
```

**Step 5: Run diagnostics end-to-end**

Run: `python scripts/taps_diagnostics.py`
Expected: Full output including new "Predictive Orientation Diagnostic" section showing per-axis match rates, surprisal, and any adpression events.

**Step 6: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: 250 PASS (no regressions)

**Step 7: Commit**

```bash
git add scripts/taps_diagnostics.py
git commit -m "feat(diagnostics): add predictive orientation diagnostic output"
```

---

### Task 6: Update empirical_targets.md and final verification

**Files:**
- Modify: `docs/empirical_targets.md`

**Step 1: Add predictive diagnostic section to empirical_targets.md**

After Section 6b (Action Modality Calibration), add a new section documenting:
- The predictive orientation diagnostic as implemented
- Its relationship to the empirical validation roadmap
- Future work: online mode, entity:ensemble scaling dial, threshold calibration

**Step 2: Update assessment table**

Add a row for the predictive diagnostic in the assessment table.

**Step 3: Run full test suite one final time**

Run: `python -m pytest tests/ -v`
Expected: 250 PASS

**Step 4: Commit**

```bash
git add docs/empirical_targets.md
git commit -m "docs: document predictive orientation diagnostic in empirical targets"
```

---

## Summary

| Task | What | Tests Added | Cumulative Total |
|------|------|-------------|-----------------|
| 1 | Data structures + predict_step | 4 | 230 |
| 2 | compute_surprisal | 5 | 235 |
| 3 | detect_adpression + detect_state_collapse | 7 | 242 |
| 4 | run_predictive_diagnostic orchestrator | 8 | 250 |
| 5 | Diagnostics output | 0 (smoke test) | 250 |
| 6 | Documentation | 0 | 250 |

**Total new tests:** 24
**Total after implementation:** ~250
**New files:** `simulator/predictive.py`, `tests/test_predictive.py`
**Modified files:** `scripts/taps_diagnostics.py`, `docs/empirical_targets.md`
